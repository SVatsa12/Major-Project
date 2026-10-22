"""
src/api/server.py
FastAPI backend for DPDP Compliance API.

Uses the final model (E5-base + 825 features) via ClauseClassifier.
Model and embedder are loaded once at startup for efficient serving.
"""
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import ClauseClassifier (single source of truth)
from src.inference.predictor import ClauseClassifier

# Global classifier instance (loaded at startup)
classifier: ClauseClassifier = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: Load model once. Shutdown: cleanup."""
    global classifier
    print("\n" + "=" * 80)
    print("DPDP COMPLIANCE API - STARTUP")
    print("=" * 80)
    classifier = ClauseClassifier()
    print("=" * 80)
    print("API READY - Model loaded and cached")
    print("=" * 80 + "\n")
    yield
    # Cleanup (if needed)
    print("[API] Shutting down...")


app = FastAPI(
    title="DPDP Statutory Compliance API",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"

class ClauseRequest(BaseModel):
    clause_text: str
    app_name: str = "Platform"  # Optional, defaults to "Platform"


class PredictionResponse(BaseModel):
    verdict: str
    confidence: float
    probabilities: Dict[str, float]
    related_rule: Dict[str, Any]


def clean_matrix_records(df: pd.DataFrame) -> list[dict]:
    """
    Normalizes statutory matrix records and ensures all possible 
    'final' column aliases exist with valid numeric float percentages.
    """
    # If the first column is unnamed or named Platform, ensure clean naming
    cols = list(df.columns)
    if cols and cols[0] != "Platform" and cols[0] != "platform":
        df = df.rename(columns={cols[0]: "Platform"})
        
    records = []
    for _, row in df.iterrows():
        rec = row.to_dict()
        
        # Extract platform name
        platform = str(rec.get("Platform") or rec.get("platform") or rec.get("app_name") or "")
        
        # Safely parse act scores
        def parse_score(val):
            if val is None or val == "":
                return 0.0
            try:
                return float(str(val).replace("%", "").strip())
            except ValueError:
                return 0.0

        dpdp_act = parse_score(rec.get("DPDP Act 2023") or rec.get("dpdp_act"))
        dpdp_rules = parse_score(rec.get("DPDP Rules 2025") or rec.get("dpdp_rules"))
        it_act = parse_score(rec.get("IT Act 2000") or rec.get("it_act"))
        
        # Calculate macro-average if missing or zero
        existing_final = (
            rec.get("Combined Final %") or 
            rec.get("Combined Final") or 
            rec.get("combined_final_pct") or 
            rec.get("final") or 
            rec.get("final_pct") or 
            rec.get("combined_statutory_pct") or 
            rec.get("effective_compliance_pct")
        )
        
        final_val = parse_score(existing_final)
        if final_val == 0.0:
            final_val = round(float(np.mean([dpdp_act, dpdp_rules, it_act])), 1)

        # Populate ALL common key aliases so the frontend can never read undefined/0
        augmented_rec = {
            "platform": platform,
            "Platform": platform,
            "app_name": platform,
            "DPDP Act 2023": dpdp_act,
            "DPDP Rules 2025": dpdp_rules,
            "IT Act 2000": it_act,
            "dpdp_act": dpdp_act,
            "dpdp_rules": dpdp_rules,
            "it_act": it_act,
            "final": final_val,
            "final_pct": final_val,
            "final_score": final_val,
            "Combined Final": final_val,
            "Combined Final %": final_val,
            "combined_final_pct": final_val,
            "combined_statutory_pct": final_val,
            "effective_compliance_pct": final_val,
        }
        records.append(augmented_rec)
        
    return records

@app.get("/api/rankings")
def get_rankings():
    # Try loading overall rankings first
    rankings_file = REPORTS_DIR / "platform_overall_rankings.csv"
    matrix_file = REPORTS_DIR / "platform_act_compliance_matrix.csv"
    
    # If the frontend is hitting /api/rankings for this specific card, return the normalized records
    if matrix_file.exists():
        df_mat = pd.read_csv(matrix_file)
        return clean_matrix_records(df_mat)
        
    if rankings_file.exists():
        df = pd.read_csv(rankings_file)
        return df.fillna("").to_dict(orient="records")

    raise HTTPException(status_code=404, detail="No ranking or matrix reports found.")

@app.get("/api/statutory-matrix")
def get_matrix():
    matrix_file = REPORTS_DIR / "platform_act_compliance_matrix.csv"
    if not matrix_file.exists():
        raise HTTPException(status_code=404, detail="Statutory matrix report not found.")
    df = pd.read_csv(matrix_file)
    return clean_matrix_records(df)

@app.get("/health")
def health_check():
    """Health check endpoint with model info."""
    if classifier is None:
        return {
            "status": "starting",
            "model_loaded": False
        }
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": classifier.model_config["model_type"],
        "feature_dimensions": classifier.feature_config["total_features"],
        "encoder": classifier.feature_config["encoder_model"],
        "active_rules": len(classifier.active_rules)
    }


@app.get("/api/metrics")
def get_metrics():
    """Return model performance metrics."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "primary_model": {
            "name": "E5-base + 825 features (Logistic Regression)",
            "accuracy": "91.66%",  # From confirmation experiments
            "macro_f1": 0.8132,
            "features": "768 E5-base + 8 actionability + 49 all-rules similarity",
            "type": "Final production model from paper"
        },
        "model_config": {
            "encoder": classifier.feature_config["encoder_model"],
            "total_features": classifier.feature_config["total_features"],
            "hyperparameters": classifier.model_config["hyperparameters"],
            "thresholds": {
                "tau_compliant": classifier.tau_compliant,
                "tau_partial": classifier.tau_partial
            }
        },
        "status": "Active Production"
    }


@app.post("/api/predict", response_model=PredictionResponse)
def predict_clause(req: ClauseRequest):
    """
    Predict compliance verdict for a policy clause.
    
    Uses the final model (E5-base + 825 features) trained on 826 clauses.
    Returns verdict, confidence, probabilities, and most related statutory rule.
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        result = classifier.predict(req.clause_text, req.app_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("\n[INFO] Starting DPDP Compliance API server...")
    print("[INFO] Model will be loaded on startup (may take ~30s for E5-base download)...")
    uvicorn.run(
        "src.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False  # Disable reload to prevent double model loading
    )