import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.model import load_model

app = FastAPI(title="Train Route Analysis Web Dashboard")

# Paths configuration
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(CURRENT_DIR, "static")
REPORTS_DIR = os.path.join(CURRENT_DIR, "reports")
MODEL_PATH = os.path.join(REPORTS_DIR, "journey_time_model.joblib")

# Ensure static directories exist
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Load the trained linear regression model
try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print("Linear regression model loaded successfully.")
    else:
        # Fallback parameters if the model hasn't been serialized yet
        # OLS parameters computed on Dataset1.csv
        class DummyModel:
            coef_ = [1.03700411, 6.09257608]
            intercept_ = -30.281898748301777
            def predict(self, df):
                dist = df.iloc[0]['Total_Distance']
                stops = df.iloc[0]['Number_of_Stops']
                return [dist * self.coef_[0] + stops * self.coef_[1] + self.intercept_]
        model = DummyModel()
        print("Warning: joblib model file not found. Loaded fallback OLS parameters.")
except Exception as e:
    print(f"Error loading model: {e}")
    raise e

# Input schema for prediction
class PredictionRequest(BaseModel):
    distance: float
    stops: int

# API prediction endpoint
@app.post("/api/predict")
def predict_journey_time(request: PredictionRequest):
    try:
        dist = request.distance
        stops = request.stops
        
        # Calculate contributions
        dist_contrib = dist * model.coef_[0]
        stops_contrib = stops * model.coef_[1]
        intercept = model.intercept_
        
        # Compute final prediction
        pred_mins = dist_contrib + stops_contrib + intercept
        
        # Journey duration cannot physically be negative (adjust to min duration if so)
        if pred_mins < 1:
            pred_mins = 1.0
            
        return {
            "predicted_minutes": pred_mins,
            "distance_contribution": dist_contrib,
            "stops_contribution": stops_contrib,
            "intercept": intercept
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount visual reports static assets (plots)
if os.path.exists(REPORTS_DIR):
    app.mount("/reports", StaticFiles(directory=REPORTS_DIR), name="reports")

# Mount frontend static files
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Serve main dashboard
@app.get("/")
def get_dashboard():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Train Route Analysis Dashboard active. Place static files inside /static folder."}
