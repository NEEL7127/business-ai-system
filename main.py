from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path

# Load trained models
BASE_DIR = Path(__file__).resolve().parent
risk_model = joblib.load(BASE_DIR / "risk_model.pkl")
risk_encoder = joblib.load(BASE_DIR / "risk_label_encoder.pkl")
revenue_model = joblib.load(BASE_DIR / "revenue_model.pkl")

app = FastAPI(title="Business AI System")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request schema
class BusinessInput(BaseModel):
    customers: int
    conversion_rate: float
    marketing_spend: float
    avg_purchase_value: float
    fixed_operational_cost: float
    variable_cost_per_customer: float

@app.get("/")
def health_check():
    return {"status": "Business AI API running"}

@app.post("/predict")
def predict(data: BusinessInput):

    # --- Revenue Prediction ---
    revenue_input = np.array([[
        data.customers,
        data.conversion_rate,
        data.marketing_spend,
        data.avg_purchase_value,
        data.avg_purchase_value * data.conversion_rate,  # customer_value_index
        data.fixed_operational_cost / (data.customers + 1),  # operational_burden
        data.variable_cost_per_customer / (data.avg_purchase_value + 1)  # cost_pressure
    ]])

    predicted_revenue = revenue_model.predict(revenue_input)[0]

    # --- Expense Calculation ---
    expenses = (
        data.fixed_operational_cost
        + (data.customers * data.variable_cost_per_customer)
        + data.marketing_spend
    )

    # --- Profit ---
    profit = predicted_revenue - expenses
    profit_margin = (profit / predicted_revenue) * 100 if predicted_revenue != 0 else 0

    # --- Risk Prediction ---
    risk_input = np.array([[
        data.customers,
        data.conversion_rate,
        data.marketing_spend,
        data.avg_purchase_value,
        data.fixed_operational_cost,
        data.variable_cost_per_customer,
        data.avg_purchase_value * data.conversion_rate,
        data.fixed_operational_cost / (data.customers + 1),
        data.variable_cost_per_customer / (data.avg_purchase_value + 1)
    ]])

    risk_encoded = risk_model.predict(risk_input)
    risk_label = risk_encoder.inverse_transform(risk_encoded)[0]

    return {
        "predicted_revenue": round(float(predicted_revenue), 2),
        "expenses": round(float(expenses), 2),
        "profit": round(float(profit), 2),
        "profit_margin": round(float(profit_margin), 2),
        "risk_level": risk_label
    }
