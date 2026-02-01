from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse



class CustomerData(BaseModel):
    Age: float
    Gender: str
    Tenure: float 
    Usage_Frequency: float 
    Support_Calls: float
    Subscription_Type: str
    Contract_Length: str
    Total_Spend: float
    Last_Interaction: float
    Payment_Delay: float


app = FastAPI(title="Churn Prediction System")


    

model = joblib.load('PolynomialLogisticRegressionModel.joblib')
transformation = joblib.load('DataTransformation.joblib')
ploytransformation = joblib.load('PloynomialTransformation.joblib')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()
    
@app.post("/predict")
def predict(data: CustomerData):
    input_dict = data.dict()
    
    # Access and categorize payment delay
    days = input_dict['Payment_Delay']
    if days < 2:
        delay_cat = 'On_Time'
    elif 2 <= days < 15:
        delay_cat = 'Late'
    else:
        delay_cat = 'Very_Late'

    # Create the dictionary with exact column names from training
    clean_data = {
        'Age': input_dict['Age'],
        'Gender': str(input_dict['Gender']), # Force string
        'Tenure': input_dict['Tenure'],
        'Usage Frequency': input_dict['Usage_Frequency'],
        'Support Calls': input_dict['Support_Calls'],
        'Subscription Type': str(input_dict['Subscription_Type']), # Force string
        'Contract Length': str(input_dict['Contract_Length']), # Force string
        'Total Spend': float(input_dict['Total_Spend']),
        'Last Interaction': input_dict['Last_Interaction'],
        'PaymentDelayCat': str(delay_cat) # Force string
    }

    # Create DataFrame
    input_df = pd.DataFrame([clean_data])

    # CRITICAL: Ensure categorical columns are treated as objects (strings)
    # to avoid the 'isnan' error in Scikit-Learn transformers
    categorical_cols = ['Gender', 'Subscription Type', 'Contract Length', 'PaymentDelayCat']
    for col in categorical_cols:
        input_df[col] = input_df[col].astype(object)

    try:
        # Transformation steps
        processed_data = transformation.transform(input_df)
        polyed_data = ploytransformation.transform(processed_data)
        
        prediction = model.predict(polyed_data)[0]
        probability = model.predict_proba(polyed_data)[0][1]
            
        return {
            "prediction": "Churn" if prediction == 1 else "Stay",
            "probability": f"{probability:.2%}",
            "risk_level": "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
        }
    except Exception as e:
        return {"error": str(e)}