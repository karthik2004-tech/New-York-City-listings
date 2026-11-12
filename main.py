from pyexpat import features
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel,Field
# import joblib
import warnings
warnings.filterwarnings("ignore")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"]

)

COLUMNS = ['latitude', 'longitude', 'price', 'minimum_nights', 'number_of_reviews',
           'reviews_per_month', 'calculated_host_listings_count', 'availability_365', 'neighbourhood_group', 'neighbourhood']

model = joblib.load('Model_pipeline.pkl')

class Features(BaseModel):
    latitude: float = Field(..., ge=-90, le=90, description="Latitude of the location")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude of the location")
    price: float = Field(..., gt=0, description="Price of the property")
    minimum_nights: int = Field(..., ge=1 , le =365, description="Minimum nights required for booking")
    number_of_reviews: int = Field(..., ge=0, description="Number of reviews for the property")
    reviews_per_month: float = Field(..., ge=0, description="Average number of reviews per month")
    calculated_host_listings_count: int = Field(..., ge=0, description="Total number of listings by the host")
    availability_365: int = Field(..., ge=0, le=365, description="Number of days the property is available in a year")
    neighbourhood_group: str = Field(..., description="Neighborhood group of the property")
    neighbourhood: str = Field(..., description="Neighborhood of the property")


@app.get('/')
def greet():
    return "hello people"



@app.post('/predict')
def predict(features: Features):
    row = pd.DataFrame([features.dict()], columns =COLUMNS)
    prediction = model.predict(row)
    probability = model.predict_proba(row)
    return {
        "predicted_room_type": prediction[0],
        "probability": probability.tolist()
    }
