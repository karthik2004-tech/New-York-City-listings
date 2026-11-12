# NYC Airbnb Room Type Predictor

Predicts whether a New York City Airbnb listing is an **entire home**, a **private room** or a **shared room**, based on its details.

**Live demo:** [your-frontend-link](https://new-york-city-listings-3.onrender.com) *(free hosting, so the first load can take about 50 seconds)*

## How it works

| | |
|---|---|
| **Input** | Latitude, longitude, borough, neighbourhood, price, minimum nights, availability, number of reviews, reviews per month, host listing count |
| **Output** | Predicted room type and the probability of each type |
| **Model** | Random Forest in a scikit-learn pipeline (imputation, skew handling, scaling, one-hot encoding) |

## Tech stack

Python, scikit-learn, pandas, FastAPI, HTML, CSS, JavaScript, Render

## Project structure

```
├── main.py               # FastAPI backend (loads the model, serves /predict)
├── Model_pipeline.pkl    # Trained model
├── requirements.txt      # Python dependencies
├── index.html            # Web interface
├── style.css
└── script.js             # Calls the API
```

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open `index.html` in your browser. If the API isn't on port 8000, change `API_BASE` at the top of `script.js`.

## API

`POST /predict` with JSON like:

```json
{
  "latitude": 40.7549,
  "longitude": -73.984,
  "price": 250,
  "minimum_nights": 3,
  "number_of_reviews": 40,
  "reviews_per_month": 1.2,
  "calculated_host_listings_count": 1,
  "availability_365": 200,
  "neighbourhood_group": "Manhattan",
  "neighbourhood": "Midtown"
}
```

Response:

```json
{
  "predicted_room_type": "Entire home/apt",
  "probability": [[0.8, 0.169, 0.032]]
}
```

The probabilities follow the order: Entire home/apt, Private room, Shared room.

## What I practiced

Exploratory data analysis, preprocessing, training a classifier, serving a model with FastAPI, building a simple web interface and deploying the project.

## Note

This is a learning project. Predictions are statistical estimates and can be wrong, especially for borderline cases such as cheap private rooms versus shared rooms.
