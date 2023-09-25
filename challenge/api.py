import fastapi
import pandas as pd
from typing import List, Optional
from challenge.model import DelayModel
# from model import DelayModel
from pydantic import BaseModel
import os

class Flight(BaseModel):
    MES: Optional[int] = None
    TIPOVUELO: Optional[str] = None
    OPERA: Optional[str] = None

class FlightInput(BaseModel):
    flights: List[Flight]

data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
csv_file_path = os.path.join(data_dir, 'data.csv')
data = pd.read_csv(filepath_or_buffer=csv_file_path, low_memory=False)

load_model = DelayModel()
features, target = load_model.preprocess(data, 'delay')
load_model.fit(features, target)

app = fastapi.FastAPI()

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def predict(data: FlightInput) -> dict:
    response = []
    flights = data.flights
    for flight in flights:
        if flight.MES < 1 or flight.MES > 12:
            raise fastapi.HTTPException(status_code=400, detail='Invalid month number')
        if flight.TIPOVUELO not in ['N', 'I']:
            raise fastapi.HTTPException(status_code=400, detail='Invalid flight type')
        panda = pd.DataFrame([flight.dict()])
        delay_prediction = load_model.predict(panda)
        response.append(int(delay_prediction[0]))
    return {"predict": response}