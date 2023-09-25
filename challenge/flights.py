from pydantic import BaseModel
from typing import List, Optional

class Flight(BaseModel):
    MES: Optional[int] = None
    TIPOVUELO: Optional[str] = None
    OPERA: Optional[str] = None

class FlightInput(BaseModel):
    flights: List[Flight]