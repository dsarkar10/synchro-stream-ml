from pydantic import BaseModel
from typing import Optional


class ProfileRequest(BaseModel):
    num_features: int = 10
    num_samples: int = 32
    memory_samples: int = 32


class SimulateRequest(BaseModel):
    strategy: str


class ProfileResponse(BaseModel):
    nps_score: float
    layer_disturbance: list
    layer_names: list
    recommended_strategy: dict
    status: str = "ok"


class SimulateResponse(BaseModel):
    strategy: str
    plasticity: float
    stability: float
    throughput: float
    safety: str
    description: str
