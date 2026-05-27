from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np

from nps_calculator import SimpleNet, calculate_nps
from traffic_controller import controller
from models import ProfileRequest, ProfileResponse, SimulateRequest, SimulateResponse

app = FastAPI(title="SynchroStream-ML")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def create_overlapping_data(num_features, num_samples, memory_samples, shift=0.0):
    base = torch.randn(memory_samples, num_features)
    base_targets = torch.randint(0, 2, (memory_samples,))

    new_x = base[:num_samples].clone() if num_samples <= memory_samples else torch.randn(num_samples, num_features)
    new_x = new_x + torch.randn_like(new_x) * shift
    new_targets = torch.randint(0, 2, (new_x.shape[0],))

    old_data = (base, base_targets)
    new_data = (new_x, new_targets)
    return old_data, new_data, base, base_targets


@app.get("/")
def root():
    return {"service": "SynchroStream-ML", "status": "ready"}


@app.post("/profile", response_model=ProfileResponse)
def profile_endpoint(req: ProfileRequest):
    model = SimpleNet(input_dim=req.num_features)

    shift = np.clip(np.random.exponential(0.3) * 0.5, 0.05, 1.5)
    old_data, new_data, _, _ = create_overlapping_data(
        req.num_features, req.num_samples, req.memory_samples, shift=shift
    )

    result = calculate_nps(model, old_data, new_data)
    strategy = controller.recommend(result["nps_score"])

    return ProfileResponse(
        nps_score=result["nps_score"],
        layer_disturbance=result["layer_disturbance"],
        layer_names=result["layer_names"],
        recommended_strategy=strategy,
    )


@app.post("/simulate", response_model=SimulateResponse)
def simulate_endpoint(req: SimulateRequest):
    metrics = controller.get_simulated_metrics(req.strategy)
    return SimulateResponse(
        strategy=metrics["strategy"],
        plasticity=metrics["plasticity"],
        stability=metrics["stability"],
        throughput=metrics["throughput"],
        safety=metrics["safety"],
        description=metrics["description"],
    )


@app.post("/profile-with-data")
def profile_with_data(
    num_features: int = 10,
    num_samples: int = 32,
    memory_samples: int = 32,
):
    return profile_endpoint(
        ProfileRequest(
            num_features=num_features,
            num_samples=num_samples,
            memory_samples=memory_samples,
        )
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
