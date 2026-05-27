import torch
import torch.nn as nn
import numpy as np


class SimpleNet(nn.Module):
    def __init__(self, input_dim=10, h1=24, h2=16, h3=8, output_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def get_layer_names(self):
        return ["Input", "Hidden 1", "Hidden 2", "Output"]

    def get_layer_for_grad(self, grad_name):
        if grad_name.startswith("fc1"):
            return "Input"
        elif grad_name.startswith("fc2"):
            return "Hidden 1"
        elif grad_name.startswith("fc3"):
            return "Hidden 2"
        elif grad_name.startswith("fc4"):
            return "Output"
        return None


def train_step(model, data, loss_fn=nn.CrossEntropyLoss(), lr=0.01, steps=5):
    x, y = data
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()


def compute_gradients(model, data_batch, loss_fn=nn.CrossEntropyLoss()):
    model.zero_grad()
    if isinstance(data_batch, (list, tuple)):
        x, y = data_batch
    else:
        x = data_batch
        y = torch.randint(0, 2, (x.shape[0],))
    output = model(x)
    loss = loss_fn(output, y)
    loss.backward()

    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.detach().flatten()
    return grads


def cosine_similarity(vec1, vec2):
    if vec1.numel() == 0 or vec2.numel() == 0:
        return 0.0
    dot = torch.dot(vec1, vec2)
    norm1 = torch.norm(vec1)
    norm2 = torch.norm(vec2)
    if norm1.item() == 0 or norm2.item() == 0:
        return 0.0
    return (dot / (norm1 * norm2)).item()


def calculate_nps(model, old_data, new_data, loss_fn=nn.CrossEntropyLoss()):
    train_step(model, old_data, loss_fn, steps=5)

    old_grads = compute_gradients(model, old_data, loss_fn)
    old_vec = torch.cat(list(old_grads.values())) if old_grads else torch.zeros(1)

    new_grads = compute_gradients(model, new_data, loss_fn)
    new_vec = torch.cat(list(new_grads.values())) if new_grads else torch.zeros(1)

    full_sim = cosine_similarity(old_vec, new_vec)
    nps = max(0.0, min(1.0, 1.0 - full_sim))

    layer_names = model.get_layer_names()
    bucket_old = {ln: [] for ln in layer_names}
    bucket_new = {ln: [] for ln in layer_names}

    for gname, gval in old_grads.items():
        ln = model.get_layer_for_grad(gname)
        if ln:
            bucket_old[ln].append(gval)
    for gname, gval in new_grads.items():
        ln = model.get_layer_for_grad(gname)
        if ln:
            bucket_new[ln].append(gval)

    layer_disturbance = []
    for ln in layer_names:
        if bucket_old[ln] and bucket_new[ln]:
            old_cat = torch.cat(bucket_old[ln])
            new_cat = torch.cat(bucket_new[ln])
            layer_sim = cosine_similarity(old_cat, new_cat)
        else:
            layer_sim = 0.0
        layer_disturbance.append(round(max(0.0, min(1.0, 1.0 - layer_sim)), 4))

    return {
        "nps_score": round(nps, 4),
        "layer_disturbance": layer_disturbance,
        "layer_names": layer_names,
    }
