import torch

# source=[1,0], target=[0,1] (parsed from filename: "target" in path)
def domain_cond_from_names(names, device, dtype=torch.float32):
    if isinstance(names, str):
        names = [names]
    rows = [[0.0, 1.0] if "target" in n else [1.0, 0.0] for n in names]
    return torch.tensor(rows, device=device, dtype=dtype)
