import sys
import os
import torch
import numpy as np

# Load parameters
import common as com
from datasets.datasets import Datasets

def main():
    parser = com.get_argparse()
    param = com.yaml_load()
    flat_param = com.param_to_args_list(params=param)
    args = parser.parse_args(args=flat_param)
    args.cuda = False  # Keep on CPU

    print("Loading ToyRCCar dataset (train_loader)...")
    datasets_obj = Datasets(args.dataset).data(args)
    train_loader = datasets_obj.train_loader

    print("\nFetching first batch of training features...")
    for batch in train_loader:
        data = batch[0]
        print(f"Batch Tensor Shape: {data.shape}")
        
        # Statistics of clean batch
        mean_val = data.mean().item()
        std_val = data.std().item()
        min_val = data.min().item()
        max_val = data.max().item()
        
        print("\n--- Feature Distribution Stats (Clean) ---")
        print(f"  Mean: {mean_val:.6f}")
        print(f"  Std:  {std_val:.6f}")
        print(f"  Min:  {min_val:.6f}")
        print(f"  Max:  {max_val:.6f}")
        
        # Adding Gaussian noise
        noise_level = 0.02
        noise = torch.randn_like(data) * noise_level
        noisy_data = data + noise
        
        print("\n--- Feature Distribution Stats (Noisy, level=0.02) ---")
        print(f"  Mean: {noisy_data.mean().item():.6f}")
        print(f"  Std:  {noisy_data.std().item():.6f}")
        print(f"  Min:  {noisy_data.min().item():.6f}")
        print(f"  Max:  {noisy_data.max().item():.6f}")
        
        # Analysis
        ratio = noise_level / std_val
        print(f"\nRatio of Noise Standard Deviation (0.02) to Feature Standard Deviation ({std_val:.4f}):")
        print(f"  Ratio = {ratio * 100:.2f}%")
        
        # Variance ratio
        var_ratio = (noise_level ** 2) / (std_val ** 2)
        print(f"Ratio of Noise Variance to Feature Variance:")
        print(f"  Ratio = {var_ratio * 100:.4f}%")
        
        break

if __name__ == "__main__":
    main()
