import torch
from torch import nn
import torch.nn.functional as F

class MemoryBank(nn.Module):
    def __init__(self, num_slots, dim):
        super(MemoryBank, self).__init__()
        self.num_slots = num_slots
        self.dim = dim
        # Initialize memory slots
        self.memory = nn.Parameter(torch.empty(num_slots, dim))
        nn.init.kaiming_uniform_(self.memory)
        self.last_attention = None

    def forward(self, z):
        # z: [batch_size, dim]
        # L2-normalize input and memory to calculate cosine similarity
        z_norm = F.normalize(z, p=2, dim=-1)
        mem_norm = F.normalize(self.memory, p=2, dim=-1)
        
        # similarity: [batch_size, num_slots]
        similarity = torch.matmul(z_norm, mem_norm.t())
        
        # Softmax over slots to get addressing weights
        attention = F.softmax(similarity, dim=-1)
        self.last_attention = attention
        
        # Reconstruct query representation from memory slots
        z_mem = torch.matmul(attention, self.memory)
        return z_mem, attention

    def get_sparsity_loss(self):
        if self.last_attention is None:
            return 0.0
        # Entropy sparsity loss: -sum(attention * log(attention + eps))
        # Minimizing this loss term directly minimizes attention entropy, encouraging 
        # sharp/peaky memory slot addressing weights.
        epsilon = 1e-12
        entropy = - torch.sum(self.last_attention * torch.log(self.last_attention + epsilon), dim=-1)
        return entropy.mean()

class AENet(nn.Module):
    def __init__(self, input_dim, block_size, use_memcae=True, num_memory_slots=64, memory_sparsity_weight=0.0):
        super(AENet, self).__init__()
        self.input_dim = input_dim
        self.cov_source = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        self.cov_target = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        
        self.use_memcae = use_memcae
        
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, 8),
            nn.BatchNorm1d(8, momentum=0.01, eps=1e-03),
            nn.ReLU(),
        )

        if self.use_memcae:
            self.memory_module = MemoryBank(num_slots=num_memory_slots, dim=8)

        self.decoder = nn.Sequential(
            nn.Linear(8, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.ReLU(),
            nn.Linear(128, self.input_dim)
        )

    def forward(self, x):
        z = self.encoder(x.view(-1, self.input_dim))
        if self.use_memcae:
            z_mem, _ = self.memory_module(z)
            return self.decoder(z_mem), z_mem
        else:
            return self.decoder(z), z

    def get_sparsity_loss(self):
        if self.use_memcae:
            return self.memory_module.get_sparsity_loss()
        return 0.0
