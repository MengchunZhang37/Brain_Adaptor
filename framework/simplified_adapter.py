"""
"""

import torch
import torch.nn as nn
from typing import Optional


class SimplifiedAdapter(nn.Module):
    """
    """
    
    def __init__(
        self,
        mvpformer_dim: int = 2048,
        llama_dim: int = 4096,
        hidden_dim: int = 2048,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_residual: bool = True,
    ):
        super().__init__()
        
        self.mvpformer_dim = mvpformer_dim
        self.llama_dim = llama_dim
        self.use_residual = use_residual
        
        layers = []
        current_dim = mvpformer_dim
        
        for i in range(num_layers):
            if i == num_layers - 1:
                layers.append(nn.Linear(current_dim, llama_dim))
                layers.append(nn.LayerNorm(llama_dim))
            else:
                layers.append(nn.Linear(current_dim, hidden_dim))
                layers.append(nn.LayerNorm(hidden_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
                current_dim = hidden_dim
        
        self.net = nn.Sequential(*layers)
        
        if use_residual:
            self.residual_proj = nn.Linear(mvpformer_dim, llama_dim)
            self.residual_scale = nn.Parameter(torch.tensor(0.1))
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        
        if self.use_residual:
            residual = self.residual_proj(x)
            out = out + self.residual_scale * residual
        
        return out
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class SimplifiedAdapterWithBottleneck(nn.Module):
    """
    """
    
    def __init__(
        self,
        mvpformer_dim: int = 2048,
        llama_dim: int = 4096,
        bottleneck_dim: int = 512,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(mvpformer_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, llama_dim),
            nn.LayerNorm(llama_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        out = self.decoder(z)
        return out
    
    def get_bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def create_simplified_adapter(config) -> SimplifiedAdapter:
    """
    """
    return SimplifiedAdapter(
        mvpformer_dim=config.mvpformer_dim,
        llama_dim=config.llama_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        use_residual=config.use_residual,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Simplified Adapter")
    print("=" * 60)
    
    adapter = SimplifiedAdapter()
    x = torch.randn(4, 2048)
    out = adapter(x)
    
    print(f"\nDefault adapter:")
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Params: {adapter.get_num_params():,}")
    
    adapter_bn = SimplifiedAdapterWithBottleneck(bottleneck_dim=256)
    out_bn = adapter_bn(x)
    z = adapter_bn.get_bottleneck(x)
    
    print(f"\nBottleneck adapter (dim=256):")
    print(f"  Input:      {x.shape}")
    print(f"  Bottleneck: {z.shape}")
    print(f"  Output:     {out_bn.shape}")
    print(f"  Params:     {adapter_bn.get_num_params():,}")
