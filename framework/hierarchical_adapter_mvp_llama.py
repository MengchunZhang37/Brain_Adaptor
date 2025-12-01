import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple


class SubjectInvariantAdapter(nn.Module):
    def __init__(
        self,
        mvpformer_dim: int = 2048,
        canonical_dim: int = 1024,
        hidden_dim: int = 2048,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_layer_scale: bool = True,
    ):
        super().__init__()
        
        self.mvpformer_dim = mvpformer_dim
        self.canonical_dim = canonical_dim
        
        self.layers = nn.ModuleList()
        current_dim = mvpformer_dim
        
        for i in range(num_layers):
            if i == num_layers - 1:
                layer = nn.Sequential(
                    nn.Linear(current_dim, canonical_dim),
                    nn.LayerNorm(canonical_dim),
                )
            else:
                layer = nn.Sequential(
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                current_dim = hidden_dim
            
            self.layers.append(layer)
            
            if use_layer_scale:
                self.layers.append(LayerScale(current_dim if i < num_layers - 1 else canonical_dim))
        
        if mvpformer_dim != canonical_dim:
            self.residual_proj = nn.Linear(mvpformer_dim, canonical_dim)
        else:
            self.residual_proj = nn.Identity()
    
    def forward(
        self, 
        mvpformer_features: torch.Tensor,
        return_intermediates: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        intermediates = {}
        x = mvpformer_features
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if return_intermediates:
                intermediates[f'layer_{i}'] = x
        
        identity = self.residual_proj(mvpformer_features)
        output = x + 0.1 * identity
        
        if return_intermediates:
            intermediates['output'] = output
            return intermediates
        
        return output


class SemanticAlignmentAdapter(nn.Module):
    def __init__(
        self,
        canonical_dim: int = 1024,
        llama_hidden_dim: int = 4096,
        num_heads: int = 16,
        num_queries: int = 64,
        dropout: float = 0.1,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        
        self.canonical_dim = canonical_dim
        self.llama_hidden_dim = llama_hidden_dim
        self.num_queries = num_queries
        self.use_cross_attention = use_cross_attention
        
        if use_cross_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=canonical_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            
            self.semantic_queries = nn.Parameter(
                torch.randn(1, num_queries, canonical_dim)
            )
            
            self.query_aggregator = nn.Sequential(
                nn.Linear(canonical_dim * num_queries, canonical_dim),
                nn.LayerNorm(canonical_dim),
                nn.GELU(),
            )
        
        bottleneck_dim = 2048
        self.semantic_proj = nn.Sequential(
            nn.Linear(canonical_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, llama_hidden_dim),
            nn.LayerNorm(llama_hidden_dim),
        )
        
        self.residual_proj = nn.Sequential(
            nn.Linear(canonical_dim, llama_hidden_dim),
            nn.LayerNorm(llama_hidden_dim),
        )
    
    def forward(
        self, 
        canonical_features: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        batch_size = canonical_features.shape[0]
        
        if self.use_cross_attention:
            queries = self.semantic_queries.expand(batch_size, -1, -1)
            canonical_seq = canonical_features.unsqueeze(1)
            
            attn_out, attn_weights = self.cross_attention(
                query=queries,
                key=canonical_seq,
                value=canonical_seq,
            )
            
            attn_flat = attn_out.reshape(batch_size, -1)
            aggregated = self.query_aggregator(attn_flat)
        else:
            aggregated = canonical_features
        
        semantic_out = self.semantic_proj(aggregated)
        identity = self.residual_proj(canonical_features)
        output = semantic_out + 0.1 * identity
        
        if return_attention and self.use_cross_attention:
            return output, attn_weights
        return output


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x


class HierarchicalAdapter(nn.Module):
    def __init__(
        self,
        mvpformer_dim: int = 2048,
        canonical_dim: int = 1024,
        llama_hidden_dim: int = 4096,
        hidden_dim: int = 2048,
        num_layers: int = 3,
        num_heads: int = 16,
        num_queries: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.config = {
            'mvpformer_dim': mvpformer_dim,
            'canonical_dim': canonical_dim,
            'llama_hidden_dim': llama_hidden_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'num_queries': num_queries,
        }
        
        self.subject_invariant_adapter = SubjectInvariantAdapter(
            mvpformer_dim=mvpformer_dim,
            canonical_dim=canonical_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        self.semantic_alignment_adapter = SemanticAlignmentAdapter(
            canonical_dim=canonical_dim,
            llama_hidden_dim=llama_hidden_dim,
            num_heads=num_heads,
            num_queries=num_queries,
            dropout=dropout,
        )
    
    def forward(
        self,
        mvpformer_features: torch.Tensor,
        stage: str = "full",
        return_all: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        outputs = {}
        
        canonical_features = self.subject_invariant_adapter(mvpformer_features)
        outputs['canonical'] = canonical_features
        
        if stage == "stage1":
            return outputs if return_all else canonical_features
        
        semantic_features = self.semantic_alignment_adapter(canonical_features)
        outputs['semantic'] = semantic_features
        
        if return_all:
            return outputs
        
        return semantic_features
    
    def get_trainable_params(self, stage: str) -> List[nn.Parameter]:
        if stage == "stage1":
            return list(self.subject_invariant_adapter.parameters())
        elif stage == "stage2":
            return list(self.semantic_alignment_adapter.parameters())
        else:
            raise ValueError(f"Unknown stage: {stage}")
    
    def freeze_stage(self, stage: str):
        if stage == "stage1":
            for param in self.subject_invariant_adapter.parameters():
                param.requires_grad = False
        elif stage == "stage2":
            for param in self.semantic_alignment_adapter.parameters():
                param.requires_grad = False
    
    def get_num_params(self) -> Dict[str, int]:
        return {
            "stage1_subject_invariant": sum(
                p.numel() for p in self.subject_invariant_adapter.parameters()
            ),
            "stage2_semantic_alignment": sum(
                p.numel() for p in self.semantic_alignment_adapter.parameters()
            ),
        }
    
    def load_stage1(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.subject_invariant_adapter.load_state_dict(
            checkpoint['adapter_state_dict'] if 'adapter_state_dict' in checkpoint 
            else checkpoint['model_state_dict']
        )
        print(f"Loaded Stage 1 from {checkpoint_path}")
    
    def load_stage2(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.semantic_alignment_adapter.load_state_dict(
            checkpoint['semantic_adapter_state_dict'] if 'semantic_adapter_state_dict' in checkpoint
            else checkpoint['model_state_dict']
        )
        print(f"Loaded Stage 2 from {checkpoint_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Hierarchical Adapter for MVPFormer + Llama2-7B")
    print("Stage 1 + Stage 2 only (no Stage 3)")
    print("=" * 60)
    
    model = HierarchicalAdapter(
        mvpformer_dim=2048,
        canonical_dim=1024,
        llama_hidden_dim=4096,
    )
    
    print("\nParameter counts:")
    params = model.get_num_params()
    for stage, num_params in params.items():
        print(f"  {stage}: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"  Total: {sum(params.values()):,} ({sum(params.values())/1e6:.2f}M)")
    
    batch_size = 4
    x = torch.randn(batch_size, 2048)
    
    print("\nForward pass:")
    canonical = model(x, stage="stage1")
    print(f"  Stage 1: {x.shape} -> {canonical.shape}")
    
    semantic = model(x, stage="full")
    print(f"  Full: {x.shape} -> {semantic.shape}")
