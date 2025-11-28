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
        
        # Residual MLP with layer scale
        self.layers = nn.ModuleList()
        current_dim = mvpformer_dim
        
        for i in range(num_layers):
            if i == num_layers - 1:
                # Last layer: project to canonical space
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
            
            # Layer scale (helps training stability)
            if use_layer_scale:
                self.layers.append(LayerScale(current_dim if i < num_layers - 1 else canonical_dim))
        
        # Residual projection
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
        
        # Forward through layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if return_intermediates:
                intermediates[f'layer_{i}'] = x
        
        # Residual connection
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
            # Multi-head cross-attention 
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=canonical_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            
            # Learnable semantic queries
            self.semantic_queries = nn.Parameter(
                torch.randn(1, num_queries, canonical_dim)
            )
            
            # Aggregation after cross-attention
            self.query_aggregator = nn.Sequential(
                nn.Linear(canonical_dim * num_queries, canonical_dim),
                nn.LayerNorm(canonical_dim),
                nn.GELU(),
            )
        
        # Projection to Llama2 space
        bottleneck_dim = 2048
        self.semantic_proj = nn.Sequential(
            # First expansion
            nn.Linear(canonical_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Second expansion to Llama2 dim
            nn.Linear(bottleneck_dim, llama_hidden_dim),
            nn.LayerNorm(llama_hidden_dim),
        )
        
        # Residual projection (direct path)
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
            # Expand queries for batch
            queries = self.semantic_queries.expand(batch_size, -1, -1)
            
            # canonical_features as key/value
            canonical_seq = canonical_features.unsqueeze(1)  # [B, 1, canonical_dim]
            
            # Cross-attention
            attn_out, attn_weights = self.cross_attention(
                query=queries,  # [B, num_queries, canonical_dim]
                key=canonical_seq,
                value=canonical_seq,
            )
            
            # Flatten and aggregate queries
            attn_flat = attn_out.reshape(batch_size, -1)  # [B, num_queries * canonical_dim]
            aggregated = self.query_aggregator(attn_flat)  # [B, canonical_dim]
        else:
            aggregated = canonical_features
        
        # Project to Llama2 space
        semantic_out = self.semantic_proj(aggregated)
        
        # Residual connection
        identity = self.residual_proj(canonical_features)
        output = semantic_out + 0.1 * identity
        
        if return_attention and self.use_cross_attention:
            return output, attn_weights
        return output


class SubjectSpecificAdapter(nn.Module):
    def __init__(
        self,
        llama_hidden_dim: int = 4096,
        lora_rank: int = 64,  
        lora_alpha: int = 128,  
        prefix_length: int = 16,  
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.llama_hidden_dim = llama_hidden_dim
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / lora_rank
        
        # LoRA adaptation
        self.lora_A = nn.Linear(llama_hidden_dim, lora_rank, bias=False)
        self.lora_B = nn.Linear(lora_rank, llama_hidden_dim, bias=False)
        self.lora_dropout = nn.Dropout(dropout)
        
        # Subject-specific prefix
        self.prefix_length = prefix_length
        self.prefix = nn.Parameter(
            torch.randn(1, prefix_length, llama_hidden_dim) * 0.02
        )
        
        # Layer norm
        self.ln = nn.LayerNorm(llama_hidden_dim)
        
        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=torch.sqrt(torch.tensor(5.0)))
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(
        self, 
        semantic_features: torch.Tensor,
        return_prefix: bool = True,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
   
        # LoRA forward
        lora_out = self.lora_dropout(semantic_features)
        lora_out = self.lora_A(lora_out)
        lora_out = self.lora_B(lora_out)
        lora_out = lora_out * self.scaling
        
        # Residual connection with LoRA
        adapted = semantic_features + lora_out
        adapted = self.ln(adapted)
        
        if return_prefix:
            batch_size = semantic_features.shape[0]
            prefix = self.prefix.expand(batch_size, -1, -1)
            return adapted, prefix
        
        return adapted


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x


class HierarchicalAdapter(nn.Module):
    def __init__(
        self,
        mvpformer_dim: int = 2048,  # From MVPFormer config
        canonical_dim: int = 1024,
        llama_hidden_dim: int = 4096,
        hidden_dim: int = 2048,
        num_layers: int = 3,
        num_heads: int = 16,
        num_queries: int = 64,
        lora_rank: int = 64,
        lora_alpha: int = 128,
        prefix_length: int = 16,
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
            'lora_rank': lora_rank,
            'lora_alpha': lora_alpha,
            'prefix_length': prefix_length,
        }
        
        # Stage 1: Subject-Invariant Adapter
        self.subject_invariant_adapter = SubjectInvariantAdapter(
            mvpformer_dim=mvpformer_dim,
            canonical_dim=canonical_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        # Stage 2: Semantic Alignment Adapter
        self.semantic_alignment_adapter = SemanticAlignmentAdapter(
            canonical_dim=canonical_dim,
            llama_hidden_dim=llama_hidden_dim,
            num_heads=num_heads,
            num_queries=num_queries,
            dropout=dropout,
        )
        
        # Stage 3: Subject-Specific Adapter
        self.subject_specific_adapter = SubjectSpecificAdapter(
            llama_hidden_dim=llama_hidden_dim,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            prefix_length=prefix_length,
            dropout=dropout,
        )
    
    def forward(
        self,
        mvpformer_features: torch.Tensor,
        stage: str = "full",  # "stage1", "stage2", "full"
        return_all: bool = False,
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
    
        outputs = {}
        
        # Stage 1: To canonical brain space
        canonical_features = self.subject_invariant_adapter(mvpformer_features)
        outputs['canonical'] = canonical_features
        
        if stage == "stage1":
            return outputs if return_all else canonical_features
        
        # Stage 2: To Llama2 semantic space
        semantic_features = self.semantic_alignment_adapter(canonical_features)
        outputs['semantic'] = semantic_features
        
        if stage == "stage2":
            return outputs if return_all else semantic_features
        
        # Stage 3: Subject-specific adaptation
        adapted_features, prefix = self.subject_specific_adapter(
            semantic_features, return_prefix=True
        )
        outputs['adapted'] = adapted_features
        outputs['prefix'] = prefix
        
        if return_all:
            return outputs
        
        return adapted_features, prefix
    
    def get_trainable_params(self, stage: str) -> List[nn.Parameter]:
        if stage == "stage1":
            return list(self.subject_invariant_adapter.parameters())
        elif stage == "stage2":
            return list(self.semantic_alignment_adapter.parameters())
        elif stage == "stage3":
            return list(self.subject_specific_adapter.parameters())
        else:
            raise ValueError(f"Unknown stage: {stage}")
    
    def freeze_stage(self, stage: str):
        if stage == "stage1":
            for param in self.subject_invariant_adapter.parameters():
                param.requires_grad = False
        elif stage == "stage2":
            for param in self.semantic_alignment_adapter.parameters():
                param.requires_grad = False
        elif stage == "stage3":
            for param in self.subject_specific_adapter.parameters():
                param.requires_grad = False
    
    def get_num_params(self) -> Dict[str, int]:
        return {
            "stage1_subject_invariant": sum(
                p.numel() for p in self.subject_invariant_adapter.parameters()
            ),
            "stage2_semantic_alignment": sum(
                p.numel() for p in self.semantic_alignment_adapter.parameters()
            ),
            "stage3_subject_specific": sum(
                p.numel() for p in self.subject_specific_adapter.parameters()
            ),
        }
    
    def load_stage1(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.subject_invariant_adapter.load_state_dict(
            checkpoint['adapter_state_dict']
        )
        print(f"Loaded Stage 1 from {checkpoint_path}")
    
    def load_stage2(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.semantic_alignment_adapter.load_state_dict(
            checkpoint['semantic_adapter_state_dict']
        )
        print(f"Loaded Stage 2 from {checkpoint_path}")


# ========================================

if __name__ == "__main__":
    print("=" * 60)
    print("Hierarchical Adapter for MVPFormer + Llama2-7B")
    print("=" * 60)
    
    model = HierarchicalAdapter(
        mvpformer_dim=512,       
        canonical_dim=1024,      
        llama_hidden_dim=4096,   
        hidden_dim=2048,
        num_layers=3,
        num_heads=16,
        num_queries=64,
        lora_rank=64,
        lora_alpha=128,
        prefix_length=16,
    )
    
    print("\n#parameters:")
    print("-" * 60)
    params = model.get_num_params()
    for stage, num_params in params.items():
        print(f"{stage:30s}: {num_params:>10,} ({num_params/1e6:>6.2f}M)")
    print("-" * 60)
    total_params = sum(params.values())
    print(f"{'Total':30s}: {total_params:>10,} ({total_params/1e6:>6.2f}M)")
    print(f"\nPer-subject overhead (Stage 3): {params['stage3_subject_specific']:,} params")
    
    # forward test
    print("\n" + "=" * 60)
    print("Forward Pass")
    print("=" * 60)
    
    batch_size = 4
    mvpformer_features = torch.randn(batch_size, 512)
    
    # Stage 1 only
    print("\n[Stage 1] MVPFormer → Canonical Brain Space")
    canonical = model(mvpformer_features, stage="stage1")
    print(f"  Input shape:  {mvpformer_features.shape}")
    print(f"  Output shape: {canonical.shape}")
    
    # Stage 2
    print("\n[Stage 2] Canonical → Llama2 Semantic Space")
    semantic = model(mvpformer_features, stage="stage2")
    print(f"  Input shape:  {mvpformer_features.shape}")
    print(f"  Output shape: {semantic.shape}")
    
    # Full pipeline
    print("\n[Full Pipeline] MVPFormer → Llama2 (with prefix)")
    adapted, prefix = model(mvpformer_features, stage="full")
    print(f"  Input shape:       {mvpformer_features.shape}")
    print(f"  Adapted shape:     {adapted.shape}")
    print(f"  Prefix shape:      {prefix.shape}")
    print(f"  Prefix tokens:     {prefix.shape[1]}")
    
    print("\n[All Intermediates]")
    all_outputs = model(mvpformer_features, return_all=True)
    for name, tensor in all_outputs.items():
        print(f"  {name:20s}: {tensor.shape}")
    
    print("\n" + "=" * 60)
    print("model test pass")
    print("=" * 60)
