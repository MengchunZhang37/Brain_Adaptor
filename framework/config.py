from dataclasses import dataclass, field
from typing import List, Optional, Dict
import yaml
import json
from pathlib import Path


@dataclass
class MVPFormerConfig:
    # Model architecture
    output_dim: int = 2048  # IBM: n_embd
    model_class: str = "HMVPFormer"  # models.mvpformer.HMVPFormer
    
    # Checkpoint - uses partial loading to handle shape mismatch
    checkpoint_path: Optional[str] = None
    repo_path: str = "./mvpformer"  

    # Input specs (from IBM config)
    n_channels: int = 128
    n_positions: int = 110
    input_format: str = "batch_time_channels"  # [B, T, C] - MVPFormer
    input_format: str = "batch_channels_time"  # or "batch_time_channels"


@dataclass
class AdapterConfig:
    # Dimensions
    mvpformer_dim: int = 2048  # Input from MVPFormer
    canonical_dim: int = 1024  # Stage 1 output
    llama_hidden_dim: int = 4096  # Llama2-7B
    hidden_dim: int = 2048  # MLP hidden dimension
    
    # Stage 1: Subject-Invariant Adapter
    stage1_num_layers: int = 2
    stage1_dropout: float = 0.5
    stage1_use_layer_scale: bool = True
    stage1_layer_scale_init: float = 1e-5
    stage1_residual_scale: float = 0.1
    
    # Stage 2: Semantic Alignment Adapter
    stage2_num_heads: int = 16
    stage2_num_queries: int = 64
    stage2_use_qformer: bool = True  
    stage2_dropout: float = 0.1
    
    # Stage 3: Subject-Specific Adapter
    stage3_lora_rank: int = 64
    stage3_lora_alpha: int = 128
    stage3_prefix_length: int = 16
    stage3_use_lora: bool = True  
    stage3_use_prefix: bool = True  


@dataclass
class Stage1TrainingConfig:
    # Loss weights
    temporal_weight: float = 1.0
    consistency_weight: float = 2.5  
    masked_weight: float = 0.0  
    
    # Loss functions
    use_temporal_contrastive: bool = True  
    use_cross_subject_consistency: bool = True  
    use_masked_modeling: bool = False  
    consistency_method: str = "cosine"
    
    # Contrastive learning
    temperature: float = 0.07
    num_negatives: int = 16
    
    # Masked modeling (default=false)
    mask_ratio: float = 0.15
    
    # Training hyperparameters
    num_epochs: int = 500
    batch_size: int = 64  
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"  # "cosine", "linear", "constant"
    gradient_clip: float = 1.0
    accumulation_steps: int = 1
    
    # Checkpointing
    save_every_n_epochs: int = 5
    keep_last_n_checkpoints: int = 3


@dataclass
class Stage2TrainingConfig:
    """Stage 2 Training 配置"""
    # Loss weights
    contrastive_weight: float = 1.0
    alignment_weight: float = 2.0  
    consistency_weight: float = 0.5  
    
    # Loss functions
    use_contrastive: bool = True  
    use_direct_alignment: bool = True  
    use_semantic_consistency: bool = True  
    
    # Contrastive learning
    temperature: float = 0.07
    
    # Training hyperparameters
    num_epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 3e-5  # Lower than Stage 1
    weight_decay: float = 0.01
    warmup_epochs: int = 2
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    gradient_clip: float = 1.0
    accumulation_steps: int = 1
    
    # Checkpointing
    save_every_n_epochs: int = 2
    keep_last_n_checkpoints: int = 3


@dataclass
class Stage3TrainingConfig:
    # Training hyperparameters
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    
    # Few-shot settings
    min_trials_per_subject: int = 10
    max_trials_per_subject: int = 100
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "constant"
    gradient_clip: float = 1.0


@dataclass
class DataConfig:
    # Paths
    data_root: str = "/user_data/yingjueb/ecog_pretrain/podcast"
    output_root: str = "./outputs"
    cache_dir: str = "./cache"
    
    # Podcast ECoG specific
    subjects: List[int] = field(default_factory=lambda: list(range(1, 10)))  # 9 subjects
    sampling_rate: int = 512  # Hz
    
    # Preprocessing
    bandpass_low: float = 1.0  # Hz
    bandpass_high: float = 200.0  # Hz
    notch_freq: List[float] = field(default_factory=lambda: [50.0, 60.0])  # Hz
    extract_high_gamma: bool = True
    high_gamma_low: float = 70.0  # Hz
    high_gamma_high: float = 150.0  # Hz
    
    # Windowing 
    window_size: float = 4.0  # seconds (±2s around center, total 4s)
    stride: float = 2.0  # seconds (50% overlap)
    
    # MVPFormer chunking (to avoid Flash Attention shared memory OOM)
    chunk_size: int = 128  # samples per chunk (1s @ 512Hz)
    # Full window = 2048 samples, split into 4 chunks of 512
    # Options: 512 (fast, may OOM), 256 (balanced), 128 (safe, slower)
    
    # Data splits
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # DataLoader
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    # Caching
    use_cache: bool = True
    force_recompute: bool = False


@dataclass
class LinguisticConfig:
    transcript_file: str = "/user_data/yingjueb/ecog_pretrain/stimuli/podcast_transcript.csv"
    embeddings_file: str = "/user_data/yingjueb/ecog_pretrain/semantic/llama2_7b_lastlayer_word_embeddings.npy"

    # Feature extraction
    use_word_embeddings: bool = True
    embedding_model: str = "llama2-7b"  # Llama2-7B last layer
    embedding_layer: int = -1  # Last layer
    embedding_dim: int = 4096  # Llama2-7B hidden size
    context_length: int = 32  # Context used in embedding extraction
    
    # Optional features
    use_phonetic_features: bool = False
    use_prosody_features: bool = False
    
    # Alignment
    alignment_method: str = "auto"  
    alignment_window: float = 0.1  


@dataclass
class SystemConfig:
    # Device
    device: str = "cuda"  
    mixed_precision: bool = True  
    
    # Distributed training
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_every_n_steps: int = 10
    
    # Weights & Biases
    use_wandb: bool = True  
    wandb_project: str = "brain-adapter"  
    wandb_entity: Optional[str] = None  
    wandb_run_name: Optional[str] = None  
    wandb_tags: List[str] = field(default_factory=list)  
    wandb_notes: Optional[str] = None  
    wandb_mode: str = "online"  
    
    # Debugging
    debug: bool = False
    fast_dev_run: bool = False  
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True


@dataclass
class ExperimentConfig:
    name: str = "stage1_full"
    stage: int = 1  # 1, 2, or 3
    
    # Sub-configs
    mvpformer: MVPFormerConfig = field(default_factory=MVPFormerConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    stage1_training: Stage1TrainingConfig = field(default_factory=Stage1TrainingConfig)
    stage2_training: Stage2TrainingConfig = field(default_factory=Stage2TrainingConfig)
    stage3_training: Stage3TrainingConfig = field(default_factory=Stage3TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    linguistic: LinguisticConfig = field(default_factory=LinguisticConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        import dataclasses
        return dataclasses.asdict(self)
    
    def save(self, path: str):
        """Save config to yaml file"""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: str):
        """Load config from yaml file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)


# ==================== Predefined Configs ====================

def get_config(name: str = "stage1_full") -> ExperimentConfig:
    """
    Available configs:
    - stage1_full: Stage 1 complete train
    - stage1_ablation_no_consistency: no use cross-subject consistency
    - stage1_ablation_no_temporal: no use temporal contrastive
    - stage1_ablation_consistency_only: only use consistency loss
    - stage1_ablation_low_consistency: consistency_weight=1.0
    - stage1_ablation_high_consistency: consistency_weight=5.0
    - stage2_full: Stage 2 complete train
    - stage2_ablation_no_alignment: no use direct alignment
    - stage2_ablation_no_qformer: no use Q-Former
    - stage3_full: Stage 3 complete train
    - stage3_ablation_lora_only: only use LoRA
    - stage3_ablation_prefix_only: only use Prefix
    """
    
    # Base config
    config = ExperimentConfig(name=name)
    
    # ==================== Stage 1 Ablations ====================
    
    if name == "stage1_full":
        config.stage = 1
        pass
    
    elif name == "stage1_ablation_no_consistency":
        config.stage = 1
        config.name = "stage1_ablation_no_consistency"
        config.stage1_training.use_cross_subject_consistency = False
        config.stage1_training.consistency_weight = 0.0
    
    elif name == "stage1_ablation_no_temporal":
        config.stage = 1
        config.name = "stage1_ablation_no_temporal"
        config.stage1_training.use_temporal_contrastive = False
        config.stage1_training.temporal_weight = 0.0
    
    elif name == "stage1_ablation_consistency_only":
        config.stage = 1
        config.name = "stage1_ablation_consistency_only"
        config.stage1_training.use_temporal_contrastive = False
        config.stage1_training.temporal_weight = 0.0
        config.stage1_training.consistency_weight = 3.0  # 增加weight
    
    elif name == "stage1_ablation_low_consistency":
        config.stage = 1
        config.name = "stage1_ablation_low_consistency"
        config.stage1_training.consistency_weight = 1.0
    
    elif name == "stage1_ablation_high_consistency":
        config.stage = 1
        config.name = "stage1_ablation_high_consistency"
        config.stage1_training.consistency_weight = 5.0
    
    elif name == "stage1_ablation_with_masking":
        config.stage = 1
        config.name = "stage1_ablation_with_masking"
        config.stage1_training.use_masked_modeling = True
        config.stage1_training.masked_weight = 1.0
    
    elif name == "stage1_ablation_mmd_consistency":
        config.stage = 1
        config.name = "stage1_ablation_mmd_consistency"
        config.stage1_training.consistency_method = "mmd"
    
    # ==================== Stage 2 Ablations ====================
    
    elif name == "stage2_full":
        config.stage = 2
        config.name = "stage2_full"
    
    elif name == "stage2_ablation_no_alignment":
        config.stage = 2
        config.name = "stage2_ablation_no_alignment"
        config.stage2_training.use_direct_alignment = False
        config.stage2_training.alignment_weight = 0.0
    
    elif name == "stage2_ablation_no_contrastive":
        config.stage = 2
        config.name = "stage2_ablation_no_contrastive"
        config.stage2_training.use_contrastive = False
        config.stage2_training.contrastive_weight = 0.0
    
    elif name == "stage2_ablation_alignment_only":
        config.stage = 2
        config.name = "stage2_ablation_alignment_only"
        config.stage2_training.use_contrastive = False
        config.stage2_training.contrastive_weight = 0.0
        config.stage2_training.alignment_weight = 3.0
    
    elif name == "stage2_ablation_no_qformer":
        config.stage = 2
        config.name = "stage2_ablation_no_qformer"
        config.adapter.stage2_use_qformer = False
    
    elif name == "stage2_ablation_high_alignment":
        config.stage = 2
        config.name = "stage2_ablation_high_alignment"
        config.stage2_training.alignment_weight = 5.0
    
    # ==================== Stage 3 Ablations ====================
    
    elif name == "stage3_full":
        config.stage = 3
        config.name = "stage3_full"
    
    elif name == "stage3_ablation_lora_only":
        config.stage = 3
        config.name = "stage3_ablation_lora_only"
        config.adapter.stage3_use_prefix = False
    
    elif name == "stage3_ablation_prefix_only":
        config.stage = 3
        config.name = "stage3_ablation_prefix_only"
        config.adapter.stage3_use_lora = False
    
    elif name == "stage3_ablation_no_adaptation":
        config.stage = 3
        config.name = "stage3_ablation_no_adaptation"
        config.adapter.stage3_use_lora = False
        config.adapter.stage3_use_prefix = False
    
    elif name == "stage3_ablation_large_lora":
        config.stage = 3
        config.name = "stage3_ablation_large_lora"
        config.adapter.stage3_lora_rank = 128
    
    elif name == "stage3_ablation_few_shot_10":
        config.stage = 3
        config.name = "stage3_ablation_few_shot_10"
        config.stage3_training.min_trials_per_subject = 10
        config.stage3_training.max_trials_per_subject = 10
    
    elif name == "stage3_ablation_few_shot_50":
        config.stage = 3
        config.name = "stage3_ablation_few_shot_50"
        config.stage3_training.min_trials_per_subject = 50
        config.stage3_training.max_trials_per_subject = 50
    
    else:
        raise ValueError(f"Unknown config name: {name}")
    
    return config


# ==================== Helper Functions ====================

def list_available_configs() -> List[str]:
    configs = [
        # Stage 1
        "stage1_full",
        "stage1_ablation_no_consistency",
        "stage1_ablation_no_temporal",
        "stage1_ablation_consistency_only",
        "stage1_ablation_low_consistency",
        "stage1_ablation_high_consistency",
        "stage1_ablation_with_masking",
        "stage1_ablation_mmd_consistency",
        
        # Stage 2
        "stage2_full",
        "stage2_ablation_no_alignment",
        "stage2_ablation_no_contrastive",
        "stage2_ablation_alignment_only",
        "stage2_ablation_no_qformer",
        "stage2_ablation_high_alignment",
        
        # Stage 3
        "stage3_full",
        "stage3_ablation_lora_only",
        "stage3_ablation_prefix_only",
        "stage3_ablation_no_adaptation",
        "stage3_ablation_large_lora",
        "stage3_ablation_few_shot_10",
        "stage3_ablation_few_shot_50",
    ]
    return configs


def print_config_diff(config1: ExperimentConfig, config2: ExperimentConfig):
    dict1 = config1.to_dict()
    dict2 = config2.to_dict()
    
    def flatten_dict(d, parent_key=''):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    flat1 = flatten_dict(dict1)
    flat2 = flatten_dict(dict2)
    
    print(f"\nDifferences between {config1.name} and {config2.name}:")
    print("=" * 80)
    
    all_keys = set(flat1.keys()) | set(flat2.keys())
    for key in sorted(all_keys):
        val1 = flat1.get(key, "N/A")
        val2 = flat2.get(key, "N/A")
        if val1 != val2:
            print(f"{key}:")
            print(f"  {config1.name}: {val1}")
            print(f"  {config2.name}: {val2}")


if __name__ == "__main__":
    print("Available configurations:")
    for name in list_available_configs():
        print(f"  - {name}")
    
    config = get_config("stage1_full")
    config.save("./configs/stage1_full.yaml")
    print(f"\nSaved config to ./configs/stage1_full.yaml")
    
    config1 = get_config("stage1_full")
    config2 = get_config("stage1_ablation_no_consistency")
    print_config_diff(config1, config2)