from dataclasses import dataclass, field
from typing import List, Optional, Dict
import yaml
import json
from pathlib import Path


@dataclass
class MVPFormerConfig:
    output_dim: int = 2048
    model_class: str = "HMVPFormer"
    checkpoint_path: Optional[str] = None
    repo_path: str = "./mvpformer"
    n_channels: int = 128
    n_positions: int = 110
    input_format: str = "batch_time_channels"
    input_format: str = "batch_channels_time"


@dataclass
class AdapterConfig:
    mvpformer_dim: int = 2048
    canonical_dim: int = 1024
    llama_hidden_dim: int = 4096
    hidden_dim: int = 2048
    stage1_num_layers: int = 3
    stage1_dropout: float = 0.1
    stage1_use_layer_scale: bool = True
    stage1_layer_scale_init: float = 1e-5
    stage1_residual_scale: float = 0.1
    stage2_num_heads: int = 16
    stage2_num_queries: int = 64
    stage2_use_qformer: bool = True
    stage2_dropout: float = 0.1
    stage3_lora_rank: int = 64
    stage3_lora_alpha: int = 128
    stage3_prefix_length: int = 16
    stage3_use_lora: bool = True
    stage3_use_prefix: bool = True


@dataclass
class Stage1TrainingConfig:
    temporal_weight: float = 1.0
    consistency_weight: float = 2.5
    infonce_weight: float = 0.0
    mmd_weight: float = 0.0
    masked_weight: float = 0.0
    use_temporal_contrastive: bool = True
    use_cross_subject_consistency: bool = True
    use_cross_subject_infonce: bool = False
    use_mmd: bool = False
    use_masked_modeling: bool = False
    use_curriculum: bool = False
    curriculum_phase1_epochs: int = 30
    temperature: float = 0.07
    num_negatives: int = 16
    mask_ratio: float = 0.15
    num_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    gradient_clip: float = 1.0
    accumulation_steps: int = 1
    save_every_n_epochs: int = 5
    keep_last_n_checkpoints: int = 3
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    use_downstream_probe: bool = True
    probe_every_n_epochs: int = 5
    probe_train_epochs: int = 1
    probe_lr: float = 1e-3
    probe_hidden_dim: int = 0


@dataclass
class Stage2TrainingConfig:
    contrastive_weight: float = 1.0
    alignment_weight: float = 2.0
    consistency_weight: float = 0.5
    use_contrastive: bool = True
    use_direct_alignment: bool = True
    use_semantic_consistency: bool = True
    temperature: float = 0.07
    num_epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_epochs: int = 2
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    gradient_clip: float = 1.0
    accumulation_steps: int = 1
    save_every_n_epochs: int = 2
    keep_last_n_checkpoints: int = 3
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001


@dataclass
class Stage3TrainingConfig:
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    min_trials_per_subject: int = 10
    max_trials_per_subject: int = 100
    optimizer: str = "adamw"
    scheduler: str = "constant"
    gradient_clip: float = 1.0


@dataclass
class DataConfig:
    data_root: str = "./dataset/Podcast/preprocessed"
    output_root: str = "./outputs"
    cache_dir: str = "./cache"
    subjects: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6, 7, 9])
    sampling_rate: int = 512
    n_channels: int = 90
    bandpass_low: float = 1.0
    bandpass_high: float = 200.0
    notch_freq: List[float] = field(default_factory=lambda: [50.0, 60.0])
    extract_high_gamma: bool = True
    high_gamma_low: float = 70.0
    high_gamma_high: float = 150.0
    window_size: float = 4.0
    stride: float = 2.0
    chunk_size: int = 512
    train_split: float = 0.75
    val_split: float = 0.25
    test_split: float = 0.0
    split_seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    use_cache: bool = True
    force_recompute: bool = False


@dataclass
class LinguisticConfig:
    transcript_file: str = "stimuli/podcast_transcript.csv"
    embeddings_file: str = "semantic/llama2_7b_lastlayer_word_embeddings.npy"
    use_word_embeddings: bool = True
    embedding_model: str = "llama2-7b"
    embedding_layer: int = -1
    embedding_dim: int = 4096
    context_length: int = 32
    use_phonetic_features: bool = False
    use_prosody_features: bool = False
    alignment_method: str = "auto"
    alignment_window: float = 0.1


@dataclass
class SystemConfig:
    device: str = "cuda"
    mixed_precision: bool = True
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    log_level: str = "INFO"
    log_to_file: bool = True
    log_every_n_steps: int = 10
    use_wandb: bool = True
    wandb_project: str = "brain-adapter"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)
    wandb_notes: Optional[str] = None
    wandb_mode: str = "online"
    debug: bool = False
    fast_dev_run: bool = False
    seed: int = 42
    deterministic: bool = True


@dataclass
class ExperimentConfig:
    name: str = "stage1_full"
    stage: int = 1
    mvpformer: MVPFormerConfig = field(default_factory=MVPFormerConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    stage1_training: Stage1TrainingConfig = field(default_factory=Stage1TrainingConfig)
    stage2_training: Stage2TrainingConfig = field(default_factory=Stage2TrainingConfig)
    stage3_training: Stage3TrainingConfig = field(default_factory=Stage3TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    linguistic: LinguisticConfig = field(default_factory=LinguisticConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    def to_dict(self) -> Dict:
        import dataclasses
        return dataclasses.asdict(self)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)


def get_config(name: str = "stage1_full") -> ExperimentConfig:
    config = ExperimentConfig(name=name)
    
    if name == "stage1_full":
        config.stage = 1
    
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
        config.stage1_training.consistency_weight = 3.0
    
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
        config.stage1_training.use_mmd = True
        config.stage1_training.mmd_weight = 2.5
        config.stage1_training.use_cross_subject_consistency = False
        config.stage1_training.consistency_weight = 0.0
    
    elif name == "stage1_temporal_only":
        config.stage = 1
        config.name = "stage1_temporal_only"
        config.stage1_training.use_temporal_contrastive = True
        config.stage1_training.temporal_weight = 1.0
        config.stage1_training.use_cross_subject_consistency = False
        config.stage1_training.consistency_weight = 0.0
        config.stage1_training.use_cross_subject_infonce = False
        config.stage1_training.infonce_weight = 0.0
    
    elif name == "stage1_infonce":
        config.stage = 1
        config.name = "stage1_infonce"
        config.stage1_training.use_temporal_contrastive = True
        config.stage1_training.temporal_weight = 1.0
        config.stage1_training.use_cross_subject_consistency = False
        config.stage1_training.consistency_weight = 0.0
        config.stage1_training.use_cross_subject_infonce = True
        config.stage1_training.infonce_weight = 1.0
    
    elif name == "stage1_infonce_curriculum":
        config.stage = 1
        config.name = "stage1_infonce_curriculum"
        config.stage1_training.use_temporal_contrastive = True
        config.stage1_training.temporal_weight = 1.0
        config.stage1_training.use_cross_subject_consistency = False
        config.stage1_training.consistency_weight = 0.0
        config.stage1_training.use_cross_subject_infonce = True
        config.stage1_training.infonce_weight = 1.0
        config.stage1_training.use_curriculum = True
        config.stage1_training.curriculum_phase1_epochs = 30
    
    elif name == "stage1_cosine_curriculum":
        config.stage = 1
        config.name = "stage1_cosine_curriculum"
        config.stage1_training.use_temporal_contrastive = True
        config.stage1_training.temporal_weight = 1.0
        config.stage1_training.use_cross_subject_consistency = True
        config.stage1_training.consistency_weight = 2.5
        config.stage1_training.use_curriculum = True
        config.stage1_training.curriculum_phase1_epochs = 30
    
    elif name == "stage1_infonce_mmd":
        config.stage = 1
        config.name = "stage1_infonce_mmd"
        config.stage1_training.use_temporal_contrastive = True
        config.stage1_training.temporal_weight = 1.0
        config.stage1_training.use_cross_subject_consistency = False
        config.stage1_training.consistency_weight = 0.0
        config.stage1_training.use_cross_subject_infonce = True
        config.stage1_training.infonce_weight = 1.0
        config.stage1_training.use_mmd = True
        config.stage1_training.mmd_weight = 0.5
    
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


def list_available_configs() -> List[str]:
    configs = [
        "stage1_full",
        "stage1_ablation_no_consistency",
        "stage1_ablation_no_temporal",
        "stage1_ablation_consistency_only",
        "stage1_ablation_low_consistency",
        "stage1_ablation_high_consistency",
        "stage1_ablation_with_masking",
        "stage1_ablation_mmd_consistency",
        "stage1_temporal_only",
        "stage1_infonce",
        "stage1_infonce_curriculum",
        "stage1_cosine_curriculum",
        "stage1_infonce_mmd",
        "stage2_full",
        "stage2_ablation_no_alignment",
        "stage2_ablation_no_contrastive",
        "stage2_ablation_alignment_only",
        "stage2_ablation_no_qformer",
        "stage2_ablation_high_alignment",
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
