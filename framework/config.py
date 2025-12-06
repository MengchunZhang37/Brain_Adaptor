from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
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
    input_format: str = "batch_channels_time"


@dataclass
class AdapterConfig:
    mvpformer_dim: int = 2048
    canonical_dim: int = 1024
    llama_hidden_dim: int = 768
    hidden_dim: int = 2048
    
    stage1_num_layers: int = 2
    stage1_dropout: float = 0.1
    stage1_use_layer_scale: bool = True
    stage1_layer_scale_init: float = 1e-5
    stage1_residual_scale: float = 0.1
    
    stage2_num_heads: int = 16
    stage2_num_queries: int = 64
    stage2_use_qformer: bool = True
    stage2_dropout: float = 0.1


@dataclass
class TimeBasedSplitConfig:
    train_start: float = 0.0
    train_end: float = 1080.0
    
    buffer1_start: float = 1080.0
    buffer1_end: float = 1140.0
    
    val_start: float = 1140.0
    val_end: float = 1440.0
    
    buffer2_start: float = 1440.0
    buffer2_end: float = 1500.0
    
    test_start: float = 1500.0
    test_end: float = 1800.0
    
    total_duration: float = 1800.0
    
    def get_split_bounds(self, split: str) -> Tuple[float, float]:
        if split == 'train':
            return (self.train_start, self.train_end)
        elif split == 'val':
            return (self.val_start, self.val_end)
        elif split == 'test':
            return (self.test_start, self.test_end)
        else:
            raise ValueError(f"Unknown split: {split}")
    
    def is_in_buffer(self, time_sec: float) -> bool:
        return (self.buffer1_start <= time_sec < self.buffer1_end or
                self.buffer2_start <= time_sec < self.buffer2_end)
    
    def get_split_for_time(self, time_sec: float) -> Optional[str]:
        if self.is_in_buffer(time_sec):
            return None
        if self.train_start <= time_sec < self.train_end:
            return 'train'
        elif self.val_start <= time_sec < self.val_end:
            return 'val'
        elif self.test_start <= time_sec < self.test_end:
            return 'test'
        return None


@dataclass
class Stage1TrainingConfig:
    temporal_weight: float = 1.0
    consistency_weight: float = 0
    infonce_weight: float = 2.5
    mmd_weight: float = 0.0
    masked_weight: float = 0.0
    
    use_temporal_contrastive: bool = True
    use_cross_subject_consistency: bool = False
    use_cross_subject_infonce: bool = True
    use_mmd: bool = False
    use_masked_modeling: bool = False
    
    use_curriculum: bool = True
    curriculum_phase1_epochs: int = 0
    
    temperature: float = 0.07
    num_negatives: int = 16
    mask_ratio: float = 0.15
    
    num_epochs: int = 200
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
    
    num_epochs: int = 100
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
    
    early_stopping: bool = False
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001


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
    
    split_config: TimeBasedSplitConfig = field(default_factory=TimeBasedSplitConfig)
    
    train_split: float = 0.6
    val_split: float = 0.2
    test_split: float = 0.2
    split_seed: int = 42
    
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    use_cache: bool = True
    force_recompute: bool = False


@dataclass
class LinguisticConfig:
    transcript_file: str = "stimuli/podcast_transcript.csv"
    embeddings_file: str = "semantic/gpt2_layer12_word_embeddings.npy"
    
    use_word_embeddings: bool = True
    embedding_model: str = "llama2-7b"
    embedding_layer: int = -1
    embedding_dim: int = 768
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
    
    elif name == "stage1_infonce_mmd":
        config.stage = 1
        config.name = "stage1_infonce_mmd"
        config.stage1_training.use_temporal_contrastive = True
        config.stage1_training.temporal_weight = 1.0
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
    
    else:
        raise ValueError(f"Unknown config name: {name}")
    
    return config


@dataclass
class SimplifiedTrainingConfig:
    alignment_weight: float = 1.0
    temporal_weight: float = 0.3
    
    num_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    gradient_clip: float = 1.0
    
    save_every_n_epochs: int = 5
    keep_last_n_checkpoints: int = 3
    
    early_stopping: bool = True
    early_stopping_patience: int = 15


@dataclass
class SimplifiedAdapterConfig:
    mvpformer_dim: int = 2048
    llama_dim: int = 768
    hidden_dim: int = 2048
    num_layers: int = 3
    dropout: float = 0.1
    use_residual: bool = True


@dataclass
class SimplifiedExperimentConfig:
    name: str = "simplified"
    
    adapter: SimplifiedAdapterConfig = field(default_factory=SimplifiedAdapterConfig)
    training: SimplifiedTrainingConfig = field(default_factory=SimplifiedTrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    linguistic: LinguisticConfig = field(default_factory=LinguisticConfig)
    mvpformer: MVPFormerConfig = field(default_factory=MVPFormerConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    def to_dict(self) -> Dict:
        import dataclasses
        return dataclasses.asdict(self)
    
    def save(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


def get_simplified_config(name: str = "simplified") -> SimplifiedExperimentConfig:
    config = SimplifiedExperimentConfig(name=name)
    
    if name == "simplified":
        pass
    
    elif name == "simplified_light":
        config.name = "simplified_light"
        config.adapter.num_layers = 2
        config.adapter.hidden_dim = 1024
    
    elif name == "simplified_deep":
        config.name = "simplified_deep"
        config.adapter.num_layers = 4
        config.adapter.hidden_dim = 2048
    
    elif name == "simplified_no_temporal":
        config.name = "simplified_no_temporal"
        config.training.temporal_weight = 0.0
    
    elif name == "simplified_high_temporal":
        config.name = "simplified_high_temporal"
        config.training.temporal_weight = 0.5
    
    else:
        raise ValueError(f"Unknown simplified config: {name}")
    
    return config


def list_available_configs() -> List[str]:
    return [
        "stage1_full",
        "stage1_ablation_no_consistency",
        "stage1_ablation_no_temporal",
        "stage1_temporal_only",
        "stage1_infonce",
        "stage1_infonce_curriculum",
        "stage1_infonce_mmd",
        "stage2_full",
        "stage2_ablation_no_alignment",
        "stage2_ablation_no_contrastive",
        "stage2_ablation_alignment_only",
        "stage2_ablation_no_qformer",
        "simplified",
        "simplified_light",
        "simplified_deep",
        "simplified_no_temporal",
        "simplified_high_temporal",
    ]


def print_split_info(config: DataConfig):
    split_config = config.split_config
    
    print("\n" + "="*60)
    print("TIME-BASED DATA SPLIT")
    print("="*60)
    print(f"Total duration: {split_config.total_duration/60:.1f} min")
    print()
    print(f"Train:  {split_config.train_start/60:.1f} - {split_config.train_end/60:.1f} min "
          f"({(split_config.train_end - split_config.train_start)/60:.1f} min)")
    print(f"Buffer: {split_config.buffer1_start/60:.1f} - {split_config.buffer1_end/60:.1f} min (discarded)")
    print(f"Val:    {split_config.val_start/60:.1f} - {split_config.val_end/60:.1f} min "
          f"({(split_config.val_end - split_config.val_start)/60:.1f} min)")
    print(f"Buffer: {split_config.buffer2_start/60:.1f} - {split_config.buffer2_end/60:.1f} min (discarded)")
    print(f"Test:   {split_config.test_start/60:.1f} - {split_config.test_end/60:.1f} min "
          f"({(split_config.test_end - split_config.test_start)/60:.1f} min)")
    print()
    print(f"All {len(config.subjects)} subjects appear in ALL splits")
    print("="*60 + "\n")


if __name__ == "__main__":
    print("Available configurations:")
    for name in list_available_configs():
        print(f"  - {name}")
    
    config = get_config("stage1_full")
    print_split_info(config.data)
