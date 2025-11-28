import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
import wandb

from config import get_config, ExperimentConfig
from brain_datasets import create_stage1_dataloader
from hierarchical_adapter_mvp_llama import SubjectInvariantAdapter


# ==================== Loss Functions ====================

class TemporalContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, anchor, positive, negatives):
        # anchor, positive: [B, D]
        # negatives: [B, N, D]
        
        anchor = nn.functional.normalize(anchor, dim=1)
        positive = nn.functional.normalize(positive, dim=1)
        negatives = nn.functional.normalize(negatives, dim=2)
        
        # Positive similarity
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature  # [B]
        
        # Negative similarities
        neg_sim = torch.bmm(negatives, anchor.unsqueeze(2)).squeeze(2) / self.temperature  # [B, N]
        
        # InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [B, 1+N]
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        loss = nn.functional.cross_entropy(logits, labels)
        return loss


class CrossSubjectConsistencyLoss(nn.Module):
    def __init__(self, method: str = "cosine"):
        super().__init__()
        self.method = method
    
    def forward(self, features):
        # features: [B, n_subjects, D]
        if self.method == "cosine":
            # Pairwise cosine similarity
            features = nn.functional.normalize(features, dim=2)
            similarity = torch.bmm(features, features.transpose(1, 2))  # [B, n_subjects, n_subjects]
            
            # Maximize similarity (minimize 1 - similarity)
            loss = 1.0 - similarity.mean()
            return loss
        
        elif self.method == "mmd":
            # Maximum Mean Discrepancy
            # Simplified implementation
            mean_feat = features.mean(dim=1, keepdim=True)  # [B, 1, D]
            diff = features - mean_feat  # [B, n_subjects, D]
            mmd = (diff ** 2).mean()
            return mmd


# ==================== Trainer ====================

class Stage1Trainer:
    """Stage 1 Trainer"""
    def __init__(
        self,
        adapter: SubjectInvariantAdapter,
        train_loader,
        val_loader,
        config: ExperimentConfig,
    ):
        self.adapter = adapter
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.device = torch.device(config.system.device)
        self.adapter = self.adapter.to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.adapter.parameters(),
            lr=config.stage1_training.learning_rate,
            weight_decay=config.stage1_training.weight_decay,
        )
        
        total_steps = len(train_loader) * config.stage1_training.num_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps
        )
        
        # Loss functions
        if config.stage1_training.use_temporal_contrastive:
            self.temporal_loss_fn = TemporalContrastiveLoss(
                temperature=config.stage1_training.temperature
            )
        
        if config.stage1_training.use_cross_subject_consistency:
            self.consistency_loss_fn = CrossSubjectConsistencyLoss(
                method=config.stage1_training.consistency_method
            )
        
        self.output_dir = Path(config.data.output_root) / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(self.output_dir / "logs")
        
        if config.system.use_wandb:
            self._init_wandb()
        
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def _init_wandb(self):

        run_name = self.config.system.wandb_run_name or self.config.name
        
        wandb.init(
            project=self.config.system.wandb_project,
            entity=self.config.system.wandb_entity,
            name=run_name,
            config=self.config.to_dict(),
            tags=self.config.system.wandb_tags,
            notes=self.config.system.wandb_notes,
            mode=self.config.system.wandb_mode,
            dir=str(self.output_dir),
        )
        
        wandb.watch(self.adapter, log='all', log_freq=100)
        
        print(f" Weights & Biases initialized")
        print(f"  Project: {self.config.system.wandb_project}")
        print(f"  Run: {run_name}")
        if wandb.run:
            print(f"  URL: {wandb.run.url}")
    
    def train_epoch(self, epoch: int):
        self.adapter.train()
        
        total_loss = 0.0
        total_temporal = 0.0
        total_consistency = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            feature = batch['feature'].to(self.device)
            temporal_positive = batch['temporal_positive'].to(self.device)
            cross_subject_group = batch['cross_subject_group'].to(self.device)
            
            canonical = self.adapter(feature)
            canonical_pos = self.adapter(temporal_positive)
            
            batch_size, n_subjects, feat_dim = cross_subject_group.shape
            cross_subject_flat = cross_subject_group.view(-1, feat_dim)
            canonical_cross = self.adapter(cross_subject_flat)
            canonical_cross = canonical_cross.view(batch_size, n_subjects, -1)
            
            loss = 0.0
            losses_dict = {}
            
            # Temporal contrastive loss
            if self.config.stage1_training.use_temporal_contrastive:
                # Sample negatives from batch
                negatives = canonical[torch.randperm(len(canonical))][:self.config.stage1_training.num_negatives]
                negatives = negatives.unsqueeze(0).expand(len(canonical), -1, -1)
                
                temporal_loss = self.temporal_loss_fn(canonical, canonical_pos, negatives)
                loss += self.config.stage1_training.temporal_weight * temporal_loss
                losses_dict['temporal'] = temporal_loss.item()
                total_temporal += temporal_loss.item()
            
            # Cross-subject consistency loss
            if self.config.stage1_training.use_cross_subject_consistency:
                consistency_loss = self.consistency_loss_fn(canonical_cross)
                loss += self.config.stage1_training.consistency_weight * consistency_loss
                losses_dict['consistency'] = consistency_loss.item()
                total_consistency += consistency_loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.adapter.parameters(),
                self.config.stage1_training.gradient_clip
            )
            
            self.optimizer.step()
            self.scheduler.step()  
            
            total_loss += loss.item()
            self.global_step += 1
            
            if self.global_step % self.config.system.log_every_n_steps == 0:
                self.writer.add_scalar('train/total_loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
                for k, v in losses_dict.items():
                    self.writer.add_scalar(f'train/{k}_loss', v, self.global_step)
                
                if self.config.system.use_wandb:
                    wandb.log({
                        'train/total_loss': loss.item(),
                        'train/lr': self.optimizer.param_groups[0]['lr'],
                        **{f'train/{k}_loss': v for k, v in losses_dict.items()},
                        'global_step': self.global_step,
                        'epoch': epoch,
                    }, step=self.global_step)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                **{k: f'{v:.4f}' for k, v in losses_dict.items()}
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_temporal = total_temporal / len(self.train_loader)
        avg_consistency = total_consistency / len(self.train_loader)
        
        return {
            'total_loss': avg_loss,
            'temporal_loss': avg_temporal,
            'consistency_loss': avg_consistency,
        }
    
    def validate(self, epoch: int):
        self.adapter.eval()
        
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                feature = batch['feature'].to(self.device)
                cross_subject_group = batch['cross_subject_group'].to(self.device)
                
                canonical = self.adapter(feature)
                
                # Cross-subject features
                batch_size, n_subjects, feat_dim = cross_subject_group.shape
                cross_subject_flat = cross_subject_group.view(-1, feat_dim)
                canonical_cross = self.adapter(cross_subject_flat)
                canonical_cross = canonical_cross.view(batch_size, n_subjects, -1)
                
                # Consistency loss (main validation metric)
                if self.config.stage1_training.use_cross_subject_consistency:
                    loss = self.consistency_loss_fn(canonical_cross)
                    total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        
        if self.config.system.use_wandb:
            wandb.log({
                'val/loss': avg_loss,
                'epoch': epoch,
            }, step=self.global_step)
        
        return {'val_loss': avg_loss}
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.adapter.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'global_step': self.global_step,
        }
        
        path = self.output_dir / f"checkpoint_epoch{epoch}.pt"
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f" Saved best model to {best_path}")
        
        checkpoints = list(self.output_dir.glob("checkpoint_epoch*.pt"))

        def _extract_epoch(path: Path) -> int:
            # "checkpoint_epoch123.pt" -> 123
            name = path.stem  # "checkpoint_epoch123"
            # 也可以更稳健用 split("epoch")[1]
            epoch_str = name.split("epoch")[-1]
            return int(epoch_str)

        checkpoints.sort(key=_extract_epoch)

        if len(checkpoints) > self.config.stage1_training.keep_last_n_checkpoints:
            for ckpt in checkpoints[:-self.config.stage1_training.keep_last_n_checkpoints]:
                ckpt.unlink()
    
    def train(self):
        print(f"\n{'='*70}")
        print(f"Starting Stage 1 Training: {self.config.name}")
        print(f"{'='*70}\n")
        
        for epoch in range(1, self.config.stage1_training.num_epochs + 1):

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            
            print(f"\nEpoch {epoch}/{self.config.stage1_training.num_epochs}")
            print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
            print(f"    Temporal: {train_metrics['temporal_loss']:.4f}")
            print(f"    Consistency: {train_metrics['consistency_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            
            if self.config.system.use_wandb:
                wandb.log({
                    'epoch_metrics/train_loss': train_metrics['total_loss'],
                    'epoch_metrics/train_temporal': train_metrics['temporal_loss'],
                    'epoch_metrics/train_consistency': train_metrics['consistency_loss'],
                    'epoch_metrics/val_loss': val_metrics['val_loss'],
                    'epoch': epoch,
                }, step=self.global_step)
            
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
                
                if self.config.system.use_wandb:
                    wandb.run.summary['best_val_loss'] = self.best_val_loss
                    wandb.run.summary['best_epoch'] = epoch
            
            if epoch % self.config.stage1_training.save_every_n_epochs == 0:
                self.save_checkpoint(epoch, is_best)
        
        print(f"\n Training complete!")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print(f"  Outputs saved to: {self.output_dir}")
        
        if self.config.system.use_wandb:
            wandb.finish()
        print(f"  Outputs saved to: {self.output_dir}")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description="Stage 1 Training")
    parser.add_argument('--config', type=str, default='stage1_full',
                       help='Config name (see config.py for available configs)')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to Podcast ECoG dataset')
    parser.add_argument('--mvpformer_checkpoint', type=str, required=True,
                       help='Path to MVPFormer checkpoint')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory')
    
    args = parser.parse_args()
    
    config = get_config(args.config)
    config.data.data_root = args.data_root
    config.data.output_root = args.output_dir
    config.mvpformer.checkpoint_path = args.mvpformer_checkpoint
    
    print(f"Loaded config: {args.config}")
    print(f"Key ablation settings:")
    print(f"  use_temporal_contrastive: {config.stage1_training.use_temporal_contrastive}")
    print(f"  use_cross_subject_consistency: {config.stage1_training.use_cross_subject_consistency}")
    print(f"  consistency_weight: {config.stage1_training.consistency_weight}")
    
    config_path = Path(args.output_dir) / config.name / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config.save(config_path)
    
    print("\nLoading MVPFormer...")
    
    # Disable Flash Attention to avoid shared memory OOM
    print("  ⚠ Disabling Flash Attention (using standard attention to avoid OOM)")
    original_get_device_capability = torch.cuda.get_device_capability
    torch.cuda.get_device_capability = lambda *args, **kwargs: (7, 0)  # Pretend GPU capability < 8
    
    sys.path.insert(0, config.mvpformer.repo_path)
    from models.mvpformer import HMVPFormer
    
    chunk_size = config.data.chunk_size
    full_window = int(config.data.window_size * config.data.sampling_rate)
    num_chunks = full_window // chunk_size
    
    print(f"  Full window: {config.data.window_size}s × {config.data.sampling_rate}Hz = {full_window} samples")
    print(f"  Processing in {num_chunks} chunks of {chunk_size} samples each")
    
    # Build MVPFormer from scratch with correct chunk_size
    import yaml
    config_path = Path(config.mvpformer.repo_path) / "configs" / "mvpformer_generative.yaml"
    
    print(f"  Loading config from: {config_path}")
    with open(config_path) as f:
        mvp_yaml_config = yaml.safe_load(f)
    

    model_args = mvp_yaml_config['model']['init_args']
    
    print(f"  Building MVPFormer with size_input={chunk_size}, n_channels=90")
    

    if 'encoder' not in model_args:
        model_args['encoder'] = {'class_path': 'models.fftencoder.WaveEncoder', 'init_args': {}}
    if 'init_args' not in model_args['encoder']:
        model_args['encoder']['init_args'] = {}
    
    model_args['encoder']['init_args']['size_input'] = chunk_size
    
    if 'gpt_config' in model_args:
        if 'init_args' in model_args['gpt_config']:
            model_args['gpt_config']['init_args']['n_channels'] = 90
            print(f"  Modified n_channels: 128 → 90 (balanced channel selection)")
    
    encoder_class_path = model_args['encoder']['class_path']
    encoder_module, encoder_class_name = encoder_class_path.rsplit('.', 1)
    
    import importlib
    encoder_module = importlib.import_module(encoder_module)
    EncoderClass = getattr(encoder_module, encoder_class_name)
    
    encoder = EncoderClass(**model_args['encoder']['init_args'])
    print(f"   Encoder built: {encoder_class_path}")
    
    gpt_config_args = model_args['gpt_config']['init_args']
    from models.mvpformer import MVPFormerConfig
    gpt_config = MVPFormerConfig(**gpt_config_args)
    print(f"   GPT config built: n_embd={gpt_config.n_embd}, n_layer={gpt_config.n_layer}")
    
    head_args = model_args['head']['init_args']
    from models.mvpformer import MVPFormerHead
    head = MVPFormerHead(**head_args)
    print(f"   Head built")
    
    hmvp_args = {k: v for k, v in model_args.items() 
                 if k not in ['gpt_config', 'encoder', 'head', 'base_model']}
    
    mvpformer = HMVPFormer(
        gpt_config=gpt_config,
        encoder=encoder,
        head=head,
        **hmvp_args
    )
    
    print(f"   MVPFormer built with Standard Attention and chunk_size={chunk_size}")
    
    print(f"  Loading pretrained weights from: {config.mvpformer.checkpoint_path}")
    from mvpformer_utils import load_mvpformer_partial
    mvpformer, stats = load_mvpformer_partial(
        mvpformer,
        config.mvpformer.checkpoint_path,
        verbose=True
    )
    
    print(f"   Loaded {stats['loaded_keys']}/{stats['total_keys']} keys ({stats['loaded_keys']/stats['total_keys']*100:.1f}%)")
    
    # Freeze all parameters
    mvpformer.eval()
    for param in mvpformer.parameters():
        param.requires_grad = False
    
    n_params = sum(p.numel() for p in mvpformer.parameters())
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  All parameters frozen")
    
    torch.cuda.get_device_capability = original_get_device_capability
    
    print("\nCreating datasets...")
    train_loader = create_stage1_dataloader(
        data_root=config.data.data_root,
        subjects=config.data.subjects,
        mvpformer_model=mvpformer,
        config=config,
        split='train',
    )
    
    val_loader = create_stage1_dataloader(
        data_root=config.data.data_root,
        subjects=config.data.subjects,
        mvpformer_model=mvpformer,
        config=config,
        split='val',
    )
    
    print(f" Train samples: {len(train_loader.dataset)}")
    print(f" Val samples: {len(val_loader.dataset)}")
    

    print("\nCreating adapter...")
    adapter = SubjectInvariantAdapter(
        mvpformer_dim=config.adapter.mvpformer_dim,
        canonical_dim=config.adapter.canonical_dim,
        hidden_dim=config.adapter.hidden_dim,
        num_layers=config.adapter.stage1_num_layers,
        dropout=config.adapter.stage1_dropout,
        use_layer_scale=config.adapter.stage1_use_layer_scale,
    )
    
    n_params = sum(p.numel() for p in adapter.parameters())
    print(f" Adapter parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    

    trainer = Stage1Trainer(
        adapter=adapter,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )
    
    trainer.train()


if __name__ == "__main__":
    main()