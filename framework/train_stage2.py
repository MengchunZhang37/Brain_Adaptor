import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from config import get_config, ExperimentConfig
from brain_datasets import create_stage2_dataloader
from hierarchical_adapter_mvp_llama import SemanticAlignmentAdapter


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, brain_features, language_features, word_indices=None):
        brain_features = nn.functional.normalize(brain_features, dim=1)
        language_features = nn.functional.normalize(language_features, dim=1)
        
        logits = torch.matmul(brain_features, language_features.t()) / self.temperature
        
        if word_indices is not None:
            word_indices = word_indices.view(-1, 1)
            positive_mask = (word_indices == word_indices.t()).float()
            
            positive_mask_b2l = positive_mask / positive_mask.sum(dim=1, keepdim=True).clamp(min=1)
            positive_mask_l2b = positive_mask.t() / positive_mask.t().sum(dim=1, keepdim=True).clamp(min=1)
            
            log_softmax_b2l = nn.functional.log_softmax(logits, dim=1)
            loss_b2l = -(positive_mask_b2l * log_softmax_b2l).sum(dim=1).mean()
            
            log_softmax_l2b = nn.functional.log_softmax(logits.t(), dim=1)
            loss_l2b = -(positive_mask_l2b * log_softmax_l2b).sum(dim=1).mean()
        else:
            labels = torch.arange(logits.size(0), device=logits.device)
            loss_b2l = nn.functional.cross_entropy(logits, labels)
            loss_l2b = nn.functional.cross_entropy(logits.t(), labels)
        
        loss = (loss_b2l + loss_l2b) / 2
        return loss


class DirectAlignmentLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, brain_features, language_features):
        similarity = nn.functional.cosine_similarity(brain_features, language_features, dim=1)
        loss = 1.0 - similarity.mean()
        return loss


class SemanticConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, features_t, features_t1):
        similarity = nn.functional.cosine_similarity(features_t, features_t1, dim=1)
        loss = 1.0 - similarity.mean()
        return loss


class Stage2Trainer:
    def __init__(
        self,
        adapter: SemanticAlignmentAdapter,
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
            lr=config.stage2_training.learning_rate,
            weight_decay=config.stage2_training.weight_decay,
        )
        
        total_steps = len(train_loader) * config.stage2_training.num_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps
        )
        
        if config.stage2_training.use_contrastive:
            self.contrastive_loss_fn = ContrastiveLoss(
                temperature=config.stage2_training.temperature
            )
        
        if config.stage2_training.use_direct_alignment:
            self.alignment_loss_fn = DirectAlignmentLoss()
        
        if config.stage2_training.use_semantic_consistency:
            self.consistency_loss_fn = SemanticConsistencyLoss()
        
        self.output_dir = Path(config.data.output_root) / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.output_dir / "logs")
        
        if config.system.use_wandb and WANDB_AVAILABLE:
            self._init_wandb()
        
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.early_stopping_counter = 0
    
    def _init_wandb(self):
        run_name = self.config.system.wandb_run_name or f"{self.config.name}"
        
        wandb.init(
            project=self.config.system.wandb_project,
            entity=self.config.system.wandb_entity,
            name=run_name,
            config={
                'stage': 2,
                'name': self.config.name,
                'learning_rate': self.config.stage2_training.learning_rate,
                'batch_size': self.config.stage2_training.batch_size,
                'num_epochs': self.config.stage2_training.num_epochs,
                'contrastive_weight': self.config.stage2_training.contrastive_weight,
                'alignment_weight': self.config.stage2_training.alignment_weight,
                'consistency_weight': self.config.stage2_training.consistency_weight,
                'use_contrastive': self.config.stage2_training.use_contrastive,
                'use_direct_alignment': self.config.stage2_training.use_direct_alignment,
                'use_semantic_consistency': self.config.stage2_training.use_semantic_consistency,
            },
            dir=str(self.output_dir),
        )
        
        wandb.watch(self.adapter, log='all', log_freq=100)
    
    def train_epoch(self, epoch: int):
        self.adapter.train()
        
        total_loss = 0.0
        total_contrastive = 0.0
        total_alignment = 0.0
        total_consistency = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            brain_feature = batch['brain_feature'].to(self.device)
            word_embedding = batch['word_embedding'].to(self.device)
            word_idx = batch['word_idx'].to(self.device)
            
            semantic_feature = self.adapter(brain_feature)
            
            loss = 0.0
            losses_dict = {}
            
            if self.config.stage2_training.use_contrastive:
                contrastive_loss = self.contrastive_loss_fn(
                    semantic_feature, word_embedding, word_indices=word_idx
                )
                loss += self.config.stage2_training.contrastive_weight * contrastive_loss
                losses_dict['contrastive'] = contrastive_loss.item()
                total_contrastive += contrastive_loss.item()
            
            if self.config.stage2_training.use_direct_alignment:
                alignment_loss = self.alignment_loss_fn(semantic_feature, word_embedding)
                loss += self.config.stage2_training.alignment_weight * alignment_loss
                losses_dict['alignment'] = alignment_loss.item()
                total_alignment += alignment_loss.item()
            
            if self.config.stage2_training.use_semantic_consistency:
                next_brain_feature = batch['next_brain_feature'].to(self.device)
                next_semantic_feature = self.adapter(next_brain_feature)
                
                consistency_loss = self.consistency_loss_fn(
                    semantic_feature,
                    next_semantic_feature
                )
                loss += self.config.stage2_training.consistency_weight * consistency_loss
                losses_dict['consistency'] = consistency_loss.item()
                total_consistency += consistency_loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.adapter.parameters(),
                self.config.stage2_training.gradient_clip
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
                
                if self.config.system.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        'train/total_loss': loss.item(),
                        'train/lr': self.optimizer.param_groups[0]['lr'],
                        **{f'train/{k}_loss': v for k, v in losses_dict.items()},
                    }, step=self.global_step)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                **{k: f'{v:.4f}' for k, v in losses_dict.items()}
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_contrastive = total_contrastive / len(self.train_loader)
        avg_alignment = total_alignment / len(self.train_loader)
        avg_consistency = total_consistency / len(self.train_loader)
        
        return {
            'total_loss': avg_loss,
            'contrastive_loss': avg_contrastive,
            'alignment_loss': avg_alignment,
            'consistency_loss': avg_consistency,
        }
    
    def validate(self, epoch: int):
        self.adapter.eval()
        
        total_loss = 0.0
        total_contrastive = 0.0
        total_alignment = 0.0
        total_consistency = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                brain_feature = batch['brain_feature'].to(self.device)
                word_embedding = batch['word_embedding'].to(self.device)
                word_idx = batch['word_idx'].to(self.device)
                
                semantic_feature = self.adapter(brain_feature)
                
                loss = 0.0
                
                if self.config.stage2_training.use_contrastive:
                    contrastive_loss = self.contrastive_loss_fn(
                        semantic_feature, word_embedding, word_indices=word_idx
                    )
                    loss += self.config.stage2_training.contrastive_weight * contrastive_loss
                    total_contrastive += contrastive_loss.item()
                
                if self.config.stage2_training.use_direct_alignment:
                    alignment_loss = self.alignment_loss_fn(semantic_feature, word_embedding)
                    loss += self.config.stage2_training.alignment_weight * alignment_loss
                    total_alignment += alignment_loss.item()
                
                if self.config.stage2_training.use_semantic_consistency:
                    next_brain_feature = batch['next_brain_feature'].to(self.device)
                    next_semantic_feature = self.adapter(next_brain_feature)
                    
                    consistency_loss = self.consistency_loss_fn(
                        semantic_feature,
                        next_semantic_feature
                    )
                    loss += self.config.stage2_training.consistency_weight * consistency_loss
                    total_consistency += consistency_loss.item()
                
                total_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
                n_batches += 1
        
        avg_loss = total_loss / n_batches if n_batches > 0 else 0
        avg_contrastive = total_contrastive / n_batches if n_batches > 0 else 0
        avg_alignment = total_alignment / n_batches if n_batches > 0 else 0
        avg_consistency = total_consistency / n_batches if n_batches > 0 else 0
        
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        self.writer.add_scalar('val/contrastive', avg_contrastive, epoch)
        self.writer.add_scalar('val/alignment', avg_alignment, epoch)
        self.writer.add_scalar('val/consistency', avg_consistency, epoch)
        
        if self.config.system.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'val/loss': avg_loss,
                'val/contrastive': avg_contrastive,
                'val/alignment': avg_alignment,
                'val/consistency': avg_consistency,
                'epoch': epoch,
            }, step=self.global_step)
        
        return {
            'val_loss': avg_loss,
            'val_contrastive': avg_contrastive,
            'val_alignment': avg_alignment,
            'val_consistency': avg_consistency,
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.adapter.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.to_dict(),
            'global_step': self.global_step,
        }
        
        path = self.output_dir / f"checkpoint_epoch{epoch}.pt"
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
        
        checkpoints = sorted(self.output_dir.glob("checkpoint_epoch*.pt"))
        if len(checkpoints) > self.config.stage2_training.keep_last_n_checkpoints:
            for ckpt in checkpoints[:-self.config.stage2_training.keep_last_n_checkpoints]:
                ckpt.unlink()
    
    def train(self):
        print(f"\n{'='*70}")
        print(f"Starting Stage 2 Training: {self.config.name}")
        print(f"{'='*70}\n")
        
        for epoch in range(1, self.config.stage2_training.num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            
            print(f"\nEpoch {epoch}/{self.config.stage2_training.num_epochs}")
            print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
            print(f"    Contrastive: {train_metrics['contrastive_loss']:.4f}")
            print(f"    Alignment: {train_metrics['alignment_loss']:.4f}")
            print(f"    Consistency: {train_metrics['consistency_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"    Contrastive: {val_metrics['val_contrastive']:.4f}")
            print(f"    Alignment: {val_metrics['val_alignment']:.4f}")
            print(f"    Consistency: {val_metrics['val_consistency']:.4f}")
            
            if self.config.system.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'epoch_metrics/train_loss': train_metrics['total_loss'],
                    'epoch_metrics/train_contrastive': train_metrics['contrastive_loss'],
                    'epoch_metrics/train_alignment': train_metrics['alignment_loss'],
                    'epoch_metrics/train_consistency': train_metrics['consistency_loss'],
                    'epoch_metrics/val_loss': val_metrics['val_loss'],
                    'epoch_metrics/val_contrastive': val_metrics['val_contrastive'],
                    'epoch_metrics/val_alignment': val_metrics['val_alignment'],
                    'epoch_metrics/val_consistency': val_metrics['val_consistency'],
                    'epoch': epoch,
                }, step=self.global_step)
            
            is_best = val_metrics['val_loss'] < (self.best_val_loss - self.config.stage2_training.early_stopping_min_delta)
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
                self.best_epoch = epoch
                self.early_stopping_counter = 0
                print(f"  New best model! (val_loss: {self.best_val_loss:.4f})")
                
                self.save_checkpoint(epoch, is_best=True)
                
                if self.config.system.use_wandb and WANDB_AVAILABLE:
                    wandb.run.summary['best_val_loss'] = self.best_val_loss
                    wandb.run.summary['best_epoch'] = epoch
            else:
                if self.config.stage2_training.early_stopping:
                    self.early_stopping_counter += 1
                    if self.early_stopping_counter >= self.config.stage2_training.early_stopping_patience:
                        print(f"\nEarly stopping triggered! No improvement for {self.early_stopping_counter} epochs.")
                        print(f"  Best val loss: {self.best_val_loss:.4f} (epoch {self.best_epoch})")
                        break
            
            if epoch % self.config.stage2_training.save_every_n_epochs == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        print(f"\nTraining complete!")
        print(f"  Best val loss: {self.best_val_loss:.4f} (epoch {self.best_epoch})")
        print(f"  Best model saved to: {self.output_dir / 'best_model.pt'}")
        print(f"  Outputs saved to: {self.output_dir}")
        
        if self.config.system.use_wandb and WANDB_AVAILABLE:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Stage 2 Training")
    parser.add_argument('--config', type=str, default='stage2_full',
                       help='Config name (see config.py for available configs)')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to Podcast ECoG dataset')
    parser.add_argument('--mvpformer_checkpoint', type=str, required=True,
                       help='Path to MVPFormer checkpoint')
    parser.add_argument('--stage1_checkpoint', type=str, required=True,
                       help='Path to trained Stage 1 adapter checkpoint')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory')
    
    args = parser.parse_args()
    
    config = get_config(args.config)
    config.data.data_root = args.data_root
    config.data.output_root = args.output_dir
    config.mvpformer.checkpoint_path = args.mvpformer_checkpoint
    
    print(f"Loaded config: {args.config}")
    print(f"Split seed: {config.data.split_seed}")
    print(f"Key ablation settings:")
    print(f"  use_contrastive: {config.stage2_training.use_contrastive}")
    print(f"  use_direct_alignment: {config.stage2_training.use_direct_alignment}")
    print(f"  use_semantic_consistency: {config.stage2_training.use_semantic_consistency}")
    print(f"  alignment_weight: {config.stage2_training.alignment_weight}")
    
    config_path = Path(args.output_dir) / config.name / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config.save(config_path)
    
    print("\nLoading MVPFormer...")
    
    print("  Disabling Flash Attention (using standard attention to avoid OOM)")
    original_get_device_capability = torch.cuda.get_device_capability
    torch.cuda.get_device_capability = lambda *args, **kwargs: (7, 0)
    
    sys.path.insert(0, config.mvpformer.repo_path)
    from models.mvpformer import HMVPFormer
    
    chunk_size = config.data.chunk_size
    full_window = int(config.data.window_size * config.data.sampling_rate)
    num_chunks = full_window // chunk_size
    
    print(f"  Full window: {config.data.window_size}s × {config.data.sampling_rate}Hz = {full_window} samples")
    print(f"  Processing in {num_chunks} chunks of {chunk_size} samples each")
    
    import yaml
    yaml_config_path = Path(config.mvpformer.repo_path) / "configs" / "mvpformer_generative.yaml"
    
    print(f"  Loading config from: {yaml_config_path}")
    with open(yaml_config_path) as f:
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
    encoder_module_name, encoder_class_name = encoder_class_path.rsplit('.', 1)
    
    import importlib
    encoder_module = importlib.import_module(encoder_module_name)
    EncoderClass = getattr(encoder_module, encoder_class_name)
    
    encoder = EncoderClass(**model_args['encoder']['init_args'])
    print(f"  Encoder built: {encoder_class_path}")
    
    gpt_config_args = model_args['gpt_config']['init_args']
    from models.mvpformer import MVPFormerConfig
    gpt_config = MVPFormerConfig(**gpt_config_args)
    print(f"  GPT config built: n_embd={gpt_config.n_embd}, n_layer={gpt_config.n_layer}")
    
    head_args = model_args['head']['init_args']
    from models.mvpformer import MVPFormerHead
    head = MVPFormerHead(**head_args)
    print(f"  Head built")
    
    hmvp_args = {k: v for k, v in model_args.items() 
                 if k not in ['gpt_config', 'encoder', 'head', 'base_model']}
    
    mvpformer = HMVPFormer(
        gpt_config=gpt_config,
        encoder=encoder,
        head=head,
        **hmvp_args
    )
    
    print(f"  MVPFormer built with Standard Attention and chunk_size={chunk_size}")
    
    print(f"  Loading pretrained weights from: {args.mvpformer_checkpoint}")
    from mvpformer_utils import load_mvpformer_partial
    mvpformer, stats = load_mvpformer_partial(
        mvpformer,
        args.mvpformer_checkpoint,
        verbose=True
    )
    
    print(f"  Loaded {stats['loaded_keys']}/{stats['total_keys']} keys ({stats['loaded_keys']/stats['total_keys']*100:.1f}%)")
    
    mvpformer.eval()
    for param in mvpformer.parameters():
        param.requires_grad = False
    
    n_params = sum(p.numel() for p in mvpformer.parameters())
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  All parameters frozen")
    
    torch.cuda.get_device_capability = original_get_device_capability
    
    print("\nLoading Stage 1 adapter...")
    from hierarchical_adapter_mvp_llama import SubjectInvariantAdapter
    
    stage1_checkpoint = torch.load(args.stage1_checkpoint, map_location='cpu')
    stage1_adapter = SubjectInvariantAdapter(
        mvpformer_dim=config.adapter.mvpformer_dim,
        canonical_dim=config.adapter.canonical_dim,
        hidden_dim=config.adapter.hidden_dim,
        num_layers=config.adapter.stage1_num_layers,
        dropout=config.adapter.stage1_dropout,
    )
    stage1_adapter.load_state_dict(stage1_checkpoint['model_state_dict'])
    stage1_adapter.eval()
    for param in stage1_adapter.parameters():
        param.requires_grad = False
    
    epoch_info = stage1_checkpoint.get('epoch', 'unknown')
    print(f"Stage 1 adapter loaded from epoch {epoch_info} and frozen")
    
    print("\nCreating datasets...")
    print("  Expected linguistic files:")
    print(f"    - {{data_root}}/{config.linguistic.transcript_file}")
    print(f"    - {{data_root}}/{config.linguistic.embeddings_file}")
    print("")
    
    try:
        train_loader = create_stage2_dataloader(
            data_root=config.data.data_root,
            subjects=config.data.subjects,
            stage1_adapter=stage1_adapter,
            mvpformer_model=mvpformer,
            config=config,
            split='train',
        )
        
        val_loader = create_stage2_dataloader(
            data_root=config.data.data_root,
            subjects=config.data.subjects,
            stage1_adapter=stage1_adapter,
            mvpformer_model=mvpformer,
            config=config,
            split='val',
        )
        
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
    
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nTo prepare linguistic features, see:")
        print("   STAGE2_LINGUISTIC_FEATURES_GUIDE.md")
        return
    
    print("\nCreating Stage 2 adapter...")
    adapter = SemanticAlignmentAdapter(
        canonical_dim=config.adapter.canonical_dim,
        llama_hidden_dim=config.adapter.llama_hidden_dim,
        num_heads=config.adapter.stage2_num_heads,
        num_queries=config.adapter.stage2_num_queries,
        use_cross_attention=config.adapter.stage2_use_qformer,
        dropout=config.adapter.stage2_dropout,
    )
    
    n_params = sum(p.numel() for p in adapter.parameters())
    print(f"Adapter parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    trainer = Stage2Trainer(
        adapter=adapter,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )
    
    trainer.train()


if __name__ == "__main__":
    main()
