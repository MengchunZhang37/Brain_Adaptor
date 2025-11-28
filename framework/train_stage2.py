import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
from tqdm import tqdm
import sys

from config import get_config, ExperimentConfig
from brain_datasets import create_stage2_dataloader
from hierarchical_adapter_mvp_llama import SemanticAlignmentAdapter


# ==================== Loss Functions ====================

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, brain_features, language_features):
        # brain_features: [B, D_brain]
        # language_features: [B, D_lang]
        
        # Normalize
        brain_features = nn.functional.normalize(brain_features, dim=1)
        language_features = nn.functional.normalize(language_features, dim=1)
        
        # Compute similarity matrix
        logits = torch.matmul(brain_features, language_features.t()) / self.temperature  # [B, B]
        
        # Labels: diagonal elements are positives
        labels = torch.arange(logits.size(0), device=logits.device)
        
        # Symmetric loss
        loss_b2l = nn.functional.cross_entropy(logits, labels)
        loss_l2b = nn.functional.cross_entropy(logits.t(), labels)
        
        loss = (loss_b2l + loss_l2b) / 2
        return loss


class DirectAlignmentLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, brain_features, language_features):
        # brain_features: [B, D]
        # language_features: [B, D]
        
        # Cosine similarity loss
        similarity = nn.functional.cosine_similarity(brain_features, language_features, dim=1)
        loss = 1.0 - similarity.mean()
        
        return loss


class SemanticConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, features_t, features_t1):
        # features_t: [B, D] at time t
        # features_t1: [B, D] at time t+1
        
        # Features should be similar temporally
        similarity = nn.functional.cosine_similarity(features_t, features_t1, dim=1)
        loss = 1.0 - similarity.mean()
        
        return loss


# ==================== Trainer ====================

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
        
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def train_epoch(self, epoch: int):
        self.adapter.train()
        
        total_loss = 0.0
        total_contrastive = 0.0
        total_alignment = 0.0
        total_consistency = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            brain_feature = batch['brain_feature'].to(self.device)  # [B, 1024] canonical
            word_embedding = batch['word_embedding'].to(self.device)  # [B, 4096] semantic
            
            semantic_feature = self.adapter(brain_feature)  # [B, 4096]
            
            loss = 0.0
            losses_dict = {}
            
            # Contrastive loss
            if self.config.stage2_training.use_contrastive:
                contrastive_loss = self.contrastive_loss_fn(semantic_feature, word_embedding)
                loss += self.config.stage2_training.contrastive_weight * contrastive_loss
                losses_dict['contrastive'] = contrastive_loss.item()
                total_contrastive += contrastive_loss.item()
            
            # Direct alignment loss
            if self.config.stage2_training.use_direct_alignment:
                alignment_loss = self.alignment_loss_fn(semantic_feature, word_embedding)
                loss += self.config.stage2_training.alignment_weight * alignment_loss
                losses_dict['alignment'] = alignment_loss.item()
                total_alignment += alignment_loss.item()
            
            # Semantic consistency (temporal)
            if self.config.stage2_training.use_semantic_consistency and len(brain_feature) > 1:
                # Use consecutive samples in batch
                consistency_loss = self.consistency_loss_fn(
                    semantic_feature[:-1],
                    semantic_feature[1:]
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
        """Validate"""
        self.adapter.eval()
        
        total_loss = 0.0
        total_alignment = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                brain_feature = batch['brain_feature'].to(self.device)
                word_embedding = batch['word_embedding'].to(self.device)
                
                semantic_feature = self.adapter(brain_feature)
                
                # Alignment loss (main validation metric)
                if self.config.stage2_training.use_direct_alignment:
                    loss = self.alignment_loss_fn(semantic_feature, word_embedding)
                    total_loss += loss.item()
                    total_alignment += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        avg_alignment = total_alignment / len(self.val_loader)
        
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        self.writer.add_scalar('val/alignment_loss', avg_alignment, epoch)
        
        return {'val_loss': avg_loss, 'val_alignment': avg_alignment}
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save checkpoint"""
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
            print(f"✓ Saved best model to {best_path}")
        
        checkpoints = sorted(self.output_dir.glob("checkpoint_epoch*.pt"))
        if len(checkpoints) > self.config.stage2_training.keep_last_n_checkpoints:
            for ckpt in checkpoints[:-self.config.stage2_training.keep_last_n_checkpoints]:
                ckpt.unlink()
    
    def train(self):
        """Full training loop"""
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
            print(f"    Alignment: {val_metrics['val_alignment']:.4f}")
            
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
            
            if epoch % self.config.stage2_training.save_every_n_epochs == 0:
                self.save_checkpoint(epoch, is_best)
        
        print(f"\n✓ Training complete!")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print(f"  Outputs saved to: {self.output_dir}")


# ==================== Main ====================

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
    print(f"Key ablation settings:")
    print(f"  use_contrastive: {config.stage2_training.use_contrastive}")
    print(f"  use_direct_alignment: {config.stage2_training.use_direct_alignment}")
    print(f"  use_semantic_consistency: {config.stage2_training.use_semantic_consistency}")
    print(f"  alignment_weight: {config.stage2_training.alignment_weight}")
    
    config_path = Path(args.output_dir) / config.name / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)  
    config.save(config_path)
    

    print("\nLoading MVPFormer...")
    sys.path.insert(0, config.mvpformer.repo_path)
    
    from test_data_loading import build_mvpformer_from_yaml
    from mvpformer_utils import load_mvpformer_partial
    
    size_input = int(config.data.window_size * config.data.sampling_rate)
    
    mvpformer = build_mvpformer_from_yaml(
        repo_path=config.mvpformer.repo_path,
        size_input=size_input
    )
    
    mvpformer, stats = load_mvpformer_partial(
        mvpformer,
        config.mvpformer.checkpoint_path,
        verbose=True
    )
    
    mvpformer.eval()
    for param in mvpformer.parameters():
        param.requires_grad = False
    
    print(f"MVPFormer loaded: {stats['loaded_keys']}/{stats['total_keys']} keys")
    

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
    
    print("Stage 1 adapter loaded and frozen")
    

    print("\nCreating datasets...")
    print("  Warning: Stage 2 requires linguistic features!")
    print("  Expected files:")
    print("    - {data_root}/derivatives/linguistic_features/word_onsets.tsv")
    print("    - {data_root}/derivatives/linguistic_features/word_embeddings.npy")
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
        print(f"\n Error: {e}")
        print("\n To prepare linguistic features, see:")
        print("   STAGE2_LINGUISTIC_FEATURES_GUIDE.md")
        return
    

    print("\nCreating Stage 2 adapter...")
    adapter = SemanticAlignmentAdapter(
        canonical_dim=config.adapter.canonical_dim,
        semantic_dim=config.adapter.llama_hidden_dim,
        num_heads=config.adapter.stage2_num_heads,
        num_queries=config.adapter.stage2_num_queries,
        use_qformer=config.adapter.stage2_use_qformer,
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