import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
import numpy as np

from config import get_config, ExperimentConfig
from brain_datasets import create_stage3_dataloader
from hierarchical_adapter_mvp_llama import SubjectSpecificAdapter


# ==================== Trainer ====================

class Stage3Trainer:
    def __init__(
        self,
        adapter: SubjectSpecificAdapter,
        train_loader,
        val_loader,
        config: ExperimentConfig,
        subject_id: int,
    ):
        self.adapter = adapter
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.subject_id = subject_id
        
        self.device = torch.device(config.system.device)
        self.adapter = self.adapter.to(self.device)
        
        # Optimizer (only for LoRA and Prefix parameters)
        trainable_params = [p for p in self.adapter.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.stage3_training.learning_rate,
            weight_decay=config.stage3_training.weight_decay,
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.output_dir = Path(config.data.output_root) / config.name / f"subject_{subject_id:02d}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.output_dir / "logs")
        
        self.global_step = 0
        self.best_val_acc = 0.0
        
        print(f"\nTrainable parameters: {sum(p.numel() for p in trainable_params):,}")
    
    def train_epoch(self, epoch: int):
        self.adapter.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f"Subject {self.subject_id:02d} - Epoch {epoch}")
        
        for batch in pbar:
            data = batch['data'].to(self.device)
            labels = batch['label'].to(self.device)
            
            logits = self.adapter(data)
            loss = self.loss_fn(logits, labels)
            
            self.optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [p for p in self.adapter.parameters() if p.requires_grad],
                self.config.stage3_training.gradient_clip
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            self.global_step += 1
            
            if self.global_step % 10 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/acc', (preds == labels).float().mean().item(), self.global_step)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{total_correct/total_samples:.3f}'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = total_correct / total_samples
        
        return {'loss': avg_loss, 'acc': avg_acc}
    
    def validate(self, epoch: int):
        """Validate"""
        self.adapter.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                data = batch['data'].to(self.device)
                labels = batch['label'].to(self.device)
                
                logits = self.adapter(data)
                loss = self.loss_fn(logits, labels)
                
                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = total_correct / total_samples
        
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        self.writer.add_scalar('val/acc', avg_acc, epoch)
        
        return {'val_loss': avg_loss, 'val_acc': avg_acc}
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'subject_id': self.subject_id,
            'model_state_dict': self.adapter.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
        }
        
        path = self.output_dir / f"checkpoint_epoch{epoch}.pt"
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
    
    def train(self):
        """Full training loop"""
        print(f"\n{'='*70}")
        print(f"Stage 3 Training - Subject {self.subject_id:02d}")
        print(f"{'='*70}\n")
        
        for epoch in range(1, self.config.stage3_training.num_epochs + 1):

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            
            print(f"\nEpoch {epoch}/{self.config.stage3_training.num_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.3f}")
            print(f"  Val   - Loss: {val_metrics['val_loss']:.4f}, Acc: {val_metrics['val_acc']:.3f}")

            is_best = val_metrics['val_acc'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['val_acc']
            
            self.save_checkpoint(epoch, is_best)
        
        print(f"\n Training complete!")
        print(f"  Best val accuracy: {self.best_val_acc:.3f}")
        print(f"  Outputs saved to: {self.output_dir}")
        
        return self.best_val_acc


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description="Stage 3 Training")
    parser.add_argument('--config', type=str, default='stage3_full',
                       help='Config name')
    parser.add_argument('--subject_id', type=int, required=True,
                       help='Subject ID to train on')
    parser.add_argument('--subject_data', type=str, required=True,
                       help='Path to subject-specific data (.npy or .npz)')
    parser.add_argument('--stage2_checkpoint', type=str, required=True,
                       help='Path to trained Stage 2 adapter checkpoint')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory')
    
    args = parser.parse_args()
    
    config = get_config(args.config)
    config.data.output_root = args.output_dir
    
    print(f"Loaded config: {args.config}")
    print(f"Subject: {args.subject_id}")
    print(f"Key settings:")
    print(f"  use_lora: {config.adapter.stage3_use_lora}")
    print(f"  use_prefix: {config.adapter.stage3_use_prefix}")
    print(f"  lora_rank: {config.adapter.stage3_lora_rank}")
    print(f"  max_trials: {config.stage3_training.max_trials_per_subject}")
    
    config_path = Path(args.output_dir) / config.name / f"subject_{args.subject_id:02d}" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)  
    config.save(config_path)
    
    print("\nLoading subject data...")
    if args.subject_data.endswith('.npy'):
        data = np.load(args.subject_data)  # [n_trials, ...]
        labels = np.arange(len(data)) % 10  # need to be replaced with actual
    elif args.subject_data.endswith('.npz'):
        data_dict = np.load(args.subject_data)
        data = data_dict['data']
        labels = data_dict['labels']
    else:
        raise ValueError("Unsupported data format. Use .npy or .npz")
    
    print(f"  Data shape: {data.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Unique labels: {len(np.unique(labels))}")
    
    train_loader = create_stage3_dataloader(
        subject_data=data,
        labels=labels,
        config=config,
        split='train',
    )
    
    val_loader = create_stage3_dataloader(
        subject_data=data,
        labels=labels,
        config=config,
        split='val',
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    

    print("\nLoading Stage 2 adapter...")
    from hierarchical_adapter_mvp_llama import SemanticAlignmentAdapter
    
    stage2_checkpoint = torch.load(args.stage2_checkpoint, map_location='cpu')
    stage2_adapter = SemanticAlignmentAdapter(
        canonical_dim=config.adapter.canonical_dim,
        semantic_dim=config.adapter.llama_hidden_dim,
        num_heads=config.adapter.stage2_num_heads,
        num_queries=config.adapter.stage2_num_queries,
        use_qformer=config.adapter.stage2_use_qformer,
    )
    stage2_adapter.load_state_dict(stage2_checkpoint['model_state_dict'])
    stage2_adapter.eval()
    for param in stage2_adapter.parameters():
        param.requires_grad = False
    
    print("Stage 2 adapter loaded and frozen")
    

    print("\nCreating Stage 3 adapter...")
    
    class DummyLLM(nn.Module):
        def __init__(self, hidden_dim, num_classes):
            super().__init__()
            self.head = nn.Linear(hidden_dim, num_classes)
        
        def forward(self, x):
            return self.head(x)
    
    num_classes = len(np.unique(labels))
    dummy_llm = DummyLLM(config.adapter.llama_hidden_dim, num_classes)
    
    adapter = SubjectSpecificAdapter(
        semantic_adapter=stage2_adapter,
        llm=dummy_llm,
        lora_rank=config.adapter.stage3_lora_rank,
        lora_alpha=config.adapter.stage3_lora_alpha,
        prefix_length=config.adapter.stage3_prefix_length,
        use_lora=config.adapter.stage3_use_lora,
        use_prefix=config.adapter.stage3_use_prefix,
    )
    
    trainer = Stage3Trainer(
        adapter=adapter,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        subject_id=args.subject_id,
    )
    
    best_acc = trainer.train()
    
    print(f"\n{'='*70}")
    print(f"Subject {args.subject_id:02d} - Final Accuracy: {best_acc:.3f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()