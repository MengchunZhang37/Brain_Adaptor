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


class TemporalContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, anchor, positive, negatives):
        anchor = nn.functional.normalize(anchor, dim=1)
        positive = nn.functional.normalize(positive, dim=1)
        negatives = nn.functional.normalize(negatives, dim=2)
        
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature
        
        neg_sim = torch.bmm(negatives, anchor.unsqueeze(2)).squeeze(2) / self.temperature
        
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        loss = nn.functional.cross_entropy(logits, labels)
        return loss


class CrossSubjectConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, features):
        n_subjects = features.shape[1]
        
        if n_subjects < 2:
            return torch.tensor(0.0, device=features.device)
        
        features = nn.functional.normalize(features, dim=2)
        similarity = torch.bmm(features, features.transpose(1, 2))
        
        mask = 1.0 - torch.eye(n_subjects, device=similarity.device)
        mask = mask.unsqueeze(0)
        
        n_off_diag = mask.sum()
        if n_off_diag == 0:
            return torch.tensor(0.0, device=features.device)
        
        masked_sim = similarity * mask
        loss = 1.0 - masked_sim.sum() / (n_off_diag * similarity.shape[0])
        return loss


class CrossSubjectInfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, time_indices):
        batch_size = features.shape[0]
        device = features.device
        
        if batch_size < 2:
            return torch.tensor(0.0, device=device)
        
        features = nn.functional.normalize(features, dim=1)
        
        similarity = torch.matmul(features, features.t()) / self.temperature
        
        time_indices = time_indices.view(-1, 1)
        positive_mask = (time_indices == time_indices.t()).float()
        
        positive_mask = positive_mask - torch.eye(batch_size, device=device)
        
        if positive_mask.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        mask_self = 1.0 - torch.eye(batch_size, device=device)
        similarity = similarity * mask_self - (1 - mask_self) * 1e9
        
        log_sum_exp = torch.logsumexp(similarity, dim=1)
        
        total_loss = 0.0
        n_valid_anchors = 0
        
        for i in range(batch_size):
            pos_indices = torch.where(positive_mask[i] > 0)[0]
            if len(pos_indices) == 0:
                continue
            
            pos_similarities = similarity[i, pos_indices]
            anchor_loss = -pos_similarities + log_sum_exp[i]
            total_loss += anchor_loss.mean()
            n_valid_anchors += 1
        
        if n_valid_anchors == 0:
            return torch.tensor(0.0, device=device)
        
        return total_loss / n_valid_anchors


class MMDLoss(nn.Module):
    def __init__(self, kernel: str = "rbf", bandwidth: float = 1.0):
        super().__init__()
        self.kernel = kernel
        self.bandwidth = bandwidth
    
    def forward(self, features):
        n_subjects = features.shape[1]
        
        if n_subjects < 2:
            return torch.tensor(0.0, device=features.device)
        
        total_mmd = 0.0
        n_pairs = 0
        
        for i in range(n_subjects):
            for j in range(i + 1, n_subjects):
                mmd = self._compute_mmd(features[:, i, :], features[:, j, :])
                total_mmd += mmd
                n_pairs += 1
        
        if n_pairs == 0:
            return torch.tensor(0.0, device=features.device)
        
        return total_mmd / n_pairs
    
    def _compute_mmd(self, x, y):
        if self.kernel == "rbf":
            xx = self._rbf_kernel(x, x)
            yy = self._rbf_kernel(y, y)
            xy = self._rbf_kernel(x, y)
            
            mmd = xx.mean() + yy.mean() - 2 * xy.mean()
        else:
            xx = torch.matmul(x, x.t()).mean()
            yy = torch.matmul(y, y.t()).mean()
            xy = torch.matmul(x, y.t()).mean()
            
            mmd = xx + yy - 2 * xy
        
        return mmd
    
    def _rbf_kernel(self, x, y):
        x_norm = (x ** 2).sum(dim=1, keepdim=True)
        y_norm = (y ** 2).sum(dim=1, keepdim=True)
        
        dist = x_norm + y_norm.t() - 2 * torch.matmul(x, y.t())
        kernel = torch.exp(-dist / (2 * self.bandwidth ** 2))
        
        return kernel


class DownstreamProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 0):
        super().__init__()
        
        if hidden_dim > 0:
            self.probe = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            self.probe = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.probe(x)


class Stage1Trainer:
    def __init__(
        self,
        adapter: SubjectInvariantAdapter,
        train_loader,
        val_loader,
        config: ExperimentConfig,
        probe_data: dict = None,
    ):
        self.adapter = adapter
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.probe_data = probe_data
        
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
        
        if config.stage1_training.use_temporal_contrastive:
            self.temporal_loss_fn = TemporalContrastiveLoss(
                temperature=config.stage1_training.temperature
            )
        
        if config.stage1_training.use_cross_subject_consistency:
            self.consistency_loss_fn = CrossSubjectConsistencyLoss()
        
        if config.stage1_training.use_cross_subject_infonce:
            self.infonce_loss_fn = CrossSubjectInfoNCELoss(
                temperature=config.stage1_training.temperature
            )
        
        if config.stage1_training.use_mmd:
            self.mmd_loss_fn = MMDLoss()
        
        self.probe = None
        self.probe_optimizer = None
        if config.stage1_training.use_downstream_probe and probe_data is not None:
            self._setup_probe()
        
        self.output_dir = Path(config.data.output_root) / config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(self.output_dir / "logs")
        
        if config.system.use_wandb:
            self._init_wandb()
        
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_probe_cosine = -float('inf')
        self.best_epoch = 0
        self.early_stopping_counter = 0
    
    def _setup_probe(self):
        input_dim = self.config.adapter.canonical_dim
        output_dim = self.config.linguistic.embedding_dim
        hidden_dim = self.config.stage1_training.probe_hidden_dim
        
        self.probe = DownstreamProbe(input_dim, output_dim, hidden_dim).to(self.device)
        self.probe_optimizer = torch.optim.Adam(
            self.probe.parameters(),
            lr=self.config.stage1_training.probe_lr
        )
    
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
        
        print(f"Weights & Biases initialized")
        print(f"  Project: {self.config.system.wandb_project}")
        print(f"  Run: {run_name}")
        if wandb.run:
            print(f"  URL: {wandb.run.url}")
    
    def train_epoch(self, epoch: int):
        self.adapter.train()
        
        total_loss = 0.0
        total_temporal = 0.0
        total_consistency = 0.0
        total_infonce = 0.0
        total_mmd = 0.0
        
        use_cross_subject = True
        if self.config.stage1_training.use_curriculum:
            phase1_epochs = self.config.stage1_training.curriculum_phase1_epochs
            if epoch <= phase1_epochs:
                use_cross_subject = False
                if epoch == 1:
                    print(f"  Curriculum Phase 1 (epoch 1-{phase1_epochs}): Only temporal loss")
            elif epoch == phase1_epochs + 1:
                print(f"  Curriculum Phase 2 (epoch {phase1_epochs+1}+): Adding cross-subject losses")
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            feature = batch['feature'].to(self.device)
            temporal_positive = batch['temporal_positive'].to(self.device)
            cross_subject_group = batch['cross_subject_group'].to(self.device)
            time_indices = batch['time_idx']
            
            canonical = self.adapter(feature)
            canonical_pos = self.adapter(temporal_positive)
            
            batch_size, n_subjects, feat_dim = cross_subject_group.shape
            cross_subject_flat = cross_subject_group.view(-1, feat_dim)
            canonical_cross = self.adapter(cross_subject_flat)
            canonical_cross = canonical_cross.view(batch_size, n_subjects, -1)
            
            loss = 0.0
            losses_dict = {}
            
            if self.config.stage1_training.use_temporal_contrastive:
                num_neg = self.config.stage1_training.num_negatives
                temporal_margin = 3
                
                all_negatives = []
                for i in range(batch_size):
                    current_time = time_indices[i].item()
                    time_diff = torch.abs(time_indices.float() - current_time)
                    valid_mask = time_diff > temporal_margin
                    candidates = canonical[valid_mask]
                    
                    if len(candidates) >= num_neg:
                        perm = torch.randperm(len(candidates), device=canonical.device)[:num_neg]
                        neg_samples = candidates[perm]
                    else:
                        if len(candidates) > 0:
                            repeat_times = (num_neg // len(candidates)) + 1
                            expanded = candidates.repeat(repeat_times, 1)
                            perm = torch.randperm(len(expanded), device=canonical.device)[:num_neg]
                            neg_samples = expanded[perm]
                        else:
                            perm = torch.randperm(batch_size, device=canonical.device)[:num_neg]
                            neg_samples = canonical[perm]
                    
                    all_negatives.append(neg_samples)
                
                negatives = torch.stack(all_negatives)
                temporal_loss = self.temporal_loss_fn(canonical, canonical_pos, negatives)
                loss += self.config.stage1_training.temporal_weight * temporal_loss
                losses_dict['temporal'] = temporal_loss.item()
                total_temporal += temporal_loss.item()
            
            if use_cross_subject:
                if self.config.stage1_training.use_cross_subject_consistency and \
                   self.config.stage1_training.consistency_weight > 0:
                    consistency_loss = self.consistency_loss_fn(canonical_cross)
                    loss += self.config.stage1_training.consistency_weight * consistency_loss
                    losses_dict['consistency'] = consistency_loss.item()
                    total_consistency += consistency_loss.item()
                
                if self.config.stage1_training.use_cross_subject_infonce and \
                   self.config.stage1_training.infonce_weight > 0:
                    infonce_loss = self.infonce_loss_fn(canonical, time_indices.to(self.device))
                    loss += self.config.stage1_training.infonce_weight * infonce_loss
                    losses_dict['infonce'] = infonce_loss.item()
                    total_infonce += infonce_loss.item()
                
                if self.config.stage1_training.use_mmd and \
                   self.config.stage1_training.mmd_weight > 0:
                    mmd_loss = self.mmd_loss_fn(canonical_cross)
                    loss += self.config.stage1_training.mmd_weight * mmd_loss
                    losses_dict['mmd'] = mmd_loss.item()
                    total_mmd += mmd_loss.item()
            
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
                **{k: f'{v:.4f}' for k, v in losses_dict.items()}
            })
        
        n_batches = len(self.train_loader)
        return {
            'total_loss': total_loss / n_batches,
            'temporal_loss': total_temporal / n_batches,
            'consistency_loss': total_consistency / n_batches if total_consistency > 0 else 0,
            'infonce_loss': total_infonce / n_batches if total_infonce > 0 else 0,
            'mmd_loss': total_mmd / n_batches if total_mmd > 0 else 0,
        }
    
    def validate(self, epoch: int):
        self.adapter.eval()
        
        total_loss = 0.0
        total_temporal = 0.0
        total_consistency = 0.0
        total_infonce = 0.0
        total_mmd = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                feature = batch['feature'].to(self.device)
                cross_subject_group = batch['cross_subject_group'].to(self.device)
                time_indices = batch['time_idx']
                
                canonical = self.adapter(feature)
                
                batch_size, n_subjects, feat_dim = cross_subject_group.shape
                cross_subject_flat = cross_subject_group.view(-1, feat_dim)
                canonical_cross = self.adapter(cross_subject_flat)
                canonical_cross = canonical_cross.view(batch_size, n_subjects, -1)
                
                loss = 0.0
                
                if 'temporal_positive' in batch and self.config.stage1_training.use_temporal_contrastive:
                    temporal_positive = batch['temporal_positive'].to(self.device)
                    canonical_pos = self.adapter(temporal_positive)
                    
                    batch_size = len(canonical)
                    num_neg = self.config.stage1_training.num_negatives
                    temporal_margin = 3
                    
                    all_negatives = []
                    for i in range(batch_size):
                        current_time = time_indices[i].item()
                        time_diff = torch.abs(time_indices.float() - current_time)
                        valid_mask = time_diff > temporal_margin
                        candidates = canonical[valid_mask]
                        
                        if len(candidates) >= num_neg:
                            perm = torch.randperm(len(candidates), device=canonical.device)[:num_neg]
                            neg_samples = candidates[perm]
                        else:
                            if len(candidates) > 0:
                                repeat_times = (num_neg // len(candidates)) + 1
                                expanded = candidates.repeat(repeat_times, 1)
                                perm = torch.randperm(len(expanded), device=canonical.device)[:num_neg]
                                neg_samples = expanded[perm]
                            else:
                                perm = torch.randperm(batch_size, device=canonical.device)[:num_neg]
                                neg_samples = canonical[perm]
                        
                        all_negatives.append(neg_samples)
                    
                    negatives = torch.stack(all_negatives)
                    temporal_loss = self.temporal_loss_fn(canonical, canonical_pos, negatives)
                    loss += self.config.stage1_training.temporal_weight * temporal_loss
                    total_temporal += temporal_loss.item()
                
                if self.config.stage1_training.use_cross_subject_consistency and \
                   self.config.stage1_training.consistency_weight > 0:
                    consistency_loss = self.consistency_loss_fn(canonical_cross)
                    loss += self.config.stage1_training.consistency_weight * consistency_loss
                    total_consistency += consistency_loss.item()
                
                if self.config.stage1_training.use_cross_subject_infonce and \
                   self.config.stage1_training.infonce_weight > 0:
                    infonce_loss = self.infonce_loss_fn(canonical, time_indices.to(self.device))
                    loss += self.config.stage1_training.infonce_weight * infonce_loss
                    total_infonce += infonce_loss.item()
                
                if self.config.stage1_training.use_mmd and \
                   self.config.stage1_training.mmd_weight > 0:
                    mmd_loss = self.mmd_loss_fn(canonical_cross)
                    loss += self.config.stage1_training.mmd_weight * mmd_loss
                    total_mmd += mmd_loss.item()
                
                total_loss += loss.item() if isinstance(loss, torch.Tensor) else loss
                n_batches += 1
        
        avg_loss = total_loss / n_batches if n_batches > 0 else 0
        avg_temporal = total_temporal / n_batches if n_batches > 0 else 0
        avg_consistency = total_consistency / n_batches if n_batches > 0 else 0
        avg_infonce = total_infonce / n_batches if n_batches > 0 else 0
        avg_mmd = total_mmd / n_batches if n_batches > 0 else 0
        
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        self.writer.add_scalar('val/temporal', avg_temporal, epoch)
        self.writer.add_scalar('val/consistency', avg_consistency, epoch)
        self.writer.add_scalar('val/infonce', avg_infonce, epoch)
        self.writer.add_scalar('val/mmd', avg_mmd, epoch)
        
        if self.config.system.use_wandb:
            wandb.log({
                'val/loss': avg_loss,
                'val/temporal': avg_temporal,
                'val/consistency': avg_consistency,
                'val/infonce': avg_infonce,
                'val/mmd': avg_mmd,
                'epoch': epoch,
            }, step=self.global_step)
        
        return {
            'val_loss': avg_loss,
            'val_temporal': avg_temporal,
            'val_consistency': avg_consistency,
            'val_infonce': avg_infonce,
            'val_mmd': avg_mmd,
        }
    
    def run_downstream_probe(self, epoch: int):
        if self.probe_data is None or self.probe is None:
            return None
        
        self.adapter.eval()
        self.probe.train()
        
        brain_features = self.probe_data['brain_features'].to(self.device)
        word_embeddings = self.probe_data['word_embeddings'].to(self.device)
        
        with torch.no_grad():
            canonical_features = self.adapter(brain_features)
        
        probe_epochs = self.config.stage1_training.probe_train_epochs
        batch_size = 64
        n_samples = len(canonical_features)
        
        for _ in range(probe_epochs):
            indices = torch.randperm(n_samples)
            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i+batch_size]
                batch_canonical = canonical_features[batch_idx]
                batch_target = word_embeddings[batch_idx]
                
                predicted = self.probe(batch_canonical)
                
                loss = nn.functional.mse_loss(predicted, batch_target)
                
                self.probe_optimizer.zero_grad()
                loss.backward()
                self.probe_optimizer.step()
        
        self.probe.eval()
        with torch.no_grad():
            predicted = self.probe(canonical_features)
            
            pred_norm = nn.functional.normalize(predicted, dim=1)
            target_norm = nn.functional.normalize(word_embeddings, dim=1)
            cosine_sim = (pred_norm * target_norm).sum(dim=1).mean().item()
            
            mse = nn.functional.mse_loss(predicted, word_embeddings).item()
        
        self.writer.add_scalar('probe/cosine_similarity', cosine_sim, epoch)
        self.writer.add_scalar('probe/mse', mse, epoch)
        
        if self.config.system.use_wandb:
            wandb.log({
                'probe/cosine_similarity': cosine_sim,
                'probe/mse': mse,
                'epoch': epoch,
            }, step=self.global_step)
        
        return {
            'cosine_similarity': cosine_sim,
            'mse': mse,
        }
        
        if self.config.system.use_wandb:
            wandb.log({
                'val/loss': avg_loss,
                'val/temporal': avg_temporal,
                'val/consistency': avg_consistency,
                'epoch': epoch,
            }, step=self.global_step)
        
        return {
            'val_loss': avg_loss,
            'val_temporal': avg_temporal,
            'val_consistency': avg_consistency
        }
    
    def save_checkpoint(self, epoch: int, is_best_probe: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.adapter.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'global_step': self.global_step,
            'best_probe_cosine': self.best_probe_cosine,
        }
        
        if is_best_probe:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  Saved best model (probe_cosine={self.best_probe_cosine:.4f}) to {best_path}")
        
        if epoch % self.config.stage1_training.save_every_n_epochs == 0:
            path = self.output_dir / f"checkpoint_epoch{epoch}.pt"
            torch.save(checkpoint, path)
            
            checkpoints = sorted(self.output_dir.glob("checkpoint_epoch*.pt"))
            if len(checkpoints) > self.config.stage1_training.keep_last_n_checkpoints:
                for ckpt in checkpoints[:-self.config.stage1_training.keep_last_n_checkpoints]:
                    ckpt.unlink()
    
    def train(self):
        print(f"\n{'='*70}")
        print(f"Starting Stage 1 Training: {self.config.name}")
        print(f"{'='*70}")
        
        print(f"\nLoss Configuration:")
        print(f"  Temporal Contrastive: {self.config.stage1_training.use_temporal_contrastive} (weight={self.config.stage1_training.temporal_weight})")
        print(f"  Cosine Consistency:   {self.config.stage1_training.use_cross_subject_consistency} (weight={self.config.stage1_training.consistency_weight})")
        print(f"  Cross-Subject InfoNCE:{self.config.stage1_training.use_cross_subject_infonce} (weight={self.config.stage1_training.infonce_weight})")
        print(f"  MMD:                  {self.config.stage1_training.use_mmd} (weight={self.config.stage1_training.mmd_weight})")
        
        if self.config.stage1_training.use_curriculum:
            print(f"\nCurriculum Learning: Phase 1 (temporal only) for {self.config.stage1_training.curriculum_phase1_epochs} epochs")
        
        if self.config.stage1_training.use_downstream_probe and self.probe_data is not None:
            print(f"\nDownstream Probe: Every {self.config.stage1_training.probe_every_n_epochs} epoch(s)")
            print(f"   best_model.pt will be saved based on probe_cosine (alignment metric)")
        else:
            print(f"\nDownstream Probe: Disabled (no probe data)")
            print(f"   best_model.pt will be saved based on val_temporal")
        
        print()
        
        for epoch in range(1, self.config.stage1_training.num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            
            val_metrics = self.validate(epoch)
            
            print(f"\nEpoch {epoch}/{self.config.stage1_training.num_epochs}")
            print(f"  Train Loss: {train_metrics['total_loss']:.4f}")
            print(f"    Temporal:    {train_metrics['temporal_loss']:.4f}")
            if train_metrics.get('consistency_loss', 0) > 0:
                print(f"    Consistency: {train_metrics['consistency_loss']:.4f}")
            if train_metrics.get('infonce_loss', 0) > 0:
                print(f"    InfoNCE:     {train_metrics['infonce_loss']:.4f}")
            if train_metrics.get('mmd_loss', 0) > 0:
                print(f"    MMD:         {train_metrics['mmd_loss']:.4f}")
            
            print(f"  Val Temporal: {val_metrics['val_temporal']:.4f}")
            
            probe_metrics = None
            is_best_probe = False
            
            if self.config.stage1_training.use_downstream_probe and \
               self.probe_data is not None and \
               epoch % self.config.stage1_training.probe_every_n_epochs == 0:
                probe_metrics = self.run_downstream_probe(epoch)
                if probe_metrics:
                    print(f"  Probe: cosine={probe_metrics['cosine_similarity']:.4f}, mse={probe_metrics['mse']:.4f}")
                    
                    if probe_metrics['cosine_similarity'] > self.best_probe_cosine:
                        self.best_probe_cosine = probe_metrics['cosine_similarity']
                        self.best_epoch = epoch
                        is_best_probe = True
                        print(f"  New best! (probe_cosine: {self.best_probe_cosine:.4f})")
            
            if self.probe_data is None:
                if val_metrics['val_temporal'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_temporal']
                    self.best_epoch = epoch
                    is_best_probe = True
                    print(f"  New best! (val_temporal: {self.best_val_loss:.4f})")
            
            if self.config.system.use_wandb:
                log_dict = {
                    'epoch_metrics/train_loss': train_metrics['total_loss'],
                    'epoch_metrics/train_temporal': train_metrics['temporal_loss'],
                    'epoch_metrics/val_loss': val_metrics['val_loss'],
                    'epoch_metrics/val_temporal': val_metrics['val_temporal'],
                    'epoch': epoch,
                }
                
                if train_metrics.get('consistency_loss', 0) > 0:
                    log_dict['epoch_metrics/train_consistency'] = train_metrics['consistency_loss']
                if train_metrics.get('infonce_loss', 0) > 0:
                    log_dict['epoch_metrics/train_infonce'] = train_metrics['infonce_loss']
                if train_metrics.get('mmd_loss', 0) > 0:
                    log_dict['epoch_metrics/train_mmd'] = train_metrics['mmd_loss']
                
                if val_metrics.get('val_consistency', 0) > 0:
                    log_dict['epoch_metrics/val_consistency'] = val_metrics['val_consistency']
                if val_metrics.get('val_infonce', 0) > 0:
                    log_dict['epoch_metrics/val_infonce'] = val_metrics['val_infonce']
                if val_metrics.get('val_mmd', 0) > 0:
                    log_dict['epoch_metrics/val_mmd'] = val_metrics['val_mmd']
                
                if probe_metrics:
                    log_dict['epoch_metrics/probe_cosine'] = probe_metrics['cosine_similarity']
                    log_dict['epoch_metrics/probe_mse'] = probe_metrics['mse']
                
                wandb.log(log_dict, step=self.global_step)
                
                if is_best_probe:
                    if self.probe_data is not None:
                        wandb.run.summary['best_probe_cosine'] = self.best_probe_cosine
                    else:
                        wandb.run.summary['best_val_temporal'] = self.best_val_loss
                    wandb.run.summary['best_epoch'] = self.best_epoch
            
            self.save_checkpoint(epoch, is_best_probe=is_best_probe)
        
        print(f"\n{'='*70}")
        print(f"Training complete!")
        print(f"{'='*70}")
        if self.probe_data is not None:
            print(f"  Best probe_cosine: {self.best_probe_cosine:.4f} (epoch {self.best_epoch})")
        else:
            print(f"  Best val_temporal: {self.best_val_loss:.4f} (epoch {self.best_epoch})")
        print(f"  Best model: {self.output_dir / 'best_model.pt'}")
        print(f"  Outputs: {self.output_dir}")
        
        if self.config.system.use_wandb:
            wandb.finish()


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
    print(f"Split seed: {config.data.split_seed}")
    print(f"Key ablation settings:")
    print(f"  use_temporal_contrastive: {config.stage1_training.use_temporal_contrastive}")
    print(f"  use_cross_subject_consistency: {config.stage1_training.use_cross_subject_consistency}")
    print(f"  consistency_weight: {config.stage1_training.consistency_weight}")
    
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
    
    print(f"  Loading pretrained weights from: {config.mvpformer.checkpoint_path}")
    from mvpformer_utils import load_mvpformer_partial
    mvpformer, stats = load_mvpformer_partial(
        mvpformer,
        config.mvpformer.checkpoint_path,
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
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    probe_data = None
    if config.stage1_training.use_downstream_probe:
        print("\nLoading probe data...")
        
        data_root_parent = Path(config.data.data_root).parent
        embeddings_path = data_root_parent / config.linguistic.embeddings_file
        transcript_path = data_root_parent / config.linguistic.transcript_file
        
        print(f"  Embeddings: {embeddings_path}")
        print(f"  Transcript: {transcript_path}")
        
        if not embeddings_path.exists() or not transcript_path.exists():
            print(f"  Files not found, disabling probe...")
            config.stage1_training.use_downstream_probe = False
        else:
            try:
                import numpy as np
                import pandas as pd
                
                word_embeddings = np.load(embeddings_path)
                transcript_df = pd.read_csv(transcript_path)
                print(f"  Word embeddings: {word_embeddings.shape}")
                print(f"  Transcript: {len(transcript_df)} words")
                
                val_dataset = val_loader.dataset
                
                window_size = config.data.window_size
                stride = config.data.stride
                
                probe_brain_features = []
                probe_word_embeddings = []
                
                for word_idx, row in transcript_df.iterrows():
                    if word_idx >= len(word_embeddings):
                        continue
                    
                    word_start = row['start']
                    word_end = row['end']
                    word_center = (word_start + word_end) / 2
                    
                    window_idx = int((word_center - window_size / 2) / stride)
                    window_idx = max(0, window_idx)
                    
                    if word_idx < len(probe_brain_features):
                        continue
                    
                    for i in range(len(val_dataset)):
                        sample = val_dataset[i]
                        if sample['time_idx'] == window_idx:
                            probe_brain_features.append(sample['feature'])
                            probe_word_embeddings.append(
                                torch.from_numpy(word_embeddings[word_idx]).float()
                            )
                            break
                    
                    if len(probe_brain_features) >= 500:
                        break
                
                if len(probe_word_embeddings) > 0:
                    probe_data = {
                        'brain_features': torch.stack(probe_brain_features),
                        'word_embeddings': torch.stack(probe_word_embeddings),
                    }
                    print(f"  Probe data ready: {len(probe_brain_features)} aligned pairs")
                else:
                    print("  No aligned pairs found, disabling probe...")
                    config.stage1_training.use_downstream_probe = False
                    
            except Exception as e:
                print(f"  Error: {e}")
                config.stage1_training.use_downstream_probe = False
    
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
    print(f"Adapter parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    trainer = Stage1Trainer(
        adapter=adapter,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        probe_data=probe_data,
    )
    
    trainer.train()


if __name__ == "__main__":
    main()
