import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import argparse
from tqdm import tqdm
import sys
import numpy as np
import pickle
import pandas as pd

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from config import get_simplified_config, print_split_info
from simplified_adapter import SimplifiedAdapter, create_simplified_adapter


class SimplifiedDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        subjects: list,
        mvpformer_model,
        config,
        split: str = 'train',
        device: str = 'cuda',
    ):
        self.data_root = Path(data_root)
        self.subjects = subjects
        self.mvpformer = mvpformer_model
        self.config = config
        self.split = split
        self.device = device

        self.split_config = config.data.split_config
        self.time_start, self.time_end = self.split_config.get_split_bounds(split)

        self.sampling_rate = config.data.sampling_rate
        self.window_size = config.data.window_size
        self.stride = config.data.stride
        self.cache_dir = Path(config.data.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{split.upper()}] Time: {self.time_start/60:.1f} - {self.time_end/60:.1f} min")
        print(f"[{split.upper()}] Subjects: {subjects}")

        self._load_ecog()
        self._extract_features()
        self._load_linguistic()
        self._create_samples()

        print(f"[{split.upper()}] Ready: {len(self)} samples")

    def _load_ecog(self):
        self.raw_data = {}

        ch_file = self.data_root.parent / "selected_channel_indices.json"
        if ch_file.exists():
            import json
            with open(ch_file) as f:
                selected_indices = json.load(f)
            use_balanced = True
        else:
            use_balanced = False

        for subj_id in self.subjects:
            fpath = self.data_root / f"sub-{subj_id:02d}_hg_z.npy"
            if not fpath.exists():
                print(f"Not found: {fpath}")
                continue

            data = np.load(fpath)

            if use_balanced:
                indices = selected_indices.get(str(subj_id), list(range(90)))
                valid = [i for i in indices if i < data.shape[0]]
                data = data[valid, :]
            else:
                data = data[:90, :]

            self.raw_data[subj_id] = data

        print(f"Loaded {len(self.raw_data)} subjects")

    def _extract_features(self):
        self.mvpformer.eval()
        self.mvpformer = self.mvpformer.to(self.device)

        self.features = {}
        self.window_times = None

        chunk_size = self.config.data.chunk_size

        for subj_id in self.subjects:
            if subj_id not in self.raw_data:
                continue

            cache_file = self.cache_dir / f"sub{subj_id:02d}_features_all.pkl"

            if cache_file.exists() and not self.config.data.force_recompute:
                with open(cache_file, 'rb') as f:
                    cached = pickle.load(f)
                    self.features[subj_id] = cached['features']
                    if self.window_times is None:
                        self.window_times = cached['window_times']
            else:
                print(f"Extracting features for sub-{subj_id:02d}...")
                ecog = self.raw_data[subj_id]
                n_ch, n_samp = ecog.shape

                win_samp = int(self.window_size * self.sampling_rate)
                stride_samp = int(self.stride * self.sampling_rate)

                feats = []
                times = []

                with torch.no_grad():
                    start = 0
                    while start + win_samp <= n_samp:
                        window = ecog[:, start:start + win_samp]
                        center_time = (start + win_samp // 2) / self.sampling_rate
                        times.append(center_time)

                        window_t = torch.from_numpy(window).float()
                        n_chunks = window_t.shape[1] // chunk_size

                        chunk_feats = []
                        for c in range(n_chunks):
                            chunk = window_t[:, c*chunk_size:(c+1)*chunk_size]
                            chunk = chunk.unsqueeze(0).unsqueeze(0).to(self.device)

                            result = self.mvpformer(chunk)
                            if isinstance(result, tuple):
                                feat = result[0]
                            else:
                                feat = result
                            if hasattr(feat, 'last_hidden_state'):
                                feat = feat.last_hidden_state

                            if len(feat.shape) == 4:
                                feat = feat.mean(dim=(1, 2))
                            elif len(feat.shape) == 3:
                                feat = feat.mean(dim=1)

                            chunk_feats.append(feat.cpu())

                        win_feat = torch.stack(chunk_feats).mean(dim=0)
                        feats.append(win_feat.numpy())

                        start += stride_samp

                features = np.concatenate(feats, axis=0)
                window_times = np.array(times)

                with open(cache_file, 'wb') as f:
                    pickle.dump({'features': features, 'window_times': window_times}, f)

                self.features[subj_id] = features
                if self.window_times is None:
                    self.window_times = window_times

        self.valid_indices = []
        for idx, t in enumerate(self.window_times):
            if self.time_start <= t < self.time_end:
                if not self.split_config.is_in_buffer(t):
                    self.valid_indices.append(idx)

        min_len = min(len(f) for f in self.features.values())
        for sid in self.features:
            self.features[sid] = self.features[sid][:min_len]
        self.valid_indices = [i for i in self.valid_indices if i < min_len]

        print(f"Valid windows: {len(self.valid_indices)}")

    def _load_linguistic(self):
        root = self.data_root
        if root.name == 'preprocessed':
            root = root.parent

        transcript_path = root / self.config.linguistic.transcript_file
        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript not found: {transcript_path}")
        self.words_df = pd.read_csv(transcript_path)

        emb_path = root / self.config.linguistic.embeddings_file
        if not emb_path.exists():
            raise FileNotFoundError(f"Embeddings not found: {emb_path}")
        self.word_embeddings = np.load(emb_path)

        print(f"Words: {len(self.words_df)}, Embeddings: {self.word_embeddings.shape}")

    def _create_samples(self):
        self.samples = []

        window_to_words = {idx: [] for idx in self.valid_indices}

        for word_idx, row in self.words_df.iterrows():
            if word_idx >= len(self.word_embeddings):
                continue

            if 'start' in row and 'end' in row:
                word_time = (row['start'] + row['end']) / 2
            elif 'onset' in row:
                word_time = row['onset']
            else:
                continue

            if not (self.time_start <= word_time < self.time_end):
                continue
            if self.split_config.is_in_buffer(word_time):
                continue

            window_idx = int((word_time - self.window_size / 2) / self.stride)
            window_idx = max(0, window_idx)

            if window_idx in window_to_words:
                window_to_words[window_idx].append(word_idx)

        for window_idx in self.valid_indices:
            word_indices = window_to_words.get(window_idx, [])
            if not word_indices:
                continue

            avg_emb = np.mean([self.word_embeddings[wi] for wi in word_indices], axis=0)

            for subj_id in self.features.keys():
                self.samples.append({
                    'subject_id': subj_id,
                    'window_idx': window_idx,
                    'word_indices': word_indices,
                    'avg_word_embedding': avg_emb,
                })

        print(f"Samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        subj_id = sample['subject_id']
        window_idx = sample['window_idx']

        feature = torch.from_numpy(self.features[subj_id][window_idx]).float()
        word_embedding = torch.from_numpy(sample['avg_word_embedding']).float()

        pos_in_valid = self.valid_indices.index(window_idx)
        if pos_in_valid < len(self.valid_indices) - 1:
            next_idx = self.valid_indices[pos_in_valid + 1]
        else:
            next_idx = self.valid_indices[pos_in_valid - 1]
        next_feature = torch.from_numpy(self.features[subj_id][next_idx]).float()

        return {
            'feature': feature,
            'word_embedding': word_embedding,
            'next_feature': next_feature,
            'subject_id': subj_id,
            'window_idx': window_idx,
        }


class AlignmentLoss(nn.Module):
    def forward(self, pred, target):
        sim = nn.functional.cosine_similarity(pred, target, dim=1)
        return (1.0 - sim).mean()


class TemporalSmoothnessLoss(nn.Module):
    def forward(self, pred_t, pred_t1):
        sim = nn.functional.cosine_similarity(pred_t, pred_t1, dim=1)
        return (1.0 - sim).mean()


class SimplifiedTrainer:
    def __init__(
        self,
        adapter: SimplifiedAdapter,
        train_loader,
        val_loader,
        config,
        output_dir: Path,
        run_name: str = None,
    ):
        self.adapter = adapter
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = output_dir
        self.run_name = run_name or config.name

        self.device = torch.device(config.system.device)
        self.adapter = self.adapter.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.adapter.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        total_steps = len(train_loader) * config.training.num_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(total_steps, 1)
        )

        self.alignment_loss = AlignmentLoss()
        self.temporal_loss = TemporalSmoothnessLoss()

        self.w_align = config.training.alignment_weight
        self.w_temporal = config.training.temporal_weight

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.output_dir / "logs")

        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_cosine = -float('inf')
        self.best_epoch = 0
        self.patience_counter = 0

    def train_epoch(self, epoch):
        self.adapter.train()

        total_loss = 0
        total_align = 0
        total_temporal = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)

        for batch in pbar:
            feature = batch['feature'].to(self.device)
            word_embedding = batch['word_embedding'].to(self.device)
            next_feature = batch['next_feature'].to(self.device)

            pred = self.adapter(feature)
            pred_next = self.adapter(next_feature)

            l_align = self.alignment_loss(pred, word_embedding)
            l_temporal = self.temporal_loss(pred, pred_next)

            loss = self.w_align * l_align + self.w_temporal * l_temporal

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.adapter.parameters(),
                self.config.training.gradient_clip
            )
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            total_align += l_align.item()
            total_temporal += l_temporal.item()
            self.global_step += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'align': f'{l_align.item():.4f}',
            })

        n = len(self.train_loader)
        metrics = {
            'loss': total_loss / n if n > 0 else 0,
            'align': total_align / n if n > 0 else 0,
            'temporal': total_temporal / n if n > 0 else 0,
        }

        for k, v in metrics.items():
            self.writer.add_scalar(f'train/{k}', v, epoch)

        if self.config.system.use_wandb and WANDB_AVAILABLE:
            wandb.log({f'train/{k}': v for k, v in metrics.items()}, step=self.global_step)

        return metrics

    @torch.no_grad()
    def validate(self, epoch=None):
        self.adapter.eval()

        total_loss = 0
        total_align = 0
        total_temporal = 0
        total_cosine = 0
        n = 0

        for batch in self.val_loader:
            feature = batch['feature'].to(self.device)
            word_embedding = batch['word_embedding'].to(self.device)
            next_feature = batch['next_feature'].to(self.device)

            pred = self.adapter(feature)
            pred_next = self.adapter(next_feature)

            l_align = self.alignment_loss(pred, word_embedding)
            l_temporal = self.temporal_loss(pred, pred_next)
            loss = self.w_align * l_align + self.w_temporal * l_temporal

            cosine = nn.functional.cosine_similarity(pred, word_embedding, dim=1)

            total_loss += loss.item()
            total_align += l_align.item()
            total_temporal += l_temporal.item()
            total_cosine += cosine.mean().item()
            n += 1

        metrics = {
            'val_loss': total_loss / n if n > 0 else 0,
            'val_align': total_align / n if n > 0 else 0,
            'val_temporal': total_temporal / n if n > 0 else 0,
            'val_cosine': total_cosine / n if n > 0 else 0,
        }

        if epoch is not None:
            for k, v in metrics.items():
                self.writer.add_scalar(f'val/{k}', v, epoch)

        if self.config.system.use_wandb and WANDB_AVAILABLE:
            wandb.log({f'val/{k}': v for k, v in metrics.items()}, step=self.global_step)

        return metrics

    def save_checkpoint(self, epoch, is_best=False):
        ckpt = {
            'epoch': epoch,
            'model_state_dict': self.adapter.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_cosine': self.best_val_cosine,
        }

        if epoch % self.config.training.save_every_n_epochs == 0:
            torch.save(ckpt, self.output_dir / f"checkpoint_epoch{epoch}.pt")

        if is_best:
            torch.save(ckpt, self.output_dir / "best_model.pt")

        checkpoints = sorted(self.output_dir.glob("checkpoint_epoch*.pt"))
        if len(checkpoints) > self.config.training.keep_last_n_checkpoints:
            for ckpt_path in checkpoints[:-self.config.training.keep_last_n_checkpoints]:
                ckpt_path.unlink()

    def train(self, verbose=True):
        if verbose:
            print(f"\nTraining: {self.run_name}")
            print(f"Loss: {self.w_align} * alignment + {self.w_temporal} * temporal")

        for epoch in range(1, self.config.training.num_epochs + 1):
            train_m = self.train_epoch(epoch)
            val_m = self.validate(epoch)

            if verbose and (epoch % 10 == 0 or epoch == 1):
                print(f"Epoch {epoch}: loss={train_m['loss']:.4f}, "
                      f"val_cosine={val_m['val_cosine']:.4f}, val_align={val_m['val_align']:.4f}")

            is_best = val_m['val_cosine'] > self.best_val_cosine
            if is_best:
                self.best_val_cosine = val_m['val_cosine']
                self.best_val_loss = val_m['val_loss']
                self.best_epoch = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch, is_best)

            if self.config.training.early_stopping:
                if self.patience_counter >= self.config.training.early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.adapter.state_dict(),
        }, self.output_dir / "final_model.pt")

        if verbose:
            print(f"Best: epoch {self.best_epoch}, val_cosine={self.best_val_cosine:.4f}")

        return self.best_val_cosine


def load_mvpformer(config):
    print("\nLoading MVPFormer...")

    orig_cap = torch.cuda.get_device_capability
    torch.cuda.get_device_capability = lambda *a, **k: (7, 0)

    sys.path.insert(0, config.mvpformer.repo_path)
    from models.mvpformer import HMVPFormer

    import yaml
    yaml_path = Path(config.mvpformer.repo_path) / "configs" / "mvpformer_generative.yaml"
    with open(yaml_path) as f:
        mvp_cfg = yaml.safe_load(f)

    model_args = mvp_cfg['model']['init_args']

    if 'encoder' not in model_args:
        model_args['encoder'] = {'class_path': 'models.fftencoder.WaveEncoder', 'init_args': {}}
    if 'init_args' not in model_args['encoder']:
        model_args['encoder']['init_args'] = {}
    model_args['encoder']['init_args']['size_input'] = config.data.chunk_size

    if 'gpt_config' in model_args and 'init_args' in model_args['gpt_config']:
        model_args['gpt_config']['init_args']['n_channels'] = 90

    enc_path = model_args['encoder']['class_path']
    enc_mod, enc_cls = enc_path.rsplit('.', 1)
    import importlib
    EncoderClass = getattr(importlib.import_module(enc_mod), enc_cls)
    encoder = EncoderClass(**model_args['encoder']['init_args'])

    from models.mvpformer import MVPFormerConfig, MVPFormerHead
    gpt_config = MVPFormerConfig(**model_args['gpt_config']['init_args'])
    head = MVPFormerHead(**model_args['head']['init_args'])

    hmvp_args = {k: v for k, v in model_args.items()
                 if k not in ['gpt_config', 'encoder', 'head', 'base_model']}

    mvpformer = HMVPFormer(gpt_config=gpt_config, encoder=encoder, head=head, **hmvp_args)

    from mvpformer_utils import load_mvpformer_partial
    mvpformer, _ = load_mvpformer_partial(mvpformer, config.mvpformer.checkpoint_path)

    mvpformer.eval()
    for p in mvpformer.parameters():
        p.requires_grad = False

    torch.cuda.get_device_capability = orig_cap

    return mvpformer


def train_shared(config, mvpformer):
    print("\n" + "="*70)
    print("Shared Training: All subjects → One adapter")
    print("="*70)

    print("\nCreating datasets...")

    train_dataset = SimplifiedDataset(
        data_root=config.data.data_root,
        subjects=config.data.subjects,
        mvpformer_model=mvpformer,
        config=config,
        split='train',
        device=config.system.device,
    )

    val_dataset = SimplifiedDataset(
        data_root=config.data.data_root,
        subjects=config.data.subjects,
        mvpformer_model=mvpformer,
        config=config,
        split='val',
        device=config.system.device,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    print("\nCreating SimplifiedAdapter...")
    adapter = create_simplified_adapter(config.adapter)

    n_params = adapter.get_num_params()
    print(f"Parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    output_dir = Path(config.data.output_root) / config.name

    output_dir.mkdir(parents=True, exist_ok=True)
    config.save(str(output_dir / "config.yaml"))

    if config.system.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=config.system.wandb_project,
            name=config.name,
            config=config.to_dict(),
            tags=['simplified', 'shared', 'time-based-split'],
        )

    trainer = SimplifiedTrainer(
        adapter=adapter,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=output_dir,
        run_name=config.name,
    )

    best_cosine = trainer.train(verbose=True)

    print("\n" + "="*70)
    print(f"Done! Output: {output_dir}")
    print("="*70)

    if config.system.use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    return best_cosine


def train_per_subject(config, mvpformer, subjects=None):
    print("\n" + "="*70)
    print("Per-Subject Training: Each subject → Separate adapter")
    print("="*70)

    subjects = subjects or config.data.subjects
    print(f"Subjects: {subjects}")

    output_dir = Path(config.data.output_root) / f"{config.name}_per_subject"
    output_dir.mkdir(parents=True, exist_ok=True)

    config.save(str(output_dir / "config.yaml"))

    results = {}

    for subj_id in subjects:
        print(f"\n{'='*50}")
        print(f"Subject {subj_id:02d}")
        print(f"{'='*50}")

        try:
            train_dataset = SimplifiedDataset(
                data_root=config.data.data_root,
                subjects=[subj_id],
                mvpformer_model=mvpformer,
                config=config,
                split='train',
                device=config.system.device,
            )

            val_dataset = SimplifiedDataset(
                data_root=config.data.data_root,
                subjects=[subj_id],
                mvpformer_model=mvpformer,
                config=config,
                split='val',
                device=config.system.device,
            )

            if len(train_dataset) == 0:
                print("No training samples, skipping")
                results[subj_id] = None
                continue

            train_loader = DataLoader(
                train_dataset,
                batch_size=min(config.training.batch_size, len(train_dataset)),
                shuffle=True,
                num_workers=0,
                drop_last=len(train_dataset) > config.training.batch_size,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=min(config.training.batch_size, max(1, len(val_dataset))),
                shuffle=False,
                num_workers=0,
            )

            adapter = create_simplified_adapter(config.adapter)

            subj_output_dir = output_dir / f"sub-{subj_id:02d}"

            trainer = SimplifiedTrainer(
                adapter=adapter,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                output_dir=subj_output_dir,
                run_name=f"sub-{subj_id:02d}",
            )

            best_cosine = trainer.train(verbose=True)
            results[subj_id] = best_cosine

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            results[subj_id] = None

    print("\n" + "="*70)
    print("Summary")
    print("="*70)

    for subj_id, cosine in results.items():
        status = f"val_cosine={cosine:.4f}" if cosine is not None else "FAILED"
        print(f"Sub-{subj_id:02d}: {status}")

    valid_cosines = [v for v in results.values() if v is not None]
    if valid_cosines:
        print(f"\nMean val_cosine: {np.mean(valid_cosines):.4f} ± {np.std(valid_cosines):.4f}")

    print(f"\nOutputs: {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Simplified Training")
    parser.add_argument('--config', type=str, default='simplified')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--mvpformer_checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs')

    parser.add_argument('--per_subject', action='store_true')
    parser.add_argument('--subjects', type=int, nargs='+', default=None)

    parser.add_argument('--alignment_weight', type=float, default=None)
    parser.add_argument('--temporal_weight', type=float, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)

    args = parser.parse_args()

    config = get_simplified_config(args.config)
    config.data.data_root = args.data_root
    config.data.output_root = args.output_dir
    config.mvpformer.checkpoint_path = args.mvpformer_checkpoint

    wandb.init(
        project=config.system.wandb_project,
        name=config.name,
        config=config.to_dict(),
        tags=['simplified', 'per-subject' if args.per_subject else 'shared', 'time-based-split'],
    )

    if args.alignment_weight is not None:
        config.training.alignment_weight = args.alignment_weight
    if args.temporal_weight is not None:
        config.training.temporal_weight = args.temporal_weight
    if args.num_epochs is not None:
        config.training.num_epochs = args.num_epochs
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size

    print(f"Config: {args.config}")
    print(f"Mode: {'per-subject' if args.per_subject else 'shared'}")
    print_split_info(config.data)

    mvpformer = load_mvpformer(config)

    if args.per_subject:
        train_per_subject(config, mvpformer, subjects=args.subjects)
    else:
        train_shared(config, mvpformer)


if __name__ == "__main__":
    main()
