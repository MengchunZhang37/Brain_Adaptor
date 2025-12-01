import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Literal
import pandas as pd
from tqdm import tqdm
import pickle

from config import DataConfig, TimeBasedSplitConfig, LinguisticConfig


class PodcastECoGDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        subjects: List[int],
        mvpformer_model,
        config: DataConfig,
        split: Literal['train', 'val', 'test'] = 'train',
        device: str = 'cuda',
    ):
        self.data_root = Path(data_root)
        self.subjects = subjects
        self.mvpformer = mvpformer_model
        self.config = config
        self.split = split
        self.device = device
        
        self.split_config = config.split_config
        self.time_start, self.time_end = self.split_config.get_split_bounds(split)
        
        self.sampling_rate = config.sampling_rate
        self.window_size = config.window_size
        self.stride = config.stride
        self.use_cache = config.use_cache
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[{split.upper()}] Time range: {self.time_start/60:.1f} - {self.time_end/60:.1f} min")
        print(f"[{split.upper()}] Using ALL {len(subjects)} subjects")
        
        print("Loading preprocessed Podcast ECoG data...")
        self._load_data()
        
        print("Extracting MVPFormer features...")
        self._extract_features()
        
        print("Creating time-based windows...")
        self._create_time_based_samples()
        
        print(f" [{split.upper()}] Dataset ready: {len(self)} samples")
    
    def _load_data(self):
        self.raw_data = {}
        self.channel_info = {}
        
        channel_indices_file = self.data_root.parent / "selected_channel_indices.json"
        
        if channel_indices_file.exists():
            import json
            with open(channel_indices_file) as f:
                selected_indices = json.load(f)
            print(f"  Using balanced channel selection")
            use_balanced = True
        else:
            print(f"  No balanced channel indices, using first 90 channels")
            use_balanced = False
            target_channels = 90
        
        for subj_id in tqdm(self.subjects, desc="Loading subjects"):
            preprocessed_file = self.data_root / f"sub-{subj_id:02d}_hg_z.npy"
            
            if not preprocessed_file.exists():
                print(f"  {preprocessed_file} not found, skipping subject {subj_id}")
                continue
            
            ecog_data = np.load(preprocessed_file)
            orig_channels = ecog_data.shape[0]
            
            if use_balanced:
                indices = selected_indices.get(str(subj_id), None)
                if indices is None:
                    indices = list(range(min(90, orig_channels)))
                valid_indices = [i for i in indices if i < orig_channels]
                if len(valid_indices) == 0:
                    continue
                ecog_data = ecog_data[valid_indices, :]
            else:
                if orig_channels < target_channels:
                    continue
                ecog_data = ecog_data[:target_channels, :]
            
            self.raw_data[subj_id] = ecog_data
            self.channel_info[subj_id] = [f"ch{i}" for i in range(ecog_data.shape[0])]
        
        print(f"  Loaded {len(self.raw_data)} subjects")
    
    def _extract_features(self):
        self.mvpformer.eval()
        self.mvpformer = self.mvpformer.to(self.device)
        
        self.features = {}
        self.window_times = None
        
        for subj_id in tqdm(self.subjects, desc="Extracting features"):
            if subj_id not in self.raw_data:
                continue
            
            cache_file = self.cache_dir / f"sub{subj_id:02d}_features_all.pkl"
            
            if self.use_cache and cache_file.exists() and not self.config.force_recompute:
                with open(cache_file, 'rb') as f:
                    cached = pickle.load(f)
                    self.features[subj_id] = cached['features']
                    if self.window_times is None:
                        self.window_times = cached['window_times']
            else:
                ecog_data = self.raw_data[subj_id]
                n_channels, n_samples = ecog_data.shape
                
                window_samples = int(self.window_size * self.sampling_rate)
                stride_samples = int(self.stride * self.sampling_rate)
                
                features_list = []
                window_times_list = []
                
                with torch.no_grad():
                    start = 0
                    
                    while start + window_samples <= n_samples:
                        window = ecog_data[:, start:start + window_samples]
                        
                        center_sample = start + window_samples // 2
                        center_time = center_sample / self.sampling_rate
                        window_times_list.append(center_time)
                        
                        window_tensor = torch.from_numpy(window).float()
                        chunk_size = self.config.chunk_size
                        num_chunks = window_tensor.shape[1] // chunk_size
                        
                        chunk_features = []
                        for chunk_idx in range(num_chunks):
                            start_idx = chunk_idx * chunk_size
                            end_idx = start_idx + chunk_size
                            chunk = window_tensor[:, start_idx:end_idx]
                            chunk = chunk.unsqueeze(0).unsqueeze(0).to(self.device)
                            
                            try:
                                result = self.mvpformer(chunk)
                                if isinstance(result, tuple):
                                    feat = result[0]
                                    if hasattr(feat, 'last_hidden_state'):
                                        feat = feat.last_hidden_state
                                else:
                                    feat = result
                                    if hasattr(feat, 'last_hidden_state'):
                                        feat = feat.last_hidden_state
                                
                                if len(feat.shape) == 4:
                                    feat = feat.mean(dim=(1, 2))
                                elif len(feat.shape) == 3:
                                    feat = feat.mean(dim=1)
                                
                                chunk_features.append(feat.cpu())
                            except Exception as e:
                                print(f"  Error: {e}")
                                raise
                        
                        window_feat = torch.stack(chunk_features).mean(dim=0)
                        features_list.append(window_feat.numpy())
                        
                        start += stride_samples
                
                features = np.concatenate(features_list, axis=0)
                window_times = np.array(window_times_list)
                
                if self.use_cache:
                    with open(cache_file, 'wb') as f:
                        pickle.dump({'features': features, 'window_times': window_times}, f)
                
                self.features[subj_id] = features
                if self.window_times is None:
                    self.window_times = window_times
        
        print(f"  Feature shape per subject: {list(self.features.values())[0].shape}")
        print(f"  Total windows (all time): {len(self.window_times)}")
    
    def _create_time_based_samples(self):
        self.valid_window_indices = []
        
        for idx, center_time in enumerate(self.window_times):
            if self.time_start <= center_time < self.time_end:
                if not self.split_config.is_in_buffer(center_time):
                    self.valid_window_indices.append(idx)
        
        self.n_windows = len(self.valid_window_indices)
        
        min_total_windows = min(len(feat) for feat in self.features.values())
        for subj_id in self.features:
            self.features[subj_id] = self.features[subj_id][:min_total_windows]
        
        self.valid_window_indices = [idx for idx in self.valid_window_indices if idx < min_total_windows]
        self.n_windows = len(self.valid_window_indices)
        
        print(f"  [{self.split.upper()}] Valid windows: {self.n_windows} "
              f"(from {self.time_start/60:.1f}-{self.time_end/60:.1f} min)")
    
    def __len__(self) -> int:
        return self.n_windows * len(self.features)
    
    def __getitem__(self, idx: int) -> Dict:
        n_subjects = len(self.features)
        window_in_split_idx = idx // n_subjects
        subject_idx = idx % n_subjects
        
        actual_window_idx = self.valid_window_indices[window_in_split_idx]
        subj_id = list(self.features.keys())[subject_idx]
        
        feature = torch.from_numpy(self.features[subj_id][actual_window_idx]).float()
        
        if window_in_split_idx < self.n_windows - 1:
            next_window_idx = self.valid_window_indices[window_in_split_idx + 1]
            positive = torch.from_numpy(self.features[subj_id][next_window_idx]).float()
        else:
            prev_window_idx = self.valid_window_indices[window_in_split_idx - 1]
            positive = torch.from_numpy(self.features[subj_id][prev_window_idx]).float()
        
        cross_subject_features = []
        for other_subj in self.features.keys():
            feat = torch.from_numpy(self.features[other_subj][actual_window_idx]).float()
            cross_subject_features.append(feat)
        cross_subject_features = torch.stack(cross_subject_features)
        
        center_time = self.window_times[actual_window_idx]
        
        return {
            'feature': feature,
            'temporal_positive': positive,
            'cross_subject_group': cross_subject_features,
            'subject_id': subj_id,
            'time_idx': actual_window_idx,
            'center_time': center_time,
        }


class PodcastLinguisticDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        subjects: List[int],
        stage1_adapter,
        mvpformer_model,
        config,
        split: Literal['train', 'val', 'test'] = 'train',
        device: str = 'cuda',
    ):
        self.data_root = Path(data_root)
        self.subjects = subjects
        self.stage1_adapter = stage1_adapter
        self.mvpformer = mvpformer_model
        self.config = config
        self.split = split
        self.device = device
        
        self.split_config = config.data.split_config
        self.time_start, self.time_end = self.split_config.get_split_bounds(split)
        
        print(f"\n[{split.upper()}] Time range: {self.time_start/60:.1f} - {self.time_end/60:.1f} min")
        
        print("Loading brain features from Stage 1...")
        self._load_brain_features()
        
        print("Loading linguistic features...")
        self._load_linguistic_features()
        
        print("Aligning brain and linguistic features (time-based)...")
        self._align_features_time_based()
        
        print(f" [{split.upper()}] Dataset ready: {len(self)} samples")
    
    def _load_brain_features(self):
        preprocessed_dir = self.data_root
        if preprocessed_dir.name != 'preprocessed':
            preprocessed_dir = preprocessed_dir / 'preprocessed'
        
        ecog_dataset = PodcastECoGDataset(
            data_root=str(preprocessed_dir),
            subjects=self.subjects,
            mvpformer_model=self.mvpformer,
            config=self.config.data,
            split=self.split,
            device=self.device,
        )
        
        self.stage1_adapter.eval()
        self.stage1_adapter = self.stage1_adapter.to(self.device)
        
        self.canonical_features = {}
        self.window_times = ecog_dataset.window_times
        self.valid_window_indices = ecog_dataset.valid_window_indices
        
        with torch.no_grad():
            for subj_id in self.subjects:
                if subj_id in ecog_dataset.features:
                    mvp_feat = torch.from_numpy(ecog_dataset.features[subj_id]).float().to(self.device)
                    canonical_feat = self.stage1_adapter(mvp_feat)
                    self.canonical_features[subj_id] = canonical_feat.cpu().numpy()
        
        print(f"  Canonical features: {list(self.canonical_features.values())[0].shape}")
    
    def _load_linguistic_features(self):
        data_root = self.data_root
        if data_root.name == 'preprocessed':
            data_root = data_root.parent
        
        transcript_file = data_root / self.config.linguistic.transcript_file
        if not transcript_file.exists():
            raise FileNotFoundError(f"Transcript not found: {transcript_file}")
        
        self.words_df = pd.read_csv(transcript_file)
        print(f"  Loaded transcript: {len(self.words_df)} words")
        
        emb_file = data_root / self.config.linguistic.embeddings_file
        if not emb_file.exists():
            raise FileNotFoundError(f"Embeddings not found: {emb_file}")
        
        self.word_embeddings = np.load(emb_file)
        print(f"  Loaded embeddings: {self.word_embeddings.shape}")
    
    def _align_features_time_based(self):
        self.aligned_pairs = []
        
        window_size = self.config.data.window_size
        stride = self.config.data.stride
        
        has_start_end = 'start' in self.words_df.columns and 'end' in self.words_df.columns
        has_onset = 'onset' in self.words_df.columns or 'word_onset' in self.words_df.columns
        onset_col = 'onset' if 'onset' in self.words_df.columns else 'word_onset' if 'word_onset' in self.words_df.columns else None
        
        for idx, row in self.words_df.iterrows():
            if idx >= len(self.word_embeddings):
                continue
            
            if has_start_end:
                word_start = row['start']
                word_end = row['end']
                word_center = (word_start + word_end) / 2
            elif onset_col:
                word_center = row[onset_col]
            else:
                word_center = (idx / len(self.words_df)) * self.split_config.total_duration
            
            if not (self.time_start <= word_center < self.time_end):
                continue
            
            if self.split_config.is_in_buffer(word_center):
                continue
            
            window_idx = int((word_center - window_size / 2) / stride)
            window_idx = max(0, window_idx)
            
            if window_idx not in self.valid_window_indices:
                closest_valid = min(self.valid_window_indices, 
                                   key=lambda x: abs(x - window_idx),
                                   default=None)
                if closest_valid is None:
                    continue
                window_idx = closest_valid
            
            for subj_id in self.subjects:
                if subj_id in self.canonical_features:
                    n_windows = len(self.canonical_features[subj_id])
                    if window_idx < n_windows:
                        self.aligned_pairs.append({
                            'subject_id': subj_id,
                            'brain_idx': window_idx,
                            'word_idx': idx,
                            'word': row.get('word', f'word_{idx}'),
                            'word_time': word_center,
                        })
        
        print(f"  [{self.split.upper()}] Aligned pairs: {len(self.aligned_pairs)}")
    
    def __len__(self) -> int:
        return len(self.aligned_pairs)
    
    def __getitem__(self, idx: int) -> Dict:
        pair = self.aligned_pairs[idx]
        
        subj_id = pair['subject_id']
        brain_idx = pair['brain_idx']
        word_idx = pair['word_idx']
        
        brain_feat = torch.from_numpy(self.canonical_features[subj_id][brain_idx]).float()
        word_emb = torch.from_numpy(self.word_embeddings[word_idx]).float()
        
        n_windows = len(self.canonical_features[subj_id])
        
        current_pos = self.valid_window_indices.index(brain_idx) if brain_idx in self.valid_window_indices else 0
        if current_pos < len(self.valid_window_indices) - 1:
            next_idx = self.valid_window_indices[current_pos + 1]
        else:
            next_idx = self.valid_window_indices[current_pos - 1] if current_pos > 0 else brain_idx
        
        next_brain_feat = torch.from_numpy(self.canonical_features[subj_id][next_idx]).float()
        
        return {
            'brain_feature': brain_feat,
            'word_embedding': word_emb,
            'next_brain_feature': next_brain_feat,
            'subject_id': subj_id,
            'word_idx': word_idx,
            'brain_idx': brain_idx,
        }


def create_stage1_dataloader(
    data_root: str,
    subjects: List[int],
    mvpformer_model,
    config,
    split: Literal['train', 'val', 'test'] = 'train',
) -> DataLoader:
    print(f"\n{'='*60}")
    print(f"Creating Stage 1 DataLoader: {split.upper()}")
    print(f"{'='*60}")
    print(f"  Split method: TIME-BASED (all {len(subjects)} subjects in all splits)")
    
    dataset = PodcastECoGDataset(
        data_root=data_root,
        subjects=subjects,
        mvpformer_model=mvpformer_model,
        config=config.data,
        split=split,
        device=config.system.device,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.stage1_training.batch_size,
        shuffle=(split == 'train'),
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        prefetch_factor=config.data.prefetch_factor if config.data.num_workers > 0 else None,
        drop_last=(split == 'train'),
    )
    
    return dataloader


def create_stage2_dataloader(
    data_root: str,
    subjects: List[int],
    stage1_adapter,
    mvpformer_model,
    config,
    split: Literal['train', 'val', 'test'] = 'train',
) -> DataLoader:
    print
