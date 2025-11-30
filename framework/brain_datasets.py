import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import mne
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
from tqdm import tqdm
import pickle


class PodcastECoGDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        subjects: List[int],
        mvpformer_model,
        config,
        device: str = 'cuda',
    ):
        self.data_root = Path(data_root)
        self.subjects = subjects
        self.mvpformer = mvpformer_model
        self.config = config
        self.device = device
        
        self.sampling_rate = config.sampling_rate
        self.window_size = config.window_size
        self.stride = config.stride
        self.use_cache = config.use_cache
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print("Loading preprocessed Podcast ECoG data...")
        self._load_data()
        
        print("Extracting MVPFormer features...")
        self._extract_features()
        
        print("Creating temporal pairs and cross-subject groups...")
        self._create_pairs_and_groups()
        
        print(f"Dataset ready: {len(self)} samples")
    
    def _load_data(self):
        self.raw_data = {}
        self.channel_info = {}
        
        channel_indices_file = self.data_root.parent / "selected_channel_indices.json"
        
        if channel_indices_file.exists():
            import json
            with open(channel_indices_file) as f:
                selected_indices = json.load(f)
            print(f"\nUsing balanced channel selection from: {channel_indices_file}")
            use_balanced = True
        else:
            print(f"\nNo balanced channel indices found at {channel_indices_file}")
            print(f"  Falling back to first 90 channels")
            use_balanced = False
            target_channels = 90
        
        for subj_id in tqdm(self.subjects, desc="Loading subjects"):
            preprocessed_file = self.data_root / f"sub-{subj_id:02d}_hg_z.npy"
            
            if not preprocessed_file.exists():
                print(f"Warning: {preprocessed_file} not found, skipping subject {subj_id}")
                continue
            
            ecog_data = np.load(preprocessed_file)
            orig_channels = ecog_data.shape[0]
            
            if use_balanced:
                indices = selected_indices.get(str(subj_id), None)
                if indices is None:
                    print(f"  Subject {subj_id}: no indices in json, using first 90")
                    indices = list(range(min(90, orig_channels)))
                
                valid_indices = [i for i in indices if i < orig_channels]
                if len(valid_indices) < len(indices):
                    print(f"  Subject {subj_id}: {len(indices) - len(valid_indices)} indices out of bounds, using {len(valid_indices)} valid")
                
                if len(valid_indices) == 0:
                    print(f"  Subject {subj_id}: no valid indices, SKIPPING!")
                    continue
                
                ecog_data = ecog_data[valid_indices, :]
                print(f"  Subject {subj_id}: {orig_channels} → {len(valid_indices)} channels (balanced)")
            else:
                if orig_channels < target_channels:
                    print(f"  Subject {subj_id}: only {orig_channels} channels, SKIPPING!")
                    continue
                ecog_data = ecog_data[:target_channels, :]
                print(f"  Subject {subj_id}: {orig_channels} → {target_channels} channels")
            
            windowed_data = self._create_windows(ecog_data)
            
            self.raw_data[subj_id] = windowed_data
            self.channel_info[subj_id] = [f"ch{i}" for i in range(ecog_data.shape[0])]
        
        print(f"  Loaded {len(self.raw_data)} subjects")
        if len(self.raw_data) > 0:
            sample_shape = list(self.raw_data.values())[0].shape
            print(f"  Window shape: {sample_shape}")
    
    def _create_windows(self, ecog_data: np.ndarray) -> np.ndarray:
        n_channels, n_samples = ecog_data.shape
        
        window_samples = int(self.window_size * self.sampling_rate)
        stride_samples = int(self.stride * self.sampling_rate)
        
        windows = []
        for start in range(0, n_samples - window_samples + 1, stride_samples):
            end = start + window_samples
            window = ecog_data[:, start:end]
            windows.append(window)
        
        return np.array(windows)
    
    def _extract_features(self):
        self.mvpformer.eval()
        self.mvpformer = self.mvpformer.to(self.device)
        
        self.features = {}
        
        for subj_id in tqdm(self.subjects, desc="Extracting features"):
            if subj_id not in self.raw_data:
                continue
            
            cache_file = self.cache_dir / f"sub{subj_id:02d}_features.pkl"
            
            if self.use_cache and cache_file.exists() and not self.config.force_recompute:
                with open(cache_file, 'rb') as f:
                    self.features[subj_id] = pickle.load(f)
            else:
                windows = self.raw_data[subj_id]
                features_list = []
                
                with torch.no_grad():
                    for i, window in enumerate(windows):
                        window_tensor = torch.from_numpy(window).float()
                        
                        chunk_size = self.config.chunk_size
                        num_chunks = window_tensor.shape[1] // chunk_size
                        
                        chunk_features = []
                        for chunk_idx in range(num_chunks):
                            start_idx = chunk_idx * chunk_size
                            end_idx = start_idx + chunk_size
                            chunk = window_tensor[:, start_idx:end_idx]
                            
                            chunk = chunk.unsqueeze(0).unsqueeze(0)
                            chunk = chunk.to(self.device)
                            
                            if i == 0 and chunk_idx == 0:
                                print(f"  Processing in {num_chunks} chunks of size {chunk_size}")
                                print(f"  Chunk shape: {chunk.shape} [B, Seg, C, T]")
                            
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
                                
                                if i == 0 and chunk_idx == 0:
                                    print(f"  Chunk output: {feat.shape}")
                                
                            except Exception as e:
                                print(f"  Error processing chunk {chunk_idx}:")
                                print(f"    Chunk shape: {chunk.shape}")
                                print(f"    Error: {e}")
                                raise
                        
                        window_feat = torch.stack(chunk_features).mean(dim=0)
                        
                        if i == 0:
                            print(f"  Final window feature: {window_feat.shape}")
                        
                        features_list.append(window_feat.numpy())
                
                features = np.concatenate(features_list, axis=0)
                
                if self.use_cache:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(features, f)
                
                self.features[subj_id] = features
        
        print(f"  Feature shape: {list(self.features.values())[0].shape}")
    
    def _create_pairs_and_groups(self):
        min_windows = min(len(feat) for feat in self.features.values())
        
        for subj_id in self.features:
            self.features[subj_id] = self.features[subj_id][:min_windows]
        
        self.n_windows = min_windows
        
        self.temporal_pairs = []
        for subj_id in self.features:
            for i in range(self.n_windows - 1):
                self.temporal_pairs.append({
                    'subject': subj_id,
                    'anchor_idx': i,
                    'positive_idx': i + 1,
                })
        
        self.cross_subject_groups = []
        for time_idx in range(self.n_windows):
            group = {
                'time_idx': time_idx,
                'subjects': list(self.features.keys()),
            }
            self.cross_subject_groups.append(group)
    
    def __len__(self) -> int:
        return self.n_windows * len(self.subjects)
    
    def __getitem__(self, idx: int) -> Dict:
        subj_idx = idx % len(self.subjects)
        time_idx = idx // len(self.subjects)
        
        subj_id = self.subjects[subj_idx]
        
        feature = torch.from_numpy(self.features[subj_id][time_idx]).float()
        
        if time_idx < self.n_windows - 1:
            positive = torch.from_numpy(self.features[subj_id][time_idx + 1]).float()
        else:
            positive = torch.from_numpy(self.features[subj_id][time_idx - 1]).float()
        
        cross_subject_features = []
        for other_subj in self.subjects:
            if other_subj in self.features:
                feat = torch.from_numpy(self.features[other_subj][time_idx]).float()
                cross_subject_features.append(feat)
        
        cross_subject_features = torch.stack(cross_subject_features)
        
        return {
            'feature': feature,
            'temporal_positive': positive,
            'cross_subject_group': cross_subject_features,
            'subject_id': subj_id,
            'time_idx': time_idx,
        }


class PodcastLinguisticDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        subjects: List[int],
        stage1_adapter,
        mvpformer_model,
        config,
        device: str = 'cuda',
    ):
        self.data_root = Path(data_root)
        self.subjects = subjects
        self.stage1_adapter = stage1_adapter
        self.mvpformer = mvpformer_model
        self.config = config
        self.device = device
        
        print("Loading brain features from Stage 1...")
        self._load_brain_features()
        
        print("Loading linguistic features...")
        self._load_linguistic_features()
        
        print("Aligning brain and linguistic features...")
        self._align_features()
        
        print(f"Dataset ready: {len(self)} samples")
    
    def _load_brain_features(self):
        preprocessed_dir = self.data_root
        if preprocessed_dir.name != 'preprocessed':
            preprocessed_dir = preprocessed_dir / 'preprocessed'
        
        ecog_dataset = PodcastECoGDataset(
            data_root=str(preprocessed_dir),
            subjects=self.subjects,
            mvpformer_model=self.mvpformer,
            config=self.config.data,
            device=self.device,
        )
        
        self.stage1_adapter.eval()
        self.stage1_adapter = self.stage1_adapter.to(self.device)
        
        self.canonical_features = {}
        
        with torch.no_grad():
            for subj_id in self.subjects:
                if subj_id in ecog_dataset.features:
                    mvp_feat = torch.from_numpy(ecog_dataset.features[subj_id]).float().to(self.device)
                    canonical_feat = self.stage1_adapter(mvp_feat)
                    self.canonical_features[subj_id] = canonical_feat.cpu().numpy()
    
    def _load_linguistic_features(self):
        data_root = self.data_root
        
        if data_root.name == 'preprocessed':
            data_root = data_root.parent
        
        transcript_file = data_root / self.config.linguistic.transcript_file
        if not transcript_file.exists():
            raise FileNotFoundError(
                f"Transcript not found: {transcript_file}\n"
                f"Expected: {self.config.linguistic.transcript_file}\n"
                f"Update config.linguistic.transcript_file if needed"
            )
        
        self.words_df = pd.read_csv(transcript_file)
        print(f"  Loaded transcript: {len(self.words_df)} words")
        print(f"  Columns: {self.words_df.columns.tolist()}")
        
        emb_file = data_root / self.config.linguistic.embeddings_file
        if not emb_file.exists():
            raise FileNotFoundError(
                f"Embeddings not found: {emb_file}\n"
                f"Expected: {self.config.linguistic.embeddings_file}\n"
                f"Update config.linguistic.embeddings_file if needed"
            )
        
        self.word_embeddings = np.load(emb_file)
        print(f"  Loaded embeddings: {self.word_embeddings.shape}")
        
        expected_dim = self.config.linguistic.embedding_dim
        if self.word_embeddings.shape[1] != expected_dim:
            print(f"  Warning: Expected dim {expected_dim}, got {self.word_embeddings.shape[1]}")
            print(f"  Update config.linguistic.embedding_dim = {self.word_embeddings.shape[1]}")
        
        if len(self.words_df) != len(self.word_embeddings):
            raise ValueError(
                f"Mismatch: {len(self.words_df)} words in transcript "
                f"but {len(self.word_embeddings)} embeddings"
            )
        
        print(f"  Transcript and embeddings matched!")
    
    def _align_features(self):
        has_start_end = 'start' in self.words_df.columns and 'end' in self.words_df.columns
        has_onset = 'onset' in self.words_df.columns or 'word_onset' in self.words_df.columns
        
        if has_start_end:
            print("  Using time-based alignment (start/end columns found)")
            self._align_by_start_end()
        elif has_onset:
            print("  Using time-based alignment (onset column found)")
            self._align_by_time()
        else:
            print("  Using sequence-based alignment (no timing columns)")
            self._align_by_sequence()
        
        print(f"  Created {len(self.aligned_pairs)} aligned pairs")
    
    def _align_by_start_end(self):
        self.aligned_pairs = []
        
        window_size = self.config.data.window_size
        stride = self.config.data.stride
        
        for idx, row in self.words_df.iterrows():
            word_start = row['start']
            word_end = row['end']
            word_center = (word_start + word_end) / 2
            word_idx = idx
            
            window_idx = int((word_center - window_size / 2) / stride)
            window_idx = max(0, window_idx)
            
            for subj_id in self.subjects:
                if subj_id in self.canonical_features:
                    n_windows = len(self.canonical_features[subj_id])
                    if 0 <= window_idx < n_windows:
                        self.aligned_pairs.append({
                            'subject_id': subj_id,
                            'brain_idx': window_idx,
                            'word_idx': word_idx,
                            'word': row['word'],
                            'start': word_start,
                            'end': word_end,
                        })
    
    def _align_by_time(self):
        onset_col = 'onset' if 'onset' in self.words_df.columns else 'word_onset'
        
        self.aligned_pairs = []
        
        sampling_rate = self.config.data.sampling_rate
        stride = self.config.data.stride
        
        for idx, row in self.words_df.iterrows():
            word_onset = row[onset_col]
            word_idx = idx
            
            window_idx = int(word_onset / stride)
            
            for subj_id in self.subjects:
                if subj_id in self.canonical_features:
                    n_windows = len(self.canonical_features[subj_id])
                    if window_idx < n_windows:
                        self.aligned_pairs.append({
                            'subject_id': subj_id,
                            'brain_idx': window_idx,
                            'word_idx': word_idx,
                            'word': row['word'],
                            'onset': word_onset,
                        })
    
    def _align_by_sequence(self):
        self.aligned_pairs = []
        
        n_words = len(self.words_df)
        
        for subj_id in self.subjects:
            if subj_id in self.canonical_features:
                n_windows = len(self.canonical_features[subj_id])
                
                for word_idx in range(n_words):
                    brain_idx = int((word_idx / n_words) * n_windows)
                    brain_idx = min(brain_idx, n_windows - 1)
                    
                    self.aligned_pairs.append({
                        'subject_id': subj_id,
                        'brain_idx': brain_idx,
                        'word_idx': word_idx,
                        'word': self.words_df.iloc[word_idx]['word'],
                        'onset': None,
                    })
    
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
        if brain_idx < n_windows - 1:
            next_brain_feat = torch.from_numpy(self.canonical_features[subj_id][brain_idx + 1]).float()
        else:
            next_brain_feat = torch.from_numpy(self.canonical_features[subj_id][brain_idx - 1]).float()
        
        return {
            'brain_feature': brain_feat,
            'word_embedding': word_emb,
            'next_brain_feature': next_brain_feat,
            'subject_id': subj_id,
            'word_idx': word_idx,
            'brain_idx': brain_idx,
        }


class SubjectSpecificDataset(Dataset):
    def __init__(
        self,
        subject_data: np.ndarray,
        labels: np.ndarray,
        config,
    ):
        self.data = subject_data
        self.labels = labels
        self.config = config
        
        max_trials = min(len(self.data), config.max_trials_per_subject)
        self.data = self.data[:max_trials]
        self.labels = self.labels[:max_trials]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        return {
            'data': torch.from_numpy(self.data[idx]).float(),
            'label': torch.tensor(self.labels[idx]).long(),
        }


def create_stage1_dataloader(
    data_root: str,
    subjects: List[int],
    mvpformer_model,
    config,
    split: str = 'train',
) -> DataLoader:
    import random
    
    seed = config.data.split_seed
    shuffled_subjects = subjects.copy()
    random.Random(seed).shuffle(shuffled_subjects)
    
    n_subjects = len(shuffled_subjects)
    n_train = int(n_subjects * config.data.train_split)
    n_val = int(n_subjects * config.data.val_split)
    
    if split == 'train':
        split_subjects = shuffled_subjects[:n_train]
    elif split == 'val':
        split_subjects = shuffled_subjects[n_train:n_train + n_val]
    else:
        split_subjects = shuffled_subjects[n_train + n_val:]
    
    print(f"  [seed={seed}] {split.upper()} subjects: {split_subjects}")
    
    dataset = PodcastECoGDataset(
        data_root=data_root,
        subjects=split_subjects,
        mvpformer_model=mvpformer_model,
        config=config.data,
        device=config.system.device,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.stage1_training.batch_size,
        shuffle=(split == 'train'),
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        prefetch_factor=config.data.prefetch_factor if config.data.num_workers > 0 else None,
    )
    
    return dataloader


def create_stage2_dataloader(
    data_root: str,
    subjects: List[int],
    stage1_adapter,
    mvpformer_model,
    config,
    split: str = 'train',
) -> DataLoader:
    import random
    
    seed = config.data.split_seed
    shuffled_subjects = subjects.copy()
    random.Random(seed).shuffle(shuffled_subjects)
    
    n_subjects = len(shuffled_subjects)
    n_train = int(n_subjects * config.data.train_split)
    n_val = int(n_subjects * config.data.val_split)
    
    if split == 'train':
        split_subjects = shuffled_subjects[:n_train]
    elif split == 'val':
        split_subjects = shuffled_subjects[n_train:n_train + n_val]
    else:
        split_subjects = shuffled_subjects[n_train + n_val:]
    
    print(f"  [seed={seed}] {split.upper()} subjects: {split_subjects}")
    
    dataset = PodcastLinguisticDataset(
        data_root=data_root,
        subjects=split_subjects,
        stage1_adapter=stage1_adapter,
        mvpformer_model=mvpformer_model,
        config=config,
        device=config.system.device,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.stage2_training.batch_size,
        shuffle=(split == 'train'),
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    
    return dataloader


def create_stage3_dataloader(
    subject_data: np.ndarray,
    labels: np.ndarray,
    config,
    split: str = 'train',
) -> DataLoader:
    dataset = SubjectSpecificDataset(
        subject_data=subject_data,
        labels=labels,
        config=config.stage3_training,
    )
    
    train_size = int(len(dataset) * 0.8)
    if split == 'train':
        indices = range(train_size)
    else:
        indices = range(train_size, len(dataset))
    
    subset = torch.utils.data.Subset(dataset, list(indices))
    
    dataloader = DataLoader(
        subset,
        batch_size=config.stage3_training.batch_size,
        shuffle=(split == 'train'),
        num_workers=min(config.data.num_workers, 2),
    )
    
    return dataloader
