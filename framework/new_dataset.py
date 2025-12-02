from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from brain_datasets import PodcastLinguisticDataset
from train_simplified import SimplifiedDataset
import random
from typing import List, Dict, Optional
import sys
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM

def icl_collate_fn(batch):
    """
    ICL 下虽然 batch_size=1，但我们仍然返回带 batch 维的结构：
    - tensor → 在 dim=0 unsqueeze 一下，变成 (1, ...)
    - 非 tensor（比如字符串）→ 收集成长度为 1 的 list
    """
    assert len(batch) == 1, "ICL dataloader 目前假设 batch_size=1"
    sample = batch[0]
    collated = {}

    for k, v in sample.items():
        if torch.is_tensor(v):
            collated[k] = v.unsqueeze(0)  # (1, ...)
        else:
            # target_word 等保留为 list[...]
            collated[k] = [v]
    return collated

def mc_collate_fn(batch):
    """
    batch: List[dict]，每个元素是 BrainLLMMCDataset.__getitem__ 的返回。
    """
    brain_feature = torch.stack([item["brain_feature"] for item in batch], dim=0)  # (B, D)
    subject_ids = torch.tensor([item["subject_id"] for item in batch], dtype=torch.long)
    window_idx = torch.tensor([item["window_idx"] for item in batch], dtype=torch.long)
    
    target_words = [item["target_word"] for item in batch]            # List[str]
    candidate_words = [item["candidate_words"] for item in batch]     # List[List[str]]
    correct_choice_idx = torch.tensor(
        [item["correct_choice_idx"] for item in batch], dtype=torch.long
    )  # (B,)

    return {
        "brain_feature": brain_feature,          # (B, D)
        "subject_id": subject_ids,               # (B,)
        "window_idx": window_idx,                # (B,)
        "target_word": target_words,             # List[str]
        "candidate_words": candidate_words,      # List[List[str]]，每个内部长度 K
        "correct_choice_idx": correct_choice_idx # (B,)
    }

class BrainLLMFinetuneDataset(Dataset):
    """
    用于 LLM 微调的 Dataset：
    - 基于 SimplifiedDataset（一条样本 = 一个 (subject, window_idx)）。
    - 文本结构：不含 demos，只是一个简单的 instruction + <brain> + Word: target。
    """
    def __init__(
        self,
        base_dataset,         # SimplifiedDataset 实例
        tokenizer,
        max_length: int = 256,
    ):
        self.base = base_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.brain_token = "<brain>"
        self.words_df = self.base.words_df

    def __len__(self):
        return len(self.base)

    def _pick_word_idx_from_sample(self, sample_idx: int) -> int:
        """
        从 base.samples[sample_idx]['word_indices'] 中挑一个 word_idx。
        """
        word_indices = self.base.samples[sample_idx]["word_indices"]
        assert len(word_indices) > 0, f"Sample {sample_idx} has no word_indices."
        return random.choice(word_indices)   # 或者 word_indices[0]

    def __getitem__(self, idx):
        base_item = self.base[idx]                 # {'feature', 'word_embedding', ...}
        brain_feature = base_item["feature"]       # (D_ecog,)

        word_idx = self._pick_word_idx_from_sample(idx)
        word = self.words_df.iloc[word_idx]["word"]

        header = (
            "You are a model that predicts the word a subject heard from their brain activity.\n\n"
        )
        prompt = header + f"{self.brain_token}\nWord:"
        target = f" {word}" + self.tokenizer.eos_token

        full_text = prompt + target

        enc = self.tokenizer(
            full_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )
        input_ids = enc.input_ids[0]          # (L,)
        attention_mask = enc.attention_mask[0]

        # 重新计算 prompt 长度：不要 padding
        prompt_enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=False,
        )
        prompt_len = prompt_enc.input_ids.shape[1]
        prompt_len = min(prompt_len, self.max_length)

        labels = input_ids.clone()
        labels[:prompt_len] = -100                 # prompt 部分不计入 loss
        labels[attention_mask == 0] = -100         # padding 位置也不计入 loss

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "brain_feature": brain_feature,
            "target_word": word,
            "subject_id": base_item["subject_id"],
            "window_idx": base_item["window_idx"],
        }

class BrainLLMMCDataset(Dataset):
    """
    用于 MC 微调的 Dataset：
    - 基于 SimplifiedDataset（一条样本 = 一个 (subject, window_idx)）。
    - 不直接构造 token；只返回 brain_feature + candidate_words + correct_choice_idx。
    - 约定：正确答案在 candidate_words[correct_choice_idx]。
    """
    def __init__(
        self,
        base_dataset,         # SimplifiedDataset 实例
        tokenizer,
        mc_num_choices: int = 4,
    ):
        self.base = base_dataset
        self.tokenizer = tokenizer
        self.mc_num_choices = mc_num_choices

        self.brain_token = "<brain>"
        self.words_df = self.base.words_df
        self.num_words = len(self.words_df)

        assert self.mc_num_choices <= self.num_words, \
            f"mc_num_choices={self.mc_num_choices} > num_words={self.num_words}"

        # 用所有 word 的索引做负样本池
        self.all_word_indices = list(range(self.num_words))

    def __len__(self):
        return len(self.base)

    def _pick_word_idx_from_sample(self, sample_idx: int) -> int:
        word_indices = self.base.samples[sample_idx]["word_indices"]
        assert len(word_indices) > 0, f"Sample {sample_idx} has no word_indices."
        return random.choice(word_indices)

    def _sample_negatives(self, pos_idx: int):
        # 从全局词表中采负样本
        candidates = [i for i in self.all_word_indices if i != pos_idx]
        n_neg = self.mc_num_choices - 1
        assert len(candidates) >= n_neg, "词表太小，无法采样足够的负样本。"
        neg_indices = random.sample(candidates, n_neg)
        return neg_indices

    def _build_candidates(self, pos_word_idx: int):
        """
        返回:
            candidate_words: List[str]
            correct_choice_idx: int  (0..mc_num_choices-1)
        """
        neg_word_indices = self._sample_negatives(pos_word_idx)
        candidate_indices = [pos_word_idx] + neg_word_indices
        random.shuffle(candidate_indices)

        candidate_words = [self.words_df.iloc[i]["word"] for i in candidate_indices]
        correct_choice_idx = candidate_indices.index(pos_word_idx)
        return candidate_words, correct_choice_idx

    def __getitem__(self, idx):
        base_item = self.base[idx]
        brain_feature = base_item["feature"]       # (D_ecog,)

        # 正样本 word
        pos_word_idx = self._pick_word_idx_from_sample(idx)
        pos_word = self.words_df.iloc[pos_word_idx]["word"]

        # 候选 + 正确选项 index
        candidate_words, correct_choice_idx = self._build_candidates(pos_word_idx)

        return {
            "brain_feature": torch.as_tensor(brain_feature, dtype=torch.float32),
            "candidate_words": candidate_words,         # List[str]
            "correct_choice_idx": correct_choice_idx,   # int
            "target_word": pos_word,
            "subject_id": base_item["subject_id"],
            "window_idx": base_item["window_idx"],
        }

class BrainLLMIclDataset(Dataset):
    def __init__(
        self,
        base_dataset,      # 这里直接传 SimplifiedDataset
        tokenizer,
        n_demo: int = 4,
        max_length: int = 1024,
        use_mc: bool = False,          # 是否构造多选题
        mc_num_choices: int = 4,       # 多选题选项数（含正确答案）
    ):
        self.base = base_dataset          # SimplifiedDataset 实例
        self.tokenizer = tokenizer
        self.n_demo = n_demo
        self.max_length = max_length

        self.use_mc = use_mc
        self.mc_num_choices = mc_num_choices
        
        self.brain_token = "<brain>"      # 和 finetune 的保持一致
        
        self.words_df = self.base.words_df
    
    def __len__(self):
        # 一条样本对应 base.samples / base.__getitem__ 的一个 index
        return len(self.base)
    
    def _pick_word_idx_from_sample(self, sample_idx: int) -> int:
        """
        从 base.samples[sample_idx]["word_indices"] 中挑一个 word_idx 出来。
        可以选第一个，也可以随机。
        """
        word_indices = self.base.samples[sample_idx]["word_indices"]
        assert len(word_indices) > 0, f"Sample {sample_idx} has no word_indices."
        # 简单起见，用随机一个；也可以用 word_indices[0]
        return random.choice(word_indices)

    def _sample_mc_candidates(self, target_word_idx: int):
        """
        构造多选题的候选词列表：
        - 包含一个正确答案 (target_word)
        - 另外 (mc_num_choices - 1) 个干扰项，从 words_df 中随机采样
        
        返回: List[str]，顺序已打乱
        """
        if (not self.use_mc) or self.mc_num_choices is None or self.mc_num_choices <= 1:
            return None

        n_neg = self.mc_num_choices - 1

        all_indices = list(range(len(self.words_df)))
        # 去掉正确答案的 index
        all_indices = [i for i in all_indices if i != target_word_idx]

        assert len(all_indices) >= n_neg, "词表太小，无法采样足够的干扰词。"

        neg_indices = random.sample(all_indices, k=n_neg)
        candidate_indices = [target_word_idx] + neg_indices

        candidate_words = [self.words_df.iloc[i]["word"] for i in candidate_indices]
        random.shuffle(candidate_words)  # 打乱，使正确答案位置不固定

        return candidate_words
    
    def _build_demo_text(self, word_idx: int, idx_in_demo: int) -> str:
        """
        根据 word_idx 构造 demo 文本。
        """
        word = self.words_df.iloc[word_idx]["word"]
        text = (
            f"Example {idx_in_demo}:\n"
            f"{self.brain_token}\n"
            f"Word: {word}\n\n"
        )
        return text
    
    def __getitem__(self, idx):
        # ========= 1. query 部分 =========
        # 1.1 取 query 样本（一个 (subject, window)）
        base_item = self.base[idx]                    # SimplifiedDataset.__getitem__
        query_brain_feature = base_item["feature"]    # (D_ecog)，query 的脑特征
        
        # 1.2 从对应的 window-level 样本中选一个 word_idx 作为 query 的目标词
        query_word_idx = self._pick_word_idx_from_sample(idx)
        query_word = self.words_df.iloc[query_word_idx]["word"]

        # 如果要做 MC，这里基于 target_word_idx 构造 candidate_words
        candidate_words = self._sample_mc_candidates(query_word_idx)
        
        # ========= 2. demo 部分（索引 + 脑特征 + 文本） =========
        all_indices = list(range(len(self.base)))
        if idx in all_indices:
            all_indices.remove(idx)
        assert len(all_indices) >= self.n_demo, "base dataset 太小，无法抽取足够的 demo。"
        
        demo_indices = random.sample(all_indices, k=self.n_demo)
        
        header = (
            "You are a model that predicts the word a subject heard from their brain activity.\n\n"
            "Below are some examples.\n\n"
        )
        
        demo_texts = []
        demo_brain_features = []     # (n_demo, D_ecog)
        demo_word_indices = []       # 方便 debug / 分析
        demo_words = []
        
        for j, demo_idx in enumerate(demo_indices, start=1):
            # 每个 demo 也取对应 window 的脑特征
            demo_base_item = self.base[demo_idx]
            demo_feat = demo_base_item["feature"]     # (D_ecog)
            
            # 为这个 demo 选一个 word_idx，构造文本
            demo_word_idx = self._pick_word_idx_from_sample(demo_idx)
            word = self.words_df.iloc[demo_word_idx]["word"]
            
            demo_texts.append(self._build_demo_text(demo_word_idx, j))
            
            demo_brain_features.append(demo_feat)
            demo_word_indices.append(demo_word_idx)
            demo_words.append(word)
        
        # 将 demo 的脑特征堆叠成 (n_demo, D_ecog)
        if isinstance(demo_brain_features[0], torch.Tensor):
            demo_brain_feature = torch.stack(demo_brain_features, dim=0)
        else:
            import numpy as np
            demo_brain_feature = np.stack(demo_brain_features, axis=0)
        
        demos_block = "".join(demo_texts)
        
        # ========= 3. query 文本部分 =========
        query_prompt = (
            "Now a new example:\n"
            f"{self.brain_token}\n"
            "Word:"
        )
        
        # target：只训练/评估 query 的 word
        target = f" {query_word}" + self.tokenizer.eos_token
        
        full_text = header + demos_block + query_prompt + target
        
        enc = self.tokenizer(
            full_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        )
        input_ids = enc.input_ids[0]
        attention_mask = enc.attention_mask[0]
        
        # prompt_text：不包含 target 的部分，用来算 prompt_len
        prompt_text = header + demos_block + query_prompt
        prompt_enc = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        )
        prompt_len = prompt_enc.input_ids.shape[1]
        
        # 只在 target 区段计 loss
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        
        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_len": prompt_len,
            "target_word": query_word,
            "target_word_idx": query_word_idx,
            
            # ====== 脑特征部分 ======
            "query_brain_feature": query_brain_feature,
            "demo_brain_features": demo_brain_feature,
            "brain_feature": query_brain_feature,
            
            # ====== 一些元信息 ======
            "demo_indices": demo_indices,
            "demo_word_indices": demo_word_indices,
            "demo_words": demo_words,
            "query_index": idx,
            "subject_id": base_item["subject_id"],
            "window_idx": base_item["window_idx"],
        }

        # 只有在 use_mc=True 时才加这个字段
        if candidate_words is not None:
            item["candidate_words"] = candidate_words

        return item

def create_ICL_dataloader(
    config,
    mvpformer,
    tokenizer,
    split: str = "test",
    subjects: Optional[List[int]] = None,
    n_demo: int = 4,
    max_length: int = 1024,
    icl_mode: str = "generation",    # 新增：和 evaluate 一致
    mc_num_choices: int = 4,         # 新增：MC 模式下的选项数
) -> DataLoader:
    """
    基于 SimplifiedDataset + BrainLLMIclDataset 构建 ICL 用的 DataLoader。
    - icl_mode = "generation": 不需要 candidate_words
    - icl_mode = "mc":        构造 candidate_words（正确 + 若干干扰）
    """
    subjects = subjects or config.data.subjects

    print(f"\nCreating ICL base dataset for split={split}, subjects={subjects} ...")
    base_dataset = SimplifiedDataset(
        data_root=config.data.data_root,
        subjects=subjects,
        mvpformer_model=mvpformer,
        config=config,
        split=split,
        device=config.system.device,
    )

    # 是否开启多选题构造
    use_mc = (icl_mode == "mc")

    icl_dataset = BrainLLMIclDataset(
        base_dataset=base_dataset,
        tokenizer=tokenizer,
        n_demo=n_demo,
        max_length=max_length,
        use_mc=use_mc,                # 关键：根据 icl_mode 开关
        mc_num_choices=mc_num_choices # 关键：多选题个数
    )

    dataloader = DataLoader(
        icl_dataset,
        batch_size=1,         # ICL 场景通常 batch=1
        shuffle=False,        # 评估阶段不需要 shuffle
        num_workers=config.data.num_workers,
        pin_memory=True,
        collate_fn=icl_collate_fn,  # 使用自定义的 collate_fn
    )

    print(f"ICL dataset size: {len(icl_dataset)} samples")
    return dataloader

def create_finetune_dataloaders(
    config,
    mvpformer,
    tokenizer,
    subjects: Optional[List[int]] = None,
    max_length: int = 256,
    batch_size: int = 4,
):
    """
    复用 SimplifiedDataset 做时间/被试切分，然后包一层 BrainLLMFinetuneDataset。
    """
    subjects = subjects or config.data.subjects

    print("\n[Finetune] Creating SimplifiedDataset for TRAIN (split='train')...")
    train_base = SimplifiedDataset(
        data_root=config.data.data_root,
        subjects=subjects,
        mvpformer_model=mvpformer,
        config=config,
        split='train',
        device=config.system.device,
    )

    print("\n[Finetune] Creating SimplifiedDataset for VAL (split='val')...")
    val_base = SimplifiedDataset(
        data_root=config.data.data_root,
        subjects=subjects,
        mvpformer_model=mvpformer,
        config=config,
        split='val',
        device=config.system.device,
    )

    train_ds = BrainLLMFinetuneDataset(
        base_dataset=train_base,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    val_ds = BrainLLMFinetuneDataset(
        base_dataset=val_base,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )

    print(f"[Finetune] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    return train_loader, val_loader

def create_finetune_dataloaders_mc(
    config,
    mvpformer,
    tokenizer,
    subjects,
    max_length: int,
    batch_size: int,
    mc_num_choices: int,
):
    """
    MC 任务的 dataloader：
    - 用 SimplifiedDataset + MVPFormer 提取 feature（和 LM 一致）；
    - 再包一层 BrainLLMMCDataset + mc_collate_fn。
    这里假设你已经有 create_simplified_datasets 或类似 API；
    如果你是通过 create_finetune_dataloaders 生成 SimplifiedDataset，
    可以在那里面拆出 base_train/base_val 再复用。
    """

    # 这个函数示意：你应该根据现有代码获取 train_base / val_base
    subjects = subjects or config.data.subjects

    print("\n[Finetune] Creating SimplifiedDataset for TRAIN (split='train')...")
    base_train = SimplifiedDataset(
        data_root=config.data.data_root,
        subjects=subjects,
        mvpformer_model=mvpformer,
        config=config,
        split='train',
        device=config.system.device,
    )

    print("\n[Finetune] Creating SimplifiedDataset for VAL (split='val')...")
    base_val = SimplifiedDataset(
        data_root=config.data.data_root,
        subjects=subjects,
        mvpformer_model=mvpformer,
        config=config,
        split='val',
        device=config.system.device,
    )


    train_dataset = BrainLLMMCDataset(
        base_dataset=base_train,
        tokenizer=tokenizer,
        mc_num_choices=mc_num_choices,
    )
    val_dataset = BrainLLMMCDataset(
        base_dataset=base_val,
        tokenizer=tokenizer,
        mc_num_choices=mc_num_choices,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        collate_fn=mc_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
        collate_fn=mc_collate_fn,
    )
    return train_loader, val_loader


if __name__ == "__main__":
    pass