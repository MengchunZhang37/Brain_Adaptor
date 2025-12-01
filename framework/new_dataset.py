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

class BrainToLlamaAdapter(nn.Module):
    def __init__(self, brain_dim: int, llama_hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(brain_dim, llama_hidden_dim)

    def forward(self, brain_feature: torch.Tensor) -> torch.Tensor:
        """
        brain_feature: (B, D_brain) or (B, 1, D_brain)
        return: (B, D_llama)
        """
        if brain_feature.dim() == 3:
            # e.g. (B, 1, D) -> (B, D)
            brain_feature = brain_feature.squeeze(1)
        print("Brain feature shape:", brain_feature.shape)
        return self.proj(brain_feature)

class BrainLLMFinetuneDataset(Dataset):
    def __init__(self, base_dataset, tokenizer, max_length: int = 512):
        self.base = base_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 假设你在 tokenizer 中已经添加了一个特殊 token 用来表示脑特征
        # tokenizer.add_special_tokens({'additional_special_tokens': ['<brain>']})
        self.brain_token = "<brain>"
    
    def __len__(self):
        return len(self.base)
    
    def __getitem__(self, idx):
        base_item = self.base[idx]
        brain_feat = base_item["brain_feature"]              # (D,)
        word_idx = base_item["word_idx"]
        
        # 拿到真实单词（或你想预测的文本单位）
        word = self.base.words_df.iloc[word_idx]["word"]
        # 你也可以改成短句、上下文片段等
        
        # 构造 prompt + target 文本
        # 这里示例：只训练 “word” 这一段，前面的 prompt 不计入 loss
        prompt = (
            "You are a model that predicts the word a subject heard, "
            "given their brain activity.\n"
            f"{self.brain_token}\n"
            "The word is:"
        )
        target = f" {word}"  # 前面加空格是为了 tokenizer 对英文单词更自然
        
        # 拼接成完整输入序列，后面 + eos
        full_text = prompt + target + self.tokenizer.eos_token
        
        enc = self.tokenizer(
            full_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        )
        input_ids = enc.input_ids[0]          # (L,)
        attention_mask = enc.attention_mask[0]
        
        # 构造 labels: prompt 部分为 -100，只在 target+eos 区段训练
        # 简单做法：再单独 encode 一次 prompt，长度用来划分边界
        prompt_enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        )
        prompt_len = prompt_enc.input_ids.shape[1]
        
        labels = input_ids.clone()
        # prompt 位置不参与 loss
        labels[:prompt_len] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            # 额外输出给 adapter 用
            "brain_feature": brain_feat,     # (D,)
            "subject_id": base_item["subject_id"],
            "word_idx": word_idx,
            "brain_idx": base_item["brain_idx"],
        }

class BrainLLMIclDataset(Dataset):
    def __init__(
        self,
        base_dataset,      # 这里直接传 SimplifiedDataset
        tokenizer,
        n_demo: int = 4,
        max_length: int = 1024,
    ):
        self.base = base_dataset          # SimplifiedDataset 实例
        self.tokenizer = tokenizer
        self.n_demo = n_demo
        self.max_length = max_length
        
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
        # 1. 取 query 样本（一个 (subject, window)）
        base_item = self.base[idx]              # SimplifiedDataset.__getitem__
        brain_feature = base_item["feature"]    # 脑特征
        
        # 从对应的 window-level 样本中选一个 word_idx
        query_word_idx = self._pick_word_idx_from_sample(idx)
        query_word = self.words_df.iloc[query_word_idx]["word"]
        
        # 2. 构造 demo 样本 index（仍然在当前 Dataset 的 index 空间里采样）
        all_indices = list(range(len(self.base)))
        if idx in all_indices:
            all_indices.remove(idx)
        assert len(all_indices) >= self.n_demo, "base dataset 太小，无法抽取足够的 demo。"
        
        demo_indices = random.sample(all_indices, k=self.n_demo)
        
        # 3. 文本部分
        header = (
            "You are a model that predicts the word a subject heard from their brain activity.\n\n"
            "Below are some examples.\n\n"
        )
        
        demo_texts = []
        for j, demo_idx in enumerate(demo_indices, start=1):
            demo_word_idx = self._pick_word_idx_from_sample(demo_idx)
            demo_texts.append(self._build_demo_text(demo_word_idx, j))
        
        demos_block = "".join(demo_texts)
        
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
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_len": prompt_len,
            "target_word": query_word,
            "target_word_idx": query_word_idx,
            
            # 脑特征，用于外面映射到 LLM embedding
            "brain_feature": brain_feature,
            
            # 一些元信息
            "demo_indices": demo_indices,
            "query_index": idx,
            "subject_id": base_item["subject_id"],
            "window_idx": base_item["window_idx"],
        }

def create_ICL_dataloader(
    config,
    mvpformer,
    tokenizer,
    split: str = "test",
    subjects: Optional[List[int]] = None,
    n_demo: int = 4,
    max_length: int = 1024,
) -> DataLoader:
    """
    基于 SimplifiedDataset + BrainLLMIclDataset 构建 ICL 用的 DataLoader。
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

    icl_dataset = BrainLLMIclDataset(
        base_dataset=base_dataset,
        tokenizer=tokenizer,
        n_demo=n_demo,
        max_length=max_length,
    )

    dataloader = DataLoader(
        icl_dataset,
        batch_size=1,         # ICL 场景通常 batch=1
        shuffle=False,        # 评估阶段不需要 shuffle
        num_workers=config.data.num_workers,
        pin_memory=True,
    )
    print(f"ICL dataset size: {len(icl_dataset)} samples")
    return dataloader

if __name__ == "__main__":
    pass