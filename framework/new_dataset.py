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

if __name__ == "__main__":
    pass