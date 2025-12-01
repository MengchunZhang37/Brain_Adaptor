import sys
from pathlib import Path
import torch
import yaml
import importlib
import argparse

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


from config import get_config, get_simplified_config, print_split_info
# from hierarchical_adapter_mvp_llama import SubjectInvariantAdapter, BrainToLlamaAdapter
# from brain_datasets import create_ICL_dataloader

from new_dataset import create_ICL_dataloader
from simplified_adapter import create_simplified_adapter
from train_simplified import SimplifiedDataset, load_mvpformer  

@torch.no_grad()
def evaluate_icl_with_adapter(
    model,
    tokenizer,
    adapter,
    dataloader: DataLoader,
    device: str = "cuda",
    max_eval_samples: Optional[int] = None,
    icl_mode: str = "generation",   # "generation" 或 "mc"
):
    """
    用 adapter 把 demo + query 的 ECoG 特征映射到 LLaMA hidden space，
    替换掉 prompt 中每个 <brain> token 的 embedding，然后做 ICL 预测。

    icl_mode:
        - "generation": 直接用 logits 在 target 第一个位置上选 argmax token，
          与 target_word 比较。
        - "mc": multiple-choice 模式。要求 batch 中提供:
              batch["candidate_words"]: List[List[str]] (B 个样本，每个是若干选项)
          我们在 target 第一个位置上，对每个选项的「第一个 token」计算概率，
          选概率最高的选项，判断是否为正确答案。
    """
    assert icl_mode in ("generation", "mc"), f"Unsupported icl_mode: {icl_mode}"

    model.eval().to(device)
    adapter.eval().to(device)

    brain_token_id = tokenizer.convert_tokens_to_ids("<brain>")
    hidden_size_llama = model.get_input_embeddings().embedding_dim

    # 通过一个 batch 动态检查 adapter 输出维度
    _example_batch = next(iter(dataloader))
    _feat = _example_batch["query_brain_feature"].to(device)  # (1, D_ecog) 或 (D_ecog,)
    if _feat.ndim == 1:
        _feat = _feat.unsqueeze(0)
    _out = adapter(_feat)                                    # (1, D_out)
    adapter_out_dim = _out.shape[-1]
    if adapter_out_dim != hidden_size_llama:
        raise ValueError(
            f"adapter 输出维度 {adapter_out_dim} 与 LLaMA hidden_size {hidden_size_llama} 不匹配。"
            "请在训练 adapter 时就把输出维度设成 LLaMA hidden_size，"
            "或者在本脚本中额外加一个线性层做映射。"
        )

    total = 0
    correct = 0

    for i, batch in enumerate(tqdm(dataloader, desc=f"ICL eval ({icl_mode}) with adapter")):
        if max_eval_samples is not None and total >= max_eval_samples:
            break

        input_ids = batch["input_ids"].to(device)              # (B=1, T)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)                    # (1, T)
        target_word = batch["target_word"][0]                  # str

        demo_feats = batch["demo_brain_features"].to(device)   # (1, n_demo, D_ecog) after collate
        query_feat = batch["query_brain_feature"].to(device)   # (1, D_ecog) or (D_ecog,)

        # 处理一下形状（目前假设 batch_size = 1）
        if demo_feats.ndim == 3:
            # (B=1, n_demo, D) → (n_demo, D)
            demo_feats = demo_feats[0]
        if query_feat.ndim == 1:
            query_feat = query_feat.unsqueeze(0)               # (1, D_ecog)

        # [n_demo + 1, D_ecog]
        all_feats = torch.cat([demo_feats, query_feat], dim=0)

        # 1) adapter 映射到 LLaMA hidden 空间
        brain_embeds = adapter(all_feats)                      # (n_demo+1, hidden_size)

        n_demo = demo_feats.shape[0]

        # 2) 原始 token embedding
        input_embeds = model.get_input_embeddings()(input_ids) # (1, T, hidden_size)

        # 3) 找到所有 <brain> 位置
        brain_positions = (input_ids[0] == brain_token_id).nonzero(as_tuple=False).squeeze(1)
        if brain_positions.numel() == 0:
            # 理论上不应该发生
            total += 1
            continue

        # 期望有 n_demo + 1 个 <brain>（每个 demo 一次 + query 一次）
        if brain_positions.numel() != n_demo + 1:
            # 做个最小长度的匹配，避免直接崩掉
            min_len = min(brain_positions.numel(), n_demo + 1)
            brain_positions = brain_positions[:min_len]
            brain_embeds = brain_embeds[:min_len]

        # 按顺序把 embedding 写进去
        for k, pos in enumerate(brain_positions):
            p = pos.item()
            input_embeds[0, p, :] = brain_embeds[k]

        # 4) 用 inputs_embeds 前向
        outputs = model(inputs_embeds=input_embeds,
                        attention_mask=attention_mask)
        logits = outputs.logits                                 # (1, T, V)

        # 5) 找 target 段的第一个位置（labels != -100）
        target_positions = (labels[0] != -100).nonzero(as_tuple=False)
        if len(target_positions) == 0:
            total += 1
            continue
        first_pos = target_positions[0].item()                 # int

        # ========= icl_mode = "generation" =========
        if icl_mode == "generation":
            pred_token_id = logits[0, first_pos].argmax(dim=-1).item()
            pred_token = tokenizer.decode([pred_token_id]).strip()

            is_correct = (target_word.strip().lower() == pred_token.lower())
            total += 1
            correct += int(is_correct)

            # 你可以保留轻量的 debug 信息
            # print(f"[GEN] Sample {i}: target_word = '{target_word}' | pred = '{pred_token}' | correct = {is_correct}")

        # ========= icl_mode = "mc" =========
        else:  # "mc"
            if "candidate_words" not in batch:
                raise KeyError(
                    "icl_mode='mc' 需要 batch 中包含 'candidate_words' 字段，"
                    "形如 List[List[str]]，每个样本一个候选列表。"
                )

            candidate_words_list = batch["candidate_words"][0]  # List[str] 对应当前样本
            # 计算每个候选的第一个 token 的 logit
            candidate_token_ids = []
            for w in candidate_words_list:
                # 不加特殊 token，只取第一个 token
                ids = tokenizer.encode(w, add_special_tokens=False)
                if len(ids) == 0:
                    raise ValueError(f"候选词 '{w}' 编码结果为空，请检查 tokenizer 或候选文本。")
                candidate_token_ids.append(ids[0])  # 只看第一个 token

            candidate_token_ids_tensor = torch.tensor(candidate_token_ids, device=logits.device)  # (C,)
            # logits[0, first_pos] 形状 (V,)，选出这些候选 token 对应的 logit
            candidate_logits = logits[0, first_pos, candidate_token_ids_tensor]  # (C,)
            pred_idx = candidate_logits.argmax(dim=-1).item()
            pred_word = candidate_words_list[pred_idx]

            # 正确性判定：用 target_word 进行比较（如果你有 ground-truth index，也可以改成对比 index）
            is_correct = (pred_word.strip().lower() == target_word.strip().lower())
            total += 1
            correct += int(is_correct)

            # debug 信息
            # print(f"[MC] Sample {i}: target = '{target_word}' | pred = '{pred_word}' | "
            #       f"choices = {candidate_words_list} | correct = {is_correct}")

    acc = correct / max(total, 1)
    print(f"ICL ({icl_mode}, with adapter) accuracy: {acc:.4f}  ({correct}/{total})")
    return acc


# =========================
# 4. CLI 入口
# =========================

def main():
    parser = argparse.ArgumentParser(description="ICL evaluation with adapter and LLaMA")
    parser.add_argument('--config', type=str, default='simplified')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--mvpformer_checkpoint', type=str, required=True)
    parser.add_argument('--adapter_checkpoint', type=str, required=True)

    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--subjects', type=int, nargs='+', default=None)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])

    parser.add_argument('--llm_name', type=str,
                        default='meta-llama/Llama-3-8B-Instruct')
    parser.add_argument('--n_demo', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--max_eval_samples', type=int, default=None)

    parser.add_argument('--icl_mode', type=str, default='generation',
                        choices=['generation', 'mc'],
                        help="ICL 评估模式：'generation' 直接预测；'mc' 多选题式评分")

    # 建议加一个选项数参数
    parser.add_argument('--mc_num_choices', type=int, default=4,
                        help="多选题模式下的选项个数（包含正确答案）")

    args = parser.parse_args()

    # ---- 1) config & 数据相关 ----
    config = get_simplified_config(args.config)
    config.data.data_root = args.data_root
    config.data.output_root = args.output_dir
    config.mvpformer.checkpoint_path = args.mvpformer_checkpoint

    print(f"Config: {args.config}")
    print_split_info(config.data)

    device = config.system.device

    # ---- 2) 加载 MVPFormer ----
    mvpformer = load_mvpformer(config)

    # ---- 3) 准备 tokenizer & LLaMA ----
    print("\nLoading LLaMA & tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
    # 确保有 <brain> 这个特殊 token
    tokenizer.add_special_tokens({'additional_special_tokens': ['<brain>']})
    model = AutoModelForCausalLM.from_pretrained(args.llm_name)
    model.resize_token_embeddings(len(tokenizer))

    # ---- 4) 构建 ICL dataloader ----
    icl_loader = create_ICL_dataloader(
        config=config,
        mvpformer=mvpformer,
        tokenizer=tokenizer,
        split=args.split,
        subjects=args.subjects,
        n_demo=args.n_demo,
        max_length=args.max_length,
        icl_mode=args.icl_mode,            # 传给 dataloader
        mc_num_choices=args.mc_num_choices # 传给 dataloader
    )

    # ---- 5) 加载 adapter ----
    print("\nLoading adapter checkpoint ...")
    adapter = create_simplified_adapter(config.adapter)
    ckpt = torch.load(args.adapter_checkpoint, map_location='cpu')
    adapter.load_state_dict(ckpt['model_state_dict'])
    adapter.to(device)

    # ---- 6) ICL 评估 ----
    acc = evaluate_icl_with_adapter(
        model=model,
        tokenizer=tokenizer,
        adapter=adapter,
        dataloader=icl_loader,
        device=device,
        max_eval_samples=args.max_eval_samples,
        icl_mode=args.icl_mode,            # 和 dataloader 保持一致
    )

    print("\nDone.")
    print(f"Final ICL (with adapter) accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()