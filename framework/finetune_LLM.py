from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PrefixTuningConfig, get_peft_model
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from config import get_simplified_config, print_split_info
from train_simplified import load_mvpformer
from simplified_adapter import create_simplified_adapter
from new_dataset import create_finetune_dataloaders, create_finetune_dataloaders_mc

from tqdm import tqdm

def build_peft_llm(
    llm_name: str,
    tokenizer,
    ft_method: str = "lora",
    r: int = 4,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    prefix_length: int = 30,
):
    print(f"\nLoading base LLaMA model from {llm_name} ...")
    base_model = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.bfloat16)
    base_model.resize_token_embeddings(len(tokenizer))

    if ft_method == "lora":
        print("Using LoRA finetuning.")
        peft_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
    elif ft_method == "prefix":
        print("Using Prefix Tuning.")
        peft_config = PrefixTuningConfig(
            task_type="CAUSAL_LM",
            num_virtual_tokens=prefix_length,
        )
    else:
        raise ValueError(f"Unknown ft_method: {ft_method}")

    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    return model

def inject_brain_embeddings(
    model,
    adapter,
    input_ids: torch.Tensor,      # (B, T)
    brain_features: torch.Tensor, # (B, D_ecog)
    brain_token_id: int,
    device: torch.device,
):
    """
    将每个样本中的 <brain> token 的 embedding 替换为 adapter(feature)。
    """
    input_ids = input_ids.to(device)
    brain_features = brain_features.to(device)

    input_embeds = model.get_input_embeddings()(input_ids)  # (B, T, hidden)
    B, T, hidden = input_embeds.shape

    if brain_features.ndim == 1:
        brain_features = brain_features.unsqueeze(0)        # (1, D)

    brain_embeds = adapter(brain_features)                  # (B, hidden)

    for b in range(B):
        positions = (input_ids[b] == brain_token_id).nonzero(as_tuple=False).squeeze(1)
        if positions.numel() == 0:
            continue
        # 如果 prompt 中有多个 <brain>，全部用同一个 brain_embeds[b]
        for pos in positions:
            p = pos.item()
            input_embeds[b, p, :] = brain_embeds[b]

    return input_embeds

def compute_mc_loss_batch(
    model,
    adapter,
    tokenizer,
    batch,
    device,
    mc_num_choices: int = 4,
    max_length: int = 256,
    brain_token: str = "<brain>",
):
    """
    对一个 batch 计算 MC loss：
    - 输入：batch 来自 MC Dataset 的 collate_fn
      需要包含:
        - brain_feature: (B, D)
        - candidate_words: List[List[str]]，len = B
        - correct_choice_idx: List[int] 或 (B,) tensor
    - 输出：标量 loss
    """
    brain_features = batch["brain_feature"].to(device)      # (B, D)
    candidate_words_batch = batch["candidate_words"]        # List[List[str]]，len = B
    correct_choice_idx = batch["correct_choice_idx"]
    if not torch.is_tensor(correct_choice_idx):
        correct_choice_idx = torch.tensor(correct_choice_idx, dtype=torch.long)
    correct_choice_idx = correct_choice_idx.to(device)      # (B,)

    B = brain_features.size(0)
    K = mc_num_choices

    # sanity check：每个样本的候选数必须一致
    assert all(len(cands) == K for cands in candidate_words_batch), \
        f"Each sample must have {K} candidates."

    header = "You are a model that predicts the word a subject heard from their brain activity.\n\n"
    prompt = header + f"{brain_token}\nCandidate:"

    # 1) 计算 prompt_len —— 和 eval 逻辑保持一致，add_special_tokens=True
    prompt_enc = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=False,
        add_special_tokens=True,
    )
    prompt_len = prompt_enc.input_ids.shape[1]
    if prompt_len >= max_length:
        raise ValueError(
            f"Prompt too long: prompt_len={prompt_len}, max_length={max_length}"
        )

    # 2) 构造 BK 个 full_text
    texts = []
    for b in range(B):
        cands = candidate_words_batch[b]
        for w in cands:
            target = " " + w + tokenizer.eos_token
            full_text = prompt + target
            texts.append(full_text)


    # 3) 一次性 tokenize BK 个序列
    enc = tokenizer(
        texts,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length",
        add_special_tokens=True,
    )
    input_ids = enc.input_ids.to(device)           # (B*K, T)
    attention_mask = enc.attention_mask.to(device) # (B*K, T)

    BK, T = input_ids.shape
    if prompt_len >= T:
        raise ValueError(
            f"prompt_len={prompt_len} >= seq_len={T}; candidate token may be truncated."
        )

    # 4) brain_feature 重复 K 次，对应 BK 条序列
    brain_features_rep = brain_features.repeat_interleave(K, dim=0)  # (B*K, D)

    brain_token_id = tokenizer.convert_tokens_to_ids(brain_token)

    # 5) 注入 brain embedding
    inputs_embeds = inject_brain_embeddings(
        model=model,
        adapter=adapter,
        input_ids=input_ids,
        brain_features=brain_features_rep,
        brain_token_id=brain_token_id,
        device=device,
    )

    # 6) 前向（不需要 labels）
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = outputs.logits  # (B*K, T, V)

    # 则应该用 logits[:, candidate_pos-1, :] 来打分 input_ids[:, candidate_pos]
    shift_logits = logits[:, :-1, :].contiguous() # (B*K, T-1, V)
    shift_labels = input_ids[:, 1:].contiguous()  # (B*K, T-1)
    
    # 2. 计算整个序列的 Log Softmax
    # (B*K, T-1)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # 3. 取出真实 token 对应的 log_prob
    # gather 需要 index 维度一致，所以 unsqueeze 最后一维
    target_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    
    # 4. 构建 Mask：只计算 Candidate 部分的 loss
    # (a) Padding Mask: 忽略 padding 的部分
    # shift_labels 对应的 mask 是原 mask 去掉第一个
    if attention_mask is not None:
        active_mask = attention_mask[:, 1:].bool()
    else:
        active_mask = (shift_labels != tokenizer.pad_token_id)

    # (b) Position Mask: 忽略 Prompt 的部分
    # Candidate 从 input_ids[prompt_len] 开始
    # 对应 shift_labels 的索引是 prompt_len - 1
    seq_ids = torch.arange(shift_labels.size(1), device=device).unsqueeze(0) # (1, T-1)
    candidate_start_idx = prompt_len - 1
    candidate_mask = seq_ids >= candidate_start_idx
    
    # (c) 最终 Mask = 既不是 padding，也是 candidate 部分
    final_mask = active_mask & candidate_mask

    # 5. 求和得到每个选项的总分
    # 乘上 mask 后求和 (B*K,)
    candidate_scores = (target_log_probs * final_mask.float()).sum(dim=1)
    
    # 6. Reshape 回 (B, K)
    token_scores = candidate_scores.view(B, K)
    # 9) 使用 correct_choice_idx 作为 label（不再假定 index 0）
    labels = correct_choice_idx                             # (B,)
    loss = F.cross_entropy(token_scores, labels)

    return loss

def train_one_epoch(
    model,
    adapter,
    dataloader: DataLoader,
    tokenizer,
    optimizer,
    device: torch.device,
    epoch: int,
    freeze_adapter: bool = True,
    log_interval: int = 50,
):
    model.train()
    if freeze_adapter:
        adapter.eval()
        for p in adapter.parameters():
            p.requires_grad = False
    else:
        adapter.train()

    brain_token_id = tokenizer.convert_tokens_to_ids("<brain>")

    total_loss = 0.0
    n_steps = 0

    pbar = tqdm(dataloader, desc=f"[Train] Epoch {epoch}", leave=False)
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)           # (B, T)
        attention_mask = batch["attention_mask"].to(device) # (B, T)
        labels = batch["labels"].to(device)                 # (B, T)
        brain_feature = batch["brain_feature"].to(device)   # (B, D)

        optimizer.zero_grad()

        inputs_embeds = inject_brain_embeddings(
            model=model,
            adapter=adapter,
            input_ids=input_ids,
            brain_features=brain_feature,
            brain_token_id=brain_token_id,
            device=device,
        )

        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        valid_mask = (labels != -100)
        n_valid = valid_mask.sum().item()
        if n_valid == 0:
            print(f"[BUG] n_valid == 0 at step {step}")
            print("  input_ids[0]:", input_ids[0][:50])
            print("  labels[0]:   ", labels[0][:50])
            raise ValueError("All labels are -100 in this batch!")

        # 2. 检查 loss 是否是正常数值
        if not torch.isfinite(loss):
            with torch.no_grad():
                max_logit = outputs.logits.abs().max().item()
            print(f"[BUG] Non-finite loss at step {step}: {loss}")
            print(f"      max |logit| = {max_logit}")
            print("  n_valid:", n_valid)
            raise ValueError("Non-finite loss")

        # if step == 0 or step % 50 == 0:
        #     print(f"[DEBUG] step {step}, loss = {loss.item()}")

        if not torch.isfinite(loss):
            print(f"[WARN] Non-finite loss at step {step}: {loss}")
            # 这里可以打印 logits 范围
            with torch.no_grad():
                max_logit = outputs.logits.abs().max().item()
            print(f"[DEBUG] max |logit| = {max_logit}")
            raise ValueError("Non-finite loss, aborting to debug.")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_steps += 1

        if (step + 1) % log_interval == 0:
            avg_loss = total_loss / n_steps
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    avg_loss = total_loss / max(n_steps, 1)
    return avg_loss

def train_one_epoch_mc(
    model,
    adapter,
    tokenizer,
    dataloader,
    optimizer,
    device,
    epoch: int,
    mc_num_choices: int = 4,
    max_length: int = 256,
    freeze_adapter: bool = True,
    brain_token: str = "<brain>",
    **kwargs
):

    model.train()
    if freeze_adapter and adapter is not None:
        adapter.eval()
        for p in adapter.parameters():
            p.requires_grad = False
    elif adapter is not None:
        adapter.train()

    total_loss = 0.0
    n_steps = 0

    pbar = tqdm(dataloader, desc=f"[Train-MC] Epoch {epoch}", leave=True)
    for step, batch in enumerate(pbar):
        optimizer.zero_grad()

        loss = compute_mc_loss_batch(
            model=model,
            adapter=adapter,
            tokenizer=tokenizer,
            batch=batch,
            device=device,
            mc_num_choices=mc_num_choices,
            max_length=max_length,
            brain_token=brain_token,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_steps += 1

        avg_loss = total_loss / n_steps
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    return total_loss / max(n_steps, 1)

def train_one_epoch_unified(
    model,
    adapter,
    dataloader: DataLoader,
    tokenizer,
    optimizer,
    device: torch.device,
    epoch: int,
    llm_task: str = "lm",
    freeze_adapter: bool = True,
    log_interval: int = 50,
    mc_num_choices: int = 4,
    max_length: int = 256,
):
    """
    Unified training entry for both LM and MC tasks.

    llm_task:
        - "lm": token-level cross-entropy (language modeling)
        - "mc": multiple-choice word prediction
    """
    if llm_task == "lm":
        return train_one_epoch(
            model=model,
            adapter=adapter,
            dataloader=dataloader,
            tokenizer=tokenizer,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            freeze_adapter=freeze_adapter,
            log_interval=log_interval,
        )
    elif llm_task == "mc":
        return train_one_epoch_mc(
            model=model,
            adapter=adapter,
            tokenizer=tokenizer,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            mc_num_choices=mc_num_choices,
            max_length=max_length,
            freeze_adapter=freeze_adapter,
        )
    else:
        raise ValueError(f"Unsupported llm_task: {llm_task}")


@torch.no_grad()
def evaluate_finetune(
    model,
    adapter,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device,
    freeze_adapter: bool = True,
):
    
    model.eval()
    if freeze_adapter and adapter is not None:
        adapter.eval()
        for p in adapter.parameters():
            p.requires_grad = False

    brain_token_id = tokenizer.convert_tokens_to_ids("<brain>")

    total_loss = 0.0
    n_batches = 0

    total = 0        # 有效样本数（至少有一个 label != -100）
    correct = 0

    for batch in tqdm(dataloader, desc="[Val]", leave=False):
        input_ids = batch["input_ids"].to(device)           # (B, T)
        attention_mask = batch["attention_mask"].to(device) # (B, T)
        labels = batch["labels"].to(device)                 # (B, T)
        brain_feature = batch["brain_feature"].to(device)   # (B, D)

        # 1) 用 adapter + brain_feature 替换 <brain> 的 embedding
        inputs_embeds = inject_brain_embeddings(
            model=model,
            adapter=adapter,
            input_ids=input_ids,
            brain_features=brain_feature,
            brain_token_id=brain_token_id,
            device=device,
        )

        # 2) 前向 + loss
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        total_loss += loss.item()
        n_batches += 1

        logits = outputs.logits  # (B, T, V)
        B, T, V = logits.shape

        # 3) 构造 mask：labels != -100 的位置才参与监督
        label_mask = (labels != -100)           # (B, T)
        has_label = label_mask.any(dim=1)       # (B,)

        # 这一 batch 里可能存在完全没有监督位置的样本（极小概率），先过滤掉
        if has_label.sum().item() == 0:
            continue

        valid_idx = torch.nonzero(has_label, as_tuple=False).squeeze(1)  # (B_valid,)
        logits_valid = logits[valid_idx]         # (B_valid, T, V)
        labels_valid = labels[valid_idx]         # (B_valid, T)
        label_mask_valid = label_mask[valid_idx] # (B_valid, T)

        # 4) 每个样本中第一个 label != -100 的位置：即 target 第一个 token 的位置
        first_pos = label_mask_valid.float().argmax(dim=1)  # (B_valid,)

        batch_idx = torch.arange(first_pos.size(0), device=device)
        # 该位置上的预测 token & ground truth token id
        pred_ids = logits_valid[batch_idx, first_pos].argmax(dim=-1)  # (B_valid,)
        gt_ids = labels_valid[batch_idx, first_pos]                   # (B_valid,)

        correct += (pred_ids == gt_ids).sum().item()
        total += first_pos.size(0)

    avg_loss = total_loss / max(n_batches, 1)
    acc = correct / max(total, 1)

    return avg_loss, acc

@torch.no_grad()
def evaluate_finetune_mc(
    model,
    adapter,
    dataloader,
    tokenizer,
    device,
    mc_num_choices: int = 4,
    max_length: int = 256,
    freeze_adapter: bool = True,
):
    model.eval()
    if adapter is not None and freeze_adapter:
        adapter.eval()
        for p in adapter.parameters():
            p.requires_grad = False

    header = "You are a model that predicts the word a subject heard from their brain activity.\n\n"
    brain_token = "<brain>"
    prompt = header + f"{brain_token}\nCandidate:"

    # 用和后面完全一致的设置计算 prompt_len
    prompt_enc = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=False,
        add_special_tokens=True,
    )
    prompt_len = prompt_enc.input_ids.shape[1]
    if prompt_len >= max_length:
        raise ValueError(
            f"Prompt too long: prompt_len={prompt_len}, max_length={max_length}"
        )

    total_loss = 0.0
    n_batches = 0
    total = 0
    correct = 0

    brain_token_id = tokenizer.convert_tokens_to_ids(brain_token)

    for batch in tqdm(dataloader, desc="[Val-MC]", leave=False):
        brain_features = batch["brain_feature"].to(device)          # (B, D)
        candidate_words_batch = batch["candidate_words"]            # List[List[str]]
        correct_choice_idx = batch["correct_choice_idx"]            # List[int] or Tensor
        if not torch.is_tensor(correct_choice_idx):
            correct_choice_idx = torch.tensor(correct_choice_idx, dtype=torch.long)
        correct_choice_idx = correct_choice_idx.to(device)          # (B,)

        B = brain_features.size(0)
        K = mc_num_choices

        # 1) 构造 BK 个文本（顺序严格与 candidate_words 对齐）
        texts = []
        for b in range(B):
            cands = candidate_words_batch[b]
            assert len(cands) == K, f"Got {len(cands)} candidates, expected {K}"
            for w in cands:
                target = " " + w + tokenizer.eos_token
                full_text = prompt + target
                texts.append(full_text)

        enc = tokenizer(
            texts,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length",
            add_special_tokens=True,
        )
        input_ids = enc.input_ids.to(device)           # (B*K, T)
        attention_mask = enc.attention_mask.to(device) # (B*K, T)

        BK, T = input_ids.shape
        if prompt_len >= T:
            raise ValueError(
                f"prompt_len={prompt_len} >= seq_len={T}; candidate token may be truncated."
            )

        brain_features_rep = brain_features.repeat_interleave(K, dim=0)  # (B*K, D)

        inputs_embeds = inject_brain_embeddings(
            model=model,
            adapter=adapter,
            input_ids=input_ids,
            brain_features=brain_features_rep,
            brain_token_id=brain_token_id,
            device=device,
        )

        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = outputs.logits                        # (B*K, T, V)

        # candidate 第一个 token 的 score：注意 CausalLM 的 off-by-one
        candidate_pos = prompt_len
        logits_first = logits[:, candidate_pos - 1, :]         # (B*K, V)
        candidate_token_ids = input_ids[:, candidate_pos]      # (B*K,)
        token_scores = logits_first.gather(
            -1, candidate_token_ids.unsqueeze(-1)
        ).squeeze(-1)                                          # (B*K,)

        token_scores = token_scores.view(B, K)                 # (B, K)

        # CE loss：label 是 correct_choice_idx（0..K-1）
        labels = correct_choice_idx                            # (B,)
        loss = F.cross_entropy(token_scores, labels)

        total_loss += loss.item()
        n_batches += 1

        # MC accuracy
        pred_idx = token_scores.argmax(dim=1)                  # (B,)
        correct += (pred_idx == labels).sum().item()
        total += B

    avg_loss = total_loss / max(n_batches, 1)
    acc = correct / max(total, 1)

    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser(
        description="Brain-conditioned LLaMA finetune (LoRA / Prefix, LM / MC tasks)"
    )
    parser.add_argument('--config', type=str, default='simplified')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--mvpformer_checkpoint', type=str, required=True)
    parser.add_argument('--adapter_checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs_llm_ft')

    parser.add_argument('--subjects', type=int, nargs='+', default=None)

    parser.add_argument('--llm_name', type=str,
                        default='meta-llama/Llama-3-8B-Instruct')
    parser.add_argument('--ft_method', type=str,
                        choices=['lora', 'prefix'], default='lora')

    parser.add_argument('--llm_task', type=str,
                        choices=['lm', 'mc'], default='lm',
                        help="选择 finetune 任务类型：'lm' = 生成 CE，'mc' = 多选候选解码")
    parser.add_argument('--mc_num_choices', type=int, default=4,
                        help="MC 任务中每个样本的候选数（含正确答案）")

    parser.add_argument('--llm_lr', type=float, default=1e-4)
    parser.add_argument('--llm_epochs', type=int, default=3)
    parser.add_argument('--llm_batch_size', type=int, default=4)
    parser.add_argument('--llm_max_length', type=int, default=256)
    parser.add_argument('--freeze_adapter', action='store_true', default=True)

    args = parser.parse_args()

    # ---- 1) config & 路径 ----
    config = get_simplified_config(args.config)
    config.data.data_root = args.data_root
    config.data.output_root = args.output_dir
    config.mvpformer.checkpoint_path = args.mvpformer_checkpoint

    print(f"Config: {args.config}")
    print_split_info(config.data)

    device = torch.device(config.system.device)

    # ---- 2) MVPFormer ----
    mvpformer = load_mvpformer(config)

    # ---- 3) tokenizer & LLaMA ----
    print("\nLoading tokenizer & adding <brain> ...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_name)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<brain>']})

    model = build_peft_llm(
        llm_name=args.llm_name,
        tokenizer=tokenizer,
        ft_method=args.ft_method,
    ).to(device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"

    # ---- 4) Adapter（Stage1 学到的脑 → 语义映射）----
    print("\nLoading brain adapter checkpoint ...")
    adapter = create_simplified_adapter(config.adapter)
    ckpt = torch.load(args.adapter_checkpoint, map_location='cpu')
    adapter.load_state_dict(ckpt['model_state_dict'])
    adapter.to(device)

    # 检查 adapter 输出维度是否匹配 LLaMA hidden_size
    hidden_size_llama = model.get_input_embeddings().embedding_dim
    if hasattr(adapter, "input_dim"):
        dummy_feat = torch.randn(1, adapter.input_dim)
        with torch.no_grad():
            out = adapter(dummy_feat)
        if out.shape[-1] != hidden_size_llama:
            raise ValueError(
                f"Adapter 输出维度 {out.shape[-1]} 与 LLaMA hidden_size {hidden_size_llama} 不匹配，"
                "请在简化 adapter 设计时让输出维度 = LLaMA hidden_size，"
                "或者在这里额外加一个 Linear 映射层。"
            )

    # ---- 5) Dataloaders & Eval fn ----
    if args.llm_task == "lm":
        print("\n[Data] Using LM finetune dataloaders (token-level CE).")
        train_loader, val_loader = create_finetune_dataloaders_lm(
            config=config,
            mvpformer=mvpformer,
            tokenizer=tokenizer,
            subjects=args.subjects,
            max_length=args.llm_max_length,
            batch_size=args.llm_batch_size,
        )
        eval_fn = evaluate_finetune
        eval_extra_kwargs = {}
    else:
        print("\n[Data] Using MC finetune dataloaders (candidate-level CE).")
        train_loader, val_loader = create_finetune_dataloaders_mc(
            config=config,
            mvpformer=mvpformer,
            tokenizer=tokenizer,
            subjects=args.subjects,
            max_length=args.llm_max_length,
            batch_size=args.llm_batch_size,
            mc_num_choices=args.mc_num_choices,
        )
        for b in train_loader:
            print("Example batch keys:", b.keys())
            print("  candidate_words[0]:", b["candidate_words"][0])
            break
        eval_fn = evaluate_finetune_mc
        eval_extra_kwargs = {
            "mc_num_choices": args.mc_num_choices,
            "max_length": args.llm_max_length,
        }

    # ---- 6) Optimizer ----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not args.freeze_adapter:
        for p in adapter.parameters():
            p.requires_grad = True
        trainable_params += list(adapter.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=args.llm_lr)

    # ---- 7) Training Loop ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    best_val_acc = 0.0

    # print("\n[Baseline-MC] Evaluating before finetune ...")
    # val_loss0, val_acc0 = evaluate_finetune_mc(
    #     model=model,
    #     adapter=adapter,
    #     dataloader=val_loader,
    #     tokenizer=tokenizer,
    #     device=device,
    #     mc_num_choices=args.mc_num_choices,
    #     max_length=args.llm_max_length,
    #     freeze_adapter=args.freeze_adapter,
    # )
    # print(f"[Baseline-MC] val_loss={val_loss0:.4f}, val_acc={val_acc0:.4f}")

    for epoch in range(1, args.llm_epochs + 1):
        # break
        train_loss = train_one_epoch_unified(
            model=model,
            adapter=adapter,
            dataloader=train_loader,
            tokenizer=tokenizer,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            llm_task=args.llm_task,
            freeze_adapter=args.freeze_adapter,
            log_interval=50,
            mc_num_choices=args.mc_num_choices,
            max_length=args.llm_max_length,
        )

        val_loss, val_acc = eval_fn(
            model=model,
            adapter=adapter,
            dataloader=val_loader,
            tokenizer=tokenizer,
            device=device,
            freeze_adapter=args.freeze_adapter,
            **eval_extra_kwargs,
        )

        print(f"Epoch {epoch} [{args.llm_task}]: "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            save_path = output_dir / f"best_llm_{args.ft_method}_{args.llm_task}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "adapter_state_dict": adapter.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "llm_task": args.llm_task,
                "ft_method": args.ft_method,
            }, save_path)
            print(f"  Saved best checkpoint to {save_path}")

    print(f"\nDone. Task={args.llm_task}, best val_loss={best_val_loss:.4f}, best val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()

