from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PrefixTuningConfig, get_peft_model
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from config import get_simplified_config, print_split_info
from train_simplified import load_mvpformer
from simplified_adapter import create_simplified_adapter
from new_dataset import create_finetune_dataloaders

from tqdm import tqdm

def build_peft_llm(
    llm_name: str,
    tokenizer,
    ft_method: str = "lora",
    r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
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
    if freeze_adapter:
        adapter.eval()
        for p in adapter.parameters():
            p.requires_grad = False

    brain_token_id = tokenizer.convert_tokens_to_ids("<brain>")

    total_loss = 0.0
    n_batches = 0

    total = 0
    correct = 0

    for batch in tqdm(dataloader, desc="[Val]", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        brain_feature = batch["brain_feature"].to(device)
        target_words = batch["target_word"]  # List[str]

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
        total_loss += loss.item()
        n_batches += 1

        # 简单 top-1 word accuracy：看第一个 label token 的预测是否匹配 word 的首 token
        logits = outputs.logits  # (B, T, V)
        B = input_ids.size(0)

        for b in range(B):
            tgt_pos = (labels[b] != -100).nonzero(as_tuple=False)
            if len(tgt_pos) == 0:
                continue
            first_pos = tgt_pos[0].item()
            pred_id = logits[b, first_pos].argmax(dim=-1).item()
            pred_tok = tokenizer.decode([pred_id]).strip()
            gt_word = target_words[b]
            if gt_word.strip().lower() == pred_tok.lower():
                correct += 1
            total += 1

    avg_loss = total_loss / max(n_batches, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser(description="Brain-conditioned LLaMA finetune (LoRA / Prefix)")
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

    tokenizer.padding_side = "right"   # 或 "left"，看你后面 collate 的习惯

    # ---- 4) Adapter（Stage1 学到的脑 → 语义映射）----
    print("\nLoading brain adapter checkpoint ...")
    adapter = create_simplified_adapter(config.adapter)
    ckpt = torch.load(args.adapter_checkpoint, map_location='cpu')
    adapter.load_state_dict(ckpt['model_state_dict'])
    adapter.to(device)

    # 检查 adapter 输出维度是否匹配 LLaMA hidden_size
    hidden_size_llama = model.get_input_embeddings().embedding_dim
    # 用一个 dummy feature 检查：如果 adapter 没有 input_dim，就跳过这个检查
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

    # ---- 5) Dataloaders ----
    train_loader, val_loader = create_finetune_dataloaders(
        config=config,
        mvpformer=mvpformer,
        tokenizer=tokenizer,
        subjects=args.subjects,
        max_length=args.llm_max_length,
        batch_size=args.llm_batch_size,
    )

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

    for epoch in range(1, args.llm_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            adapter=adapter,
            dataloader=train_loader,
            tokenizer=tokenizer,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            freeze_adapter=args.freeze_adapter,
        )

        val_loss, val_acc = evaluate_finetune(
            model=model,
            adapter=adapter,
            dataloader=val_loader,
            tokenizer=tokenizer,
            device=device,
            freeze_adapter=args.freeze_adapter,
        )

        print(f"Epoch {epoch}: "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            save_path = output_dir / f"best_llm_{args.ft_method}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "adapter_state_dict": adapter.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, save_path)
            print(f"  Saved best checkpoint to {save_path}")

    print(f"\nDone. Best val_loss={best_val_loss:.4f}, best val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()
