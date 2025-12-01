import os
import argparse
from typing import Dict

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from peft import (
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    get_peft_model,
)


# ==============================
# 1. 命令行参数
# ==============================
def parse_args():
    parser = argparse.ArgumentParser(description="LLaMA PEFT fine-tuning (LoRA / Prefix / Prompt)")

    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3-8B-Instruct",
        help="base LLaMA 模型名（HF Hub 或本地路径）",
    )
    parser.add_argument(
        "--adapter_type",
        type=str,
        default="lora",
        choices=["lora", "prefix", "prompt"],
        help="选择微调方式：lora / prefix / prompt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/llama_peft_unified",
        help="输出目录",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="每条样本的最大 token 长度",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="训练 epoch 数",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="per_device_train_batch_size",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="梯度累积步数",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="学习率",
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="是否使用 4bit 量化 (QLoRA 风格)。不加此 flag 就是普通半精度",
    )

    return parser.parse_args()


# ==============================
# 2. 简单文本 Dataset（示例）
# ==============================
class SimpleTextDataset(Dataset):
    """
    这里只是示例：
    - hf_dataset 每一行有一个 'text' 字段
    - 实际使用中，你可以把 ECoG/EEG + prompt 拼成 text 放进来
    """

    def __init__(self, hf_dataset, tokenizer, max_length: int = 512):
        self.ds = hf_dataset
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.ds[idx]["text"]

        enc = self.tok(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = enc["input_ids"][0]
        attention_mask = enc["attention_mask"][0]
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ==============================
# 3. 构造 PEFT 配置（LoRA / Prefix / Prompt）
# ==============================
def build_peft_model(model, adapter_type: str, model_name: str):
    """
    根据 adapter_type 返回挂了 PEFT 的模型：
    - lora   → LoraConfig
    - prefix → PrefixTuningConfig
    - prompt → PromptTuningConfig (soft prompt)
    """
    if adapter_type == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=64,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            bias="none",
        )

    elif adapter_type == "prefix":
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=30,  # 可调 10~100
        )

    elif adapter_type == "prompt":
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=30,
            prompt_tuning_init=PromptTuningInit.TEXT,
            prompt_tuning_init_text="You are a helpful neuroscience assistant.",
            tokenizer_name_or_path=model_name,
        )

    else:
        raise ValueError(f"Unknown adapter_type: {adapter_type}")

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


# ==============================
# 4. 主程序：加载模型 + 数据 + 训练
# ==============================
def main():
    args = parse_args()

    model_name = args.model_name
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # -------- 4.1 tokenizer --------
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 如果你这里要加 <brain> 等特殊 token，可以这样：
    # num_added = tokenizer.add_special_tokens({"additional_special_tokens": ["<brain>"]})
    # print("Added special tokens:", num_added)

    # -------- 4.2 base LLaMA 模型 --------
    load_kwargs = {"device_map": "auto"}

    if args.use_4bit:
        # QLoRA：4bit 量化加载
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["torch_dtype"] = torch.bfloat16
    else:
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    # 如果上面加了新 token，需要调整 embedding 尺寸：
    # model.resize_token_embeddings(len(tokenizer))

    # 4bit 时如果你习惯用 prepare_model_for_kbit_training，可以在这里加：
    # from peft import prepare_model_for_kbit_training
    # if args.use_4bit:
    #     model = prepare_model_for_kbit_training(model)

    # -------- 4.3 挂上 PEFT adapter --------
    model = build_peft_model(model, adapter_type=args.adapter_type, model_name=model_name)

    # =============================
    # 5. 准备训练数据（这里只是 demo）
    # =============================
    # 示例：用 alpaca 的一部分数据
    raw_ds = load_dataset("tatsu-lab/alpaca", split="train[:1000]")

    def format_example(example):
        if example.get("input"):
            text = (
                f"Instruction: {example['instruction']}\n"
                f"Input: {example['input']}\n"
                f"Response: {example['output']}"
            )
        else:
            text = (
                f"Instruction: {example['instruction']}\n"
                f"Response: {example['output']}"
            )
        return {"text": text}

    raw_ds = raw_ds.map(format_example)
    train_dataset = SimpleTextDataset(raw_ds, tokenizer, max_length=args.max_length)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # =============================
    # 6. 训练参数 & Trainer
    # =============================
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=not args.use_4bit,
        bf16=args.use_4bit,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        evaluation_strategy="no",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )

    trainer.train()

    # =============================
    # 7. 保存 adapter
    # =============================
    adapter_dir = os.path.join(output_dir, f"{args.adapter_type}_adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Adapter saved to: {adapter_dir}")


if __name__ == "__main__":
    main()
