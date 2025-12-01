python framework/ICL.py \
  --config simplified \
  --data_root /user_data/yingjueb/ecog_pretrain/preprocessed \
  --mvpformer_checkpoint mvpformer/ckpts/genie-m-base.pt \
  --adapter_checkpoint outputs/simplified/best_model.pt \
  --llm_name meta-llama/Llama-2-7B-chat-hf \
  --split test \
  --n_demo 4 \
  --max_length 1024