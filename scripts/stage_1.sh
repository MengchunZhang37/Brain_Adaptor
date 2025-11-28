export CUDA_VISIBLE_DEVICES=4,5,6,7

python framework/train_stage1.py \
    --config stage1_full \
    --data_root /user_data/yingjueb/ecog_pretrain/podcast \
    --mvpformer_checkpoint mvpformer/ckpts/genie-m-base.pt