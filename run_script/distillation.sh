#!/bin/bash

output_dir=xglue_xlm_roberta_large_128shot_lang8
lr=3e-8
save_strategy=no
save_steps=0  
evaluation_strategy=steps
eval_steps=40
MODEL_DIR=xglue_nli_xlm-roberta-large_xnli_128_shot
zero_tuning=True
EXPERIMENT_NAME=$output_dir
bs=40
eval_bs=30
gradient_accumulation_steps=1
epoch=100
task_type=language_modeling

dropout=0.1
seed=100
do_train=False
do_valid=True
do_predict=False

python run.py \
    --task_name xglue \
    --dataset_name xnli \
    --dataset_config_name None \
    --max_seq_length 512 \
    --overwrite_cache False \
    --pad_to_max_length False \
    --train_file None \
    --validation_file None \
    --test_file None \
    --do_train ${do_train} \
    --do_eval ${do_valid} \
    --do_predict ${do_predict} \
    --per_device_train_batch_size ${bs} \
    --per_device_eval_batch_size ${eval_bs} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --num_train_epochs ${epoch} \
    --output_dir checkpoints/${output_dir} \
    --learning_rate ${lr} \
    --weight_decay 0.0005 \
    --bf16 True \
    --prompt \
    --seed ${seed} \
    --warmup_ratio 0.2 \
    --save_strategy ${save_strategy} \
    --eval_steps ${eval_steps} \
    --evaluation_strategy ${evaluation_strategy} \
    --remove_unused_columns False \
    --model_name_or_path ${MODEL_DIR} \
    --use_fast_tokenizer True \
    --model_revision main \
    --task_type ${task_type} \
    --prompt_type hard \
    --template_id 2 \
    --verbalizer_id 1 \
    --prompt_operation none \
    --data_augmentation none \
    --device 3 \
    --deepspeed deepspeed.json \
    --zero_tuning ${zero_tuning} \
    --ensemble_lang_connect True \
    --ensemble_weak_number 8 \
    --overwrite_output_dir
