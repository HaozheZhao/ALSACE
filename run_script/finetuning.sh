#!/bin/bash

output_dir=multilingual-xlm-r-base-ft
lr=1e-6
save_strategy=steps
save_steps=50
evaluation_strategy=steps
eval_steps=50
MODEL_DIR=xlm-roberta-base
zero_tuning=True
EXPERIMENT_NAME=multilingual-xlm-r-base-ft
bs=20
eval_bs=5
gradient_accumulation_steps=1
epoch=100
task_type=language_modeling

dropout=0.1
seed=2048
do_train=False
do_valid=True
do_predict=False
python run.py \
    --task_name xglue \
    --dataset_name xnli \
    --dataset_config_name None \
    --max_seq_length 512 \
    --overwrite_cache True \
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
    --save_total_limit 5 \
    --bf16 False \
    --prompt \
    --seed 100 \
    --warmup_ratio 0.2 \
    --save_strategy ${save_strategy} \
    --save_steps ${save_steps} \
    --evaluation_strategy ${evaluation_strategy} \
    --eval_steps ${eval_steps} \
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
    --load_best_model_at_end True \
    --metric_for_best_model accuracy \
    --greater_is_better True \
    --logging_steps 20 \
    --report_to "wandb" \
    --run_name ${output_dir} \
    --overwrite_output_dir \


