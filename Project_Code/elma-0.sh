#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python run_squad.py \
  --model_type albert \
  --model_name_or_path albert-base-v2 \
  --task_name SQuAD \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir ../SQuAD2.0 \
  --max_seq_length 128 \
  --per_gpu_eval_batch_size 1 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 3 \
  --save_steps 0 \
  --seed 42 \
  --output_dir ./saved_models/albert-base/SQuAD/teacher \
  --overwrite_cache \
  --overwrite_output_dir

python masked_run_highway_squad.py --model_type masked_albert \
  --model_name_or_path albert-base-v2 \
  --task_name SQuAD \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir ../SQuAD2.0 \
  --max_seq_length 128 \
  --per_gpu_eval_batch_size=1 \
  --per_gpu_train_batch_size=64 \
  --learning_rate 3e-5 \
  --num_train_epochs 30 \
  --overwrite_output_dir \
  --seed 42 \
  --output_dir ./saved_models/masked_albert/SQuAD/two_stage_pruned_0.5 \
  --plot_data_dir ./plotting/ \
  --save_steps 0 \
  --overwrite_cache \
  --eval_after_first_stage \
  --adaptive \
  --adaptive_span_ramp 256 \
  --max_span 512 \
  --warmup_steps 1000 \
  --mask_scores_learning_rate 1e-2 \
  --initial_threshold 1 --final_threshold 0.5 \
  --initial_warmup 2 --final_warmup 3 \
  --pruning_method magnitude --mask_init constant --mask_scale 0. \
  --fxp_and_prune \
  --prune_percentile 60 \
  --teacher_type albert_teacher --teacher_name_or_path ./saved_models/albert-base/SQuAD/teacher \
  --alpha_ce 0.1 --alpha_distil 0.9

python bertarize.py \
    --pruning_method magnitude \
    --threshold 0.5 \
    --model_name_or_path ./saved_models/masked_albert/SQuAD/two_stage_pruned_0.5

ENTROPIES="0.23 0.28 0.46"

for ENTROPY in $ENTROPIES; do
    echo $ENTROPY
    python masked_run_highway_squad.py --model_type albert \
      --model_name_or_path ./saved_models/masked_albert/SQuAD/bertarized_two_stage_pruned_0.5 \
      --task_name SQuAD \
      --do_eval \
      --do_lower_case \
      --data_dir ../SQuAD2.0 \
      --max_seq_length 128 \
      --per_gpu_eval_batch_size=1 \
      --overwrite_output_dir \
      --output_dir ./saved_models/masked_albert/SQuAD/bertarized_two_stage_pruned_0.5  \
      --plot_data_dir ./plotting/ \
      --early_exit_entropy $ENTROPY \
      --eval_highway \
      --overwrite_cache
done

for ENTROPY in $ENTROPIES; do
    echo $ENTROPY
    python masked_run_highway_squad.py --model_type albert \
      --model_name_or_path ./saved_models/masked_albert/SQuAD/bertarized_two_stage_pruned_0.5 \
      --task_name SQuAD \
      --do_eval \
      --do_lower_case \
      --data_dir ../SQuAD2.0 \
      --max_seq_length 128 \
      --per_gpu_eval_batch_size=1 \
      --overwrite_output_dir \
      --output_dir ./saved_models/masked_albert/SQuAD/bertarized_two_stage_pruned_0.5  \
      --plot_data_dir ./plotting/ \
      --early_exit_entropy $ENTROPY \
      --eval_highway \
      --entropy_predictor \
      --predict_layer 1 \
      --lookup_table_file ./sst2_lookup_table_opt.csv \
      --overwrite_cache
done
