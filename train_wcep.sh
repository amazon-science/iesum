# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/

# experiment run id, which serves as the folder name in ./checkpoints
WCEP10_NAME="wcep_10" 
# training the model
python script/primer_main.py --primer_path allenai/PRIMERA \
                             --name ${WCEP10_NAME} \
                             --gpus 8 \
                             --accelerator ddp \
                             --label_smoothing 0.1 \
                             --batch_size 1 \
                             --num_workers 12 \
                             --num_train_data -1 \
                             --num_valid_data -1 \
                             --max_length_tgt 64 \
                             --val_check_interval 1.0 \
                             --dataset_name wcep10 \
                             --total_steps 5000 \
                             --warmup_steps 500 \
                             --data_load_method pkl \
                             --compute_rouge \
                             --progress_bar_refresh_rate 8 \
                             --acc_batch 2 \
                             --summ_weight 1.0 \
                             --align_weight 0.1 \
                             --ie_weight 0.1

# testing 
python script/primer_main.py --primer_path allenai/PRIMERA \
                             --name ${WCEP10_NAME} \
                             --gpus 8 \
                             --accelerator ddp \
                             --batch_size 8 \
                             --num_workers 12 \
                             --num_test_data -1 \
                             --compute_rouge \
                             --max_length_tgt 64 \
                             --beam_size 5 \
                             --data_load_method pkl \
                             --dataset_name wcep10 \
                             --mode test

# evaluation
python ./script/evaluate.py --metric rouge --data_dir ./checkpoints/${WCEP10_NAME}/test_dumps/ --output_dir ./checkpoints/${WCEP10_NAME}/
python ./script/evaluate.py --metric entity --data_dir ./checkpoints/${WCEP10_NAME}/test_dumps/ --output_dir ./checkpoints/${WCEP10_NAME}/
python ./script/evaluate.py --metric mint --data_dir ./checkpoints/${WCEP10_NAME}/test_dumps/ --output_dir ./checkpoints/${WCEP10_NAME}/