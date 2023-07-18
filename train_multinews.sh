# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LicenseRef-.amazon.com.-AmznSL-1.0
# Licensed under the Amazon Software License  http://aws.amazon.com/asl/

# experiment run id, which serves as the folder name in ./checkpoints
MULTINEWS_NAME="multinews_256"
# training the model
python src/main.py --primer_path allenai/PRIMERA \
                             --tokenizer allenai/PRIMERA \
                             --name ${MULTINEWS_NAME} \
                             --gpus 8 \
                             --accelerator ddp \
                             --label_smoothing 0.1 \
                             --batch_size 2 \
                             --num_workers 12 \
                             --num_train_data -1 \
                             --num_valid_data -1 \
                             --max_length_tgt 256 \
                             --val_check_interval 1.0 \
                             --dataset_name multi_news \
                             --total_steps 25000 \
                             --warmup_steps 2500 \
                             --compute_rouge \
                             --progress_bar_refresh_rate 8 \
                             --acc_batch 2 \
                             --summ_weight 1.0 \
                             --align_weight 0.1 \
                             --ie_weight 0.1

testing
python script/primer_hf_main.py --primer_path allenai/PRIMERA \
                             --name ${MULTINEWS_NAME} \
                             --gpus 8 \
                             --accelerator ddp \
                             --batch_size 2 \
                             --num_workers 12 \
                             --compute_rouge \
                             --max_length_tgt 50 \
                             --beam_size 5 \
                             --accelerator ddp \
                             --dataset_name wcep10 \
                             --resume_ckpt checkpoints/${MULTINEWS_NAME}/summ_checkpoints/step=2992-vloss=3.23-avgr=0.0000.ckpt \
                             --mode test

# evaluation
python ./script/evaluate.py --metric rouge --data_dir ./checkpoints/${MULTINEWS_NAME}/test_dumps/ --output_dir ./checkpoints/${MULTINEWS_NAME}/
python ./script/evaluate.py --metric entity --data_dir ./checkpoints/${MULTINEWS_NAME}/test_dumps/ --output_dir ./checkpoints/${MULTINEWS_NAME}/
python ./script/evaluate.py --metric mint --data_dir ./checkpoints/${MULTINEWS_NAME}/test_dumps/ --output_dir ./checkpoints/${MULTINEWS_NAME}/
