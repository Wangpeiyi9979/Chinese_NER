#!/usr/bin/env bash
python train.py run --model=Bert --log_name=turn1  --gpu_id=0 --train_batch_size=8
python train.py run --model=Bert --log_name=turn2  --gpu_id=0 --train_batch_size=8
python train.py run --model=Bert --log_name=turn3  --gpu_id=0 --train_batch_size=8
python train.py run --model=Bert --log_name=turn4  --gpu_id=0 --train_batch_size=8
python train.py run --model=Bert --log_name=turn5  --gpu_id=0 --train_batch_size=8
