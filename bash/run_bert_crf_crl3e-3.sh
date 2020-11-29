#!/usr/bin/env bash
python train.py run --model=BertCrf --log_name=3e-3turn1  --gpu_id=1 --crf_lr=3e-3
python train.py run --model=BertCrf --log_name=3e-3turn2  --gpu_id=1 --crf_lr=3e-3
python train.py run --model=BertCrf --log_name=3e-3turn3  --gpu_id=1 --crf_lr=3e-3
python train.py run --model=BertCrf --log_name=3e-3turn4  --gpu_id=1 --crf_lr=3e-3
python train.py run --model=BertCrf --log_name=3e-3turn5  --gpu_id=1 --crf_lr=3e-3
