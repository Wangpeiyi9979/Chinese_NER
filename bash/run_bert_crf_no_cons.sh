#!/usr/bin/env bash
python train.py run --model=BertCrf --log_name=nocons_turn1 --use_cons=False --gpu_id=0
python train.py run --model=BertCrf --log_name=nocons_turn2 --use_cons=False --gpu_id=0
python train.py run --model=BertCrf --log_name=nocons_turn3 --use_cons=False --gpu_id=0
python train.py run --model=BertCrf --log_name=nocons_turn4 --use_cons=False --gpu_id=0
python train.py run --model=BertCrf --log_name=nocons_turn5 --use_cons=False --gpu_id=0
