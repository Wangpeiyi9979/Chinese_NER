#!/usr/bin/env bash
python train.py run --model=BertCrf --log_name=3e-3nocons_turn1 --use_cons=False --gpu_id=2 --crf_lr=3e-3
python train.py run --model=BertCrf --log_name=3e-3nocons_turn2 --use_cons=False --gpu_id=2 --crf_lr=3e-3
python train.py run --model=BertCrf --log_name=3e-3nocons_turn3 --use_cons=False --gpu_id=2 --crf_lr=3e-3
python train.py run --model=BertCrf --log_name=3e-3nocons_turn4 --use_cons=False --gpu_id=2 --crf_lr=3e-3
python train.py run --model=BertCrf --log_name=3e-3nocons_turn5 --use_cons=False --gpu_id=2 --crf_lr=3e-3
