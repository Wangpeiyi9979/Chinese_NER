#!/usr/bin/env bash
python train.py run --model=BertCrf  --gpu_id=0  --mask=True --log_name=mask_turn1
python train.py run --model=BertCrf  --gpu_id=0  --mask=True --log_name=mask_turn2
python train.py run --model=BertCrf  --gpu_id=0  --mask=True --log_name=mask_turn3
python train.py run --model=BertCrf  --gpu_id=0  --mask=True --log_name=mask_turn4
python train.py run --model=BertCrf  --gpu_id=0  --mask=True --log_name=mask_turn5
