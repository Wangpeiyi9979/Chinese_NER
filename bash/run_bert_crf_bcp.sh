#!/usr/bin/env bash
python train.py run --model=BertCrf --BCP_path=checkpoints/BCPSingle_best.pt --gpu_id=1   --log_name=turn1
python train.py run --model=BertCrf --BCP_path=checkpoints/BCPSingle_best.pt --gpu_id=1   --log_name=turn2
python train.py run --model=BertCrf --BCP_path=checkpoints/BCPSingle_best.pt --gpu_id=1   --log_name=turn3
python train.py run --model=BertCrf --BCP_path=checkpoints/BCPSingle_best.pt --gpu_id=1   --log_name=turn4
python train.py run --model=BertCrf --BCP_path=checkpoints/BCPSingle_best.pt --gpu_id=1   --log_name=turn5
