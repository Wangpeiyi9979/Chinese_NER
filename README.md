# 创建所需的label文件
```bash
python create_tool_data.py
```

# 数据预处理生成npy文件
```bash
python datamodels/DataModel
```

# 训练模型并测试结果
```bash
python train.py run --model=Bert --train_batch_size=8  
python train.py run --model=BERT_LSTM_CRF --train_batch_size=8 --gpu_id=1  
```