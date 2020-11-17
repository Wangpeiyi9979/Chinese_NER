#encoding:utf-8
"""
@Time: 2020/11/13 14:00
@Author: Wang Peiyi
@Site : 
@File : test.py
"""


import argparse
import sys

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from texttable import Texttable


import datamodels
import metric
import models
import utils
from datamodels.DataModel import collate_fn
from configs import BertConfig

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='checkpoints/Bert_Lstm_Crf_best.pt')
id2label = utils.create_label_dict('dataset/tool_data/label.txt', reverse=True)

def cut_name(data_list, max_length=5):
    res = []
    for i in data_list:
        if len(i) > max_length:
            i = i[:max_length]
        res.append(i)
    return res

if __name__ == '__main__':
    args = parser.parse_args()
    checkpoint = torch.load(args.path, map_location='cpu')
    try:
        opt = checkpoint['opt']
    except KeyError:
        opt = BertConfig()
    model: nn.Module = getattr(models, opt.model)(opt)
    try:
        model.load_state_dict(checkpoint['parameters'])
    except KeyError:
        model.load_state_dict(checkpoint)
    print("record f1: {}".format(checkpoint['best_f1']))
    if opt.use_gpu:
        model.cuda()
    DataModel = getattr(datamodels, 'DataModel')
    test_data = DataModel(opt, case='test')
    test_data_loader = DataLoader(test_data, opt.test_batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    model.eval()
    golden_label = []
    pred_label = []
    output_file = open('./badcase.txt', 'w')
    tokenizer = AutoTokenizer.from_pretrained(opt.roberta_model_path)
    with torch.no_grad():
        for idy, data in enumerate(test_data_loader):
            for key in data.keys():
                if 'id' in key and opt.use_gpu:
                    data[key] = data[key].cuda()
            batch_pred = model(data, train=False)
            for idx, label in enumerate(zip(data['label'], batch_pred)):
                label, pred = label
                l = data['length'][idx]
                label = label[:l]
                pred = [id2label[id.item() if isinstance(id, torch.Tensor) else id] for id in pred[:l]]
                golden_label.append(label)
                pred_label.append(pred)
                if label == pred:
                    continue
                table = Texttable()
                table.set_max_width(2000)
                label = cut_name(label)
                pred = cut_name(pred)
                output_file.write('raw: {}'.format(tokenizer.decode(data['token_id'][idx], skip_special_tokens=True)) + '\n')
                table.add_row(['sen'] + data['token'][idx])
                table.add_row(['pred'] + pred)
                table.add_row(['gold'] + label)
                output_file.write(table.draw())
                output_file.write('\n\n')
                # output_file.writelines('SRC : ' + tokenizer.decode(data['token_id'][idx], skip_special_tokens=True) + '\n')
                # output_file.writelines('SRC : ' + " ".join(data['token'][idx]) + '\n')
                # output_file.writelines('HYP : ' + ' '.join(pred_label[-1]) + '\n')
                # output_file.writelines('GOLD: ' + ' '.join(golden_label[-1]) + '\n\n')
            sys.stdout.flush()
    output_file.write(metric.ner_confusion_matrix(golden_label, pred_label))
    output_file.close()
    print("")
    ps, rs, f1s, p, r, f1 = metric.f1_score(golden_label, pred_label, average='macro')
    print(ps, rs, f1s, p, r, f1)
