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
import json

import datamodels
import evaluate
import models
import utils
from datamodels.DataModel import collate_fn
from configs import BertConfig
import evaluate_reference

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='checkpoints/Bert_Lstm_Crf_best.pt')
parser.add_argument('--outpath', default='./badcase/badcase.txt')
id2label = utils.create_label_dict('dataset/tool_data/label.txt', reverse=True)

def cut_name(data_list, max_length=5):
    res = []
    for i in data_list:
        if len(i) > max_length:
            i = i[:max_length]
        res.append(i)
    return res

def get_raw_text(path):
    lines = [json.loads(line.strip()) for line in open(path) if line.strip()]
    texts = [line['text'] for line in lines]
    return texts


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
    print("Dev F1: {}".format(checkpoint['best_f1']))
    if opt.use_gpu:
        model.cuda()
    DataModel = getattr(datamodels, 'DataModel')

    test_data = DataModel(opt, case='test')
    test_data_loader = DataLoader(test_data, opt.test_batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    model.eval()
    golden_label = []
    pred_label = []
    output_file = open(args.outpath, 'w')
    raw_texts = get_raw_text('./dataset/test.json')
    tokenizer = AutoTokenizer.from_pretrained(opt.roberta_model_path)

    pred_json = []
    i = 0
    with torch.no_grad():
        for idy, data in enumerate(test_data_loader):
            for key in data.keys():
                if 'id' in key and opt.use_gpu:
                    data[key] = data[key].cuda()
            batch_pred = model(data, train=False)
            for idx, label in enumerate(zip(data['label'], batch_pred)):
                pred_unit = {'label':{}}

                label, pred = label
                l = data['length'][idx]
                label = label[:l]
                pred = [id2label[id.item() if isinstance(id, torch.Tensor) else id] for id in pred[:l]]

                # 使用给定的评测脚本评测准确率, 生成每一行的字典数据
                raw = raw_texts[i]
                i += 1
                entitys = evaluate.get_entities(pred)
                for entity in entitys:
                    entity_label, start, end = entity
                    pred_unit['label'][entity_label] = pred_unit['label'].get(entity_label, {})
                    entity_name = raw[start:end+1]
                    if entity_name in pred_unit['label'][entity_label]:
                        pred_unit['label'][entity_label][entity_name].append([start, end])
                    else:
                        pred_unit['label'][entity_label][entity_name] = [[start, end]]
                pred_json.append(pred_unit)


                golden_label.append(label)
                pred_label.append(pred)
                if label == pred:
                    continue
                table = Texttable()
                table.set_max_width(2000)
                label = cut_name(label)
                pred = cut_name(pred)
                output_file.write('raw: {}'.format(raw) + '\n')
                output_file.write('{} {} {}\n'.format('sen', 'pred', 'gold'))
                for sen, p, g in zip(data['token'][idx], pred, label):
                    output_file.write('{} {} {}\n'.format(sen, p, g))
                output_file.write('\n\n')
    output_file.write(evaluate.ner_confusion_matrix(golden_label, pred_label))
    output_file.write('\n')
    if 'crf' in model.model_name.lower():
        transitions = model.crf.transitions.tolist()
        label_list = list(opt.tags.keys())
        table = Texttable()
        table.add_row([" "] + [i[:4] for i in label_list])
        table.set_max_width(2000)
        for idx, r in enumerate(transitions):
            table.add_row([label_list[idx][:4]] + [str(i)[:6] for i in transitions[idx]])
        output_file.write(table.draw())
        output_file.write('\n')

    f1, avg = evaluate_reference.get_f1_score(pred_json, 'dataset/test.json')
    output_file.write("\nReference f1 detail: {};\n F1:{}\n".format(f1, avg))
    output_file.writelines(str(opt))
    output_file.close()
    print("F1 detail:{}".format(f1))
    print('Test F1:{}'.format(avg))
