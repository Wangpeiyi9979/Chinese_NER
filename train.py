import fire
import json
import torch
import os
import numpy as np
import torch.optim as optim

import sys
from tqdm import trange
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn as nn
from datamodels.DataModel import collate_fn

import datamodels
import models
import utils
import metric
import configs
id2label = utils.create_label_dict('dataset/tool_data/label.txt', reverse=True)
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run(**keward):
    opt = getattr(configs, keward.get('model', 'BERT_LSTM_CRF') + 'Config')()
    opt.parse(keward)
    if opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)
    setup_seed(opt.seed)

    DataModel = getattr(datamodels, 'DataModel')
    train_data = DataModel(opt, case='train')
    train_data_loader = DataLoader(train_data, opt.train_batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    test_data = DataModel(opt, case='test')
    test_data_loader = DataLoader(test_data, opt.test_batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    print("train data size:{}; test data size:{}".format(len(train_data), len(test_data)))

    checkpoint = None
    if opt.continue_training:
        checkpoint = 'checkpoints/{}_last.pt'.format(opt.model)
        checkpoint = torch.load(checkpoint, map_location='cpu')
    elif opt.load_checkpoint is not None:
        checkpoint = 'checkpoints/{}_{}.pt'.format(opt.model, opt.load_checkpoint)
        checkpoint = torch.load(checkpoint, map_location='cpu')
    opt = opt if checkpoint is None else checkpoint['opt']
    model = getattr(models, opt.model)(opt)
    if opt.use_gpu:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.95 * epoch))
    train_steps = (len(train_data) + opt.train_batch_size - 1) // opt.train_batch_size
    test_steps = (len(test_data) + opt.test_batch_size - 1) // opt.test_batch_size
    best_test_f1 = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint['parameters'])
        scheduler.load_state_dict(checkpoint['optimizer'])
        best_test_f1 = checkpoint['best_f1']

    print("start training...")
    for epoch in range(opt.num_epochs):
        print("{}; epoch:{}/{}:".format(utils.now(), epoch, opt.num_epochs))
        train(model, train_data_loader, scheduler, optimizer, train_steps, opt)
        ps, rs, f1s, p, r, f1 = eval(model, test_data_loader, test_steps, opt)
        if best_test_f1 < f1:
            best_test_f1 = f1
            print('[Save]: {}_{}'.format(model.model_name, best_test_f1))
            torch.save({'opt': opt, 'parameters': model.state_dict(), 'optimizer': scheduler.state_dict(),
                        'current_f1': f1, 'best_f1': best_test_f1}, 'checkpoints/{}_best.pt'
                       .format(model.model_name))
        torch.save({'opt': opt, 'parameters': model.state_dict(), 'optimizer': scheduler.state_dict(),
                    'current_f1': f1, 'best_f1': best_test_f1}, 'checkpoints/{}_{}.pt'
                   .format(model.model_name, epoch))
        torch.save({'opt': opt, 'parameters': model.state_dict(), 'optimizer': scheduler.state_dict(),
                    'current_f1': f1, 'best_f1': best_test_f1}, 'checkpoints/{}_last.pt'
                   .format(model.model_name))
        print("[Result] p:{:.3f}%, r:{:.3f}%, f1:{:.3f}%".format(p, r, f1))
        print("[Detail] p :{}\n"
              "[Detail] r :{}\n"
              "[Detail] f1:{}".format(ps, rs, f1s))
        print('[Best] test f1:{}\n'.format(best_test_f1))


def train(model, dataLoader, scheduler, optimizer, steps, opt):
    model.train()
    lossAll = utils.RunningAverage()
    pre = utils.RunningAverage()
    rel = utils.RunningAverage()
    f1_score = utils.RunningAverage()

    for it, data in enumerate(dataLoader):
        golden_label = []
        pred_label = []
        for key in data.keys():
            if 'id' in key and opt.use_gpu:
                data[key] = data[key].cuda()
        loss, batch_pred = model(data)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=opt.clip_grad)
        optimizer.step()
        for idx, label in enumerate(zip(data['label'], batch_pred)):
            label, pred = label
            l = data['length'][idx]
            golden_label.append(label[:l])
            pred_label.append([id2label[id.item() if isinstance(id, torch.Tensor) else id] for id in pred[:l]])
        p, r, f1 = metric.f1_score(golden_label, pred_label, average='micro')
        lossAll.update(loss.item())
        pre.update(p)
        rel.update(r)
        f1_score.update(f1)
        sys.stdout.write(
            '[Train] step: {}/{} | loss: {:.6f} p:{:.3f}% r:{:.3f}% f1:{:.3f}% [micro batch mean]'.format(it + 1,
                                                                                       steps,
                                                                                       lossAll(),
                                                                                       pre(),
                                                                                       rel(),
                                                                                       f1_score()) + '\r')
        sys.stdout.flush()
    print()
    scheduler.step()

def eval(model, dataLoader, steps, opt):
    model.eval()
    golden_label = []
    pred_label = []
    with torch.no_grad():
        for it, data in enumerate(dataLoader):
            for key in data.keys():
                if 'id' in key and opt.use_gpu:
                    data[key] = data[key].cuda()
            batch_pred = model(data, train=False)
            for idx, label in enumerate(zip(data['label'], batch_pred)):
                label, pred = label
                l = data['length'][idx]
                golden_label.append(label[:l])
                pred_label.append([id2label[id.item() if isinstance(id, torch.Tensor) else id] for id in pred[:l]])
            sys.stdout.write(
                '[Eval] step: {}/{}'.format(it + 1,
                                            steps)
                + '\r')
            sys.stdout.flush()
    print("")
    ps, rs, f1s, p, r, f1 = metric.f1_score(golden_label, pred_label, average='macro')
    return ps, rs, f1s, p, r, f1


if __name__ == '__main__':
    fire.Fire()
