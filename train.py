import fire
import json
import torch
import os
import numpy as np

import sys
from tqdm import trange
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn as nn
from datamodels.DataModelOld import Collate
from models.crf import CRF
import datamodels
import models
import utils
import evaluate
import configs
id2label = utils.create_label_dict('dataset/tool_data/label.txt', reverse=True)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def run(**keward):
    opt = getattr(configs, keward.get('model', 'BertCrf') + 'Config')()
    opt.parse(keward)
    if opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)
    # setup_seed(opt.seed)

    DataModel = getattr(datamodels, 'DataModel')
    train_data = DataModel(opt, case='train')
    train_collate = Collate(0, opt.mask, )
    train_data_loader = DataLoader(train_data, opt.train_batch_size, shuffle=True, num_workers=4, collate_fn=train_collate)
    dev_data = DataModel(opt, case='dev')
    dev_collate = Collate(0, False)
    dev_data_loader = DataLoader(dev_data, opt.dev_batch_size, shuffle=False, num_workers=4, collate_fn=dev_collate)

    print("train data size:{}; dev data size:{}".format(len(train_data), len(dev_data)))

    checkpoint = None
    if opt.continue_training:
        checkpoint = 'checkpoints/{}{}_last.pt'.format(opt.model, opt.log_name)
        checkpoint = torch.load(checkpoint, map_location='cpu')
    elif opt.load_checkpoint is not None:
        checkpoint = 'checkpoints/{}{}_{}.pt'.format(opt.model, opt.log_name, opt.load_checkpoint)
        checkpoint = torch.load(checkpoint, map_location='cpu')
    opt = opt if checkpoint is None else checkpoint['opt']
    model = getattr(models, opt.model)(opt)
    if opt.use_gpu:
        model.cuda()
    crf = None
    for m in model.modules():
        if isinstance(m, CRF):
            crf = m
    if crf is None:
        optimizer = models.Adam(model.parameters(), lr=opt.lr)
    else:
        ignored_params = list(map(id, crf.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
        optimizer = models.Adam([
            {'params': base_params, 'lr':opt.lr},
            {'params': crf.parameters(), 'lr': opt.crf_lr}])
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + 0.95 * epoch))
    train_steps = (len(train_data) + opt.train_batch_size - 1) // opt.train_batch_size
    dev_steps = (len(dev_data) + opt.dev_batch_size - 1) // opt.dev_batch_size
    best_dev_f1 = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint['parameters'])
        scheduler.load_state_dict(checkpoint['optimizer'])
        best_dev_f1 = checkpoint['best_f1']

    print("start training...")
    early_stop = 0
    epoch = 0
    while early_stop < opt.early_stop:
        print("{}; epoch:{}:".format(utils.now(), epoch))
        train(model, train_data_loader, scheduler, optimizer, train_steps, opt)
        # ps, rs, f1s, p, r, f1 = eval(model, train_data_loader, train_steps, opt)
        # print("[Result Train] p:{:.3f}%, r:{:.3f}%, f1:{:.3f}%".format(p, r, f1))
        # print("[Detail Train] p :{}\n"
        #       "[Detail Train] r :{}\n"
        #       "[Detail Train] f1:{}".format(ps, rs, f1s))
        #

        ps, rs, f1s, p, r, f1 = eval(model, dev_data_loader, dev_steps, opt)
        if best_dev_f1 < f1:
            early_stop = 0
            best_dev_f1 = f1
            print('[Save]: {}_{}'.format(model.model_name, best_dev_f1))
            torch.save({'opt': opt, 'parameters': model.state_dict(), 'optimizer': scheduler.state_dict(),
                        'current_f1': f1, 'best_f1': best_dev_f1}, 'checkpoints/{}{}_best.pt'
                       .format(model.model_name, opt.log_name))
        torch.save({'opt': opt, 'parameters': model.state_dict(), 'optimizer': scheduler.state_dict(),
                    'current_f1': f1, 'best_f1': best_dev_f1}, 'checkpoints/{}{}_{}.pt'
                   .format(model.model_name, opt.log_name, epoch))
        torch.save({'opt': opt, 'parameters': model.state_dict(), 'optimizer': scheduler.state_dict(),
                    'current_f1': f1, 'best_f1': best_dev_f1}, 'checkpoints/{}{}_last.pt'
                   .format(model.model_name, opt.log_name))
        print("[Result dev] p:{:.3f}%, r:{:.3f}%, f1:{:.3f}%".format(p, r, f1))
        # print("[Detail dev] p :{}\n"
        #       "[Detail dev] r :{}\n"
        #       "[Detail dev] f1:{}".format(ps, rs, f1s))
        print('[Best] dev f1:{}\n'.format(best_dev_f1))
        epoch += 1
        early_stop += 1


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
        p, r, f1 = evaluate.f1_score(golden_label, pred_label, average='micro')
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
    ps, rs, f1s, p, r, f1 = evaluate.f1_score(golden_label, pred_label, average='macro')
    return ps, rs, f1s, p, r, f1


if __name__ == '__main__':
    fire.Fire()
