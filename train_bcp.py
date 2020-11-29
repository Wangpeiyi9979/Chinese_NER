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
from datamodels.BCPDataModel import collate_fn as collate_fn_full
from datamodels.BCPSingleDataModel import collate_fn as collate_fn_single

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
    opt = getattr(configs, keward.get('model', 'BCP') + 'Config')()
    opt.parse(keward)
    if opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)
    setup_seed(opt.seed)

    DataModel = getattr(datamodels, opt.model+'DataModel')
    if opt.model == 'BCPSingle':
        collate_fn = collate_fn_single
    elif opt.model == 'BCP':
        collate_fn = collate_fn_full
    else:
        raise RuntimeError('No correct model')
    train_data = DataModel(opt, case='train')
    print('neg_num:{}'.format(len(train_data[0][-1])))
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
    best_test_acc = 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint['parameters'])
        scheduler.load_state_dict(checkpoint['optimizer'])
        best_test_acc = checkpoint['best_acc']

    print("start training...")
    for epoch in range(opt.num_epochs):
        print("{}; epoch:{}/{}:".format(utils.now(), epoch, opt.num_epochs))
        train(model, train_data_loader, scheduler, optimizer, train_steps, opt)
        acc = eval(model, test_data_loader, test_steps, opt)
        if best_test_acc < acc:
            best_test_acc = acc
            print('[Save]: {}_{}'.format(model.model_name, best_test_acc))
            torch.save({'opt': opt, 'parameters': model.state_dict(), 'optimizer': scheduler.state_dict(),
                        'current_acc': acc, 'best_acc': best_test_acc}, 'checkpoints/{}_best.pt'
                       .format(model.model_name))
        # torch.save({'opt': opt, 'parameters': model.state_dict(), 'optimizer': scheduler.state_dict(),
        #             'current_acc': acc, 'best_acc': best_test_acc}, 'checkpoints/{}_{}.pt'
        #            .format(model.model_name, epoch))
        torch.save({'opt': opt, 'parameters': model.state_dict(), 'optimizer': scheduler.state_dict(),
                    'current_acc': acc, 'best_acc': best_test_acc}, 'checkpoints/{}_last.pt'
                   .format(model.model_name))
        print("[Result] acc:{:.3f}%".format(acc))
        print('[Best] test acc:{}\n'.format(best_test_acc))


def train(model, dataLoader, scheduler, optimizer, steps, opt):
    model.train()
    lossAll = utils.RunningAverage()
    acc_score = utils.RunningAverage()
    for it, data in enumerate(dataLoader):
        for key in data.keys():
            try:
                data[key] = data[key].cuda()
            except:
                pass
        loss, batch_pred = model(data)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=opt.clip_grad)
        optimizer.step()
        lossAll.update(loss.item())
        acc = torch.sum(batch_pred == 0).item() / len(batch_pred)
        acc_score.update(acc*100)
        sys.stdout.write(
            '[Train] step: {}/{} | loss: {:.6f}/{:.6f} acc:{:.3f}/acc:{:.3f}%'.format(it + 1,
                                                                   steps,
                                                                   lossAll(),
                                                                   loss.item(),
                                                                   acc_score(),
                                                                   acc*100) + '\r')
        sys.stdout.flush()
    print()
    scheduler.step()

def eval(model, dataLoader, steps, opt):
    model.eval()
    pred_label = []
    with torch.no_grad():
        for it, data in enumerate(dataLoader):
            for key in data.keys():
                try:
                    data[key] = data[key].cuda()
                except:
                    pass
            batch_pred = model(data, train=False)
            pred_label.extend(batch_pred.tolist())
            sys.stdout.write(
                '[Eval] step: {}/{}'.format(it + 1,
                                            steps)
                + '\r')
            sys.stdout.flush()
    print("")
    acc = (torch.tensor(pred_label).long() == 0).sum().item() / len(pred_label)
    return acc * 100


if __name__ == '__main__':
    fire.Fire()
