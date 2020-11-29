import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from utils import create_label_dict


class DataModel(Dataset):
    def __init__(self, opt, case='train'):
        self.token = np.load(os.path.join(opt.data_dir, "{}_{}.npy".format(case, 'token')), allow_pickle=True)
        self.token_id = np.load(os.path.join(opt.data_dir, "{}_{}.npy".format(case, 'token_id')), allow_pickle=True)
        self.label = np.load(os.path.join(opt.data_dir, "{}_{}.npy".format(case, 'label')), allow_pickle=True)
        self.label_id = np.load(os.path.join(opt.data_dir, "{}_{}.npy".format(case, 'label_id')), allow_pickle=True)
        self.length = np.load(os.path.join(opt.data_dir, "{}_{}.npy".format(case, 'length')), allow_pickle=True)

    def __getitem__(self, idx):
        return self.token[idx], self.token_id[idx], self.label[idx], self.label_id[idx], self.length[idx]

    def __len__(self):
        return min(8000, len(self.token))


class Collate:
    def __init__(self, pad_idx=0, mask=False, n_vocab=21128, mask_idx=103):
        self.pad_idx = pad_idx
        self.mask = mask
        self.n_vocab = n_vocab
        self.mask_idx = mask_idx

    def __call__(self, data):
        token, token_id, label, label_id, length = zip(*data)
        max_length = max(length) + 2 # [CLS] and [SEP]
        token_id_pad = []
        label_id_pad = []
        for idx in range(len(token_id)):
            pad_num = max_length - len(token_id[idx])
            tid = token_id[idx] + [self.pad_idx] * pad_num
            lid = label_id[idx].tolist() + [self.pad_idx] * pad_num
            token_id_pad.append(tid)
            label_id_pad.append(lid)
        token_id_pad = torch.tensor(token_id_pad).long()
        label_id_pad = torch.tensor(label_id_pad).long()
        length = torch.tensor(length)
        token_id_pad = self.mask_sen(token_id_pad, length)
        batch_data = {
            'token': token,
            'token_id': token_id_pad,
            'label': label,
            'label_id':label_id_pad,
            'length': length
        }
        return batch_data

    def mask_sen(self, src, length):
        if self.mask:
            pad_mask = src != self.pad_idx
            prob = torch.rand_like(src, device=src.device, dtype=torch.float32) * pad_mask
            prob[:, 0] = 0.
            prob[:, length + 1] = 0.
            mask = prob < 0.865
            src = src * mask
            mask = mask.logical_not_()
            rand_mask = prob > 0.985
            src = src + (torch.randint_like(src, self.n_vocab - 1, device=src.device) + 1) * rand_mask
            src = src + self.mask_idx * (mask ^ rand_mask)

        return src


class DataProcessor(object):
    def __init__(self, data_dir, label_path):
        self.data_dir = data_dir
        self.label2id = create_label_dict(label_path)
        self.id2label = {j: i for i, j in self.label2id.items()}
        self.tokenizer = AutoTokenizer.from_pretrained("chinese-roberta-wwm-ext")

    def create_examples(self, case='train'):
        """Creates examples for the training and dev sets."""
        json_datas = []
        origin_datas = open('./dataset/{}.json'.format(case))
        for data in origin_datas:
            data = json.loads(data)
            json_datas.append(data)
        all_data = {
            'token': [],
            'token_id':[],
            'label':[],
            'label_id':[],
            'length': []
        }
        for data in tqdm(json_datas):
            return_data = self._create_single_example(data)
            for key in all_data.keys():
                all_data[key].append(return_data[key])
        for file, data in all_data.items():
            np.save(os.path.join(self.data_dir, '{}_{}.npy'.format(case, file)), data)

    def _create_single_example(self, data):
        text = data['text']
        label_dict = data['label']
        token = [self.tokenizer.tokenize(token)[0] for token in text]
        assert len(token) == len(text)
        token_id = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + token  + ['[SEP]'])
        label_id = np.zeros(len(token)).astype(int)
        for label, pos_dict in label_dict.items():
            b_id = self.label2id['B-{}'.format(label)]
            i_id = self.label2id['I-{}'.format(label)]
            poses = []
            entitys = pos_dict.keys()
            for entity in entitys:
                poses.extend(pos_dict[entity])
            for pos in poses:
                label_id[pos[0]] = b_id
                label_id[pos[0]+1:pos[-1]+1] = i_id
        label = [self.id2label[idx] for idx in label_id]
        assert len(token) + 2 == len(token_id) # +2是因为多了Cls和Sep
        assert len(label_id) + 2 == len(token_id)
        return_data = {
            'token': token,
            'token_id': token_id,
            'label': label,
            'label_id':label_id,
            'length': len(token)
        }
        return  return_data


if __name__ == '__main__':
    data_processor = DataProcessor(data_dir='dataset/',
                                  label_path='dataset/tool_data/label.txt')
    data_processor.create_examples('train')
    data_processor.create_examples('test')
