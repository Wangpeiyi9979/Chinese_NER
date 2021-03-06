import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from utils import create_label_dict, create_id2describe
from torch.nn.utils.rnn import pad_sequence

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
        # return min(8000, len(self.token))
        return len(self.token)
        # return 200

def collate_fn(data):
    token, token_id, label, label_id, length = zip(*data)
    max_length = max(length) + 2 # [CLS] and [SEP]
    token_id_pad = []
    label_id_pad = []
    for idx in range(len(token_id)):
        pad_num = max_length - len(token_id[idx])
        tid = token_id[idx] + [0] * pad_num
        lid = label_id[idx].tolist() + [0] * pad_num
        token_id_pad.append(tid)
        label_id_pad.append(lid)
    token_id_pad = torch.tensor(token_id_pad).long()
    label_id_pad = torch.tensor(label_id_pad).long()
    batch_data = {
        'token': token,
        'token_id': token_id_pad,
        'label': label,
        'label_id':label_id_pad,
        'length': torch.tensor(length)
    }
    return batch_data


class DataProcessor(object):
    def __init__(self, data_dir, label_path):
        self.data_dir = data_dir
        self.label2id = create_label_dict(label_path)
        self.id2label = {j: i for i, j in self.label2id.items()}
        self.tokenizer = AutoTokenizer.from_pretrained("chinese-roberta-wwm-ext")
        self.label2num = {}

    def create_token_id_and_label_id(self, token, label_dict):
        token_id = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + token + ['[SEP]'])
        label_id = np.zeros(len(token)).astype(int)
        for label, pos_dict in label_dict.items():
            b_id = self.label2id['B-{}'.format(label)]
            i_id = self.label2id['I-{}'.format(label)]
            poses = []
            entitys = pos_dict.keys()
            for entity in entitys:
                self.label2num[label] = self.label2num.get(label, 0) + 1
                poses.extend(pos_dict[entity])
            for pos in poses:
                label_id[pos[0]] = b_id
                label_id[pos[0] + 1:pos[-1] + 1] = i_id
        return token_id, label_id

    def create_describe_npy(self):
        """
        Returns: 创建实体的类别描述，存储为bert_id
        """
        id2des = create_id2describe('./dataset/tool_data/label.txt')
        all_des_id = []
        for idx in range(len(id2des)):
            token_id = self.tokenizer.encode(id2des[idx])
            all_des_id.append(torch.tensor(token_id).long())
        all_des_id = pad_sequence(all_des_id, padding_value=0, batch_first=True)
        np.save('./dataset/describe_id.npy', np.array(all_des_id))

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
        print(self.label2num)
        self.label2num = {}


    def _create_single_example(self, data):
        text = data['text']
        label_dict = data['label']
        token = [self.tokenizer.tokenize(token)[0] for token in text]
        assert len(token) == len(text)
        token_id, label_id = self.create_token_id_and_label_id(token, label_dict)
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
    # data_processor.create_describe_npy()
    data_processor.create_examples('train')
    data_processor.create_examples('dev')
    data_processor.create_examples('test')
