import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import sys
import random
from utils import create_label_dict
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
"""
这里按字级别构造正负样本
"""
class BCPSingleDataModel(Dataset):
    def __init__(self, opt, case='train'):
        self.token = np.load(os.path.join(opt.data_dir, "{}_{}.npy".format(case, 'token')), allow_pickle=True)
        self.token_id = np.load(os.path.join(opt.data_dir, "{}_{}.npy".format(case, 'token_id')), allow_pickle=True)
        self.length = np.load(os.path.join(opt.data_dir, "{}_{}.npy".format(case, 'length')), allow_pickle=True)
        self.anchor_index = np.load(os.path.join(opt.data_dir, "{}_{}.npy".format(case, 'anchor_index')), allow_pickle=True)
        self.pos_index = np.load(os.path.join(opt.data_dir, "{}_{}.npy".format(case, 'pos_index')), allow_pickle=True)
        self.neg_indexs = np.load(os.path.join(opt.data_dir, "{}_{}.npy".format(case, 'neg_indexs')), allow_pickle=True)

    def __getitem__(self, idx):
        return self.token[idx], self.token_id[idx], self.length[idx], self.anchor_index[idx], self.pos_index[idx],  self.neg_indexs[idx]

    def __len__(self):
        return len(self.token)

def collate_fn(data):
    token, token_id, length, anchor_index, pos_index, neg_indexs= zip(*data)
    max_length = max(length)
    token_id_pad = []
    for idx in range(len(token_id)):
        pad_num = max_length - len(token_id[idx])
        tid = token_id[idx] + [0] * pad_num
        token_id_pad.append(tid)
    token_id_pad = torch.tensor(token_id_pad).long().clone().detach()
    batch_data = {
        'token': token,
        'token_id': token_id_pad,
        'anchor_index': anchor_index,
        'pos_index': pos_index,
        'neg_indexs': neg_indexs,
        'length': length

    }
    for key in batch_data:
        if not isinstance(batch_data[key], torch.Tensor):
            try:
                batch_data[key] = torch.tensor(batch_data[key]).long()
            except:
                pass
    return batch_data

class DataProcessor(object):
    def __init__(self, data_dir, label_path):
        self.data_dir = data_dir
        self.label2id = create_label_dict(label_path)
        self.id2label = {j: i for i, j in self.label2id.items()}
        self.tokenizer = AutoTokenizer.from_pretrained("chinese-roberta-wwm-ext")
        self.bad_data = 0

    def init_menmery(self, path):
        json_datas = []
        label2dataid = {}
        origin_datas = open(path)
        for idx, data in enumerate(origin_datas):
            data = json.loads(data)
            labels = data['label'].keys()
            for label in labels:
                if label in label2dataid:
                    label2dataid[label].append(idx)
                else:
                    label2dataid[label] = [idx]
            json_datas.append(data)
        return json_datas, label2dataid

    def create_examples(self, case, num, neg_num):
        """Creates examples for the training and dev sets."""
        self.json_data, self.label2dataid = self.init_menmery('./dataset/{}.json'.format(case))
        all_data = {
            'token': [],
            'token_id':[],
            'length': [],
            'anchor_index':[],
            'pos_index': [],
            'neg_indexs': []
        }
        sample_candidate_data = []
        for _ in tqdm(range(num)):
            return_data = self._create_single_example(neg_num)
            for key in all_data.keys():
                all_data[key].append(return_data[key])
            if case == 'test':
                sample_candidate_data.append(return_data)
        if case == 'test':
            print("sample data: ")
            sample_data = random.sample(sample_candidate_data, 5)
            for sample in sample_data:
                token = sample['token']
                anchor = sample['anchor_index']
                pos = sample['pos_index']
                negs = sample['neg_indexs']
                print("case option: {}".format(sample['case_option']))
                print(sample['data'])
                print(sample['pair_data'])
                print('token: {}'.format(token))
                print('anchor: {}; anchor token:{}'.format(anchor, token[anchor]))
                print('pos: {}; pos token:{}'.format(pos, token[pos]))
                for neg in negs:
                    print('neg: {}; neg token:{}'.format(neg, token[neg]))
        for file, data in all_data.items():
            np.save(os.path.join(self.data_dir, '{}_{}.npy'.format(case, file)), data)

    def get_o_index(self, data, pair_data, shift_num):
        """
        根据数据的到O标签数据
        Args:
            data: 原始数据的一个unit
        Returns:
            标签为O的字的index: concat以后
        """
        chose_id = random.sample([0, 1], 1)[0]
        chose_data = [data, pair_data][chose_id]
        name_range = []
        for label in chose_data['label']:
            poses = chose_data['label'][label].values()
            for pos in poses:
                pos = pos[0]
                name_range.extend(list(range(pos[0], pos[1]+1)))
        candidate_list = list(set(range(len(chose_data['text'])-1)) - set(name_range))
        if len(candidate_list) == 0:  # 全部字都是实体
            index = len(chose_data['text'])+1 # 直接将[Sep]作为O
        else:
            index = random.sample(candidate_list, 1)[0]
        if chose_data == pair_data:
            index += shift_num
        else:
            index += 1
        return index

    def get_label_index(self, data, pair_data, label, shift_num):
        if label in pair_data['label'] and label in data['label']:
            chose_id = random.sample([0, 1], 1)[0]
            chose_data = [data, pair_data][chose_id]
        elif label in pair_data['label']:
            chose_data = pair_data
        else:
            chose_data = data
        start, end = list(chose_data['label'][label].values())[0][0][:]
        if chose_data == pair_data:
            start += shift_num
            end += shift_num
        else:
            start += 1
            end += 1
        return random.sample(range(start, end+1), 1)[0]

    def sample_data_and_pair_data(self):
        data = random.sample(self.json_data, 1)[0]
        chose_label = random.sample(data['label'].keys(), 1)[0]
        pair_data = data.copy()
        while pair_data == data:
            pair_data = self.json_data[random.sample(self.label2dataid[chose_label], 1)[0]]
        return data.copy(), pair_data.copy(), chose_label

    def _create_single_example(self, neg_num):
        case_option = random.sample([0, 0, 0, 1], 1)[0]
        data, pair_data, chose_label = self.sample_data_and_pair_data()
        text1 = data['text']
        text2 = pair_data['text']
        token1 = [self.tokenizer.tokenize(token)[0] for token in text1]
        token2 = [self.tokenizer.tokenize(token)[0] for token in text2]
        token = ['[CLS]'] + token1 + ['[SEP]'] + token2
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        shift_num = len(token1) + 2
        # entity index
        if case_option == 0:
            start, end = list(data['label'][chose_label].values())[0][0][:]
            anchor_index = random.sample(range(start, end+1), 1)[0] + 1 #[CLS]
            pos_index = self.get_label_index(pair_data, pair_data, chose_label, shift_num)  # 从pair data中选pos
            neg_indexs = []
            neg_label_list = (data['label'].keys() | pair_data['label'].keys()) - {chose_label}
            for _ in range(neg_num):
                c = random.sample([0, 1], 1)[0]
                # 从entity中选neg sample
                if c == 0 and len(neg_label_list):
                    neg_label = random.sample(neg_label_list, 1)[0]
                    neg_indexs.append(self.get_label_index(data, pair_data, neg_label, shift_num))
                else:
                    neg_indexs.append(self.get_o_index(data, pair_data, shift_num))
        # o index
        else:
            anchor_index = self.get_o_index(data, pair_data, shift_num)
            pos_index = self.get_o_index(data, pair_data, shift_num)
            num = 0
            while pos_index == anchor_index and num <=100:
                if num == 99:
                    import ipdb; ipdb.set_trace()
                    xxx = 1
                num += 1
                pos_index = self.get_o_index(data, pair_data, shift_num)
            neg_indexs = []
            for _ in range(neg_num):
                neg_label_list = (data['label'].keys() | pair_data['label'].keys())
                neg_label = random.sample(neg_label_list, 1)[0]
                neg_indexs.append(self.get_label_index(data, pair_data, neg_label, shift_num))

        return_data = {}
        return_data['token'] = token
        return_data['data'] = data
        return_data['pair_data'] = pair_data
        return_data['case_option'] = case_option
        return_data['token_id'] = token_id
        return_data['anchor_index'] = anchor_index
        return_data['pos_index'] = pos_index
        return_data['neg_indexs'] = neg_indexs
        return_data['length'] = len(token_id)
        return  return_data


if __name__ == '__main__':
    data_processor = DataProcessor(data_dir='dataset/bcp_single_processed_data',
                                  label_path='dataset/tool_data/label.txt')
    data_processor.create_examples(case='train', num=16000, neg_num=50)
    data_processor.create_examples(case='test', num=3000, neg_num=50)
