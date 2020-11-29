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
用于对比学习构造正负样例。共有3种情况
1、anchor为某个类型的整个entity
    - 正样本：另一个句子的同类型entity
    - 负样本：其他非同类型的entity或者O或者不完全entity
2、anchor为某个类型不完整的entity
    - 正样本：另一个句子的同类型不完整的entity
    - 负样本：其他
3、非entity span
    - 正样本：非entity span
    - 负样本：其他
"""

class BCPDataModel(Dataset):
    def __init__(self, opt, case='train'):
        self.token = np.load(os.path.join(opt.data_dir, "{}_{}.npy".format(case, 'token')), allow_pickle=True)
        self.token_id = np.load(os.path.join(opt.data_dir, "{}_{}.npy".format(case, 'token_id')), allow_pickle=True)
        self.length = np.load(os.path.join(opt.data_dir, "{}_{}.npy".format(case, 'length')), allow_pickle=True)
        self.anchor_span = np.load(os.path.join(opt.data_dir, "{}_{}.npy".format(case, 'anchor_span')), allow_pickle=True)
        self.pos_span = np.load(os.path.join(opt.data_dir, "{}_{}.npy".format(case, 'pos_span')), allow_pickle=True)
        self.neg_spans = np.load(os.path.join(opt.data_dir, "{}_{}.npy".format(case, 'neg_spans')), allow_pickle=True)

    def __getitem__(self, idx):
        return self.token[idx], self.token_id[idx], self.length[idx], self.anchor_span[idx], self.pos_span[idx],  self.neg_spans[idx]

    def __len__(self):
        return min(10000, len(self.token))
        # return 20000

def collate_fn(data):
    token, token_id, length, anchor_span, pos_span, neg_spans= zip(*data)
    max_length = max(length)
    token_id_pad = []
    for idx in range(len(token_id)):
        pad_num = max_length - len(token_id[idx])
        tid = token_id[idx] + [0] * pad_num
        token_id_pad.append(tid)
    token_id_pad = torch.tensor(token_id_pad).long()
    batch_data = {
        'token': token,
        'token_id': token_id_pad,
        'anchor_span': anchor_span,
        'pos_span':pos_span,
        'neg_spans':neg_spans,
        'length': length

    }
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
            'anchor_span':[],
            'pos_span': [],
            'neg_spans': []
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
                anchor = sample['anchor_span']
                pos = sample['pos_span']
                negs = sample['neg_spans']
                print("case option: {}".format(sample['case_option']))
                print(sample['data'])
                print(sample['pair_data'])
                print('token: {}'.format(token))
                print('anchor: {}; anchor token:{}'.format(anchor, token[anchor[0]:anchor[1]+1]))
                print('pos: {}; pos token:{}'.format(pos, token[pos[0]:pos[1]+1]))
                for neg in negs:
                    print('neg: {}; neg token:{}'.format(neg, token[neg[0]:neg[1]+1]))
        print(self.bad_data)
        for file, data in all_data.items():
            np.save(os.path.join(self.data_dir, '{}_{}.npy'.format(case, file)), data)

    def get_o_span(self, data, pair_data, shift_num):
        """
        根据数据的到O标签数据
        Args:
            data: 原始数据的一个unit
        Returns:
            (start, end)
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
        if len(candidate_list) == 0:
            return [len(data['text'])+1, len(data['text']) + 1]  # 直接将[Sep]作为O
        start = candidate_list[0]
        end_tmp = start
        while end_tmp not in name_range and end_tmp < len(chose_data['text']):
            end_tmp += 1
        end = random.sample(range(start, end_tmp+1), 1)[0]
        if chose_data == pair_data:
            start += shift_num
            end += shift_num
        else:
            start += 1
            end += 1
        if end - start >=3: # 控制长度
            end = start + 3
        return [start, end]

    def get_f_span(self, data, pair_data, label, shift_num):
        if label in pair_data['label']:
            f_span = list(pair_data['label'][label].values())[0][0][:]
            f_span[0] += shift_num
            f_span[1] += shift_num
        else:
            f_span = list(data['label'][label].values())[0][0][:]
            f_span[0] += 1
            f_span[1] += 1
        return f_span

    def get_s_span(self, data, pair_data, label, shift_num):
        if label in pair_data['label']:
            start, end = list(pair_data['label'][label].values())[0][0][:]
        else:
            start, end = list(data['label'][label].values())[0][0][:]
        if end - start == 1:
            start = end
        elif end == start:
            if label in pair_data['label']: #实体只有一个字，加入其它token
                return [start-1, end]
            else:
                return [start, end+1]
        else:
            s_start, s_end = start, end
            mean = sum([start, end]) // 2
            while [s_start, s_end] == [start, end]:
                s_start, s_end = [random.sample(range(start, mean), 1)[0], random.sample(range(mean, end), 1)[0]]
            start, end = s_start, s_end
        if label in pair_data['label']:
            start += shift_num
            end += shift_num
        else:
            start += 1
            end += 1
        return [start, end]

    def sample_data_and_pair_data(self):
        data = random.sample(self.json_data, 1)[0]
        chose_label = random.sample(data['label'].keys(), 1)[0]
        pair_data = data.copy()
        while pair_data == data:
            pair_data = self.json_data[random.sample(self.label2dataid[chose_label], 1)[0]]
        return data.copy(), pair_data.copy(), chose_label
    def _create_single_example(self, neg_num):
        case_option = random.sample([0, 1, 2], 1)[0]
        data, pair_data, chose_label = self.sample_data_and_pair_data()
        text1 = data['text']
        text2 = pair_data['text']
        token1 = [self.tokenizer.tokenize(token)[0] for token in text1]
        token2 = [self.tokenizer.tokenize(token)[0] for token in text2]
        token = ['[CLS]'] + token1 + ['[SEP]'] + token2
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        shift_num = len(token1) + 2
        # full entity
        if case_option == 0:
            start, end = list(data['label'][chose_label].values())[0][0][:]
            anchor_span = [start+1, end+1]
            pos_span = self.get_f_span(pair_data, pair_data, chose_label, shift_num)  # 从pair data中选pos

            pos_str = list(pair_data['label'][chose_label].keys())[0][:]
            get_str = "".join(token[pos_span[0]:pos_span[1]+1])
            if 'UNK' not in get_str:
                if pos_str.lower() != get_str.lower():
                    self.bad_data += 1
                    print(pos_str, get_str)
            neg_spans = []
            neg_label_list = (data['label'].keys() | pair_data['label'].keys()) - {chose_label}
            has_chose = False
            for _ in range(neg_num):
                c = random.sample([0, 1, 2], 1)[0]
                if c == 0 and len(neg_label_list) and not has_chose:
                    neg_label = random.sample(neg_label_list, 1)[0]
                    neg_spans.append(self.get_f_span(data, pair_data, neg_label, shift_num))
                    has_chose = True
                elif c == 1 and len(neg_label_list):
                    neg_label = random.sample(neg_label_list, 1)[0]
                    neg_spans.append(self.get_s_span(data, pair_data, neg_label, shift_num))
                else:
                    neg_spans.append(self.get_o_span(data, pair_data, shift_num))
        # part entity
        elif case_option == 1:
            start, end = self.get_s_span(data, data, chose_label, 0)  # 从data中选择一个部分实体
            anchor_span = [start+1, end+1]
            pos_span = self.get_s_span(pair_data, pair_data, chose_label, shift_num) # 从pair data中选择一个部分实体
            neg_spans = []
            neg_label_list = (data['label'].keys() | pair_data['label'].keys()) - {chose_label}
            has_chose = False
            for _ in range(neg_num):
                c = random.sample([0, 1, 2], 1)[0]
                if c == 0 and len(neg_label_list) and not has_chose:
                    neg_label = random.sample(neg_label_list, 1)[0]
                    neg_spans.append(self.get_f_span(data, pair_data, neg_label, shift_num))
                    has_chose = True
                elif c == 1 and len(neg_label_list):
                    neg_label = random.sample(neg_label_list, 1)[0]
                    neg_spans.append(self.get_s_span(data, pair_data, neg_label, shift_num))
                else:
                    neg_spans.append(self.get_o_span(data, pair_data, shift_num))
        else:
            anchor_span = self.get_o_span(data, pair_data, shift_num)
            pos_span = anchor_span[:]
            idx = 1
            while pos_span == anchor_span:
                idx += 1
                if idx >=20:  # 实在没法可选，只能将anchor_span作为pos_span
                    break
                pos_span = self.get_o_span(data, pair_data, shift_num)
            neg_spans = []
            has_chose = False
            for _ in range(neg_num):
                neg_label_list = (data['label'].keys() | pair_data['label'].keys())
                c = random.sample([0, 1], 1)[0]
                if c == 0 and has_chose:
                    neg_label = random.sample(neg_label_list, 1)[0]
                    neg_spans.append(self.get_f_span(data, pair_data, neg_label, shift_num))
                    has_chose = True
                else:
                    neg_label = random.sample(neg_label_list, 1)[0]
                    neg_spans.append(self.get_s_span(data, pair_data, neg_label, shift_num))

        return_data = {}
        return_data['token'] = token
        return_data['data'] = data
        return_data['pair_data'] = pair_data
        return_data['case_option'] = case_option
        return_data['token_id'] = token_id
        return_data['anchor_span'] = anchor_span
        return_data['pos_span'] = pos_span
        return_data['neg_spans'] = neg_spans
        return_data['length'] = len(token_id)
        return  return_data


if __name__ == '__main__':
    data_processor = DataProcessor(data_dir='dataset/bcp_processed_data',
                                  label_path='dataset/tool_data/label.txt')
    data_processor.create_examples(case='train', num=50000, neg_num=50)
    data_processor.create_examples(case='test', num=5000, neg_num=50)
