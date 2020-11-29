import torch
import torch.nn as nn
from transformers import AutoModel
from .crf import CRF, get_crf_constraint
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from .BasicModule import BasicModule
from utils import create_id2describe
import numpy as np
# from torchcrf import CRF

class BertCrf(BasicModule):
    def __init__(self, opt):
        super().__init__()
        self.model_name = 'BertCrf'
        self.opt = opt
        self.bert = AutoModel.from_pretrained(opt.roberta_model_path)
        if opt.BCP_path:
            print('load parameter of BCP:')
            checkpoint = torch.load(opt.BCP_path, map_location='cpu')
            print('contrastive acc:{}'.format(checkpoint['best_acc']))
            checkpoint = {k[5:]:v for k,v in checkpoint['parameters'].items() if k[:5] == 'bert.'}
            self.bert.load_state_dict(checkpoint)
            self.model_name = 'BertCrf_BCP'
        if not opt.train_bert:
            for p in self.parameters():
                p.requires_grad = False

        self.hidden2tag = nn.Linear(opt.input_features, opt.tag_num)
        start_tags, constraints = get_crf_constraint(opt.tags)
        if opt.use_cons == True:
            self.crf = CRF(opt.tag_num, constraints=constraints, start_tags=start_tags, lamb=opt.lamb)
        else:
            self.crf = CRF(opt.tag_num, start_tags=start_tags, lamb=opt.lamb)
        # self.crf = CRF(opt.tag_num, batch_first=True)
    #     self.init()
    # def init(self):
    #     nn.init.xavier_normal_(self.hidden2tag.weight)
    #     nn.init.constant_(self.hidden2tag.bias, 0)

    def forward(self, sample, train=True):
        _, logits, encoder_padding_mask = self.extract_features(sample)
        if train:
            loss = -self.crf(logits, sample['label_id'].transpose(0, 1), encoder_padding_mask)
            pred = self.crf.decode(logits, encoder_padding_mask)
            return loss , pred
        else:
            return self.crf.decode(logits, encoder_padding_mask)

    def extract_features(self, sample):
        x = sample['token_id']
        encoder_padding_mask = x.eq(self.opt.padding_idx).logical_not_()
        bert_feature = self.bert(x, attention_mask=encoder_padding_mask)[0]
        logits = self.hidden2tag(bert_feature).permute(1,0,2)
        return bert_feature[:, 1:-1, :], logits[1:-1, :, :], encoder_padding_mask.transpose(0, 1)[1:-1, :]

