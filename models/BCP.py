import torch
import torch.nn as nn
from transformers import AutoModel
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from .BasicModule import BasicModule
import torch.nn.functional as F

class BCP(BasicModule):
    def __init__(self, opt):
        super().__init__()
        self.model_name = 'BCP'
        self.opt = opt
        self.bert = AutoModel.from_pretrained(opt.roberta_model_path)
        self.linear = nn.Sequential(
            nn.Linear(opt.input_feature, 500),
            nn.ReLU(),
            nn.Linear(500, opt.hidden_size)
        )
    def forward(self, sample, train=True):
        x = sample['token_id']
        encoder_padding_mask = x.eq(self.opt.padding_idx).logical_not_()
        batch_feature = self.bert(x, attention_mask=encoder_padding_mask)[0]  # B x L x dim
        batch_sim = []
        for idx, feature in enumerate(batch_feature):
            anchor = sample['anchor_span'][idx]
            pos = sample['pos_span'][idx]
            negs = sample['neg_spans'][idx]
            negs_emb = []
            anchor_emb = torch.mean(feature[anchor[0]: anchor[1]+1], 0) # dim
            pos_emb = torch.mean(feature[pos[0]: pos[1]+1], 0)
            for neg in negs:
                negs_emb.append(torch.mean(feature[neg[0]: neg[1]+1],0))
            contrasive_emb = torch.stack([pos_emb] + negs_emb, 0)
            anchor_emb = anchor_emb.unsqueeze(0).expand(len(contrasive_emb),-1)
            anchor_emb = self.linear(anchor_emb)
            contrasive_emb = self.linear(contrasive_emb)
            sim = torch.cosine_similarity(anchor_emb, contrasive_emb)
            batch_sim.append(sim)
        batch_sim = torch.stack(batch_sim, 0) # batch x (neg_num + 1)
        batch_label = torch.zeros(len(batch_feature)).long().cuda()  # 因为是把pos放在第一个位置的
        # batch_label = torch.ones(len(batch_feature)).long().cuda()  # 弄成1，看模型可不可以学，可以就说明任务设计的不太好，改后不能学
        pred = torch.max(batch_sim, 1)[1]

        if train:
            loss = F.cross_entropy(batch_sim / self.opt.tao, batch_label)
            return loss, pred
        else:
            return pred