import torch
import torch.nn as nn
from transformers import AutoModel
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from .BasicModule import BasicModule
import torch.nn.functional as F

class BCPSingle(BasicModule):
    def __init__(self, opt):
        super().__init__()
        self.model_name = 'BCPSingle'
        self.opt = opt
        self.bert = AutoModel.from_pretrained(opt.roberta_model_path)
        self.linear = nn.Sequential(
            nn.Linear(opt.input_feature, 500),
            nn.ReLU(),
            nn.Linear(500, opt.hidden_size)
        )


    def select_single_token_rep(self, batch_rep, token_pos):
        """
        batch_rep: B x L x dim
        token_pos: B

        Returns:
            B x dim
        """
        B, L, dim = batch_rep.size()
        shift = torch.arange(B) * L
        if self.opt.use_gpu:
            shift = shift.cuda()
        token_pos = token_pos + shift
        res = batch_rep.contiguous().view(-1, dim)[token_pos]
        return res

    def select_multi_token_rep(self, batch_rep, token_pos):
        '''
        Args:
            batch_rep: B x L x dim
            token_pos: B x K
        Returns:
            B x K x dim
        '''
        B, L, dim = batch_rep.size()
        _, K = token_pos.size()
        shift = torch.arange(B).unsqueeze(1).expand(-1, K).contiguous().view(-1) * L  # B * K
        if self.opt.use_gpu:
            shift = shift.cuda()
        token_pos = token_pos.view(-1)
        token_pos += shift
        res = batch_rep.contiguous().view(-1, dim)[token_pos]
        return res.view(B, K, dim)

    def forward(self, sample, train=True):
        x = sample['token_id']
        encoder_padding_mask = x.eq(self.opt.padding_idx).logical_not_()
        batch_feature = self.bert(x, attention_mask=encoder_padding_mask)[0]  # B x L x dim
        anchor_rep = self.select_single_token_rep(batch_feature, sample['anchor_index']) # B x dim
        pos_rep = self.select_single_token_rep(batch_feature, sample['pos_index']) # B x dim
        neg_reps = self.select_multi_token_rep(batch_feature, sample['neg_indexs']) # B x K x dim
        contrasive_rep = torch.cat([pos_rep.unsqueeze(1), neg_reps], 1) # B x (1 + K) x dim
        anchor_rep = anchor_rep.unsqueeze(1).expand(-1, contrasive_rep.size(1), -1) # B x (1+K) x dim
        anchor_rep = self.linear(anchor_rep)
        contrasive_rep = self.linear(contrasive_rep)
        batch_sim = torch.cosine_similarity(anchor_rep, contrasive_rep, dim=-1) # B x (1 + K)
        batch_label = torch.zeros(len(batch_feature)).long().cuda()  # 因为是把pos放在第一个位置的
        # batch_label = torch.ones(len(batch_feature)).long().cuda()  # 弄成1，看模型可不可以学，可以就说明任务设计的不太好，改后不能学
        pred = torch.max(batch_sim, 1)[1]
        if train:
            loss = F.cross_entropy(batch_sim / self.opt.tao, batch_label)
            return loss, pred
        else:
            return pred