import torch
import torch.nn as nn
from transformers import AutoModel
from .crf import CRF, get_crf_constraint
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from .BasicModule import BasicModule
from utils import create_id2describe
import numpy as np
# from torchcrf import CRF

class BCLabelInfo(BasicModule):
    def __init__(self, opt):
        super().__init__()
        self.model_name = 'BCLabelInfo'
        self.opt = opt
        self.bert = AutoModel.from_pretrained(opt.roberta_model_path)
        self.bert_des = AutoModel.from_pretrained(opt.roberta_model_path)
        self.des_id = torch.tensor(np.load('./dataset/describe_id.npy')).long().cuda()
        if opt.BCP_path:
            print('load parameter of BCP:')
            checkpoint = torch.load(opt.BCP_path, map_location='cpu')
            print('contrastive acc:{}'.format(checkpoint['best_acc']))
            checkpoint = {k[5:]:v for k,v in checkpoint['parameters'].items() if k[:5] == 'bert.'}
            self.bert.load_state_dict(checkpoint)
            self.model_name = 'BLCLabelInfo_BCP'
        if not opt.train_bert:
            for p in self.parameters():
                p.requires_grad = False
        self.hidden2tag_W = nn.Parameter(nn.init.xavier_normal_(torch.randn(opt.tag_num, opt.input_features)), requires_grad=True)
        self.hidden2tag_bias = nn.Parameter(nn.init.constant_(torch.randn(opt.tag_num), 0), requires_grad=True)
        # self.hidden2tag = nn.Linear(opt.input_features, opt.tag_num)
        start_tags, constraints = get_crf_constraint(opt.tags)
        self.crf = CRF(opt.tag_num, constraints=constraints, start_tags=start_tags, lamb=opt.lamb)
        # self.crf = CRF(opt.tag_num, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(opt.tag_num)
    # def init_weight(self):
    #     nn.init.xavier_normal_(self.hidden2tag.weight)
    #     nn.init.constant_(self.hidden2tag.bias, 0)

    def forward(self, sample, train=True):

        bert_feature, logits_bert, encoder_padding_mask = self.extract_features(sample)
        logits_des_sim, _ = self.get_des_logits(bert_feature)
        logits_des_sim = logits_des_sim.permute(1,0,2)
        logits = (1-self.opt.dw)*logits_bert + self.opt.dw*logits_des_sim
        # print(logits_des_sim.sum())
        # print(logits_bert.sum())
        # norm_loss = torch.norm(des_feature - self.hidden2tag.weight)
        if train:
            loss = -self.crf(logits, sample['label_id'].transpose(0, 1), encoder_padding_mask)
            pred = self.crf.decode(logits, encoder_padding_mask)
            # return loss + norm_loss, pred
            return loss, pred
        else:
            return self.crf.decode(logits, encoder_padding_mask)

    def extract_features(self, sample):
        x = sample['token_id']
        encoder_padding_mask = x.eq(self.opt.padding_idx).logical_not_()
        bert_feature = self.bert(x, attention_mask=encoder_padding_mask)[0] # B x L x hidden_size
        norm_W = self.hidden2tag_W / torch.norm(self.hidden2tag_W, dim=-1).unsqueeze(1)
        logits = torch.matmul(bert_feature, norm_W.permute(1, 0)) + self.hidden2tag_bias.unsqueeze(0).unsqueeze(0)  # B x L x tag_num
        logits = logits.permute(1,0,2)
        return bert_feature[:,1:-1,:], logits[1:-1, :, :], encoder_padding_mask.transpose(0, 1)[1:-1, :]

    def get_des_logits(self, bert_feature):
        # bert_feature: B x L  x dim
        # return: B x L x tag_num
        label_mask = self.des_id.eq(self.opt.padding_idx).logical_not_()
        des_feature = self.bert_des(self.des_id, attention_mask=label_mask)[0][:, 0, :]  # tag_nums x hidden_size
        des_feature = des_feature / torch.norm(des_feature, dim=-1).unsqueeze(1)
        # des_feature = self.batch_norm(des_feature.unsqueeze(0)) # 1 x tag_nums x hiddeen_size
        logits = torch.matmul(bert_feature, des_feature.permute(1, 0)) + self.hidden2tag_bias.unsqueeze(0).unsqueeze(0)
        # logits = torch.matmul(bert_feature, des_feature.permute(0, 2, 1))
        return logits, des_feature
        # bert_feature = bert_feature.unsqueeze(2).expand(-1, -1, self.opt.tag_num, -1)  # B x L x tag_num x dim
        # des_feature = des_feature.unsqueeze(0).expand(B * L, -1, -1).contiguous().view(B, L, self.opt.tag_num,-1)  # B x L x tag_num x dim
        # concate_feature = torch.cat([bert_feature, des_feature], dim=-1)
        # sim = self.multilinear(concate_feature).squeeze(-1) # B x L x tag_num
        # return sim
