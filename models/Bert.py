import torch
import torch.nn as nn
from transformers import AutoModel
from .crf import CRF, get_crf_constraint
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from .BasicModule import BasicModule
from torchcrf import CRF
import torch.nn.functional as F

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True, lambda_ali=1,
                sample=None):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if lambda_ali != 1:
        multi = target != 0
        multi = (lambda_ali - 1) * multi + 1
        nll_loss = nll_loss * multi
        smooth_loss = smooth_loss * multi
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss / target.size(0)

class Bert(BasicModule):
    def __init__(self, opt):
        super().__init__()
        self.model_name = 'Bert'
        self.opt = opt
        self.bert = AutoModel.from_pretrained(opt.roberta_model_path)
        # for p in self.parameters():
        #     p.requires_grad = False
        self.lstm = None
        if self.opt.use_lstm:
            self.lstm = nn.LSTM(opt.input_features, opt.input_features // 2,
                    num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(opt.input_features, opt.tag_num)

    def forward(self, sample, train=True):
        features, encoder_padding_mask = self.extract_features(sample)
        logits = features[:,1:-1,].contiguous()
        if train:
            no_pad_logits = []
            no_pad_labels = []
            length = sample['length']
            for idx in range(len(logits)):
                no_pad_logits.append(logits[idx][:length[idx]])
                no_pad_labels.append(sample['label_id'][idx][:length[idx]])
            no_pad_logits = torch.cat(no_pad_logits, 0)
            lprob = F.log_softmax(no_pad_logits, dim=-1)
            no_pad_labels = torch.cat(no_pad_labels, 0)
            loss = label_smoothed_nll_loss(lprob,
                                           no_pad_labels,
                                           self.opt.smoothing, lambda_ali=self.opt.lamb)
            pred = torch.max(logits, -1)[1]
            return loss, pred
        else:
            pred = torch.max(logits, -1)[1]
            return pred

        # return self.nll_loss(features[1:-1,:,], sample['label_id'], encoder_padding_mask[:-2,:], 'sum')

    def extract_features(self, sample):
        x = sample['token_id']
        length = sample['length']
        encoder_padding_mask = x.eq(self.opt.padding_idx).logical_not_()
        x = self.bert(x, attention_mask=encoder_padding_mask)[0]
        if self.lstm is not None:
            x = pack_padded_sequence(x, length, True, False)
            x, _ = self.lstm(x)
            x, length = pad_packed_sequence(x, batch_first=True)
        x = self.hidden2tag(x)
        return x, encoder_padding_mask

    def decode(self, sample):
        features, encoder_padding_mask = self.extract_features(sample)
        return self.crf.decode(features, mask=encoder_padding_mask)

    # def nll_loss(self, features, tags, encoder_padding_mask, reduction):
    #     return -self.crf(features, tags.transpose(0, 1), encoder_padding_mask, reduction=reduction)

