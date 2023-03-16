import os, sys, pickle
import numpy as np
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from itertools import combinations, permutations

from utils import *
from layers import *
from functions import *
from data import *

from .base import *
from torch.distributions import Normal
from torch.distributions.utils import _standard_normal
import copy


# pytorch 官网transformer位置信息的计算
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        device = torch.device('cuda:0')

        pe = torch.zeros(max_len, d_model)

        pe = pe.to(device)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        # 自己添加的两行
        device = torch.device('cuda:0')
        x = x.to(device)

        x = x + self.pe[:x.size(0), :]
        # x = torch.Tensor.permute(x, (1, 0, -1))#调整和S一样的维度：B，N，H
        return self.dropout(x)


class TabEncoding(nn.Module):

    def __init__(self, config, mdrnn, emb_dim=None):
        super().__init__()

        self.config = config
        if emb_dim is None:
            self.emb_dim = config.hidden_dim
        else:
            self.emb_dim = emb_dim
        self.hidden_dim = config.hidden_dim
        self.extra_hidden_dim = config.head_emb_dim
        if config.tab_reduce.lower() == "att":
            ReduceModel = AttReduce
        elif config.tab_reduce.lower() == "cat":
            ReduceModel = CatReduce
        else:
            raise ValueError(config.tab_reduce)
        self.seq2mat = ReduceModel(n_in=self.emb_dim, n_out=self.hidden_dim)  # 句子到矩阵
        if self.extra_hidden_dim > 0:
            self.reduce_X = nn.Linear(self.emb_dim + self.extra_hidden_dim, self.hidden_dim)

        self.mdrnn = mdrnn

    def forward(
            self,
            S: torch.Tensor,
            Tlm: torch.Tensor = None,
            Tstates: torch.Tensor = None,
            masks: torch.Tensor = None,
    ):

        if masks is not None:
            masks = masks.unsqueeze(1) & masks.unsqueeze(2)

            # 拟添加位置信息
        pos_encoder = PositionalEncoding(self.emb_dim)
        pos_output = pos_encoder(S)
        # torch.Tensor.permute(pos_output, (1, 0, -1))
        S = pos_output + S

        X = self.seq2mat(S, S)  # (B, N, N, H)
        X = F.relu(X, inplace=False)  # True
        if self.extra_hidden_dim > 0:
            X = self.reduce_X(torch.cat([X, Tlm], -1))
            X = F.relu(X, inplace=False)  # True

        T, Tstates = self.mdrnn(X, states=Tstates, masks=masks)

        # the content of T and Tstates is the same;
        # keep Tstates is to save some slice and concat operations

        return T, Tstates


class SeqEncoding(nn.Module):

    def __init__(self, config, emb_dim=None, n_heads=8):
        super().__init__()

        self.config = config
        if emb_dim is None:
            self.emb_dim = config.hidden_dim
        else:
            self.emb_dim = emb_dim

        self.hidden_dim = config.hidden_dim
        self.n_heads = n_heads

        self.linear_qk = nn.Linear(self.emb_dim, self.n_heads, bias=None)
        self.linear_v = nn.Linear(self.emb_dim, self.hidden_dim, bias=True)
        self.linear_o = nn.Linear(self.emb_dim, self.hidden_dim, bias=True)

        self.norm0 = nn.LayerNorm(self.config.hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(self.config.dropout, inplace=True),
            nn.Linear(self.config.hidden_dim * 4, self.config.hidden_dim),
        )
        self.norm1 = nn.LayerNorm(self.config.hidden_dim)

        self.dropout_layer = nn.Dropout(self.config.dropout, inplace=True)

    def forward(
            self,
            S: torch.Tensor,
            T: torch.Tensor,
            masks: torch.Tensor = None,
    ):
        # 拟添加位置信息
        pos_encoder = PositionalEncoding(self.emb_dim)
        pos_output = pos_encoder(S)
        # torch.Tensor.permute(pos_output, (1, 0, -1))
        S = pos_output + S

        B, N, H = S.shape
        n_heads = self.n_heads
        subH = H // n_heads

        if masks is not None:
            masks = masks.unsqueeze(1) & masks.unsqueeze(2)
            masks_addictive = -1000. * (1. - masks.float())
            masks_addictive = masks_addictive.unsqueeze(-1)

        S_res = S

        # Table-Guided Attention
        attn = self.linear_qk(T)
        attn = attn + masks_addictive
        attn = attn.permute(0, -1, 1, 2)  # (B, n_heads, N, T)
        attn = attn.softmax(-1)  # (B, n_heads, N, T)

        v = self.linear_v(S)  # (B, N, H)
        v = v.view(B, N, n_heads, subH).permute(0, 2, 1, 3)  # B, n_heads, N, subH

        S = attn.matmul(v)  # B, n_heads, N, subH
        S = S.permute(0, 2, 1, 3).reshape(B, N, subH * n_heads)

        S = self.linear_o(S)
        S = F.relu(S, inplace=False)
        S = self.dropout_layer(S)

        S = S_res + S
        S = self.norm0(S)

        S_res = S

        # Position-wise FeedForward
        S = self.ffn(S)
        S = self.dropout_layer(S)
        S = S_res + S
        S = self.norm1(S)

        return S, attn


class TwoWayEncoding(nn.Module):

    def __init__(self, config, emb_dim=None, n_heads=8, first=False, direction='B'):
        super().__init__()

        self.config = config
        if emb_dim is None:
            self.emb_dim = config.hidden_dim
        else:
            self.emb_dim = emb_dim
        self.hidden_dim = config.hidden_dim
        self.n_heads = n_heads
        self.extra_hidden_dim = config.head_emb_dim
        self.first = first  # first MD-RNN only needs 2D, others should be 3D
        self.direction = direction

        self.tab_encoding = TabEncoding(
            config=self.config,
            mdrnn=get_mdrnn_layer(
                config, direction=direction, norm='', Md='25d', first_layer=first),
            emb_dim=self.emb_dim, )
        self.seq_encoding = SeqEncoding(
            config=self.config, emb_dim=self.emb_dim, n_heads=n_heads)

    def forward(
            self,
            S: torch.Tensor,
            Tlm: torch.Tensor = None,  # T^{\ell} attention weights of BERT
            Tstates: torch.Tensor = None,
            masks: torch.Tensor = None
    ):

        T, Tstates = self.tab_encoding(S=S.clone(), Tlm=Tlm, Tstates=Tstates, masks=masks)

        S, attn = self.seq_encoding(S=S, T=T.clone(), masks=masks)

        return S, attn, T, Tstates


class StackedTwoWayEncoding(nn.Module):

    def __init__(self, config, n_heads=8, direction='B'):
        super().__init__()

        self.config = config
        self.depth = config.num_layers
        self.hidden_dim = config.hidden_dim
        self.extra_hidden_dim = config.head_emb_dim
        self.n_heads = n_heads
        self.direction = direction

        self.layer0 = TwoWayEncoding(
            self.config, n_heads=n_heads, first=True, direction=direction,
        )

        self.layers = nn.ModuleList([
            TwoWayEncoding(
                self.config, n_heads=n_heads, first=False, direction=direction,
            ) for i in range(self.depth - 1)
        ])

    def forward(
            self,
            S: torch.Tensor,
            Tlm: torch.Tensor = None,
            masks: torch.Tensor = None
    ):
        S_list = []
        T_list = []
        attn_list = []

        Tstates = None

        S, attn, T, Tstates = self.layer0(S=S, Tlm=Tlm, Tstates=Tstates, masks=masks)

        S_list.append(S)
        T_list.append(T)
        attn_list.append(attn)

        #test the parameters
        model_tmp = self.layers
        total = sum([param.nelement() for param in model_tmp.parameters()])
        print("****************************")
        print("Number of parameter: %.2fM" % (total/1e6))
        print("****************************")



        for layer in self.layers:
            S, attn, T, Tstates = layer(S=S, Tlm=Tlm, Tstates=Tstates, masks=masks)

            S_list.append(S)
            T_list.append(T)
            attn_list.append(attn)

        return S_list, T_list, attn_list


class JointModel(Tagger):

    def __init__(self, config):
        super(JointModel, self).__init__(config)
        n_dim = config.hidden_dim
        self.tab_mu_layer = nn.Linear(n_dim, n_dim).to(config.device)
        self.seq_mu_layer = nn.Linear(n_dim, n_dim).to(config.device)

        self.tab_logvar_layer = nn.Linear(n_dim, n_dim).to(config.device)
        self.seq_logvar_layer = nn.Linear(n_dim, n_dim).to(config.device)
        self.IB = config.IB
        self.beta = config.beta

    def _variational_layer(self, hidden, mu_layer, logvar_layer):
        sampled_z = hidden
        kld = 0.0
        if self.training and self.IB is True:
            mu = hidden
            logvar = logvar_layer(hidden)
            # TODO 训练次数设置的大点, 超5w, 8w起吧
            std = F.softplus(logvar)
            # std = torch.exp(0.5 * logvar)
            posterior = Normal(loc=mu, scale=std)

            zeros = torch.zeros_like(mu, device=mu.device)
            ones = torch.ones_like(std, device=std.device)
            prior = Normal(zeros, ones)

            eps = std.new_empty(std.shape)
            eps.normal_()
            sampled_z = mu + std * eps

            kld = posterior.log_prob(sampled_z).sum(-1) - prior.log_prob(sampled_z).sum(-1)

        return sampled_z, kld

    def check_attrs(self):
        assert hasattr(self, 'ner_tag_indexing')
        assert hasattr(self, 're_tag_indexing')

    def get_default_trainer_class(self):
        return JointTrainer

    def set_embedding_layer(self):

        if self.config.tag_form.lower() == 'iob2':
            self.one_entity_n_tags = 2
        elif self.config.tag_form.lower() == 'iobes':
            self.one_entity_n_tags = 4
        else:
            raise Exception('no such tag form.')
        self.ner_tag_indexing = get_tag_indexing(self.config)
        self.re_tag_indexing = IndexingMatrix(depth=3)  # 2d
        self.re_tag_indexing.vocab = {'O': 0}

        self.token_embedding = AllEmbedding(self.config)
        self.token_indexing = self.token_embedding.preprocess_sentences

    def set_encoding_layer(self):

        emb_dim = self.config.token_emb_dim + self.config.char_emb_dim + self.config.lm_emb_dim
        # emb_dim = self.config.lm_emb_dim

        self.reduce_emb_dim = nn.Linear(emb_dim, self.config.hidden_dim)
        init_linear(self.reduce_emb_dim)

        self.bi_encoding = StackedTwoWayEncoding(self.config)

        self.dropout_layer = nn.Dropout(self.config.dropout, inplace=True)

    def set_logits_layer(self):

        self.ner_tag_logits_layer = nn.Linear(self.config.hidden_dim, self.config.ner_tag_vocab_size)
        init_linear(self.ner_tag_logits_layer)

        self.re_tag_logits_layer = nn.Linear(self.config.hidden_dim, self.config.re_tag_vocab_size)
        init_linear(self.re_tag_logits_layer)

    def set_loss_layer(self):

        if self.config.crf:
            self.crf_layer = eval(self.config.crf)(self.config)

        self.soft_loss_layer = LabelSmoothLoss(0.02, reduction='none')
        self.loss_layer = nn.CrossEntropyLoss(reduction='none')

    def forward_embeddings(self, inputs):

        if '_tokens' in inputs:
            sents = inputs['_tokens']
        else:
            sents = inputs['tokens']

        embeddings, masks, embeddings_dict = self.token_embedding(sents, return_dict=True)

        embeddings = self.dropout_layer(embeddings)
        embeddings = self.reduce_emb_dim(embeddings)

        lm_heads = embeddings_dict.get('lm_heads', None)
        inputs['lm_heads'] = lm_heads

        ## relation encoding
        seq_embeddings, tab_embeddings, attns = self.bi_encoding(embeddings, masks=masks, Tlm=lm_heads)
        seq_embeddings = seq_embeddings[-1]
        tab_embeddings = tab_embeddings[-1]
        seq_embeddings = self.dropout_layer(seq_embeddings)
        tab_embeddings = self.dropout_layer(tab_embeddings)

        seq_embeddings, seq_kld = self._variational_layer(seq_embeddings, self.seq_mu_layer, self.seq_logvar_layer)
        tab_embeddings, tab_kld = self._variational_layer(tab_embeddings, self.tab_mu_layer, self.tab_logvar_layer)

        # use diag as seq representation
        # seq_embeddings = tab_embeddings.diagonal(dim1=1, dim2=2).permute(0, -1, 1)

        inputs['attns'] = attns
        inputs['masks'] = masks
        inputs['tab_embeddings'] = tab_embeddings
        inputs['seq_embeddings'] = seq_embeddings
        inputs['seq_kld'] = seq_kld
        inputs['tab_kld'] = tab_kld
        return inputs

    def forward_step(self, inputs):
        '''
        inputs: {
            'tokens': List(List(str)),
            '_tokens'(*): [Tensor, Tensor],
        }
        outputs: +{
            'logits': Tensor,
            'masks': Tensor
        }
        '''

        inputs = self.forward_embeddings(inputs)
        tab_embeddings = inputs['tab_embeddings']
        seq_embeddings = inputs['seq_embeddings']
        re_tag_logits = self.re_tag_logits_layer(tab_embeddings)

        # use diagonal elements
        # ner_tag_embeddings = relation_embeddings.diagonal(dim1=1, dim2=2).permute(0, -1, 1)
        ner_tag_logits = self.ner_tag_logits_layer(seq_embeddings)

        rets = inputs
        rets['ner_tag_logits'] = ner_tag_logits
        rets['re_tag_logits'] = re_tag_logits

        return rets

    def forward(self, inputs):

        rets = self.forward_step(inputs)
        ner_tag_logits = rets['ner_tag_logits']
        re_tag_logits = rets['re_tag_logits']

        mask = rets['masks']
        mask_float = mask.float()  # B, T

        if '_ner_tags' in rets:
            ner_tags = rets['_ner_tags'].to(self.device)
        else:
            ner_tags = self.ner_tag_indexing(rets['ner_tags']).to(self.device)

        if self.config.crf == 'CRF':
            e_loss = - self.crf_layer(ner_tag_logits, tags, mask=mask, reduction=self.config.loss_reduction)
        elif not self.config.crf:
            e_loss = self.loss_layer(ner_tag_logits.permute(0, 2, 1), ner_tags)  # B, T
            e_loss = (e_loss * mask_float).sum()
            seq_kld = (rets['seq_kld'] * mask_float).sum()
        else:
            raise Exception('not a compatible loss')

        if '_re_tags' in rets:
            re_tags = rets['_re_tags'].to(self.device)
        else:
            re_tags = self.re_tag_indexing(rets['re_tags']).to(self.device)

        matrix_mask_float = mask_float[:, None] * mask_float[:, :, None]  # B, N, N
        r_loss = self.loss_layer(re_tag_logits.permute(0, -1, 1, 2), re_tags)
        r_loss = (r_loss * matrix_mask_float).sum()
        tab_kld = (rets['tab_kld'] * matrix_mask_float).sum()
        loss = e_loss + r_loss

        rets['_ner_tags'] = ner_tags
        rets['_re_tags'] = re_tags
        rets['loss'] = loss
        rets['kld_loss'] = seq_kld + tab_kld

        try:
            torch.isfinite(re_tag_logits).all()
        except:
            print('error!')

        assert torch.isfinite(re_tag_logits).all()
        assert torch.isfinite(ner_tag_logits).all()
        assert torch.isfinite(loss).all()
        return rets

    def _postprocess_entities(self, tags_list):

        entities = []

        for tags in tags_list:
            spans, etypes = tag2span(tags, True, True)
            entities.append([(i_start, i_end, etype) for (i_start, i_end), etype in zip(spans, etypes)])

        return entities

    def _postprocess_relations(self, relation_logits, entities):

        # convert relation_logits to relations on entities           [9.9870e-01, 7.2231e-04, 5.4624e-04,  ..., 7.6478e-09,
        #            3.4560e-09, 4.0777e-09],
        #           [9.9890e-01, 6.0854e-04, 4.6359e-04,  ..., 6.8057e-09,
        #            3.3664e-09, 3.9135e-09]],

        # except for 'O', a typical relation tag is in form of 'a:b:c', where
        # a === 'I' constantly, means the word-pair is inside a relation. We can also extent it to scheme like BIO or BIEOS;
        # b in {'fw', 'bw'}, indicating the direction of a relation;
        # c is the relation type.

        relations = []

        fw_togglemap, bw_togglemap = self._get_togglemaps()

        relation_logits = relation_logits.detach().softmax(-1).cpu()
        for i_batch, _entities in enumerate(entities):

            curr = set()
            for (ib, ie, it), (jb, je, jt) in permutations(_entities, 2):  # permutations()全排列
                fw_logit = relation_logits[i_batch, ib:ie, jb:je].sum(0).sum(0) @ fw_togglemap
                bw_logit = relation_logits[i_batch, jb:je, ib:ie].sum(0).sum(0) @ bw_togglemap

                logit = fw_logit + bw_logit

                rid = int(logit.argmax())
                rtag = self.re_tag_indexing.idx2token(rid)  # 从关系标记找到对应关系的开始结束索引及其类型

                if rtag != 'O':
                    _, direction, rtype = rtag.split(':')
                    curr.add((ib, ie, jb, je, rtype))

            relations.append(curr)

        return relations

    def _get_togglemaps(self):

        # fw_togglemap only keep forward relation
        # bw_togglemap maps backward relation to forward relation

        # self.need_update = True # force update

        # the togglemap need to be updated in case the tag vocab changes.

        if not hasattr(self, 'need_update'):
            self.need_update = True

        if not self.need_update:
            return self.fw_togglemap, self.bw_togglemap

        fw_togglemap = torch.zeros(self.config.re_tag_vocab_size, self.config.re_tag_vocab_size)
        bw_togglemap = torch.zeros(self.config.re_tag_vocab_size, self.config.re_tag_vocab_size)

        for head_tag in self.re_tag_indexing.vocab:
            head_id = self.re_tag_indexing.token2idx(head_tag)
            if head_tag == 'O' or head_tag.split(':')[-1] == 'O':
                fw_togglemap[head_id, 0] = 1
                bw_togglemap[head_id, 0] = 1
            else:
                a, b, c = head_tag.split(':')
                if b == 'fw':
                    tail_id = self.re_tag_indexing.token2idx(f"{a}:bw:{c}")
                    fw_togglemap[head_id, head_id] = 1
                    fw_togglemap[tail_id, 0] = 1
                    bw_togglemap[head_id, 0] = 1
                    bw_togglemap[tail_id, head_id] = 1
                elif b == 'bw':
                    tail_id = self.re_tag_indexing.token2idx(f"{a}:fw:{c}")
                    fw_togglemap[tail_id, tail_id] = 1
                    fw_togglemap[head_id, 0] = 1
                    bw_togglemap[tail_id, 0] = 1
                    bw_togglemap[head_id, tail_id] = 1

        self.fw_togglemap = fw_togglemap
        self.bw_togglemap = bw_togglemap

        self.need_update = False

        return self.fw_togglemap, self.bw_togglemap

    def predict_step(self, inputs):

        rets = self.forward_step(inputs)
        re_tag_logits = rets['re_tag_logits']
        ner_tag_logits = rets['ner_tag_logits']
        mask = rets['masks']
        mask_np = mask.cpu().detach().numpy()

        if self.config.crf == 'CRF':
            ner_tag_preds = self.crf_layer.decode(ner_tag_logits, mask=mask)
        elif not self.config.crf:
            ner_tag_preds = ner_tag_logits.argmax(dim=-1).cpu().detach().numpy()
        else:
            raise Exception('not a compatible decode')

        ner_tag_preds = np.array(ner_tag_preds)
        ner_tag_preds *= mask_np
        ner_tag_preds = self.ner_tag_indexing.inv(ner_tag_preds)
        rets['ner_tag_preds'] = ner_tag_preds

        matrix_mask_np = mask_np[:, np.newaxis] + mask_np[:, :, np.newaxis]

        ## uncomment below to return relation tag for each entry by argmax
        # re_tag_preds = relation_logits.argmax(dim=-1).cpu().detach().numpy()
        # re_tag_preds *= matrix_mask_np
        # re_tag_preds = self.re_tag_indexing.inv(re_tag_preds) # str:(B, N, N)
        # rets['re_tag_preds'] = re_tag_preds

        entity_preds = self._postprocess_entities(ner_tag_preds)
        # relation_preds = self._postprocess_relations(re_tag_logits, rets[
        #     'entities'])  # use GOLD entity spans:(relation_logits, rets['entities'])
        relation_preds = self._postprocess_relations(re_tag_logits, entity_preds)
        rets['entity_preds'] = entity_preds
        rets['relation_preds'] = relation_preds

        return rets

    def train_step(self, inputs):
        sys.stdout.flush()
        self.need_update = True

        rets = self(inputs)
        loss = rets['loss']
        kld_loss = rets['kld_loss']

        g_steps = self.global_steps.data.cpu().detach().numpy()[0]
        if self.config.anneal_steps > 0:
            beta = self.beta * min(1., 1. * g_steps / self.config.anneal_steps)
        else:
            beta = self.beta

        loss = loss + beta * kld_loss
        with torch.autograd.detect_anomaly():
            loss.backward()

        grad_period = self.config.grad_period if hasattr(self.config, 'grad_period') else 1
        if grad_period == 1 or self.global_steps.data % grad_period == grad_period - 1:
            nn.utils.clip_grad_norm_(self.parameters(), 5)

        return rets

    def save_ckpt(self, path):
        torch.save(self.state_dict(), path + '.pt')
        with open(path + '.vocab.pkl', 'wb') as f:
            pickle.dump(self.token_embedding.token_indexing.vocab, f)
        with open(path + '.char_vocab.pkl', 'wb') as f:
            pickle.dump(self.token_embedding.char_indexing.vocab, f)
        with open(path + '.ner_tag_vocab.pkl', 'wb') as f:
            pickle.dump(self.ner_tag_indexing.vocab, f)
        with open(path + '.re_tag_vocab.pkl', 'wb') as f:
            pickle.dump(self.re_tag_indexing.vocab, f)

    def load_ckpt(self, path):
        self.load_state_dict(torch.load(path + '.pt'))
        with open(path + '.vocab.pkl', 'rb') as f:
            self.token_embedding.token_indexing.vocab = pickle.load(f)
            self.token_embedding.token_indexing.update_inv_vocab()
        with open(path + '.char_vocab.pkl', 'rb') as f:
            self.token_embedding.char_indexing.vocab = pickle.load(f)
            self.token_embedding.char_indexing.update_inv_vocab()
        with open(path + '.ner_tag_vocab.pkl', 'rb') as f:
            self.ner_tag_indexing.vocab = pickle.load(f)
            self.ner_tag_indexing.update_inv_vocab()
        with open(path + '.re_tag_vocab.pkl', 'rb') as f:
            self.re_tag_indexing.vocab = pickle.load(f)
            self.re_tag_indexing.update_inv_vocab()


class JointModelMacroF1(JointModel):

    def get_default_trainer_class(self):
        return JointTrainerMacroF1
