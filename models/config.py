__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@foxmail.com"

__all__ = ["TBTOConfig"]

import argparse
import copy


class OrderedNamespace(argparse.Namespace):
    def __init__(self, **kwargs):
        self._ordered_key = []
        super(OrderedNamespace, self).__init__(**kwargs)

    def _get_kwargs(self):
        # retrieve (key, value) pairs in the order they were initialized using _keys
        return [(k, self.__dict__[k]) for k in self._ordered_key]

    def __setattr__(self, key, value):
        # store new attribute (key, value) pairs in builtin __dict__
        self.__dict__[key] = value
        # store the keys in self._keys in the order that they are initialized
        # do not store '_keys' itself and don't enter any key more than once
        if key not in ['_ordered_key'] + self._ordered_key:
            self._ordered_key.append(key)

    def items(self):
        for key, value in self._get_kwargs():
            yield (key, value)


class Config(OrderedNamespace):
    def __init__(self, **kwargs):
        super(Config, self).__init__(**kwargs)

    def _validate(self):
        pass

    def to_printable(self):
        arg_strings = [f"{key}={value}" for key, value in self.items()]
        return "\n".join(arg_strings)

    def parse_args(self, args=None):
        parser = argparse.ArgumentParser()
        for key, value in self.items():
            parser.add_argument(f"--{key}", type=type(value), default=value, required=False)

        return parser.parse_args(args=args, namespace=self)


class TBTOConfig(Config):
    def __init__(self, **kwargs):
        super(TBTOConfig, self).__init__(**kwargs)
        self.model_class = "JointModel"
        self.model_read_ckpt = None
        self.model_write_ckpt = None
        self.pretrained_wv = "./wv/glove.6B.100d.conll04.txt"  # conll04,ade
        self.dataset = "CoNLL04"  # CoNLL04,ADE0,...,ADE9,
        self.label_config = None
        self.batch_size = 4  # scierc_all需要减小batch_size
        self.evaluate_interval = 1000  # 1000
        self.max_steps = 500000#120000
        self.max_epoches = 12000
        self.decay_rate = 0.05
        self.token_emb_dim = 100
        self.char_encoder = "lstm"
        self.char_emb_dim = 30
        self.tag_emb_dim = 50
        self.cased = False
        self.hidden_dim = 200
        self.num_layers = 3
        self.max_depth = None
        self.crf = None
        self.loss_reduction = "sum"
        self.maxlen = None
        self.dropout = 0.5
        self.optimizer = "adam"
        self.lr = 1e-3
        self.vocab_size = 15000
        self.vocab_file = None
        self.ner_tag_vocab_size = 32  # 32
        self.re_tag_vocab_size = 64  # 64
        self.lm_emb_dim = 4096  # 4096-albert,768-luke,1024-biobert,768-scibert
        self.lm_emb_path = './wv/albert.CoNLL04_with_heads.pkl'  # CoNLL04,ADE0-9
        self.pos_emb_dim = 0
        self.head_emb_dim = 768  # 768-albert,144-luke,384-biobert,144-scibert
        self.tag_form = "iob2"
        self.warm_steps = 1000
        self.grad_period = 1
        self.device = None
        self.IB = False
        self.beta = 0.99
        self.anneal_steps = 50000
        self.tab_reduce = "att"  # att, cat
        self.tab_model = "RNN"  # CNN, RNN
        self.cnn_skip: bool = False
        self.cnn_model = "hybrid"  # plain, dilation, hybrid
        self.cnn_kernel_size = 3  # 3, 5

    def __call__(self, **kargs):

        obj = copy.copy(self)
        for k, v in kargs.items():
            setattr(obj, k, v)
        return obj

    def copy(self, deep=False):
        if deep:
            return copy.deepcopy(self)
        else:
            return copy.copy(self)
