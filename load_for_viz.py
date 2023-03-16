#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[2]:

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
print(f"my pid: {os.getpid()}")
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from utils import *
from data import *
from models import *
from models.config import TBTOConfig

import glob

torch.cuda.set_device(0)
# torch.autograd.set_detect_anomaly(True)
torch.set_num_threads(4)




import argparse

def none_or_str(value):
    if value == 'None':
        return None
    return value

def none_or_int(value):
    if value == 'None':
        return None
    return int(value)


# In[5]:

args = TBTOConfig()
args.parse_args()

if args.device is not None and args.device != 'cpu':
    torch.cuda.set_device(args.device)
elif args.device is None:
    if torch.cuda.is_available():
        gpu_idx, gpu_mem = set_max_available_gpu()
        args.device = f"cuda:{gpu_idx}"
    else:
        args.device = "cpu"


# In[6]:


# config = Config(**args.__dict__)
config = args
ModelClass = eval(args.model_class)
model = ModelClass(config)


# load model
model = model.load('./save_models/conll04_model/conll04')
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

total = sum([param.nelement() for param in model.parameters()])
print ("Number of parameter: %.2fM" % (total/1e6))

# for name in glob.glob('save_inputs/CoNLL04_TEST/*.pt'):
#     data_ori = torch.load(name)
#     input_data = {
#         'tokens': data_ori['tokens'],
#         'ner_tags': data_ori['ner_tags'],
#         're_tags': data_ori['re_tags'],
#         'relations': data_ori['relations'],
#         'entities': data_ori['entities'],
#         '_tokens': data_ori['_tokens'],
#         '_ner_tags': data_ori['_ner_tags'],
#         '_re_tags': data_ori['_re_tags']
#    }
#
#     attention = model(input_data)
#     name_re = name.split('/')[-1]
#     path = os.path.join("save_inputs/Co_ATT_TEST", "att" + f"_{name_re}.pt")
#     torch.save(attention, path)











"""
sentence_a = ' '.join(input_data['tokens'][0])
# sentence_b = "The cat lay on the rug"
inputs = tokenizer.encode_plus(sentence_a,sentence_a, return_tensors='pt', add_special_tokens=True)
token_type_ids = inputs['token_type_ids']
input_ids = inputs['input_ids']


attention = model(input_data)
input_id_list = input_ids[0].tolist()  # Batch index 0
tokens = tokenizer.convert_ids_to_tokens(input_id_list)

# model_view(attention, tokens)
"""


print()

#

# In[7]:




