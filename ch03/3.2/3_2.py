from janome.tokenizer import Tokenizer

tkz = Tokenizer()
s = '私は犬が好き。'
ws = [w for w in tkz.tokenize(s,wakati=True)]
print('ws: {}'.format(ws))

from gensim.models.keyedvectors import KeyedVectors

w2v = KeyedVectors.load_word2vec_format(
    '../../part2/bin/entity_vector.model.bin',binary=True
)

import numpy as np
import torch

xn = torch.tensor([w2v[w] for w in ws])
print('xn.shape: {}'.format(xn.shape))

xn = xn.unsqueeze(0)
print('xn.shape: {}'.format(xn.shape))

import torch.nn as nn

lstm = nn.LSTM(200,200,batch_first=True)
h0 = torch.rand(1,1,200)
c0 = torch.rand(1,1,200)
yn, (hn, cn) = lstm(xn, (h0, c0))
print('yn.shape: {}'.format(yn.shape))
print('hn.shape: {}'.format(hn.shape))
print('cn.shape: {}'.format(cn.shape))

