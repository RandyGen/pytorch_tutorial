import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import BertModel, BertConfig
import numpy as np
import pickle

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Data setting
with open('./data/src/xtrain.pkl', 'br') as fr:
    xtrain = pickle.load(fr)

with open('./data/src/ytrain.pkl', 'br') as fr:
    ytrain = pickle.load(fr)


# Define model
bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese')


class DocCls(nn.Model):
    def __init__(self,bert):
        super(DocCls, self).__init__()
        self.bert = bert
        self.cls = nn.Linear(768,9)
    def forward(self,x):
        bout = self.bert(x)
        bs = len(bout[0])
        h0 = [bout[0][i][0] for i in range(bs)]
        h0 = torch.stack(h0,dim=0)
        return self.cls(h0)


# setting optim
net = DocCls(bert).to(device)
optimizer = optim.SGD(net.parameters(),lr=0.001)
criterion = nn.CrossEntropyLoss()


# Learning
net.train()
for ep in range(30):
    lossK = 0.0
    for i in range(len(xtrain)):
        x = torch.LongTensor(xtrain[i]).unsqueeze(0).to(device)
        y = torch.LongTensor([ytrain[i]]).to(device)
        out = net(x)
        loss = criterion(out,y)
        lossK += loss.item()
        if (i % 50 == 0):
            print(ep, i, lossK)
            lossK = 0.0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    outfile = './bin/doccls-' + str(ep) + '.model'
    torch.save(net.state_dict(), outfile)


# Test
with open('./data/src/xtest.pkl', 'br') as fr:
    xtest = pickle.load(fr)

with open('./data/src/ytest.pkl', 'br') as fr:
    ytest = pickle.load(fr)

config = BertConfig.from_pretrained('cl-tohoku/bert-base-japanese')
bert = BertModel(config=config)

real_data_num, ok = 0, 0
net.eval()
with torch.no_grad():
    for i in range(len(xtest)):
        ans = net(x)
        ans1 = torch.argmax(ans, dim=1).item()
        if (ans1 == ytest[i]):
            ok += 1
        real_data_num += 1

print(ok, real_data_num, ok/real_data_num)

