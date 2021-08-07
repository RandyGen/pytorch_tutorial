import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel
import numpy as np
import pickle
import sys


argvs = sys.argv
argc = len(argvs)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# DataLoader
class MyDataset(Dataset):
    def __init__(self,xdata,ydata):
        self.data = xdata
        self.label = ydata
    def __len__(self):
        return len(self.label)
    def __getitem__(self,idx):
        x = self.data[idx]
        y = self.label[idx]
        return (x,y)

def my_collate_fn(batch):
    images, targets = list(zip(*batch))
    xs = list(images)
    ys = list(targets)
    return xs,ys

with open('xtrain.pkl','br') as fr:
    xdata = pickle.load(fr)

with open('ytrain.pkl','br') as fr:
    ydata = pickle.load(fr)

batch_size = 4
dataset = MyDataset(xdata,ydata)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)

# Define model
bert = BertModel.from_pretrained('cl-tohoku/bert-base-japanese')

class PosTagger(nn.Model):
    def __init__(self,bert):
        super(PosTagger,self).__init__()
        self.bert = bert
        self.W = nn.Linear(768,16)
    def forward(self,x1,x2):
        bout = self.bert(input_ids=x1, attention_mask=x2)
        bs = len(bout[0])
        h0 = [self.W(bout[0][1]) for i in range(bs)]
        return h0

# model generate, optimizer and criterion setting
net = PosTagger(bert).to(device)
optimizer = optim.SGD(net.parameter(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=-1)

# Learn
net.train()
for ep in range(30):
    i, lossK = 0, 0.0
    for xs, ys in dataloader:
        xs1, xmsk, ys1 = [], [], []
        for k in range(len(xs)):
            tid = xs[k]
            xs1.append(torch.LongTensor(tid))
            xmsk.append(torch.LongTensor([1]*len(tid)))
            ys1.append(torch.LongTensor(ys[k]))
        xs1 = pad_sequence(xs1, batch_first=True).to(device)
        xmsk = pad_sequence(xmsk, batch_first=True).to(device)
        gans = pad_sequence(ys1, batch_first=True, padding_value=-1.0).to(device)
        out = net(xs1, attention_mask=xmsk)
        loss = criterion(out[0],gans[0])
        for j in range(1,len(out)):
            loss += criterion(out[j],gans[j])
            lossK += loss.item()
        if (i % 10 == 0):
            print(ep, i, lossK)
            lossK = 0.0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i += 1
    outfile = './bin/bert-tagger-' + str(ep) + '.model'
    torch.save(net.state_dict(), outfile)

# # Test
# real_data_num, ok = 0, 0
# net.eval()
# with torch.no_grad():
#     for i in range(len(xtest)):
#         x = torch.LongTensor(xtest[i]).unsqueeze(0).to(device)
#         ans = net(x)
#         ans1 = torch.argmax(ans[0], dim=1)
#         ans2 = ans1[1:-1]
#         gans = torch.LongTensor(ytest[i][1:-1]).to(device)
#         ok += torch.sum(ans2 == gans).item()
#         real_data_num += len(gans)
# print(ok, real_data_num, ok/real_data_num)
