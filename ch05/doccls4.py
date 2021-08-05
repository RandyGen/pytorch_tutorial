import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import optimizer
from torch.utils.data import Dataset, DataLoader, dataloader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertForSequenceClassification, BertConfig
import numpy as np
import pickle


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
        return x, y

def my_collate_fn(batch):
    images, targets = list(zip(*batch))
    xs = list(images)
    ys = list(targets)
    return xs, ys


# Data setting
with open('./data/src/xtrain.pkl', 'br') as fr:
    xdata = pickle.load(fr)

with open('./data/src/ytrain.pkl', 'br') as fr:
    ydata = pickle.load(fr)

batch_size = 3
dataset = MyDataset(xdata, ydata)
dataloder = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=my_collate_fn)


# Define model
net = BertForSequenceClassification.from_pretrained(
    'cl-tohoku/bert-base-japanese',
    num_labels=9).to(device)
optimizer = optim.SGD(net.parameters(),lr=0.001)


# Learning
net.train()
for ep in range(30):
    i, lossK = 0, 0.0
    for xs, ys in dataloader:
        xs1, xmsk = [], []
        for k in range(len(xs)):
            tid = xs[k]
            xs1.append(torch.LongTensor(tid))
            xmsk.append(torch.LongTensor([1]*len(tid)))
        xs1 = pad_sequence(xs1, batch_first=True).to(device)
        xmsk = pad_sequence(xmsk, batch_first=True).to(device)
        ys = torch.LongTensor(ys).to(device)
        out = net(xs1, attention_mask=xmsk,label=ys)
        loss = out.loss
        lossK += loss.item()
        if (i % 10 == 0):
            print(ep, i, lossK)
            lossK = 0.0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i += 1
    outfile = './bin/doccls4-' + str(ep) + '.model'
    torch.save(net.state_dict(), outfile)


# # Test
# config = BertConfig.from_pretrained('config.json')
# config.num_labels = 9
# net = BertForSequenceClassification(config=config).to(device)
# net.load_state_dict(torch.load(argvs[1]))

# out = net(x)
# ans = out.logits
# ans1 = torch.argmax(ans,dim=1).item()
