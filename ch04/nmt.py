import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda:0'
                        if torch.cuda.is_available() else 'cpu')

## Data setting
# 英単語idをkey, 英単語をvalueとした辞書eid2wの作成
# 英単語をkey, 英単語idをvalueとした辞書ew2idの作成
id, eid2w, ew2id = 1, {}, {}
with open('data/train.en.vocab.4k', 'r', encoding='utf-8') as f:
    for w in f:
        w = w.strip()
        eid2w[id] = w
        ew2id[w] = id
        id += 1

ev = id

# for key, value in eid2w.items():
#     print(key)
#     print(value)
#     if key == 10:
#         break

# print(ev)

# train.evに書かれた文章を上で作成したdictと紐付け
# 文章をidで表したものに変換する
edata = []
with open('data/train.en', 'r', encoding='utf-8') as f:
    for sen in f:
        w1 = [ew2id['<s>']]
        for w in sen.strip().split():
            if w in ew2id:
                w1.append(ew2id[w])
            else:
                w1.append(ew2id['<unk>'])
            w1.append(ew2id['</s>'])
            edata.append(w1)

# print(w1[:5])
# print(edata[:5])

# 日単語idをkey, 日単語をvalueとした辞書jid2wの作成
# 日単語をkey, 日単語idをvalueとした辞書jw2idの作成
id, jid2w, jw2id = 1, {}, {}
with open('data/train.ja.vocab.4k', 'r', encoding='utf-8') as f:
    for w in f:
        w = w.strip()
        jid2w[id] = w
        jw2id[w] = id
        id += 1

jv = id

for key, value in jid2w.items():
    print(key)
    print(value)
    if key == 10:
        break


# print(jv)

# train.evに書かれた文章を上で作成したdictと紐付け
# 文章をidで表したものに変換する
jdata = []
with open('data/train.ja', 'r', encoding='utf-8') as f:
    for sen in f:
        w1 = [jw2id['<s>']]
        for w in sen.strip().split():
            if w in jw2id:
                w1.append(jw2id[w])
            else:
                w1.append(jw2id['<unk>'])
            w1.append(jw2id['</s>'])
            jdata.append(w1)


# Define model
class MyNMT(nn.Module):
    def __init__(self, jv, ev, k):
        super(MyNMT, self).__init__()
        self.jemb = nn.Embedding(jv, k)
        self.eemb = nn.Embedding(ev, k)
        self.lstm1 = nn.LSTM(k, k, num_layers=2, batch_first=True)
        self.lstm2 = nn.LSTM(k, k, num_layers=2, batch_first=True)
        self.W = nn.Linear(k, ev)
    def forward(self, jline, eline):
        x = self.jemb(jline)
        ox, (hnx, cnx) = self.lstm1(x)
        y = self.eemb(eline)
        oy, (hny, cny) = self.lstm1(y,(hnx, cnx))
        out = self.W(oy)
        return out


# model generate, optimizer and criterion setting
demb = 200
net = MyNMT(jv, ev, demb).to(device)
optimizer = optim.SGD(net.parameters(),lr=0.01)
criterion = nn.CrossEntropyLoss()


# Learn
net.train()
for epoch in range(200):
    loss1k = 0.0
    for i in range(len(jdata)):
        jinput = torch.LongTensor([jdata[i][1:]]).to(device)
        einput = torch.LongTensor([edata[i][:-1]]).to(device)
        out = net(jinput, einput)
        gans = torch.LongTensor([edata[i][1:]]).to(device)
        loss = criterion(out[0], gans[0])
        loss1k += loss.item()
        if (i % 100 == 0):
            print(epoch, i, loss1k)
            loss1k = 0.0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    outfile = "model/nmt-" +str(epoch) + ".model"
    torch.save(net.state_dict().outfile)

