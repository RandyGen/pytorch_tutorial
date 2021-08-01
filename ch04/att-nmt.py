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
class MyAttNMT(nn.Module):
    def __init__(self, jv, ev, k):
        super(MyAttNMT, self).__init__()
        self.jemb = nn.Embedding(jv, k)
        self.eemb = nn.Embedding(ev, k)
        self.lstm1 = nn.LSTM(k, k, num_layers=2, batch_first=True)
        self.lstm2 = nn.LSTM(k, k, num_layers=2, batch_first=True)
        self.Wc = nn.Linear(2*k, k)
        self.W = nn.Linear(k, ev)
    def forward(self, jline, eline):
        x = self.jemb(jline)
        ox, (hnx, cnx) = self.lstm1(x)
        y = self.eemb(eline)
        oy, (hny, cny) = self.lstm1(y,(hnx, cnx))
        ox1 = ox.permute(0,2,1)
        sim = torch.bmm(oy, ox1)
        bs, yws, xws = sim.shape
        sim2 = sim.reshape(bs*yws, xws)
        alpha = F.softmax(sim2, dim=1).reshape(bs, yws, xws)
        ct = torch.bmm(alpha, ox)
        oy1 = torch.cat([ct, oy], dim=2)
        oy2 = self.Wc(oy1)
        return self.W(oy2)


# model generate, optimizer and criterion setting
demb = 200
net = MyAttNMT(jv, ev, demb).to(device)
optimizer = optim.SGD(net.parameters(),lr=0.01)
criterion = nn.CrossEntropyLoss()


# test
esid = ew2id['<s>']
eeid = ew2id['</s>']
net.eval()
with torch.no_grad():
    for i in range(len(jdata)):
        jinput = torch.LongTensor([jdata[i][1:]]).to(device)
        x = net.jemb(jinput)
        ox, (hnx, cnx) = net.lstm1(x)
        wid = esid
        s1 = 0
        while True:
            wids = torch.LongTensor([[wid]]).to(device)
            y = net.eemb(wids)
            oy, (hnx, cnx) = net.lstm2(y, (hnx, cnx))
            ox1 = ox.permute(0, 2, 1)
            sim = torch.bmm(oy, ox1)
            bs, yws, xws = sim.shape
            sim2 = sim.reshape(bs*yws, xws)
            alpha =F.softmax(sim2,dim=1).reshape(bs, yws, xws)
            ct = torch.bmm(alpha, ox)
            oy1 = torch.cat([ct, oy1],dim=2)
            oy2 = net.Wc(oy1)
            oy3 = net.W(oy2)
            wid = torch.argmax(oy3[0]).item()
            if (wid == eeid):
                break
            print(eid2w[wid], ' ', end='')
            s1 += 1
            if (s1 == 30):
                break
        print()


# Learn
net.train()
for ep in range(20):
    i = 0
    for xs, ys in dataloader:
        xs1, ys1, ys2 = [], [], []
        for k in range(len(xs)):
            tid = xs[k]
            xs1.append(torch.LongTensor(tid[1:]))
            tid = ys[k]
            ys1.append(torch.LongTensor(tid[:-1]))
            ys2.append(torch.LongTensor(tid[1:]))
        jinput = pad_sequence(xs1, batch_first=True).to(device)
        einput = pad_sequence(ys1, batch_first=True).to(device)
        gans = pad_sequence(ys1, batch_first=True, padding_value=-1.0).to(device)
        out = net(jinput, einput)
        loss = criterion(out[0], gans[0])
        for h in range(1, len(gans)):
            loss += criterion(out[h], gans[h])
        print(ep, i, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i += 1
    outfile = "model/attnmt2-" +str(ep) + ".model"
    torch.save(net.state_dict().outfile)
