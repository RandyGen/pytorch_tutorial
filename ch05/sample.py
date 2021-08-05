# 5.3 ====================================
from transformers import BertConfig, BertForPreTraining
bertcfg = BertConfig.from_pretrained('bert_config.json')
net = BertForPreTraining(bertcfg)
net.load_tf_weights(bertcfg, 'model.ckpt-3900000')
import torch
torch.save(net.bert.state_dict(), 'laboro.bin')

# 5.4 ====================================
dic = {}
with open('data/vocab.txt', 'r', encoding='utf-8') as f:
    vocab = f.read()
    for id, word in enumerate(vocab.split('\n')):
        dic[word] = id

text = '[CLS] 私 は 犬 が 好き 。 [SEP]'
x = [dic[w] for w in text.split()]
print(x)

import torch
x = torch.LongTensor(x).unsqueeze(0)
print(x)

from transformers import BertModel
model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese')
a = model(x)

print(a[0].shape)
print(a[0][0][3])

# 5.5 ====================================
from transformers import BertModel, BertConfig
config = BertConfig.from_pretrained('cl-tohoku/bert-base-japanese')
config.output_hidden_states = True
model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese', config=config)
x = [2, 1325, 9, 2928, 14, 3596, 8, 3] # 私は犬が好き

import torch
x = torch.LongTensor(x).unsqueeze(0)
a = model(x)
print(len(a))
print(len(a[2]))
print(a[2][12].shape)
print(a[2][-1].shape)
print(torch.sum(a[0][0] == a[2][-1]))

# 5.6 ====================================
from transformers import BertJapaneseTokenizer
tknz = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
print(tknz.tokenize('私は犬が好き'))
print(tknz.encode('私は犬が好き'))
print(tknz.encode('私は犬が好き', add_special_tokens=False))

import MeCab
m = MeCab.Tagger('-Owakati -r /etc/mecabrc')
print(m.parse('私の名前は浩幸です。'))
print(tknz.tokenize('私の名前は浩幸です。'))
print(tknz.encode('私の名前は浩幸です。'))

# 5.7 ====================================
ids = tknz.encode('私 は [MASK] が 好き 。')
print(ids)

mskpos = ids.index(tknz.mask_token_id)
print(mskpos)

from transformers import BertForMaskedLM
model = BertForMaskedLM.from_pretrained('cl-tohoku/bert-base-japanese')
model.save_pretrained('bin/mybert.bin')
model = BertForMaskedLM.from_pretrained('bin/mybert.bin')
import torch
x = torch.LongTensor(ids).unsqueeze(0)
a = model(x)
print(a[0].shape)
print(tknz.vocab_size)

k = 100
print(a[0][0][mskpos][k])

b = torch.topk(a[0][0][mskpos], k=5)
print(b[0])
print(b[1])

ans = tknz.convert_ids_to_tokens(b[1])
print(ans)

# 5.8 ====================================
from transformers import BertModel, BertConfig
config = BertConfig.from_pretrained('bin/mybert.bin/config.json')
model = BertModel.from_pretrained('bin/mybert.bin/pytorch_model.bin', config=config)

config = BertConfig.from_pretrained('bin/mybert.bin/config.json')
model = BertForMaskedLM.from_pretrained('bin/mybert.bin/pytorch_model.bin', config=config)

from transformers import BertJapaneseTokenizer
tknz = BertJapaneseTokenizer(vocab_file='data/vocab.txt',do_lower_case=False,do_casic_tokenize=False)

from transformers.models.bert_japanese import tokenizezation_bert_japanese
tknz.word_tokenizer = tokenizezation_bert_japanese.MecabTokenizer()

import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load('webcorpus.model')
x = sp.EncodeAsIds('私は犬が大好き')
print(x)
x = [4] + x + [5]
print(x)

import torch
x = torch.LongTensor(x).unsqeeze(0)
from transformers import BertModel, BertConfig

# OSError: Can't load config for 'bert_config.json'
# - 'bert_config.json' is a correct model identifier listed on 'https://huggingface.co/models'
# - or 'bert_config.json' is the correct path to a directory containing a config.json file

config = BertConfig.from_json_file('bert_config.json')
model = BertModel.from_pretrained('laboro.bin', config=config)
a = model(x)
print(a[0].shape)

