from gensim.models import word2vec
from gensim.models import Word2Vec

# 2.3
print('----------2.3----------')

# オブジェクト作成
sens = word2vec.LineSentence('dataset/wakati.txt')

# 分散表現の構築
model = Word2Vec(sens)

# モデルの保存
model.save('bin/myw2v.bin')

# モデルの読み込み
# model = Word2Vec.load('bin/myw2v.bin')

# 2.4
print('----------2.4----------')

from gensim.models.keyedvectors import KeyVectors

model = KeyedVectors.load_wprd2vec_format('bin/entity_vector.model.bin', binary=True)

model = KeyedVectors.load('chive-1.1-mc5-aunit.kv')

# 2.5
print('----------2.5----------')

a = model['犬']
print(type(a))
print(a.shape)
print(a.dtype)

model.similarity('犬', '人')

import numpy as np

v1 = model['犬']
v2 = model['人']
nr1 = np.linalg.norm(v1, ord=2)
nr2 = np.linalg.norm(v2, ord=2)
print(np.dot(v1,v2) / (nr1 * nr2))

vocab = model.vocab
print(len(vocab))

v = vocab['犬']
print(type(v))
print(v.index)

wlist = model.index2wrod
print(wlist[793])

print(model.most_similar('犬', topn=5))
print(model.most_similar(positive=['日本','ニューヨーク'],
         negtive=['東京'], topn=3))

# 2.6
print('----------2.6----------')

from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

f = open('wakati.txt', 'r')
docs = [ TaggedDocument(word=data.split(),tag=[i])
        for i,data in enumerate(f)]
model = Doc2Vec(docs)
model.save('model/myd2v.model')
# model = Doc2Vec.load('model/myd2v.model')

print(model.docvecs[10])
print(model.docvecs.most_similar(10,topn=2))

newdoc = ['私','は','犬','が','好き']
print(model.infer_vector(newdoc))

# 2.7
print('----------2.7----------')

s1 = '私 は 犬 が 好き'
s2 = '日本 人 は みんな 動物 が 好き'
s3 = 'この 本 は 面白い'
print(model.wmdistance(s1,s2))
print(model.wmdistance(s1,s3))

# 2.8
print('----------2.8----------')

import fasttext

model = fasttext.load_model('bin/cc.ja.300.bin')
a = model['犬']
print(type(a))
print(a.shape)

