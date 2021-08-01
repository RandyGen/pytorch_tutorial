import sys
argvs = sys.argv

gold = []
with open('data/test.en', 'r', encoding='utf-8') as f:
    for sen in f:
        w = sen.strip().split()
        gold.append([w])

myans = []
# nmt-X.modelファイルを指定してBLUEスコアをを算出
with open(argvs[1], 'r') as f:
    for sen in f:
        w = sen.strip().split()
        myans.append([w])

from nltk.translate.bleu_score import corpus_bleu
score = corpus_bleu(gold, myans)
print(100*score)

