from nltk.translate.bleu_score import corpus_bleu


myans = [
    ['It', 'is', 'a', 'cat', 'at', 'room'],
    ['I', 'like', 'my', 'dog']
    ]
gold =[
    [['It', 'is', 'a', 'cat', 'inside', 'the', 'room']],
    [['I', 'love', 'a', 'dog']]
]
score = corpus_bleu(gold, myans)
print(score)
