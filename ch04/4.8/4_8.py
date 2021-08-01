import sentencepiece as spm


sp = spm.SentencePieceProcessor()
spm.SentencePieceTrainer.Train("--unput=mai08.txt --model_prefix=mai08model --vocab_size=8000")

sp.Load("mai08model.model")
s = "私は犬が大大大好き"
print(sp.EncodeAsPieces(s))
print(sp.EncodeAsIds(s))
