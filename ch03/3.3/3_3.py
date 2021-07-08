labels = {
	'名詞': 0, '助詞': 1, '形容詞': 2, '助動詞': 3, '補助記号': 4, '動詞': 5,
	'代名詞': 6, '接尾辞': 7, '副詞': 8, '形状詞': 9, '記号': 10,
	'連体詞': 11, '接頭辞': 12, '接続詞': 13, '感動詞': 14, '空白': 15
}

import pickle

# FileNotFoundError: No such file or directory 'dic.pkl'
with open('dic.pkl', 'br') as f:
	dic = pickle.load(f)
print("dic['犬']: {}".format(dic['犬']))

with open('xtrain.pkl','br') as f:
	xdata = pickle.load(f)
with open('ytrain.pkl','br') as f:
	ydata = pickle.load(f)
print("xdata: {}".format(xdata))
print("ydata: {}".format(ydata))

