class MyLSTM(nn.Module):
	def __init__(self, vocsize, posn. hdim):
		auper(MyLSTM, self).__init__()
		self.embd = nn.Embedding(vocsize, hdim)
		self.lstm = nn.LSTM(hdim, hdim, batch_first=True)
		self.ln = nn.Linear(hdim, posn)
	def forward(self, x):
		ex = self.embd(x)
		lo = self.lstm(ex)
		out = self.ln(lo)
		return out
		
		
if __name__ == '__main__':
	net = MyLSTM(len(dic), len(labels), 100).to(device)
	optimizer = optim.SGD(net,parameters(),lr=0.01)
	criterion = nn.CrossEntropyLoss()
	
	for ep in range(10):
		loss1K = 0.0
		for i in range(len(xtrain)):
			x = [xdata[i]]
			x = torch.LongTensor(ydata[i]).to(device)
			loss = criterion(output[0],y)
			if (i % 1000 == 0):
				print(i,loss1K)
				loss1K = loss.items()
			else:
			loss1K += loss.item()
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	outfile = 'lstmO-' + str(ep) + '.model'
	tourch.save(net.state_dixt().outfile)
	
