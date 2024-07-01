import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

class Netflix(Dataset):
    	def __init__(self):
		self.csv = pd.read_csv("C:/Users/zlalf/Downloads/CH06.csv")
		self.data = self.csv.iloc[:, 1:4].values
		self.data = self.data / np.max(self.data)
		self.label = self.csv["Close"].values
		self.label = self.label / np.max(self.label)
	
	def __len__(self):
		return len(self.data) - 30

	def __getitem__(self, i):
		data = self.data[i:i+30]
		label = self.label[i+30]
		return data, label
dataset = Netflix()
loader = DataLoader(dataset, batch_size=10, shuffle=True)

class RNN(nn.Module):
	def __init__(self):
		super(RNN, self).__init__()
		self.rnn = nn.RNN(input_size=3, hidden_size=8, num_layers=5, batch_first=True)
		self.fc1 = nn.Linear(in_features=240, out_features=64)
		self.fc2 = nn.Linear(in_features=64, out_features=1)
		self.relu = nn.ReLU()

	def forward(self, x, h0):
		x, hn = self.rnn(x, h0)
		x = torch.reshape(x, (x.shape[0], -1))
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.relu(x)
		x = torch.flatten(x, 1)
		return x

model = RNN()
optim = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(200):
	iterator = tqdm.tqdm(loader)
	for data, label in iterator:
		optim.zero_grad()
		h0 = torch.zeros(5, data.shape[0], 8).to(device)
		data = data.type(torch.FloatTensor).to(device)
		label = label.type(torch.FloatTensor).to(device)
		pred = model(data, h0)
		loss = nn.MSELoss()(pred, label)
		loss.backward()
		optim.step()
		iterator.set_description(f"epoch {epoch} loss: {loss.item()}")

import matplotlib.pyplot as plt
loader = DataLoader(dataset, batch_size=1)

preds = []
total_loss = 0

with torch.no_grad():
	model.load_state_dict(torch.load("rnn.pth", map_location=device))
    
	for data, label in loader:
		h0 = torch.zeros(5, data.shape[0], 8).to(device)
		pred = model(data,type.(torch.FloatTensor).to(device), h0)
		preds.append(pred.item())
		loss = nn.MSELoss()(pred, label.type(torch.FloatTensor).to(device))
		total_loss += loss/len(loader)
total_loss.item()

plt.plot(preds, label="prediction")
plt.plot(dataset.label[30:], label="actual")
plt.legend()
plt.show()