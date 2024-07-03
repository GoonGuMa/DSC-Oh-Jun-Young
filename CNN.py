import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
import shutil

# 경로 설정
data_dir = '/mnt/data/augmented_images_crop_only_fixed'
train_dir = '/mnt/data/train'
test_dir = '/mnt/data/test'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

classes = ['black_ice', 'snowy_road', 'puddle', 'normal_road']
for cls in classes:
	os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
	os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

file_paths = []
for cls in classes:
	cls_dir = os.path.join(data_dir, cls)
	file_paths += [(os.path.join(cls_dir, file), cls) for file in os.listdir(cls_dir) if file.endswith('.png')]

train_files, test_files = train_test_split(file_paths, test_size=0.2, random_state=42, stratify=[f[1] for f in file_paths])

for file_path, cls in train_files:
	shutil.copy(file_path, os.path.join(train_dir, cls))
for file_path, cls in test_files:
	shutil.copy(file_path, os.path.join(test_dir, cls))

transform = transforms.Compose([
	transforms.Resize((150, 150)),
	transforms.ToTensor(),
	transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

trainset = ImageFolder(root=train_dir, transform=transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

testset = ImageFolder(root=test_dir, transform=transform)
testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

class BasicBlock(nn.Module):
	def __init__(self, in_channels, out_channels, hidden_dim):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)
		self.relu = nn.ReLU()
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

	def forward(self, x):
		x = self.conv1(x)
		x = self.relu(x)
		x = self.conv2(x)
		x = self.relu(x)
		x = self.pool(x)
		return x

class CNN(nn.Module):
	def __init__(self, num_classes=4):
		super(CNN, self).__init__()
		self.block1 = BasicBlock(in_channels=3, out_channels=32, hidden_dim=16)
		self.block2 = BasicBlock(in_channels=32, out_channels=128, hidden_dim=64)
		self.block3 = BasicBlock(in_channels=128, out_channels=256, hidden_dim=128)
		self.fc1 = nn.Linear(in_features=256*18*18, out_features=2048)  # 수정된 입력 크기
		self.fc2 = nn.Linear(in_features=2048, out_features=256)
		self.fc3 = nn.Linear(in_features=256, out_features=num_classes)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.block1(x)
		x = self.block2(x)
		x = self.block3(x)
		x = torch.flatten(x, start_dim=1)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.relu(x)
		x = self.fc3(x)
		return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN(num_classes=len(classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(model, trainloader, criterion, optimizer, device, epochs=10):
	model.train()
	for epoch in range(epochs):
		running_loss = 0.0
		for data, labels in trainloader:
			data, labels = data.to(device), labels.to(device)
			optimizer.zero_grad()
			outputs = model(data)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
		print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(trainloader):.4f}')

def evaluate(model, testloader, device):
	model.eval()
	correct = 0
	total = 0
	with torch.no_grad():
		for data, labels in testloader:
			data, labels = data.to(device), labels.to(device)
			outputs = model(data)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	accuracy = 100 * correct / total
	print(f'Accuracy of the network on the test images: {accuracy:.2f}%')

train(model, trainloader, criterion, optimizer, device, epochs=10)
evaluate(model, testloader, device)
