import pytorch
import torch.nn as nn
import tqdm
from torchvision.datasets.cifar import CIFAR10
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, RandomCrop, Normalize
from torch.utils.data import DataLoader
from torch.optim import Adam


class ResNet(nn.Module):
	def __init(self, num_classes=10):
		super(ResNet, self).__init__()
		self.b1 = BasicBlock(in_channels=3, out_channels=64)
		self.b2 = BasicBlock(in_channels=64, out_channels=128)
		self.b3 = BasicBlock(in_channels=128, out_channels=256)
		self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
		self.fc1 = nn.Linear(in_features=4096, out_features=2048)
		self.fc2 = nn.Linear(in_features=2048, out_features=512)
		self.fc3 = nn.Linear(in_features=512, out_features=num_classes)
		self.relu = nn.ReLU()
	def forward(self, x):
		x = self.b1(x)
		x = self.pool(x)
		x = self.b2(x)
		x = self.pool(x)
		x = self.b3(x)
		x = self.pool(x)
		x = torch.flatten(x, start_dim=1)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.relu(x)
		x = self.fc3(x)
		return x

transforms = Compose([
	RandomCrop((32, 32), padding=4),
	RandomHorizontalFlip(p=0.5),
	ToTensor(),
	Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
])

training_data = CIFAR10(root="./", train=True, download=True, transform=transforms)
test_data = CIFAR10(root="./", train=False, download=True, transform=transforms)

train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ResNet(num_classes=10)
model.to(device)

lr = 1e-4
optim = Adam(model.parameters(), lr=lr)

for epoch in range(30):
	iterator = tqdm.tqdm(train_loader)
	for data, label in iterator:
		optim.zero_grad()
		preds = model(data.to(device))
		loss = nn.CrossEntropyLoss()(preds, label.to(device))
		loss.backward()
		optim.step()
		iterator.set_description(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "ResNet.pth")

model.load_state_dict(torch.load("ResNet.pth", map_location=device))

num_corr = 0

with torch.no_grad():
	for data, label in test_loader:
		output = model(data.to(device))
		preds = output.data.max(1)[1]
		corr = preds.eq(label.to(device).data).sum().item()
		num_corr += corr

print(f"Accuracy: {num_corr / len(test_data)}")
