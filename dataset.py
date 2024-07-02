import os
import shutil
from sklearn.model_selection import train_test_split

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

train_dir, test_dir