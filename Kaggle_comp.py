
#Author: Taban Akbarzadeh Shafarudi -- <Taban.Akbarzadeh-Shafarudi@polymtl.ca>
#Classifying the hand drawn images, Kaggle Competition 

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from tqdm import tqdm
from scipy import ndimage


class KaggleDataset(Dataset):

    def __init__(self, images_file, labels_file=None, transform=None):
        images_array = np.load(images_file, encoding='latin1')
        self.images = [image.reshape(100, 100) for image in images_array[:, 1]]
        self.labels_file = labels_file
        if self.labels_file:
            self.labels_list = pd.read_csv(labels_file)
            self.labels = self.labels_list['Category'].values.tolist()
            self.classes = sorted(set(self.labels_list['Category'].values.tolist()))
            self.class2idx = {cl: idx for idx, cl in enumerate(self.classes)}
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = Image.fromarray(image).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.labels_file:
            label = self.labels[idx]
            label = self.class2idx[label]
            # label = self.labels.iloc[idx, 2]  # 2 is the 'Code'
            return image, label
        else:
            return image


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tqdm.write('CUDA is not available!' if not torch.cuda.is_available() else 'CUDA is available!')

num_classes = 31

validation_split = 0.1
shuffle_dataset = True
random_seed = 192

batch_size = 32
num_workers = 2

momentum = 0.9
weight_decay = 5e-3

epochs = 45
learning_rate = 0.0001
step_size = 15
gamma = 0.1

train_dataset = KaggleDataset(images_file='train_images.npy', labels_file='train_labels.csv',
                              transform=transforms.Compose([
                                  transforms.Resize(299),
                                  transforms.RandomRotation(10),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor()
                              ]))

test_dataset = KaggleDataset(images_file='test_images.npy', transform=transforms.Compose([
    transforms.Resize(299),
    transforms.ToTensor()
]))

dataset_size = len(train_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
valid_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

model = models.inception_v3(pretrained=True)
# model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_in = model.fc.in_features
model.fc = nn.Linear(num_in, num_classes)

model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


def train(_epoch):
    scheduler.step()
    model.train()
    num_images = 0.
    running_loss = 0.
    running_accuracy = 0.
    monitor = tqdm(train_loader, desc='Training')
    for i, (train_images, train_labels) in enumerate(monitor):
        train_images, train_labels = train_images.to(device, dtype=torch.float), train_labels.to(device,
                                                                                                 dtype=torch.long)
        outputs, aux = model(train_images)
        _, preds = torch.max(outputs.data, 1)
        oloss = criterion(outputs, train_labels)
        aloss = criterion(aux, train_labels)
        loss = 0.5 * oloss + 0.5 * aloss

        num_images += train_images.size(0)
        running_loss += loss.item() * train_images.size(0)
        running_accuracy += torch.sum(preds == train_labels).item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        monitor.set_postfix(loss=running_loss / num_images, accuracy=running_accuracy / num_images, epoch=_epoch)

    epoch_loss = running_loss / num_images
    epoch_accuracy = running_accuracy / num_images

    return epoch_loss, epoch_accuracy


def valid(_epoch):
    model.eval()
    with torch.no_grad():
        num_images = 0.
        running_loss = 0.
        running_accuracy = 0.
        monitor = tqdm(valid_loader, desc='Validation')
        for i, (test_images, test_labels) in enumerate(monitor):
            test_images, test_labels = test_images.to(device, dtype=torch.float), test_labels.to(device,
                                                                                                 dtype=torch.long)

            outputs = model(test_images)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, test_labels)

            num_images += test_images.size(0)
            running_loss += loss.item() * test_images.size(0)
            running_accuracy += torch.sum(preds == test_labels).item()

            monitor.set_postfix(loss=running_loss / num_images, corrects=running_accuracy / num_images, epoch=_epoch)

        epoch_loss = running_loss / num_images
        epoch_accuracy = running_accuracy / num_images

        return epoch_loss, epoch_accuracy


def test():
    model.eval()
    with torch.no_grad():
        all_preds = list()
        monitor = tqdm(test_loader, desc='Testing')
        for i, (test_images) in enumerate(monitor):
            test_images = test_images.to(device, dtype=torch.float)

            outputs = model(test_images)
            preds = outputs.max(1, keepdim=True)[1].tolist()

            all_preds += preds

        return all_preds

bestAcc = 0.0

for epoch in range(epochs):
    valid_loss, valid_accuracy = valid(epoch)
    if valid_accuracy > bestAcc:
        bestAcc = valid_accuracy
    elif epoch > 15:
        break
    tqdm.write('EPOCH[{}] --> Valid Loss: {}, Valid Accuracy: {}'.format(epoch, valid_loss, valid_accuracy))
    train_loss, train_accuracy = train(epoch)
    tqdm.write('EPOCH[{}] --> Train Loss: {}, Train Accuracy: {}'.format(epoch, train_loss, train_accuracy))

all_test_preds = test()

labels_file = pd.read_csv('train_labels.csv')
_labels = labels_file['Category'].values.tolist()
_classes = sorted(set(labels_file['Category'].values.tolist()))
_idx2class = {idx: cl for idx, cl in enumerate(_classes)}

idxs = list()
for idx in np.asarray(all_test_preds).reshape(-1):
    idxs.append(_idx2class[idx])

s = pd.Series(idxs, index=range(len(all_test_preds)))

df = s.to_frame().reset_index()
df.columns = ['Id', 'Category']

df.to_csv('out.csv', index=False)
