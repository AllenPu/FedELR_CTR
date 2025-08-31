import random
import numpy as np
from PIL import Image
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import copy
class client_dataset(Dataset):
    def __init__(self, Subset , dataset_name,num_classes, client_idx, noise_type, noise_ratio,is_train = True, transform = transforms.Compose([transforms.ToTensor()]), sample_idx = None):
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.data, self.label = self.getdata(Subset)
        if type(self.label) != np.array:
            self.label = np.array(self.label)
        self.label = torch.from_numpy(self.label)
        self.label = self.label.to(torch.int64)
        self.noise_type = noise_type
        self.noise_ratio = noise_ratio
        self.client_idx = client_idx
        self.is_train = is_train
        self.transform = transform
        self.len = len(self.data)
        self.sample_idx = sample_idx


        if not self.is_train:
            self.noise_type = 'clean'
            self.noise_ratio = 0
        if self.noise_type not in ['clean', 'real', 'human']:
            self.label_noise = self.add_noise()
            self.label_noise = self.label_noise.to(torch.int64)
        if self.noise_type == 'human':
            self.label_noise = self.load_human_noise(dataset_name)
            self.label_noise = self.label_noise.to(torch.int64)
            self.noise_ratio = self.count_noise_ratio()
    def __getitem__(self, index):
        if self.dataset_name in ['clothing1M','tinyimagenet']:
            image = Image.open(self.data[index]).convert('RGB')
        elif self.dataset_name in ['cifar10','cifar100','cifar10NA','cifar10NW','cifar100N']:
            image = Image.fromarray(self.data[index])
        elif self.dataset_name in ['mnist','fmnist']:
            image = Image.fromarray(self.data[index].numpy(), mode='L')
        else:
            raise ValueError('Dataset not found')
        if self.transform is not None:
            image = self.transform(image)
        label = self.label[index]
        if self.noise_type not in ['clean', 'real']:
            label_noise = self.label_noise[index]
            return index, image, label, label_noise
        else:
            return index, image, label
    def __len__(self):
        return self.len

    def getdata(self,Subset):
        data = Subset.dataset.data[Subset.indices]
        if type(Subset.dataset.targets) != np.array:
            label = np.array(Subset.dataset.targets)[Subset.indices]
        else:
            label = Subset.dataset.targets[Subset.indices]
        return data, label

    def add_noise(self):
        if self.dataset_name in ['mnist']:
            return add_noise_mnist(self.label, self.num_classes, self.noise_type, self.noise_ratio)
        elif self.dataset_name in ['fmnist']:
            return add_noise_fmnist(self.label, self.num_classes, self.noise_type, self.noise_ratio)
        elif self.dataset_name in ['cifar10','cifar100']:
            return add_noise_cifar(self.label, self.num_classes, self.noise_type, self.noise_ratio)
        else:
            return add_noise_pairflip(self.label, self.num_classes, self.noise_type, self.noise_ratio)

    def load_human_noise(self, dataset_name):
        if dataset_name == 'cifar10NA':
            noise_file = torch.load('./Datasets/cifar10/CIFAR-10_human.pt')
            human_noise = noise_file['aggre_label']
        elif dataset_name == 'cifar10NW':
            noise_file = torch.load('./Datasets/cifar10/CIFAR-10_human.pt')
            human_noise = noise_file['worse_label']
        elif dataset_name == 'cifar100N':
            noise_file = torch.load('./Datasets/cifar100/CIFAR-100_human.pt')
            human_noise = noise_file['noisy_label']
        return torch.tensor(human_noise[self.sample_idx])

    def count_noise_ratio(self):
        label = np.array(self.label)
        label_noise = np.array(self.label_noise)
        return np.sum(label != label_noise) / len(label)

class centralized_dataset(Dataset):
    def __init__(self, Dataset , dataset_name,num_classes, noise_type, noise_ratio,is_train = True, transform = transforms.Compose([transforms.ToTensor()])):
        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.data = Dataset.data
        self.label = Dataset.targets
        if type(self.label) != np.array:
            self.label = np.array(self.label)
        self.label = torch.from_numpy(self.label)
        self.label = self.label.to(torch.int64)
        self.noise_type = noise_type
        self.noise_ratio = noise_ratio
        self.is_train = is_train
        if not self.is_train:
            self.noise_type = 'clean'
            self.noise_ratio = 0
        if self.noise_type not in ['clean', 'real']:
            self.label_noise = self.add_noise()
            if type(self.label_noise) == np.array:
                self.label_noise = torch.from_numpy(self.label_noise)
            self.label_noise = self.label_noise.to(torch.int64)
        self.transform = transform
        self.len = len(self.data)

    def __getitem__(self, index):
        if self.dataset_name in ['clothing1M','tinyimagenet']:
            image = Image.open(self.data[index]).convert('RGB')
        elif self.dataset_name in ['cifar10','cifar100','cifar10NA','cifar10NW','cifar100N']:
            image = Image.fromarray(self.data[index])
        elif self.dataset_name in ['mnist','fmnist']:
            image = Image.fromarray(self.data[index].numpy(), mode='L')
        else:
            raise ValueError('Dataset not found')
        if self.transform is not None:
            image = self.transform(image)
        label = self.label[index]
        if self.noise_type not in ['clean', 'real']:
            label_noise = self.label_noise[index]
            return index, image, label, label_noise
        else:
            return index, image, label
    def __len__(self):
        return self.len

    def add_noise(self):
        if self.dataset_name in ['mnist','fmnist']:
            return add_noise_mnist(self.label, self.num_classes, self.noise_type, self.noise_ratio)
        elif self.dataset_name in ['cifar10','cifar100']:
            return add_noise_cifar(self.label, self.num_classes, self.noise_type, self.noise_ratio)
        else:
            return add_noise_pairflip(self.label, self.num_classes, self.noise_type, self.noise_ratio)


class dataset_CTR(Dataset):
    def __init__(self, Dataset,noise_type, transform1,transform2,is_train=True):
        # self.data = np.transpose(data, (0, 2, 3, 1))
        # self.data = (self.data * 255).astype(np.uint8)
        self.dataset_name = Dataset.dataset_name
        self.data = Dataset.data
        if noise_type in ['clean', 'real']:
            self.label = Dataset.label
        else:
            self.label = Dataset.label_noise

        self.transform1 = transform1
        self.transform2 = transform2
        self.is_train = is_train
        self.label_clean = Dataset.label
        self.label_noise = Dataset.label_noise
        # self.sample_idx = Dataset.idx

    def __len__(self):
        return len(self.data)
    def get_noise_index(self):
        noise = self.label_clean == self.label_noise
        noise = noise.numpy()
        index = np.where(noise == False)[0]
        return index
    def __getitem__(self, index):

        if self.dataset_name in ['clothing1M','tinyimagenet']:
            x = Image.open(self.data[index]).convert('RGB')
        elif self.dataset_name in ['cifar10','cifar100','cifar10NA','cifar10NW','cifar100N']:
            x = Image.fromarray(self.data[index])
        elif self.dataset_name in ['mnist','fmnist']:
            x = Image.fromarray(self.data[index].numpy(), mode='L')
        else:
            raise ValueError('Dataset not found')


        y1 = self.label_clean[index]
        y2 = self.label_noise[index]
        if self.is_train:
            x1 = self.transform1(x)
            x2 = self.transform1(x)
            x3 = self.transform2(x)
            return index,[x1,x2,x3], [y1,y2]
        else:
            x = self.transform1(x)
            return x, y1





def add_noise_cifar(y, num_classes , noise_type, noise_rate):
    y_noised = copy.deepcopy(y)
    if noise_type == 'sym':
        noisy_num = int(noise_rate * len(y))
        noisy_idx = random.sample(range(len(y)), noisy_num)
        for i in noisy_idx:
            y_noised[i] = random.randint(0, num_classes - 1)
    elif noise_type == 'asym':
        if num_classes == 10:
            transition = {0: 0, 2: 0, 4: 7, 7: 7, 1: 1, 9: 1, 3: 5, 5: 3, 6: 6, 8: 8}
            noisy_num = int(noise_rate * len(y))
            noisy_idx = random.sample(range(len(y)), noisy_num)
            for i in noisy_idx:
                y_noised[i] = transition[int(y_noised[i])]
        elif num_classes == 100:
            noisy_num = int(noise_rate * len(y))
            noisy_idx = random.sample(range(len(y)), noisy_num)
            for i in noisy_idx:
                if (y_noised[i] + 1) % 5 == 0 :
                    y_noised[i] = int(y_noised[i]) - 4
                else:
                    y_noised[i] = (int(y_noised[i]) + 1) % 100
    return y_noised

def add_noise_mnist(y, num_classes , noise_type, noise_rate):
    y_noised = copy.deepcopy(y)
    if noise_type == 'sym':
        noisy_num = int(noise_rate * len(y))
        noisy_idx = random.sample(range(len(y)), noisy_num)
        for i in noisy_idx:
            y_noised[i] = random.randint(0, num_classes - 1)
    elif noise_type == 'asym':
        noisy_num = int(noise_rate * len(y))
        noisy_idx = random.sample(range(len(y)), noisy_num)
        transition = {0: 0, 1: 1, 2: 7, 3: 8, 4: 4, 5: 6, 6: 5, 7: 7, 8: 8, 9: 9}
        #  2 -> 7, 3 -> 8, 5 <->  6
        for i in noisy_idx:
            y_noised[i] = transition[int(y_noised[i])]
    return y_noised

def add_noise_fmnist(y, num_classes , noise_type, noise_rate):
    y_noised = copy.deepcopy(y)
    if noise_type == 'sym':
        noisy_num = int(noise_rate * len(y))
        noisy_idx = random.sample(range(len(y)), noisy_num)
        for i in noisy_idx:
            y_noised[i] = random.randint(0, num_classes - 1)
    elif noise_type == 'asym':
        noisy_num = int(noise_rate * len(y))
        noisy_idx = random.sample(range(len(y)), noisy_num)
        transition = {0: 6, 1: 1, 2: 4, 3: 8, 4: 4, 5: 7, 6: 6, 7: 5, 8: 8, 9: 9}
        # 0 -> 6, 5 <-> 7, 3 -> 8
        for i in noisy_idx:
            y_noised[i] = transition[int(y_noised[i])]
    return y_noised

def add_noise_pairflip(y, num_classes , noise_type, noise_rate):
    y_noised = copy.deepcopy(y)
    if noise_type == 'sym':
        noisy_num = int(noise_rate * len(y))
        noisy_idx = random.sample(range(len(y)), noisy_num)
        for i in noisy_idx:
            y_noised[i] = random.randint(0, num_classes - 1)
    elif noise_type == 'asym':
        noisy_num = int(noise_rate * len(y))
        noisy_idx = random.sample(range(len(y)), noisy_num)
        for i in noisy_idx:
            y_noised[i] = (y_noised[i] + 1) % num_classes
    return y_noised