import time
import torch
import numpy as np
from Core.Clients.Client_base import Client
from Core.utils.data_utils import get_dataloder,get_global_testloader
from torchvision import transforms
from Datasets.utils.custom_dataset import dataset_CTR
import os
class FedTESClient(Client):

    def __init__(self, args, id, train_samples, test_samples):
        super().__init__(args, id, train_samples, test_samples)

    def get_data_transform(self, args):
        if args.dataset in ['cifar10','cifar10NW','cifar10NA']:
            train_transforms = transforms.Compose([
                transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

            train_cls_transformcon = transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
        elif args.dataset in ['cifar100','cifar100N']:
            train_transforms = transforms.Compose([
                transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.267, 0.256, 0.276))
            ])
            train_cls_transformcon = transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5071, 0.4865, 0.4409], [0.267, 0.256, 0.276])])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5071, 0.4865, 0.4409], [0.267, 0.256, 0.276])])
        return train_transforms, train_cls_transformcon, test_transform
    def get_ctr_dataloader(self,path,idx,batch_size,isTrain):
        dataset_path = os.path.join(path, str(idx) + '.pkl')
        dataset = torch.load(dataset_path)

        dataset = dataset_CTR(dataset, self.noise_type, self.train_transform, self.train_cls_transform, isTrain)

        if isTrain:
            return torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, drop_last=True, shuffle=True)
        else:
            return torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, drop_last=False, shuffle=False)


    def train(self):
        if self.args.criterion == 'ctr_elr':
            trainloader = self.get_ctr_dataloader(self.train_path, self.id, self.batch_size, True)
        else:
            trainloader = get_dataloder(self.train_path, self.id, self.batch_size, True)
        self.model.train()
        start_time = time.time()
        train_num = 0
        losses = 0

        if self.args.criterion == 'ctr_elr':

            for step in range(self.local_epochs):
                for i, (index, x, y) in enumerate(trainloader):

                    x[0] = x[0].to(self.device)
                    x[1] = x[1].to(self.device)
                    x[2] = x[2].to(self.device)
                    if len(y) == 1:
                        y = y[0].to(self.device)
                    else:
                        y = y[1].to(self.device)

                    p1, z2, outputs = self.model(x[0], x[1], x[2])
                    loss = self.criterion(index, outputs, y, p1, z2)
                    train_num += y.shape[0]
                    losses += loss.item() * y.shape[0]

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
        else:

            for step in range(self.local_epochs):
                if self.noise_type in ['clean', 'real']:
                    for i,(index, image, label) in enumerate(trainloader):
                        label = label.view(-1)
                        image = image.to(self.device)
                        label = label.to(self.device)
                        output = self.model(image)
                        if self.args.criterion == 'elr':
                            loss = self.criterion(index, output, label)
                        else:
                            loss = self.criterion(output, label)

                        train_num += label.shape[0]
                        losses += loss.item() * label.shape[0]

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                else:
                    for i,(index, image, label, label_noise) in enumerate(trainloader):
                        label_noise = label_noise.view(-1)


                        image = image.to(self.device)
                        label_noise = label_noise.to(self.device)

                        output = self.model(image)
                        if self.args.criterion == 'elr':
                            loss = self.criterion(index, output, label_noise)
                        else:
                            loss = self.criterion(output, label_noise)

                        train_num += label_noise.shape[0]
                        losses += loss.item() * label_noise.shape[0]

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()


        self.train_loss.append(losses / train_num)
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        return losses, train_num


    def eval(self):
        testloader = get_global_testloader(self.dataset_path, self.batch_size)
        self.model.eval()
        test_acc = 0
        test_num = 0
        with torch.no_grad():
            for i,(index, image, label) in enumerate(testloader):
                label = label.view(-1)
                image = image.to(self.device)
                label = label.to(self.device)
                output = self.model(image)
                test_acc += (torch.sum(torch.argmax(output, dim=1) == label)).item()
                test_num += label.shape[0]
        self.test_acc.append(test_acc / test_num)
        return test_acc , test_num

    def memo(self):
        trainloader = get_dataloder(self.train_path, self.id, self.batch_size, True)
        self.model.eval()
        noise_num = 0

        memo_num = 0
        for i, (index, image, label, label_noise) in enumerate(trainloader):
            label_noise = label_noise.view(-1)

            noise_index = label != label_noise
            noise_index = noise_index.numpy()
            noise_index = np.where(noise_index == True)[0]

            image = image.to(self.device)
            label_noise = label_noise.to(self.device)

            output = self.model(image)


            pred = torch.argmax(output, dim=1)
            noise_num += noise_index.shape[0]
            memo = pred[noise_index] == label_noise[noise_index]

            memo_num += np.where(memo.cpu().numpy() == True)[0].shape[0]

        # print('client:',self.id,'memo_num:',memo_num,'noise_num:',noise_num,'memo_rate:',memo_num/noise_num)

        return memo_num, noise_num
