import time
from Core.Clients.Client_base import Client
from Core.utils.data_utils import get_dataloder
import torch
from Core.utils.criteria import get_criterion

class FedAvgELRClient(Client):

    def __init__(self, args, id, train_samples, test_samples):
        super().__init__(args, id, train_samples, test_samples)
        self.criterion = get_criterion('elr', self.num_classes, train_samples, args, True)

    def warm_up_train(self):
        if self.warm_up_steps != 0:
            trainloader = get_dataloder(self.train_path, self.id, self.batch_size, True)
            self.model.train()
            for step in range(self.warm_up_step):
                if self.noise_type in ['clean', 'real']:
                    for i, (index, image, label) in enumerate(trainloader):
                        label = label.view(-1)
                        image = image.to(self.device)
                        label = label.to(self.device)
                        output = self.model(image)
                        loss = self.criterion(index,output, label)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                else:
                    for i, (index, image, label, label_noise) in enumerate(trainloader):
                        label_noise = label_noise.view(-1)
                        image = image.to(self.device)
                        label_noise = label_noise.to(self.device)
                        output = self.model(image)
                        loss = self.criterion(index,output, label_noise)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

    def train(self):
        trainloader = get_dataloder(self.train_path, self.id, self.batch_size, True)
        self.model.train()
        start_time = time.time()
        train_num = 0
        losses = 0

        for step in range(self.local_epochs):
            if self.noise_type in ['clean', 'real']:
                for i,(index, image, label) in enumerate(trainloader):
                    label = label.view(-1)
                    image = image.to(self.device)
                    label = label.to(self.device)
                    output = self.model(image)
                    loss = self.criterion(index,output, label)

                    train_num += label.shape[0]
                    losses += loss.item() * label.shape[0]

                    self.optimizer.zero_grad()
                    # self.model.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            else:
                for i,(index, image, label, label_noise) in enumerate(trainloader):
                    label_noise = label_noise.view(-1)
                    image = image.to(self.device)
                    label_noise = label_noise.to(self.device)
                    output = self.model(image)
                    loss = self.criterion(index,output,label_noise)

                    train_num += label_noise.shape[0]
                    losses += loss.item() * label_noise.shape[0]

                    self.optimizer.zero_grad()
                    # self.model.zero_grad()
                    loss.backward()
                    self.optimizer.step()
        self.train_loss.append(losses / train_num)
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        return losses, train_num