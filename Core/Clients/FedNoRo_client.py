import copy
import time
import torch
import numpy as np
from Core.Clients.Client_base import Client
from Core.utils.data_utils import get_dataloder, get_dataset
from collections import Counter
from Core.utils.criteria import LogitAdjust, LA_KD

class FedNoRoClient(Client):

    def __init__(self, args, id, train_samples, test_samples):
        super().__init__(args, id, train_samples, test_samples)

        self.class_num_list = get_num_of_each_class(self.noise_type, get_dataset(self.train_path, self.id), self.num_classes)


    def train_LA(self):
        self.model.train()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, betas= (0.9,0.999), weight_decay=5e-4)
        trainloader = get_dataloder(self.train_path, self.id, self.batch_size, True)
        ce_criterion = LogitAdjust(cls_num_list= self.class_num_list)

        start_time = time.time()
        train_num = 0
        losses = 0

        for step in range(self.local_epochs):
            if self.noise_type in ['clean', 'real']:
                for i,(index, image, label) in enumerate(trainloader):
                    label = label.view(-1)
                    label = label.long()
                    image = image.to(self.device)
                    label = label.to(self.device)
                    output = self.model(image)
                    loss = ce_criterion(output, label)

                    train_num += label.shape[0]
                    losses += loss.item() * label.shape[0]

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            else:
                for i,(index, image, label, label_noise) in enumerate(trainloader):
                    label_noise = label_noise.view(-1)
                    label_noise = label_noise.long()
                    image = image.to(self.device)
                    label_noise = label_noise.to(self.device)

                    output = self.model(image)
                    loss = ce_criterion(output, label_noise)

                    train_num += label_noise.shape[0]
                    losses += loss.item() * label_noise.shape[0]

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
        self.train_loss.append(losses / train_num)
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        return losses, train_num

    def train_FedNoRo(self,weight_kd):
        student_net = copy.deepcopy(self.model).to(self.device)
        teacher_net = copy.deepcopy(self.model).to(self.device)
        student_net.train()
        teacher_net.eval()
        self.optimizer = torch.optim.Adam(
            student_net.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), weight_decay=5e-4)
        trainloader = get_dataloder(self.train_path, self.id, self.batch_size, True)
        criterion = LA_KD(cls_num_list=self.class_num_list)

        start_time = time.time()
        train_num = 0
        losses = 0

        for step in range(self.local_epochs):
            if self.noise_type in ['clean', 'real']:
                for i,(index, image, label) in enumerate(trainloader):
                    label = label.view(-1)
                    image = image.to(self.device)
                    label = label.to(self.device)
                    output = student_net(image)
                    with torch.no_grad():
                        teacher_output = teacher_net(image)
                        soft_label = torch.softmax(teacher_output/0.8, dim=1)
                    loss = criterion(output, label, soft_label, weight_kd)

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

                    output = student_net(image)
                    with torch.no_grad():
                        teacher_output = teacher_net(image)
                        soft_label = torch.softmax(teacher_output/0.8, dim=1)
                    loss = criterion(output, label_noise, soft_label, weight_kd)


                    train_num += label_noise.shape[0]
                    losses += loss.item() * label_noise.shape[0]

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
        self.train_loss.append(losses / train_num)
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        self.model = copy.deepcopy(student_net)
        del student_net
        del teacher_net
        return losses, train_num

    def get_sample_idx(self):
        train_dataset = get_dataloder(self.train_path, self.id, self.batch_size, True).dataset
        return train_dataset.sample_idx
    def get_noise_ratio(self):
        return get_dataloder(self.train_path, self.id, self.batch_size, True).dataset.noise_ratio
def get_num_of_each_class(noise_type,dataset,num_classes):
    if noise_type in ['clean', 'real']:
        label = dataset.label
    else:
        label = dataset.label_noise

    labels = label.tolist()
    class_sum = np.array([0] * num_classes)

    label_counts = Counter(labels)


    for label in label_counts:
        class_sum[label] = label_counts[label]

    return class_sum.tolist()



