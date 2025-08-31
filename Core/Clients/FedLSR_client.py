from Core.Clients.Client_base import Client
import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
import copy
import time
import random
import matplotlib.pyplot as plt
from Core.utils.data_utils import get_dataloder
from Core.utils.criteria import get_criterion
from Core.utils.optimizers import get_optimizer
from torchvision import transforms
import torchvision
class FedLSRClient(Client):

    def __init__(self, args, id, train_samples, test_samples):
        super().__init__(args, id, train_samples, test_samples)
        if 'mnist' in args.dataset:
            self.tt_transform = transforms.Compose([
                transforms.RandomRotation(30)])
        elif 'cifar10' in args.dataset:
            s = 1
            color_jitter = torchvision.transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
            self.tt_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2)])
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,momentum=0.9,weight_decay = 0.0001)
        self.sm = torch.nn.Softmax(dim=1)
        self.lsm = torch.nn.LogSoftmax(dim=1)
        self.gamma = 0.4
        self.lambda_e = 0.6
        self.T_d_reverse = 3
        self.t_w = args.global_rounds * 0.2
        self.loss = nn.CrossEntropyLoss()

    def js(self, p_output, q_output):
        """
        :param predict: last round logits
        :param target: this round logits
        :return: loss
        """
        KLDivLoss = nn.KLDivLoss(reduction='mean')
        log_mean_output = ((p_output + q_output) / 2).log()
        return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)) / 2
    def train(self,current_round):
        trainloader = get_dataloder(self.train_path, self.id, self.batch_size, True)
        self.model.train()
        start_time = time.time()
        train_num = 0
        losses = 0

        for step in range(self.local_epochs):
            if self.noise_type in ['clean', 'real']:
                for i,(index, image, label) in enumerate(trainloader):
                    label = label.view(-1)
                    image_aug = self.tt_transform(image)
                    image_aug = image_aug.to(self.device)
                    image = image.to(self.device)
                    label = label.to(self.device)

                    output1 = self.model(image)
                    output2 = self.model(image_aug)

                    mix_1 = np.random.beta(1, 1)
                    mix_2 = 1 - mix_1
                    logits1, logits2 = torch.softmax(output1 * 3, dim=1), torch.softmax(output2 * 3, dim=1)
                    logits1, logits2 = torch.clamp(logits1, min=1e-6, max=1.0), torch.clamp(logits2, min=1e-6, max=1.0)
                    L_e = - (torch.mean(torch.sum(self.sm(logits1) * self.lsm(logits1), dim=1)) + torch.mean(
                        torch.sum(self.sm(logits1) * self.lsm(logits1), dim=1))) * 0.5
                    p = torch.softmax(output1, dim=1) * mix_1 + torch.softmax(output2, dim=1) * mix_2
                    pt = p ** (2)
                    pred_mix = pt / pt.sum(dim=1, keepdim=True)
                    betaa = self.gamma
                    if (current_round < self.t_w):
                        betaa = self.gamma * current_round / self.t_w
                    loss = self.loss(pred_mix, label) + self.js(logits1, logits2) * betaa + L_e * self.lambda_e


                    train_num += label.shape[0]
                    losses += loss.item() * label.shape[0]

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            else:
                for i,(index, image, label, label_noise) in enumerate(trainloader):
                    label_noise = label_noise.view(-1)
                    image_aug = self.tt_transform(image)
                    image_aug = image_aug.to(self.device)
                    image = image.to(self.device)
                    label_noise = label_noise.to(self.device)

                    output1 = self.model(image)
                    output2 = self.model(image_aug)

                    mix_1 = np.random.beta(1, 1)
                    mix_2 = 1 - mix_1
                    logits1, logits2 = torch.softmax(output1 * 3, dim=1), torch.softmax(output2 * 3, dim=1)
                    logits1, logits2 = torch.clamp(logits1, min=1e-6, max=1.0), torch.clamp(logits2, min=1e-6, max=1.0)
                    L_e = - (torch.mean(torch.sum(self.sm(logits1) * self.lsm(logits1), dim=1)) + torch.mean(
                        torch.sum(self.sm(logits1) * self.lsm(logits1), dim=1))) * 0.5
                    p = torch.softmax(output1, dim=1) * mix_1 + torch.softmax(output2, dim=1) * mix_2
                    pt = p ** (2)
                    pred_mix = pt / pt.sum(dim=1, keepdim=True)
                    betaa = self.gamma
                    if (current_round < self.t_w):
                        betaa = self.gamma * current_round / self.t_w
                    loss = self.loss(pred_mix, label_noise) + self.js(logits1, logits2) * betaa + L_e * self.lambda_e

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
        testloader = get_dataloder(self.test_path, self.id, self.batch_size, False)
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