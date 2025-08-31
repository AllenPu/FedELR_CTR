import time
from Core.Clients.Client_base import Client
import numpy as np
import torch
import copy
from Core.utils.data_utils import get_dataloder
from Core.utils.criteria import get_criterion

class FedCOClient(Client):

    def __init__(self, args, id, train_samples, test_samples):
        super().__init__(args, id, train_samples, test_samples)

        self.model2 = copy.deepcopy(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr = self.learning_rate)
        self.optimizer2 = torch.optim.Adam(self.model2.parameters(),lr=self.learning_rate)
        self.criterion = get_criterion('coteaching', self.num_classes, train_samples, args, True)
        if args.forget_rate is None:
            if args.noise_ratio == 0.0 and args.noise_type not in ['clean', 'real','human']:
                self.forget_rate = args.max_noise_ratio
            self.forget_rate = args.noise_ratio
        else:
            self.forget_rate = args.forget_rate

        self.rate_schedule = np.ones(args.global_rounds + 5) * self.forget_rate

        self.rate_schedule[:args.num_gradual] = np.linspace(0, self.forget_rate**args.exponent, args.num_gradual)

        self.mom1 = 0.9
        self.mom2 = 0.1
        self.alpha_plan = [self.learning_rate] * (args.global_rounds + 5)
        self.beta1_plan = [self.mom1] * (args.global_rounds + 5)
        for i in range(args.epoch_decay_start,args.global_rounds):
            self.alpha_plan[i] = float(args.global_rounds - i)/(args.global_rounds - args.epoch_decay_start) * self.learning_rate
            self.beta1_plan[i] = self.mom2

    def train(self, current_round):
        self.model.train()
        trainloader = get_dataloder(self.train_path, self.id, self.batch_size, True)
        self.model.train()
        start_time = time.time()
        train_num = 0
        losses = 0
        self.adjust_learning_rate(self.optimizer, current_round)
        self.adjust_learning_rate(self.optimizer2, current_round)

        for step in range(self.local_epochs):
            if self.noise_type in ['clean', 'real']:
                for i,(index, image, label) in enumerate(trainloader):
                    label = label.view(-1)
                    image = image.to(self.device)
                    label = label.to(self.device)
                    output1 = self.model(image)
                    output2 = self.model2(image)

                    loss1, loss2 = self.criterion(output1, output2, label,self.rate_schedule[current_round])


                    train_num += label.shape[0]
                    losses += loss1.item() * label.shape[0]

                    self.optimizer.zero_grad()
                    self.optimizer2.zero_grad()
                    loss1.backward()
                    loss2.backward()
                    self.optimizer.step()
                    self.optimizer2.step()
            else:
                for i,(index, image, label, label_noise) in enumerate(trainloader):
                    label_noise = label_noise.view(-1)
                    image = image.to(self.device)
                    label_noise = label_noise.to(self.device)

                    output1 = self.model(image)
                    output2 = self.model2(image)

                    loss1, loss2 = self.criterion(output1, output2, label_noise, self.rate_schedule[current_round])

                    train_num += label.shape[0]
                    losses += loss1.item() * label.shape[0]

                    train_num += label_noise.shape[0]
                    losses += loss1.item() * label_noise.shape[0]

                    self.optimizer.zero_grad()
                    self.optimizer2.zero_grad()
                    loss1.backward()
                    loss2.backward()
                    self.optimizer.step()
                    self.optimizer2.step()

        self.train_loss.append(losses / train_num)
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        return losses, train_num

    def adjust_learning_rate(self,optimizer,epoch):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch]
            param_group['betas'] = (self.beta1_plan[epoch], 0.999)

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()