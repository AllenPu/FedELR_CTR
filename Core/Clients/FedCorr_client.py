import copy
import time
from Core.Clients.Client_base import Client
import torch
import numpy as np
import torch.nn.functional as F
from Core.utils.data_utils import get_dataloder, get_dataset, del_dataset, save_dataset
import shutil
import os
class FedCorrClient(Client):

    def __init__(self, args, id, train_samples, test_samples):
        super().__init__(args, id, train_samples, test_samples)

        clear_folder_contents(self.dataset_path + '_FedCorr')
        copy_folder_contents(self.dataset_path, self.dataset_path + '_FedCorr')
        self.dataset_path = self.dataset_path + '_FedCorr'

        self.train_path = os.path.join(self.dataset_path, "train")
        self.test_path = os.path.join(self.dataset_path, "test")

        self.beta_corr = args.beta_corr

    def relabel_dataset(self, label):
        label = torch.from_numpy(label)
        label = label.to(torch.int64)
        dataset = get_dataset(self.train_path, self.id)
        if self.noise_type in ['clean', 'real']:
            dataset.label = label
        else:
            dataset.label_noise = label
        del_dataset(self.train_path, self.id)
        save_dataset(self.train_path, self.id, dataset)

    def train(self, mu):
        global_model = copy.deepcopy(self.model)
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
                    if self.args.mixup:
                        inputs, targets_a, targets_b, lam = mixup_data(image, label, self.args.alpha_corr)

                        log_probs = self.model(inputs)
                        loss = mixup_criterion(self.criterion, log_probs, targets_a, targets_b, lam)
                    else:

                        log_probs = self.model(image)
                        loss = self.criterion(log_probs, label)

                    if self.beta_corr > 0:
                        if i > 0:
                            w_diff = torch.tensor(0.).to(self.device)
                            for w, w_t in zip(global_model.parameters(), self.model.parameters()):
                                w_diff += torch.pow(torch.norm(w - w_t), 2)
                            w_diff = torch.sqrt(w_diff)
                            loss += self.beta_corr * mu * w_diff

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

                    if self.args.mixup:
                        inputs, targets_a, targets_b, lam = mixup_data(image, label_noise, self.args.alpha_corr)

                        log_probs = self.model(inputs)
                        loss = mixup_criterion(self.criterion, log_probs, targets_a, targets_b, lam)
                    else:

                        log_probs = self.model(image)
                        loss = self.criterion(log_probs, label_noise)

                    if self.beta_corr > 0:
                        if i > 0:
                            w_diff = torch.tensor(0.).to(self.device)
                            for w, w_t in zip(global_model.parameters(), self.model.parameters()):
                                w_diff += torch.pow(torch.norm(w - w_t), 2)
                            w_diff = torch.sqrt(w_diff)
                            loss += self.beta_corr * mu * w_diff

                    train_num += label.shape[0]
                    losses += loss.item() * label.shape[0]

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
        self.train_loss.append(losses / train_num)
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        del global_model
        return losses, train_num

    def get_output_frac1(self, latent=False, criterion=torch.nn.CrossEntropyLoss(reduction='none')):
        self.model.eval()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        trainloader = get_dataloder(self.train_path,self.id, self.batch_size,True)
        with torch.no_grad():
            if self.noise_type in ['clean', 'real']:
                for i, (index, images, labels) in enumerate(trainloader):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    labels = labels.long()
                    if latent == False:
                        outputs = self.model(images)
                        outputs = F.softmax(outputs, dim=1)
                    else:
                        outputs = self.model(images, True)
                    loss = criterion(outputs, labels)
                    if i == 0:
                        output_whole = np.array(outputs.cpu())
                        loss_whole = np.array(loss.cpu())
                        index_whole = np.array(index.cpu())
                    else:
                        output_whole = np.concatenate((output_whole, outputs.cpu()), axis=0)
                        loss_whole = np.concatenate((loss_whole, loss.cpu()), axis=0)
                        index_whole = np.concatenate((index_whole, index.cpu()), axis=0)
            else:
                for i,(index, images, label, label_noise) in enumerate(trainloader):
                    images = images.to(self.device)
                    label_noise = label_noise.to(self.device)
                    label_noise = label_noise.long()
                    if latent == False:
                        outputs = self.model(images)
                        outputs = F.softmax(outputs, dim=1)
                    else:
                        outputs = self.model(images, True)
                    loss = criterion(outputs, label_noise)
                    if i == 0:
                        output_whole = np.array(outputs.cpu())
                        loss_whole = np.array(loss.cpu())
                        index_whole = np.array(index.cpu())
                    else:
                        output_whole = np.concatenate((output_whole, outputs.cpu()), axis=0)
                        loss_whole = np.concatenate((loss_whole, loss.cpu()), axis=0)
                        index_whole = np.concatenate((index_whole, index.cpu()), axis=0)
        if criterion is not None:
            return index_whole, output_whole, loss_whole
        else:
            return index_whole, output_whole



    def get_sample_idx(self):
        train_dataset = get_dataloder(self.train_path, self.id, self.batch_size, True).dataset
        return train_dataset.sample_idx

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    y_a = y_a.long()
    y_b = y_b.long()
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def copy_folder_contents(src_folder, dst_folder):
    """
    复制一个文件夹内的所有内容到新的文件夹。

    :param src_folder: 源文件夹的路径。
    :param dst_folder: 目标文件夹的路径。
    """
    # 确保目标文件夹存在
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # 遍历源文件夹中的所有文件和子文件夹
    for item in os.listdir(src_folder):
        src_item = os.path.join(src_folder, item)
        dst_item = os.path.join(dst_folder, item)

        # 如果是文件，复制文件
        if os.path.isfile(src_item):
            shutil.copy2(src_item, dst_item)
        # 如果是文件夹，递归复制文件夹
        elif os.path.isdir(src_item):
            copy_folder_contents(src_item, dst_item)

def clear_folder_contents(folder_path):
    """
    清空指定文件夹中的所有内容，但保留文件夹本身。

    :param folder_path: 要清空的文件夹的路径。
    """
    # 检查文件夹是否存在
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # 遍历文件夹中的所有文件和子文件夹
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            # 如果是文件，删除文件
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            # 如果是文件夹，递归删除文件夹中的所有内容
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
    else:
        # print(f"文件夹 {folder_path} 不存在或不是一个文件夹。")
        pass