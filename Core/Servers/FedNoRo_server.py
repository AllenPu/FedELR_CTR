import os.path
import time
import numpy as np
import torch
import copy
from torch.utils.tensorboard import SummaryWriter
from Core.Servers.Server_base import Server
from Core.Clients.FedNoRo_client import FedNoRoClient
from sklearn.mixture import GaussianMixture
from Core.utils.data_utils import get_dataloder, get_dataset, get_global_trainloader, get_global_train_dataset
import logging
import sys

class FedNoRo(Server):

    def __init__(self, args):
        super().__init__(args)

        self.set_clients(FedNoRoClient)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        if self.resume:
            self.load_model()

        self.Budget = []
        self.dataset = get_global_train_dataset(self.dataset_path)

        self.s1 = args.s1
        self.begin = args.begin
        self.end = args.end
        self.a_noro = args.a_noro

    def train(self):
        # self.save_model_init()
        rounds = 0
        if not os.path.exists(os.path.join(self.result_dir, 'log')):
            os.makedirs(os.path.join(self.result_dir, 'log'))
        logging.basicConfig(filename = os.path.join(self.result_dir, 'log') + '/log.txt',level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S'))
        console_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(console_handler)
        if self.tensorboard:
            tensorboard_path = os.path.join(self.result_dir,'log')
            writer = SummaryWriter(tensorboard_path)
            logging.info('tensorboard log file path: ' + tensorboard_path)

        # ------------------------ Stage 1: warm up ------------------------
        logging.info("-------------Stage 1: warm up-------------")
        for rnd in range(self.s1):

            self.selected_clients = self.select_clients()
            self.send_models()
            s_t = time.time()
            averaged_loss = self.local_train_s1()
            self.receive_models()
            self.aggregate_parameters()
            # averaged_acc = self.local_eval()
            averaged_acc = self.global_eval()

            self.rs_train_loss.append(averaged_loss)
            self.rs_test_acc.append(averaged_acc)
            if self.tensorboard:
                writer.add_scalar('train_loss', averaged_loss, rounds)
                writer.add_scalar('train_acc', averaged_acc, rounds)
            if rounds % self.eval_gap == 0:
                logging.info(f"\n-------------Round number: {rounds}-------------")
                logging.info("\nEvaluate global model")
                logging.info("Averaged Train loss:{:.4f}".format(averaged_loss))
                logging.info("Averaged Test acc:{:.4f}".format(averaged_acc))



            self.Budget.append(time.time() - s_t)
            logging.info('-' * 25 + 'time cost' + '-' * 25 + str(self.Budget[-1]))
            rounds += 1
        self.save_model_s1()

        #  ------------------------ client selection ------------------------

        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        local_output, loss = self.get_output(latent=False, criterion=criterion)



        metrics = np.zeros((self.num_clients, self.num_classes)).astype("float")
        num = np.zeros((self.num_clients, self.num_classes)).astype("float")

        for id in range(self.num_clients):
            idxs = np.array(self.clients[id].get_sample_idx())
            for idx in idxs:
                if self.noise_type in ['clean', 'real']:
                    c = (self.dataset.label).numpy()[idx]
                else:
                    c = (self.dataset.label_noise).numpy()[idx]
                num[id, c] += 1
                metrics[id, c] += loss[idx]

        metrics = metrics / num
        for i in range(metrics.shape[0]):
            for j in range(metrics.shape[1]):
                if np.isnan(metrics[i, j]):
                    metrics[i, j] = np.nanmin(metrics[:, j])
        for j in range(metrics.shape[1]):
            metrics[:, j] = (metrics[:, j] - metrics[:, j].min()) / \
                            (metrics[:, j].max() - metrics[:, j].min())
        logging.info("metrics:")
        logging.info(metrics)
        vote = []
        for i in range(9):
            gmm = GaussianMixture(n_components=2, random_state=i).fit(metrics)
            gmm_pred = gmm.predict(metrics)
            noisy_clients = np.where(gmm_pred == np.argmax(gmm.means_.sum(1)))[0]
            noisy_clients = set(list(noisy_clients))
            vote.append(noisy_clients)
        cnt = []
        for i in vote:
            cnt.append(vote.count(i))
        noisy_clients = list(vote[cnt.index(max(cnt))])
        real_noise_ratio = [i.get_noise_ratio() for i in self.clients]
        logging.info(
            f"selected noisy clients: {noisy_clients}, real noisy clients: {np.where(np.array(real_noise_ratio) > 0.)[0]}")
        real_noise__ = {}
        for _ in range(len(real_noise_ratio)):
            real_noise__[_]= real_noise_ratio[_]
        #real_noise__按noise_ratio从大到小排序
        real_noise__ = dict(sorted(real_noise__.items(), key=lambda x: x[1], reverse=False))
        logging.info(f'real_noise_ratio:{real_noise__}')
        clean_clients = list(set(list(range(self.num_clients))) - set(noisy_clients))
        logging.info(f"selected clean clients: {clean_clients}")
        logging.info("------------------------ Stage 2 ------------------------")
        # ------------------------ Stage 2: ------------------------
        for rnd in range(self.s1, self.global_rounds + 1):
            weight_kd = get_current_consistency_weight(
            rnd, self.begin, self.end) * self.a_noro

            self.selected_clients = self.select_clients()
            self.send_models()
            s_t = time.time()

            averaged_acc = self.local_eval()
            averaged_loss = self.local_train_s2(noisy_clients,  weight_kd)

            self.rs_train_loss.append(averaged_loss)
            self.rs_test_acc.append(averaged_acc)

            if self.tensorboard:
                writer.add_scalar('train_loss', averaged_loss, rounds)
                writer.add_scalar('train_acc', averaged_acc, rounds)
            if rounds % self.eval_gap == 0:
                logging.info(f"\n-------------Round number: {rounds}-------------")
                logging.info("\nEvaluate global model")
                logging.info("Averaged Train loss:{:.4f}".format(averaged_loss))
                logging.info("Averaged Test acc:{:.4f}".format(averaged_acc))

            self.receive_models()
            self.aggregate_parameters_DaAgg(noisy_clients, clean_clients)

            self.Budget.append(time.time() - s_t)
            logging.info('-' * 25 + 'time cost' + '-' * 25 + str(self.Budget[-1]))
            rounds += 1

        logging.info("\nBest accuracy.")
        logging.info(max(self.rs_test_acc))
        logging.info("\nAverage time cost per round.")
        logging.info(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()





    def get_output(self, latent=False, criterion=torch.nn.CrossEntropyLoss(reduction='none')):
        self.global_model.to(self.device)
        self.global_model.eval()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        trainloader = get_global_trainloader(self.dataset_path, 32)
        with torch.no_grad():
            if self.noise_type in ['clean', 'real']:
                for i, (index, images, labels) in enumerate(trainloader):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    if latent == False:
                        outputs = self.global_model(images)
                        outputs = torch.nn.functional.softmax(outputs, dim=1)
                    else:
                        outputs = self.global_model(images)
                    loss = criterion(outputs, labels)
                    if i == 0:
                        output_whole = np.array(outputs.cpu().detach())
                        loss_whole = np.array(loss.cpu().detach())
                    else:
                        output_whole = np.concatenate((output_whole, outputs.cpu().detach()), axis=0)
                        loss_whole = np.concatenate((loss_whole, loss.cpu().detach()), axis=0)
            else:
                for i, (index, images, label, label_noise) in enumerate(trainloader):
                    images = images.to(self.device)
                    label_noise = label_noise.to(self.device)
                    if latent == False:
                        outputs = self.global_model(images)
                        outputs = torch.nn.functional.softmax(outputs, dim=1)
                    else:
                        outputs = self.global_model(images)
                    loss = criterion(outputs, label_noise)
                    if i == 0:
                        output_whole = np.array(outputs.cpu().detach())
                        loss_whole = np.array(loss.cpu().detach())
                    else:
                        output_whole = np.concatenate((output_whole, outputs.cpu().detach()), axis=0)
                        loss_whole = np.concatenate((loss_whole, loss.cpu().detach()), axis=0)

        # unique_elements, counts = np.unique(loss_whole, return_counts=True)
        # for unique, count in zip(unique_elements, counts):
        #     print(f"元素 {unique} 出现了 {count} 次")
        if criterion is not None:
            return output_whole, loss_whole
        else:
            return output_whole

    def local_train_s1(self):
        id = []
        total_losses = []
        total_train_num = []
        for client in self.clients:
            losses,train_num = client.train_LA()
            # losses,train_num = client.train()
            id.append(client.id)
            total_losses.append(losses)
            total_train_num.append(train_num)

        averaged_loss = sum(total_losses) * 1.0 / sum(total_train_num)
        return averaged_loss

    def local_train_s2(self, noisy_clients, weight_kd):
        id = []
        total_losses = []
        total_train_num = []
        for client in self.clients:
            if client.id in noisy_clients:
                losses, train_num = client.train_FedNoRo(weight_kd)
            else:
                losses,train_num = client.train_LA()
            id.append(client.id)
            total_losses.append(losses)
            total_train_num.append(train_num)

        averaged_loss = sum(total_losses) * 1.0 / sum(total_train_num)
        return averaged_loss

    def save_model_s1(self):
        model_path = os.path.join(self.result_dir, "model")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, "s1_model.pt")
        torch.save(self.global_model , model_path)

    def save_model_init(self):
        model_path = os.path.join(self.result_dir, "model")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, "init_model.pt")
        torch.save(self.global_model , model_path)

    def aggregate_parameters_DaAgg(self, noisy_clients, clean_clients):
        assert (len(self.uploaded_models) > 0)
        w = self.uploaded_models
        client_weight = self.uploaded_weights
        distance = np.zeros(len(w))
        for n_idx in noisy_clients:
            dis = []
            for c_idx in clean_clients:
                dis.append(model_dist(w[n_idx], w[c_idx]))
            distance[n_idx] = min(dis)
        distance = distance / distance.max()
        client_weight = client_weight * np.exp(-distance)
        client_weight = client_weight / client_weight.sum()


        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for w, client_model in zip(client_weight, self.uploaded_models):
            self.add_parameters(w, client_model)

def sigmoid_rampup(current, begin, end):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    current = np.clip(current, begin, end)
    phase = 1.0 - (current-begin) / (end-begin)
    return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(rnd, begin, end):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return sigmoid_rampup(rnd, begin, end)

def model_dist(w_1, w_2):
    w_1 = w_1.state_dict()
    w_2 = w_2.state_dict()
    assert w_1.keys() == w_2.keys(), "Error: cannot compute distance between dict with different keys"
    dist_total = torch.zeros(1).float()
    for key in w_1.keys():
        if "int" in str(w_1[key].dtype):
            continue
        dist = torch.norm(w_1[key] - w_2[key])
        dist_total += dist.cpu()

    return dist_total.cpu().item()