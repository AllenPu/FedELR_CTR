import os.path
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from Core.Servers.Server_base import Server
from Core.Clients.FedCorr_client import FedCorrClient
from Core.utils.data_utils import get_global_train_dataset
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
import shutil
import os
import logging
import sys
class FedCorr(Server):

    def __init__(self, args):
        super().__init__(args)

        self.set_clients(FedCorrClient)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        if self.resume:
            self.load_model()

        clear_folder_contents(self.dataset_path + '_FedCorr')
        copy_folder_contents(self.dataset_path, self.dataset_path + '_FedCorr')
        self.dataset_path = self.dataset_path + '_FedCorr'

        self.train_path = os.path.join(self.dataset_path, "train")
        self.test_path = os.path.join(self.dataset_path, "test")

        self.Budget = []



        self.dataset = get_global_train_dataset(self.dataset_path)

        self.iteration1 = args.iteration1
        self.frac1 = args.frac1
        self.frac2 = args.frac2
        self.correction = args.correction
        self.relabel_ratio = args.relabel_ratio
        self.confidence_thres = args.confidence_thres
        self.beta_corr = args.beta_corr
        self.fine_tuning = args.fine_tuning
        self.clean_set_thres = args.clean_set_thres
        self.rounds1 = args.rounds1
        self.rounds2 = args.rounds2

    def train(self):
        rounds = 0

        if not os.path.exists(os.path.join(self.result_dir, 'log')):
            os.makedirs(os.path.join(self.result_dir, 'log'))
        logging.basicConfig(filename = os.path.join(self.result_dir, 'log') + '/log.txt',level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S'))
        console_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(console_handler)

        id = []
        total_losses = []
        total_train_num = []

        y_train = self.dataset.label
        LID_accumulative_client = np.zeros(self.num_clients)
        estimated_noisy_level = np.zeros(self.num_clients)
        if self.tensorboard:
            tensorboard_path = os.path.join(self.result_dir,'log')
            writer = SummaryWriter(tensorboard_path)
            logging.info('tensorboard log file path: ' + tensorboard_path)
        for iteration in range(self.iteration1):
            logging.info(f"\n-------------Iteration1:-------------")
            LID_whole = np.zeros(len(y_train))
            loss_whole = np.zeros(len(y_train))
            LID_client = np.zeros(self.num_clients)
            loss_accumulative_whole = np.zeros(len(y_train))

            if iteration == 0:
                mu_list = np.zeros(self.num_clients)
            else:
                mu_list = estimated_noisy_level

            prob = [1 / self.num_clients] * self.num_clients

            for _ in range(int(1 / self.frac1)):
                rounds += 1
                self.selected_clients = self.select_clients_round1(prob)
                self.send_models()

                for client in self.selected_clients:
                    prob[client.id] = 0
                    if sum(prob) > 0:
                        prob = [prob[i] / sum(prob) for i in range(len(prob))]
                    mu_i = mu_list[client.id]

                    losses, train_num = client.train(mu_i)
                    id.append(client.id)
                    total_losses.append(losses)
                    total_train_num.append(train_num)


                    index_whole, local_output, loss = client.get_output_frac1(latent=False)
                    sample_idx = client.get_sample_idx()
                    LID_local = list(lid_term(local_output, local_output))
                    LID_whole[index_whole] = LID_local
                    loss_whole[index_whole] = loss
                    LID_client[client.id] = np.mean(LID_local)

            averaged_acc = self.local_eval()
            averaged_loss = sum(total_losses) * 1.0 / sum(total_train_num)
            id = []
            total_losses = []
            total_train_num = []

            self.receive_models()
            self.aggregate_parameters()

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


            LID_accumulative_client = LID_accumulative_client + np.array(LID_client)
            loss_accumulative_whole = loss_accumulative_whole + np.array(loss_whole)

            # Apply Gaussian Mixture Model to LID
            gmm_LID_accumulative = GaussianMixture(n_components=2).fit(
                np.array(LID_accumulative_client).reshape(-1, 1))
            labels_LID_accumulative = gmm_LID_accumulative.predict(np.array(LID_accumulative_client).reshape(-1, 1))
            clean_label = np.argsort(gmm_LID_accumulative.means_[:, 0])[0]

            noisy_set = np.where(labels_LID_accumulative != clean_label)[0]
            clean_set = np.where(labels_LID_accumulative == clean_label)[0]

            estimated_noisy_level = np.zeros(self.num_clients)

            noisy_clients = self.select_client_from_set(noisy_set)

            for client in noisy_clients:
                sample_idx = client.get_sample_idx()
                loss = np.array(loss_accumulative_whole[sample_idx])
                gmm_loss = GaussianMixture(n_components=2).fit(np.array(loss).reshape(-1, 1))
                labels_loss = gmm_loss.predict(np.array(loss).reshape(-1, 1))
                gmm_clean_label_loss = np.argsort(gmm_loss.means_[:, 0])[0]

                pred_n = np.where(labels_loss.flatten() != gmm_clean_label_loss)[0]
                estimated_noisy_level[client.id] = len(pred_n) / len(sample_idx)


            if self.correction:
                for client in noisy_clients:
                    sample_idx = np.array(client.get_sample_idx())
                    loss = np.array(loss_accumulative_whole[sample_idx])
                    _, local_output, _ = client.get_output_frac1(latent=False)
                    relabel_idx = (-loss).argsort()[:int(len(sample_idx) * estimated_noisy_level[client.id] * self.relabel_ratio)]
                    relabel_idx = list(set(np.where(np.max(local_output, axis=1) > self.confidence_thres)[0]) & set(relabel_idx))
                    y_train_noisy_new = np.array(self.dataset.label)



                    y_train_noisy_new[sample_idx[relabel_idx]] = np.argmax(local_output, axis=1)[relabel_idx]

                    client.relabel_dataset(y_train_noisy_new[sample_idx])

        self.beta_corr = 0
        for client in self.clients:
            client.beta_corr = self.beta_corr

        # ---------------------------- second stage training -------------------------------
        if self.fine_tuning:
            selected_clean_idx = np.where(estimated_noisy_level <= self.clean_set_thres)[0]

            prob = np.zeros(self.num_clients)
            prob[selected_clean_idx] = 1 / len(selected_clean_idx)
            m = max(int(self.frac2 * self.num_clients), 1)  # num_select_clients
            m = min(m, len(selected_clean_idx))

            logging.info(f"\n-------------Rounds1:-------------")

            for rnd in range(self.rounds1):
                rounds += 1
                self.selected_clients = self.select_clients_round2(m, prob)
                self.send_models()
                averaged_acc = self.local_eval()
                for client in self.selected_clients:
                    losses, train_num = client.train(mu = 0)
                    id.append(client.id)
                    total_losses.append(losses)
                    total_train_num.append(train_num)
                self.receive_models()
                self.aggregate_parameters()



                averaged_loss = sum(total_losses) * 1.0 / sum(total_train_num)
                id = []
                total_losses = []
                total_train_num = []

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

            if self.correction:
                relabel_idx_whole = []
                for client in noisy_clients:
                    sample_idx = np.array(client.get_sample_idx())
                    _, global_output, _ = client.get_output_frac1(latent=False)
                    y_predicted = np.argmax(global_output, axis=1)
                    relabel_idx = np.where(np.max(global_output, axis=1) > self.confidence_thres)[0]
                    y_train_noisy_new = np.array(self.dataset.label)
                    y_train_noisy_new[sample_idx[relabel_idx]] = y_predicted[relabel_idx]

                    client.relabel_dataset(y_train_noisy_new[sample_idx])

        # ---------------------------- third stage training -------------------------------
        # third stage hyper-parameter initialization
        m = max(int(self.frac2 * self.num_clients), 1)  # num_select_clients
        prob = [1 / self.num_clients for i in range(self.num_clients)]

        logging.info(f"\n-------------Rounds2:-------------")
        for rnd in range(self.rounds2):
            rounds += 1
            self.selected_clients = self.select_clients_round3(m,prob)
            self.send_models()
            averaged_acc = self.local_eval()
            for client in self.selected_clients:
                losses, train_num = client.train(mu=0)
                id.append(client.id)
                total_losses.append(losses)
                total_train_num.append(train_num)
            self.receive_models()
            self.aggregate_parameters()


            averaged_loss = sum(total_losses) * 1.0 / sum(total_train_num)
            id = []
            total_losses = []
            total_train_num = []

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

        logging.info("\nBest accuracy.")
        logging.info(max(self.rs_test_acc))
        # logging.info("\nAverage time cost per round.")
        # logging.info(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

    # def train(self):
    #     if self.tensorboard:
    #         tensorboard_path = os.path.join(self.result_dir,'log')
    #         writer = SummaryWriter(tensorboard_path)
    #         print('tensorboard log file path: ', tensorboard_path)
    #
    #     for i in range(self.global_rounds + 1):
    #         self.current_round = i
    #         s_t = time.time()
    #         self.selected_clients = self.select_clients()
    #         self.send_models()
    #         if self.warm_up_steps > 0:
    #             print(f"\n-------------Warm up-------------")
    #             for client in self.selected_clients:
    #                 client.warm_up_train()
    #             self.receive_models()
    #             self.aggregate_parameters()
    #             self.warm_up_steps = 0
    #             print(f"\n-------------Warm up end-------------")
    #
    #         averaged_loss = self.local_train()
    #         averaged_acc = self.local_eval()
    #
    #         if self.just_eval_global_model:
    #             averaged_acc = self.global_eval()
    #
    #
    #         self.rs_train_loss.append(averaged_loss)
    #         self.rs_test_acc.append(averaged_acc)
    #         if self.tensorboard:
    #             writer.add_scalar('train_loss', averaged_loss, i)
    #             writer.add_scalar('train_acc', averaged_acc, i)
    #         if i % self.eval_gap == 0:
    #             print(f"\n-------------Round number: {i}-------------")
    #             print("\nEvaluate global model")
    #             print("Averaged Train loss:{:.4f}".format(averaged_loss))
    #             print("Averaged Test acc:{:.4f}".format(averaged_acc))
    #
    #         self.receive_models()
    #         self.aggregate_parameters()
    #
    #         self.Budget.append(time.time() - s_t)
    #         print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])
    #
    #         # if self.early_stop():
    #         #     break
    #
    #     print("\nBest accuracy.")
    #     print(max(self.rs_test_acc))
    #     print("\nAverage time cost per round.")
    #     print(sum(self.Budget[1:])/len(self.Budget[1:]))
    #
    #     self.save_results()
    #     self.save_global_model()

    def select_clients_round1(self,prob):

        selected_clients = list(np.random.choice(self.clients, int(self.num_clients * self.frac1), p=prob))

        return selected_clients

    def select_clients_round2(self,m,prob):

        selected_clients = list(np.random.choice(self.clients, m, replace=False, p=prob))

        return selected_clients

    def select_clients_round3(self,m,prob):

        selected_clients = list(np.random.choice(self.clients, m, replace=False, p=prob))

        return selected_clients

    def select_client_from_set(self, set):
        select_clients = [self.clients[i] for i in set]
        return select_clients

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = self.selected_clients


        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            # try:
            #     client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
            #                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            # except ZeroDivisionError:
            #     client_time_cost = 0
            # if client_time_cost <= self.time_threthold:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples


def lid_term(X, batch, k=20):
    eps = 1e-6
    X = np.asarray(X, dtype=np.float32)

    batch = np.asarray(batch, dtype=np.float32)
    f = lambda v: - k / (np.sum(np.log(v / (v[-1]+eps)))+eps)
    distances = cdist(X, batch)

    # get the closest k neighbours
    sort_indices = np.apply_along_axis(np.argsort, axis=1, arr=distances)[:, 1:k + 1]
    m, n = sort_indices.shape
    idx = np.ogrid[:m, :n]
    idx[1] = sort_indices
    # sorted matrix
    distances_ = distances[tuple(idx)]
    lids = np.apply_along_axis(f, axis=1, arr=distances_)
    return lids

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
        # logging.info(f"文件夹 {folder_path} 不存在或不是一个文件夹。")
        pass