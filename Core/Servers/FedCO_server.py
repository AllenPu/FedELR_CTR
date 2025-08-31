import os.path
import time
from torch.utils.tensorboard import SummaryWriter
from Core.Servers.Server_base import Server
from Core.Clients.FedCO_client import FedCOClient
import logging
import sys
import copy
import random
class FedCO(Server):

    def __init__(self, args):
        super().__init__(args)

        self.set_clients(FedCOClient)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.global_model2 = copy.deepcopy(self.global_model)
        if self.resume:
            self.load_model()

        self.Budget = []


    def train(self):

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

        for i in range(self.global_rounds + 1):
            self.current_round = i
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            if self.warm_up_steps > 0:
                logging.info(f"\n-------------Warm up-------------")
                for client in self.selected_clients:
                    client.warm_up_train()
                self.receive_models()
                self.aggregate_parameters()
                self.warm_up_steps = 0
                logging.info(f"\n-------------Warm up end-------------")

            averaged_acc = self.local_eval()
            averaged_loss = self.local_train(i)


            if self.just_eval_global_model:
                averaged_acc = self.global_eval()


            self.rs_train_loss.append(averaged_loss)
            self.rs_test_acc.append(averaged_acc)
            if self.tensorboard:
                writer.add_scalar('train_loss', averaged_loss, i)
                writer.add_scalar('train_acc', averaged_acc, i)
            if i % self.eval_gap == 0:
                logging.info(f"\n-------------Round number: {i}-------------")
                logging.info("\nEvaluate global model")
                logging.info("Averaged Train loss:{:.4f}".format(averaged_loss))
                logging.info("Averaged Test acc:{:.4f}".format(averaged_acc))

            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            logging.info('-' * 25 + 'time cost' + '-' * 25 + str(self.Budget[-1]))

            # if self.early_stop():
            #     break

        logging.info("\nBest accuracy.")
        logging.info(max(self.rs_test_acc))
        logging.info("\nAverage time cost per round.")
        logging.info(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

    def local_train(self,current_round):
        id = []
        total_losses = []
        total_train_num = []
        for client in self.clients:
            losses,train_num = client.train(current_round)
            id.append(client.id)
            total_losses.append(losses)
            total_train_num.append(train_num)

        averaged_loss = sum(total_losses) * 1.0 / sum(total_train_num)
        return averaged_loss

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_parameters(self.global_model)
            client.clone_model(self.global_model2, client.model2)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        self.uploaded_model2s = []
        tot_samples = 0
        for client in active_clients:
        #     try:
        #         client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
        #                            client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
        #     except ZeroDivisionError:
        #         client_time_cost = 0
        #     if client_time_cost <= self.time_threthold:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model)
            self.uploaded_model2s.append(client.model2)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples


    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        self.global_model2 = copy.deepcopy(self.uploaded_model2s[0])
        for param in self.global_model.parameters():
            param.data.zero_()
        for param in self.global_model2.parameters():
            param.data.zero_()
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)
        for w, client_model in zip(self.uploaded_weights, self.uploaded_model2s):
            self.add_parameters2(w, client_model)

    def add_parameters2(self, w, client_model):
        for server_param, client_param in zip(self.global_model2.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w