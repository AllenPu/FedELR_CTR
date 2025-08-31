import os.path
import time
from torch.utils.tensorboard import SummaryWriter
from Core.Servers.Server_base import Server
from Core.Clients.FedCTR_client import FedCTRClient
import logging
import sys

class FedCTR(Server):

    def __init__(self, args):
        super().__init__(args)

        self.set_clients(FedCTRClient)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

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

            memo = self.local_memo()
            averaged_acc = self.local_eval()
            averaged_loss = self.local_train()




            if self.just_eval_global_model:
                averaged_acc = self.global_eval()


            self.rs_train_loss.append(averaged_loss)
            self.rs_test_acc.append(averaged_acc)
            if self.tensorboard:
                writer.add_scalar('train_loss', averaged_loss, i)
                writer.add_scalar('train_acc', averaged_acc, i)
                writer.add_scalar('memo', memo, i)
            if i % self.eval_gap == 0:
                logging.info(f"\n-------------Round number: {i}-------------")
                logging.info("\nEvaluate global model")
                logging.info("Averaged Train loss:{:.4f}".format(averaged_loss))
                logging.info("Averaged Test acc:{:.4f}".format(averaged_acc))
                logging.info("Memo rate:{:.4f}".format(memo))

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

    def local_train(self):
        id = []
        total_losses = []
        total_train_num = []

        for client in self.clients:
            losses,train_num = client.train()
            id.append(client.id)
            total_losses.append(losses)
            total_train_num.append(train_num)


        averaged_loss = sum(total_losses) * 1.0 / sum(total_train_num)
        return averaged_loss

    def local_memo(self):
        id = []
        total_memo = []
        total_noise = []

        for client in self.clients:
            memo_num,noise_num = client.memo()
            id.append(client.id)
            total_memo.append(memo_num)
            total_noise.append(noise_num)


        averaged_loss = sum(total_memo) * 1.0 / sum(total_noise)
        return averaged_loss

