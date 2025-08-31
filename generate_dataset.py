import argparse
import os
import torch
from Datasets.utils.get_datasets import get_dataset
from Datasets.utils.split_data import separate_data
from Datasets.utils.custom_dataset import client_dataset,centralized_dataset
from torchvision import transforms
import random
import ujson

def read_args():
    parser = argparse.ArgumentParser(description="Dataset Preparation")
    parser.add_argument(
        "--centralized",
        default=False,
        help="Centralized setting or federated setting. True for centralized "
        "setting, while False for federated setting.",
    )
    # ----Federated Partition----
    parser.add_argument(
        "--partition",
        default="iid",
        type=str,
        choices=["iid", "dir", "pat"],
        help="Data partition scheme for federated setting.",
    )
    parser.add_argument(
        "--balance",
        default=True,
        type=bool,
        help="All clients have the same number of data.",
    )
    parser.add_argument(
        "--num_clients",
        default=20,
        type=int,
        help="Number for clients in federated setting.",
    )
    parser.add_argument(
        "--dir_alpha",
        default=1,
        type=float,
        help="Parameter for Dirichlet distribution.",
    )
    parser.add_argument(
        "--class_per_client",
        default=2,
        type=int,
        help="class_per_client number for 'pat' partition.",
    )
    parser.add_argument(
        "--max_samples",
        default=None,
        type=int,
        help="max_samples sample in one dataset(e.g. clothing1M).",
    )
    parser.add_argument(
        "--least_samples",
        default=25,
        type=int,
        help="least_samples for each client each class.",
    )


    # ----Noise setting options----
    parser.add_argument(
        "--globalize",
        action="store_true",
        help="Federated noisy label setting, globalized noise or localized noise.",
    )
    parser.add_argument(
        "--noise_type",
        default="sym",
        type=str,
        choices=["clean", "sym", "asym", "real","human"],
        help="Noise type for centralized setting: 'sym' for symmetric noise; "
        "'asym' for asymmetric noise; 'real' for real-world noise. Only works "
        "if --centralized=True.",
    )

    parser.add_argument(
        "--noise_ratio",
        default=0.0,
        type=float,
        help="Noise ratio for symmetric noise or asymmetric noise.",
    )

    parser.add_argument(
        "--min_noise_ratio",
        default=0.0,
        type=float,
        help="Minimum noise ratio for symmetric noise or asymmetric noise. Only works when 'globalize' is Flase",
    )
    parser.add_argument(
        "--max_noise_ratio",
        default=1.0,
        type=float,
        help="Maximum noise ratio for symmetric noise or asymmetric noise. Only works when 'globalize' is Flase",
    )

    # ----Dataset path options----
    parser.add_argument(
        "--dataset",
        default="cifar10",
        type=str,
        choices=["mnist", "fmnist", "cifar10", "cifar100", "clothing1M", "tinyimagenet","cifar10NA","cifar10NW","cifar100N"],
        help="Dataset for experiment. Current support: ['mnist', 'cifar10', 'cifar100`, 'clothing1m', 'tinyimagenet']",
    )
    parser.add_argument(
        "--data_dir",
        default="./Datasets",
        type=str,
        help="Directory for dataset.",
    )
    # ----Miscs options----
    parser.add_argument("--seed", default=0, type=int, help="Random seed")


    args = parser.parse_args()
    return args

class federated_dataset(object):
    def __init__(self, args):
        self.args = args
        self.centralized = args.centralized
        self.globalize = args.globalize
        self.dataset = args.dataset
        self.num_clients = args.num_clients
        self.partition = args.partition
        self.balance = args.balance
        self.dir_alpha = args.dir_alpha
        self.class_per_client = args.class_per_client
        self.max_samples = args.max_samples
        self.least_samples = args.least_samples

        self.noise_type = args.noise_type
        self.noise_ratio = args.noise_ratio
        self.min_noise_ratio = args.min_noise_ratio
        self.max_noise_ratio = args.max_noise_ratio
        if self.noise_type == 'clean':
            self.noise_ratio = 0.0
            self.min_noise_ratio = 0.0
            self.max_noise_ratio = 0.0
        self.seed = args.seed

        self.data_dir = args.data_dir

        balance_str = "balance" if self.balance else "imbalance"

        if args.partition != "iid":
            balance_str = "imbalance"

        if self.globalize:
            self.dataset_path = os.path.join(self.data_dir, self.dataset, self.noise_type+"_"+str(self.noise_ratio)+"_"+self.partition+"_"+balance_str+"_"+str(self.num_clients))
        else:
            self.dataset_path = os.path.join(self.data_dir, self.dataset, self.noise_type+"_max"+str(self.max_noise_ratio)+"_min"+str(self.min_noise_ratio)+self.partition+"_"+balance_str+"_"+str(self.num_clients))

        if self.noise_type == 'human':
            self.dataset_path = os.path.join(self.data_dir, self.dataset, self.noise_type+"_"+self.partition+"_"+balance_str+"_"+str(self.num_clients))
        # else:
        #     raise ValueError("Globalize should be 'global' or 'local'")
        if not os.path.exists(self.dataset_path):
            try:
                os.makedirs(self.dataset_path)
            except FileNotFoundError:
                os.makedirs(self.dataset_path)

        self.train_path = os.path.join(self.dataset_path, "train")
        self.test_path = os.path.join(self.dataset_path, "test")

        if not os.path.exists(self.train_path):
            try:
                os.makedirs(self.train_path)
            except FileNotFoundError:
                os.makedirs(self.train_path)
        if not os.path.exists(self.test_path):
            try:
                os.makedirs(self.test_path)
            except FileNotFoundError:
                os.makedirs(self.test_path)

        self.num_classes = {'mnist': 10,'fmnist' : 10, 'cifar10': 10, 'cifar100': 100, 'clothing1M': 14, 'tinyimagenet': 200,'cifar10NA': 10, 'cifar10NW': 10, 'cifar100N': 100}
        self.transform = {
            'mnist': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
            'fmnist': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
            'cifar10': transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])]),
            'cifar10NA': transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]),
            'cifar10NW': transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]),
            'cifar100': transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5071, 0.4867, 0.4408],[0.2675, 0.2565, 0.2761])]),
            'cifar100N': transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])]),
            'tinyimagenet': transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
            'clothing1M': transforms.Compose([transforms.Resize(256),transforms.RandomCrop(224),transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.6959, 0.6537, 0.6371],[0.3113, 0.3192, 0.3214])])
        }
    def generate_client_dataset(self):
        if not self.check_dataset():
            raw_train_set,raw_test_set = get_dataset(self.dataset)
            if self.centralized:
                train_set = centralized_dataset(raw_train_set, self.dataset,self.num_classes[self.dataset], self.noise_type, self.noise_ratio, True , self.transform[self.dataset])
                test_set = centralized_dataset(raw_test_set, self.dataset,self.num_classes[self.dataset], self.noise_type, self.noise_ratio, False , self.transform[self.dataset] )
                torch.save(train_set, os.path.join(self.train_path, "0.pkl"))
                torch.save(test_set, os.path.join(self.test_path, "0.pkl"))
                return True
            test_set = centralized_dataset(raw_test_set, self.dataset,self.num_classes[self.dataset], self.noise_type, self.noise_ratio, False,self.transform[self.dataset])
            train_set = centralized_dataset(raw_train_set, self.dataset,self.num_classes[self.dataset], self.noise_type, self.noise_ratio, True,self.transform[self.dataset])
            torch.save(test_set, os.path.join(self.dataset_path, "test_dataset.pkl"))
            torch.save(train_set,os.path.join(self.dataset_path, "train_dataset.pkl"))
            del test_set
            del train_set
            train_set_Subset,train_sample_idx = separate_data(raw_train_set, self.num_clients, self.num_classes[self.dataset], self.balance, self.partition, self.class_per_client, self.max_samples, self.least_samples, self.dir_alpha, self.seed)
            test_set_Subset,test_sample_idx = separate_data(raw_test_set, self.num_clients, self.num_classes[self.dataset], self.balance, self.partition, self.class_per_client, self.max_samples, self.least_samples, self.dir_alpha, self.seed)
            for i in range(self.num_clients):
                if not self.globalize:
                    noise_ratio = round(random.uniform(self.min_noise_ratio, self.max_noise_ratio),1)
                else:
                    noise_ratio = self.noise_ratio
                train_set = client_dataset(train_set_Subset[i], self.dataset,self.num_classes[self.dataset], i , self.noise_type, noise_ratio, True, self.transform[self.dataset], train_sample_idx[i])
                test_set = client_dataset(test_set_Subset[i], self.dataset,self.num_classes[self.dataset], i , self.noise_type, noise_ratio, False, self.transform[self.dataset], test_sample_idx[i])
                torch.save(train_set, os.path.join(self.train_path, str(i)+".pkl"))
                torch.save(test_set, os.path.join(self.test_path, str(i)+".pkl"))
                self.save_config()

    def check_dataset(self):
        config_path = os.path.join(self.dataset_path, "dataset_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = ujson.load(f)
            if config == self.args.__dict__:
                print("\nDataset already generated.\n")
                return True
        else:
            return False

    def save_config(self):
        with open(os.path.join(self.dataset_path, "dataset_config.json"), 'w') as f:
            ujson.dump(self.args.__dict__, f, indent=2)

if __name__ == "__main__":
    args = read_args()
    random.seed(args.seed)

    dataset = federated_dataset(args)
    dataset.generate_client_dataset()
    print("Dataset generated successfully!")



