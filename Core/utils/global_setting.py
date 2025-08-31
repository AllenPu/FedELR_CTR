import argparse
import os
import ujson
import random
import numpy as np
import torch
import torchvision
from Core.Networks.models import *
from Core.Networks.AlexNet import *
from Core.Networks.ResNet import *
from Core.Networks.MobileNet_V2 import *
from Core.Networks.simsiam.model_factory import SimSiam
from Core.Networks.simsiam.resnet_cifar import ResNet18_1c

from Core.Servers.FedAvg_server import FedAvg
from Core.Servers.FedAvg_ELR_server import FedAvgELR
from Core.Servers.FedTBD_server import FedTBD
from Core.Servers.FedCTR_server import FedCTR
from Core.Servers.FedTES_server import FedTES
from Core.Servers.FedCorr_server import FedCorr
from Core.Servers.FedNoRo_server import FedNoRo
from Core.Servers.FedLSR_server import FedLSR
from Core.Servers.FedCO_server import FedCO
from Core.Servers.FedSCE_server import FedSCE


def read_save_federated_args():
    parser = argparse.ArgumentParser(description="Federated setting")


    parser.add_argument(
        "--global_rounds",
        type=int,
        default=100,
        help="Number of rounds of global training."
    )
    parser.add_argument(
        "--local_epochs",
        type=int,
        default=1,
        help="Number of local epochs in each round."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cnn9l",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        default="FedAvg",
    )
    parser.add_argument(
        "--join_ratio",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--random_join_ratio",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--eval_gap",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--just_eval_global_model",
        type=bool,
        default=False,
        help="Just evaluate global model in a large test_dataset."
    )
    parser.add_argument(
        "--client_drop_rate",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--warm_up_steps",
        type=int,
        default=0,
        help="Number of warm up steps for clients before training."
    )
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
        default=0.1,
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
        default=64000,
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
        action='store_true',
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
        help="Minimum noise ratio for symmetric noise or asymmetric noise. Only works when 'globalize' is False",
    )
    parser.add_argument(
        "--max_noise_ratio",
        default=1.0,
        type=float,
        help="Maximum noise ratio for symmetric noise or asymmetric noise. Only works when 'globalize' is False",
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
    parser.add_argument(
        "--result_dir",
        default="./Results",
        type=str,
        help="Directory for results.",
    )
    # ------------------------------------------------------------------------criterion setting_________________________________________________________________________
    parser.add_argument(
        "--criterion",
        type=str,
        default="ce",
    )
    parser.add_argument(
        "--sce_alpha",
        type=float,
        default=0.1,
        help="Symmetric cross entropy loss: alpha * CE + beta * RCE",
    )
    parser.add_argument(
        "--sce_beta",
        type=float,
        default=1.0,
        help="Symmetric cross entropy loss: alpha * CE + beta * RCE",
    )
    parser.add_argument(
        "--loss_scale",
        type=float,
        default=1.0,
        help="scale parameter for loss, for example, scale * RCE, scale * NCE, scale * normalizer * RCE.",
    )
    parser.add_argument(
        "--gce_q",
        type=float,
        default=0.7,
        help="q parametor for Generalized-Cross-Entropy, Normalized-Generalized-Cross-Entropy.",
    )
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=None,
        help="alpha parameter for Focal loss and Normalzied Focal loss.",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=0.0,
        help="gamma parameter for Focal loss and Normalzied Focal loss.",
    )
    parser.add_argument(
        "--elr_beta",
        type=float,
        default=0.1,
        help="beta parameter for ELR",
    )
    parser.add_argument(
        "--elr_lamb",
        type=float,
        default=2,
        help="lamb parameter for ELR",
    )

    # ----Miscs options----
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument(
        "--resume",
        default=False,
        type=bool,
        help="Resume from previous checkpoint.",
    )
    parser.add_argument(
        "--tensorboard",
        default=True,
        type=bool,
        help="Use tensorboard to record training process.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        choices=["cuda", "cpu"],
        help="Device for training.",
    )
    parser.add_argument(
        "--device_id",
        default="0",
        type=str,
        help="GPU device id for training.",
    )

    parser.add_argument(
        "--goal",
        default="test",
        type=str,
        help="goal for this simulation.",
    )

    parser.add_argument(
        "--plot",
        default=True,
        type=bool,
        help="plot result or not.",
    )

    ##################################CTR
    parser.add_argument(
        "--ema",
        default=0.99,
        type=float,
    )
    parser.add_argument(
        "--tau",
        default=0.4,
        type=float,
        help="contrastive threshold(tau)",
    )
    parser.add_argument(
        "--lamb",
        default=90,
        type=float,
        help="lamb for contrastive learning.",
    )
    parser.add_argument(
        "--arch",
        default="resnet18",
        type=str,
        help="model for contrastive learning.",
    )


    ##########################################################################FedCorr

    parser.add_argument('--iteration1', type=int, default=5, help="enumerate iteration in preprocessing stage")
    parser.add_argument('--rounds1', type=int, default=200, help="rounds of training in fine_tuning stage")
    parser.add_argument('--rounds2', type=int, default=200, help="rounds of training in usual training stage")
    parser.add_argument('--frac1', type=float, default=1, help="fration of selected clients in preprocessing stage")
    parser.add_argument('--frac2', type=float, default=1, help="fration of selected clients in fine-tuning and usual training stage")
    parser.add_argument('--relabel_ratio', type=float, default=0.5, help="proportion of relabeled samples among selected noisy samples")
    parser.add_argument('--confidence_thres', type=float, default=0.5, help="threshold of model's confidence on each sample")
    parser.add_argument('--clean_set_thres', type=float, default=0.1, help="threshold of estimated noise level to filter 'clean' set used in fine-tuning stage")
    parser.add_argument('--fine_tuning', action='store_false', help='whether to include fine-tuning stage')
    parser.add_argument('--correction', action='store_false', help='whether to correct noisy labels')
    parser.add_argument('--beta_corr', type=float, default=0, help="coefficient for local proximal，0 for fedavg, 1 for fedprox, 5 for noise fl")
    parser.add_argument('--alpha_corr', type=float, default=1, help="0.1,1,5")
    parser.add_argument('--mixup', action='store_true')

    ##########################################################################FedNoRo
    parser.add_argument('--s1', type=int,  default=10, help='stage 1 rounds')
    parser.add_argument('--begin', type=int,  default=10, help='ramp up begin')
    parser.add_argument('--end', type=int,  default=49, help='ramp up end')
    parser.add_argument('--a_noro', type=float,  default=0.8, help='a')
    ##########################################################co-teaching
    #co-teaching
    parser.add_argument('--forget_rate', type=float, help='forget rate', default=None)
    parser.add_argument('--num_gradual', type=int, default=10,
                        help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
    parser.add_argument('--exponent', type=float, default=1,
                        help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
    parser.add_argument('--epoch_decay_start',type=int,default=80)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"


    centralized = "centralized" if args.centralized else "federated"
    globalize = "global" if args.globalize else "local"
    balanced = "balanced" if args.balance else "imbalance"

    if args.partition != 'iid':
        balanced = 'imbalance'


    if args.algorithm == "FedTBD":
        args.model = args.arch

    if args.globalize:
        unique = args.model + "_" + str(args.num_clients) + "_" +str(args.partition) + "_" + balanced + "_" + str(args.noise_type) + "_" + str(args.noise_ratio)+"_lr" + str(args.learning_rate)+"_bs" + str(args.batch_size)+"_ep" + str(args.local_epochs)+"_loss"+ str(args.criterion)+ "_" + args.goal
    else:
        unique = args.model + "_" + str(args.num_clients) + "_" +str(args.partition) + "_" + balanced + "_" + str(args.noise_type) + "_max" + str(args.max_noise_ratio) + "_min" + str(args.min_noise_ratio) + "_lr" + str(args.learning_rate) + "_bs" + str(args.batch_size) + "_ep" + str(args.local_epochs) + "_loss" + str(args.criterion) + "_" + args.goal
    if args.criterion == "elr":
        unique = unique + "_beta" + str(args.elr_beta) + "_lamb" + str(args.elr_lamb)
    elif args.criterion == "ctr":
        unique = unique + "_tau" + str(args.tau) + "_lamb" + str(args.lamb)
    elif args.criterion == "ctr_elr":
        unique = unique + "_elrbeta" + str(args.elr_beta) + "_elrlamb" + str(args.elr_lamb) + "_tau" + str(args.tau) + "_ctrlamb" + str(args.lamb)


    args.result_dir = os.path.join(args.result_dir, str(args.algorithm) + "_" + str(args.dataset) + "_" + centralized + "_" + globalize, unique)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    config_path = os.path.join(args.result_dir, "config.json")
    with open(config_path, "w") as f:
        ujson.dump(args.__dict__, f, indent=2)


    return args


def setup_seed(seed: int = 0):
    """
    Args:
        seed (int): random seed value.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_model(args):
    if "mnist" in args.dataset:
        input_channel = 1
    else:
        input_channel = 3
    num_classes = {'mnist': 10, 'fmnist': 10, 'cifar10': 10, 'cifar100': 100, 'clothing1M': 14,
                   'tinyimagenet': 200, 'cifar10NA':10, 'cifar10NW':10, 'cifar100N':100}
    output_channel = num_classes[args.dataset]

    ##################################################for CTR################################################
    if args.algorithm == "FedTBD":
        return SimSiam(output_channel,args.ema,args).to(args.device)
    if args.algorithm == "FedCTR":
        return SimSiam(output_channel,args.ema,args).to(args.device)
    #########################################################################################################
    # if args.algorithm == "FedLSR":
    #     if args.model == "resnet18":
    #         model = resnet18_lsr()
    #         in_feature = model.fc.in_features
    #         model.fc = nn.Linear(in_feature, output_channel)
    #         return model.to(args.device)
    #     elif args.model == "resnet34":
    #         model = resnet34_lsr()
    #         in_feature = model.fc.in_features
    #         model.fc = nn.Linear(in_feature, output_channel)
    #         return model.to(args.device)
    if args.model == "cnn9l":
        return CNN_9layer(input_channel, output_channel).to(args.device)
    elif args.model == "resnet18":
        if args.dataset in ['mnist', 'fmnist']:
            model = ResNet18_1c(output_channel)
            return model.to(args.device)
        model = torchvision.models.resnet18(pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, output_channel)
        return model.to(args.device)
    elif args.model == "resnet34":
        model = torchvision.models.resnet34(pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, output_channel)
        return model.to(args.device)
    elif args.model == "resnet50":
        base_model = torchvision.models.resnet50(pretrained=True)
        in_features = base_model.fc.in_features
        base_model.fc = nn.Linear(in_features, output_channel)
        base_model = base_model.to(args.device)
        return base_model
    else:
        raise NotImplementedError("Model {} is not implemented.".format(args.model))

def get_server(args):
    if args.algorithm == "FedAvg":
        server = FedAvg(args)
    elif args.algorithm == "FedAvgELR":
        server = FedAvgELR(args)
    elif args.algorithm == "FedTBD":
        server = FedTBD(args)
    elif args.algorithm == "FedCTR":
        server = FedCTR(args)
    elif args.algorithm == "FedTES":
        server = FedTES(args)
    elif args.algorithm == "FedCorr":
        server = FedCorr(args)
    elif args.algorithm == "FedNoRo":
        server = FedNoRo(args)
    elif args.algorithm == "FedLSR":
        server = FedLSR(args)
    elif args.algorithm == "FedCO":
        server = FedCO(args)
    elif args.algorithm == "FedSCE":
        server = FedSCE(args)
    else:
        raise NotImplementedError("Algorithm {} is not implemented.".format(args.algorithm))
    return server