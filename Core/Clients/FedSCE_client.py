import time
from Core.Clients.Client_base import Client
from Core.utils.criteria import get_criterion
class FedSCEClient(Client):

    def __init__(self, args, id, train_samples, test_samples):
        super().__init__(args, id, train_samples, test_samples)
        self.criterion = get_criterion('sce', self.num_classes, train_samples, args, True)