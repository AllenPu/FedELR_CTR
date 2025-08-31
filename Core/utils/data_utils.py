import os
import torch



def get_dataset(path,idx):
    dataset_path = os.path.join(path, str(idx) + '.pkl')
    return torch.load(dataset_path)

def del_dataset(path,idx):
    dataset_path = os.path.join(path, str(idx) + '.pkl')
    os.remove(dataset_path)

def save_dataset(path,idx,dataset):
    dataset_path = os.path.join(path, str(idx) + '.pkl')
    torch.save(dataset,dataset_path)

def get_dataloder(path,idx,batch_size,isTrain):
    dataset_path = os.path.join(path, str(idx) + '.pkl')
    dataset = torch.load(dataset_path)
    if isTrain:
        return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,drop_last=True, shuffle=True)
    else:
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=False, shuffle=False)


def get_global_testloader(path,batch_size):
    dataset_path = os.path.join(path, 'test_dataset.pkl')
    dataset = torch.load(dataset_path)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=False, shuffle=False)

def get_global_train_dataset(path,):
    dataset_path = os.path.join(path, 'train_dataset.pkl')
    dataset = torch.load(dataset_path)
    return dataset

def get_global_trainloader(path,batch_size):
    dataset_path = os.path.join(path, 'train_dataset.pkl')
    dataset = torch.load(dataset_path)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, drop_last=False, shuffle=False)

def count_client_data(path,idx):
    dataset_path = os.path.join(path, str(idx) + '.pkl')
    dataset = torch.load(dataset_path)
    return len(dataset)