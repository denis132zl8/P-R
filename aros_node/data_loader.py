import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, Subset, SubsetRandomSampler, TensorDataset, ConcatDataset
from torchvision import datasets, transforms, models
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, SVHN, FashionMNIST

# 1
transform_euro = transforms.Compose([
    transforms.ToTensor(),

])


class LabelChangedDataset(Dataset):
    def __init__(self, original_dataset, new_label):
        self.original_dataset = original_dataset
        self.new_label = new_label

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, _ = self.original_dataset[idx]
        return image, self.new_label


def get_subsampled_subset(dataset, subset_ratio=0.1):
    subset_size = int(len(dataset) * subset_ratio)
    remaining_size = len(dataset) - subset_size

    subset_indices, _ = torch.utils.data.random_split(dataset, [subset_size, remaining_size])
    subset_testset = Subset(dataset, subset_indices.indices)

    return subset_testset


'''
transform_tensor = transforms.Compose([ transforms.ToTensor()])


trainset_CIFAR10 = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_tensor)


testset_CIFAR10 = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_tensor)


trainloader_CIFAR10 = DataLoader(trainset_CIFAR10, batch_size=65, shuffle=True, num_workers=2)

testloader_CIFAR10 = DataLoader(testset_CIFAR10, batch_size=16, shuffle=False, num_workers=2)




trainset_CIFAR100 = torchvision.datasets.CIFAR100(
    root='./data', train=True, download=True, transform=transform_tensor)

testset_CIFAR100 = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_tensor)



trainloader_CIFAR100 = DataLoader(trainset_CIFAR100, batch_size=64, shuffle=True, num_workers=2)

testloader_CIFAR100 = DataLoader(testset_CIFAR100, batch_size=16, shuffle=False, num_workers=2)







testset_CIFAR10_relabled = LabelChangedDataset(testset_CIFAR10, new_label=100)
testset_CIFAR100_relabled = LabelChangedDataset(testset_CIFAR100, new_label=10)


testloader_CIFAR10_vs_CIFAR100 = DataLoader(ConcatDataset([testset_CIFAR10, testset_CIFAR100_relabled]), shuffle=False, batch_size=8)
testloader_CIFAR100_vs_CIFAR10 = DataLoader(ConcatDataset([testset_CIFAR100, testset_CIFAR10_relabled]), shuffle=False, batch_size=8)
'''


def get_loaders(in_dataset='CIFAR10'):
    # 2
    if in_dataset == 'eurosat':
        # Завантажуємо повний датасет
        full_dataset = datasets.ImageFolder(root='./data/eurosat', transform=transform_euro)

        # Розбиваємо на train/test (80/20)
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_set, test_set = torch.utils.data.random_split(full_dataset, [train_size, test_size])

        trainloader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
        testloader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)
        transform_euro1 = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        ood_dataset = torchvision.datasets.SVHN(
            root='./data',
            split='test',
            download=True,
            transform=transform_euro1
        )

        ood_set = LabelChangedDataset(
            ood_dataset,
            new_label=10
        )

        id_ood_loader = DataLoader(
            ConcatDataset([test_set, ood_set]),
            shuffle=False,
            batch_size=128
        )

        return trainloader, testloader, test_set, id_ood_loader
    '''if in_dataset == 'cifar10':
        return trainloader_CIFAR10, testloader_CIFAR10,testset_CIFAR10, testloader_CIFAR10_vs_CIFAR100
    if in_dataset == 'cifar100':
        return trainloader_CIFAR100, testloader_CIFAR100,testset_CIFAR100, testloader_CIFAR100_vs_CIFAR10
    else:
        raise ValueError(f"Dataset '{in_dataset}' is not supported.")'''





