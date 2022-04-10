import torch
import torchvision

if __name__ == "__main__":
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=None)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                              shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=None)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=2)