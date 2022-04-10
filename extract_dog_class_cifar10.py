import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

if __name__ == "__main__":
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                              shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, num_workers=2)

    for data, label in trainloader:
        if label.item() == 5:
            print(data.numpy().shape)
            plt.imshow(data[0].numpy().T)
            plt.show()
            print(label)