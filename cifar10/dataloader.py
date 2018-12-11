import os

import torch
import torchvision
import torchvision.transforms as transforms

def get_data_loaders(args):
    if args.trainloader and args.testloader:
        assert os.path.exists(args.trainloader), 'trainloader does not exist'
        assert os.path.exists(args.testloader), 'testloader does not exist'
        trainloader = torch.load(args.trainloader)
        testloader = torch.load(args.testloader)
        return trainloader, testloader

    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    if args.raw_data:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        if not args.noaug:
            # with data augmentation
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            # no data agumentation
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])


    kwargs = {'num_workers': 2, 'pin_memory': True} if args.ngpu else {}
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                            transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                           transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, **kwargs)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, **kwargs)

    return trainloader, testloader
