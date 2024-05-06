from . import cifarn_human
from . import cifarn_synthesis

from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100

root = '~/data/'


def get_transform(dataset):
    if dataset == 'cifar10':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif dataset == 'cifar100':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    else:
        assert False, f'Undefined dataset {dataset}'
    return train_transform, test_transform


def human_noisy_cifar(args, train_transform):
    noise_type_map = {'clean': 'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label',
                      'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3',
                      'clean100': 'clean_label', 'noisy100': 'noisy_label'}
    dataset = args.dataset
    args.noise_type = noise_type_map[args.noise_type]
    print(f'Human noise type: {args.noise_type}')
    if dataset == 'cifar10':
        noise_path = './data/CIFAR-10_human.pt'
        train_dataset = cifarn_human.CIFAR10N(root, train=True, transform=train_transform, download=True,
                                              noise_type=args.noise_type, noise_path=noise_path)

    else:
        noise_path = './data/CIFAR-100_human.pt'
        train_dataset = cifarn_human.CIFAR100N(root, train=True, transform=train_transform, download=True,
                                               noise_type=args.noise_type, noise_path=noise_path)
    return train_dataset


def synthesis_cifar_noise(args, train_transform):
    print(f'Synthetic noise rate: {args.noise_rate}, Symmetric: {args.sym}')
    dataset = args.dataset
    if dataset == 'cifar10':

        train_dataset = cifarn_synthesis.CIFAR10N(root, train=True, transform=train_transform,
                                                  download=True, noise_rate=args.noise_rate,
                                                  sym=args.sym, random_seed=args.seed)
    else:
        train_dataset = cifarn_synthesis.CIFAR100N(root, train=True, transform=train_transform,
                                                   download=True, noise_rate=args.noise_rate,
                                                   sym=args.sym, random_seed=args.seed)
    return train_dataset


def get_dataloader(args):
    dataset = args.dataset
    train_transform, test_transform = get_transform(dataset)
    # load transform
    if dataset == 'cifar10':
        test_dataset = CIFAR10(root=root, train=False, transform=test_transform)
    else:
        test_dataset = CIFAR100(root=root, train=False, transform=test_transform)

    if args.human:  # args.noise_type is required
        train_dataset = human_noisy_cifar(args, train_transform)

    else:
        train_dataset = synthesis_cifar_noise(args, train_transform)

    num_classes = train_dataset.num_classes
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                              pin_memory=True, shuffle=True)

    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                             pin_memory=True, shuffle=False)

    return train_loader, test_loader, num_classes
