import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'], default='cifar10')

    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--n_epoch', type=int, default=80)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--human', action='store_true', default=False)
    parser.add_argument('--backbone', type=str, default='resnet18')
    parser.add_argument('--criterion', type=str, default='CE')
    parser.add_argument('--torchvision', action='store_true', default=False)

    '''Noisy label arguments'''
    # --human is used , noise_type should be set:
    parser.add_argument('--noise_type', type=str,
                        choices=['clean', 'worst', 'aggre', 'rand1', 'rand2', 'rand3', 'clean100', 'noisy100'],
                        default='aggre')

    # else noise_rate  and asym should be set:
    parser.add_argument('--noise_rate', type=float, default=0.4)
    parser.add_argument('--sym', action='store_true', default=False)  # Default is symmetric noise


    return parser.parse_args()
