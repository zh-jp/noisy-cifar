import torch
import numpy as np
from torchvision.datasets import CIFAR10
from PIL import Image


class CIFAR10N(CIFAR10):
    num_classes = 10

    def __init__(self, root='~/data', train=True,
                 transform=None, target_transform=None,
                 download=False, noise_type=None, noise_path=None):
        super().__init__(root, train, transform, target_transform, download)
        self.noise_type = noise_type
        self.noise_path = noise_path

        idx_each_class_noisy = [[] for _ in range(self.num_classes)]

        if noise_type != 'clean':
            # load human noisy labels
            noisy_targets = self.load_label()
            self.noisy_targets = noisy_targets.tolist()
            print(f'Noisy labels loaded from {noise_path}')

            trans_mat = np.zeros((self.num_classes, self.num_classes))
            for i in range(len(noisy_targets)):
                trans_mat[self.targets[i], noisy_targets[i]] += 1
            trans_mat = trans_mat / np.sum(trans_mat, axis=1)
            with np.printoptions(formatter={'float': '{:0.3f},'.format}):
                print(f'Noise transition matrix(Shape: {trans_mat.shape}):\n{trans_mat}')

            for i in self.noisy_targets:
                idx_each_class_noisy[i].append(i)
            class_size_noisy = [len(i) for i in idx_each_class_noisy]
            self.noise_prior = np.array(class_size_noisy) / sum(class_size_noisy)
            print(f'Noise data prior per class: {[round(i, 3) for i in self.noise_prior]}')
            self.noise_or_not = np.array(noisy_targets) != np.array(self.targets)
            self.noisy_rate = np.sum(self.noise_or_not) / len(self.targets)
            print(f'Noisy rate: {self.noisy_rate:.4%}')

    def load_label(self):
        # NOTE: only load manual training label
        noisy_label = torch.load(self.noise_path)
        assert isinstance(noisy_label, dict), 'noisy_label should be dict'
        if 'clean_label' in noisy_label.keys():
            clean_label = torch.tensor(noisy_label['clean_label'])
            assert torch.sum(torch.tensor(self.targets) - clean_label) == 0, 'clean label is not correct'
            print(f'Loaded {self.noise_type} from {self.noise_path}')
            noise_rate = 1 - np.mean(clean_label.numpy() == noisy_label[self.noise_type])
            print(f'The overall noise rate is {noise_rate:.2%}')
        return noisy_label[self.noise_type].reshape(-1)

    def __getitem__(self, index):
        if self.noise_type == 'clean':
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.noisy_targets[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100N(CIFAR10N):
    num_classes = 100
    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }
