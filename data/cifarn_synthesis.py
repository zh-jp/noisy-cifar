from torchvision.datasets import CIFAR10
from PIL import Image
from .utils_noisy import (noisify_multiclass_symmetric, noisify_cifar10_asymmetric,
                          noisify_cifar100_asymmetric)


class CIFAR10N(CIFAR10):
    num_classes = 10

    def __init__(self, root='~/data', train=True, transform=None, target_transform=None,
                 download=False, noise_rate=0.0, sym=True, random_seed=0):
        super().__init__(root, train, transform, target_transform, download)
        assert noise_rate > 0., "No 'noise_rate' is provided."
        if self.num_classes != 10:
            return
        if sym:
            self.noisy_targets, self.actual_noise = (
                noisify_multiclass_symmetric(self.targets, noise_rate, self.num_classes))
        else:
            self.noisy_targets, self.actual_noise = (
                noisify_cifar10_asymmetric(self.targets, noise_rate, random_seed)
            )

        self.noisy_targets = self.noisy_targets.tolist()

    def __getitem__(self, index):
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

    def __init__(self, root='~/data', train=True, transform=None, target_transform=None, download=False,
                 noise_rate=0., sym=True, random_seed=0):
        super().__init__(root, train, transform, target_transform, download, noise_rate, sym)
        if sym:
            self.noisy_targets, self.actual_noise = (
                noisify_multiclass_symmetric(self.targets, noise_rate, self.num_classes))
        else:
            """mistakes are inside the same superclass of 5 classes, e.g. 'fish' """
            self.noisy_targets, self.actual_noise = (
                noisify_cifar100_asymmetric(self.targets, noise_rate, random_seed)
            )
        self.noisy_targets = self.noisy_targets.tolist()
