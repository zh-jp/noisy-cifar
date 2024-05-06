import numpy as np
from numpy.testing import assert_array_almost_equal


def build_for_cifar100(size: int, noise: float):
    """
    random flip between two random classes.
    returns a matrix with (1 - corruption_prob) on the diagonals, and corruption_prob
    concentrated in only one other entry for each row
    """

    assert 0. <= noise < 1.

    prob = (1. - noise) * np.eye(size)
    for i in range(size - 1):
        prob[i, i + 1] = noise

    # adjust last row
    prob[size - 1, 0] = noise

    assert_array_almost_equal(prob.sum(axis=1), 1, 1)
    return prob


def multiclass_noisify(y, prob, random_seed=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """

    assert prob.shape[0] == prob.shape[1]
    assert np.max(y) < prob.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(prob.sum(axis=1), np.ones(prob.shape[1]))
    assert (prob >= 0.0).all()

    new_y = y.copy()
    flipper = np.random.RandomState(random_seed)

    for idx in range(len(y)):
        i = y[idx]
        # draw a vector with only an 1, Shape: (1, len(prob[i]))
        flipped = flipper.multinomial(1, prob[i], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


def other_class(num_classes, current_class):
    """get an others class index except current class"""
    assert 0 <= current_class < num_classes, "class_ind must be within the range (0, num_classes - 1)"

    other_class_list = list(range(num_classes))
    other_class_list.remove(current_class)
    return np.random.choice(other_class_list)


def noisify_multiclass_symmetric(y, noisy_rate, num_classes):
    n_samples = len(y)
    n_noisy = int(noisy_rate * n_samples)
    print(f'The number of symmetric noisy samples: {n_noisy}, num_classes: {num_classes}')
    y = np.array(y)
    cls_idx = [np.where(y == i)[0] for i in range(num_classes)]
    n_cls_noisy = [int(len(i) * noisy_rate) for i in cls_idx]
    noisy_idx = []
    for i in range(num_classes):
        noisy_cls_idx = np.random.choice(cls_idx[i], n_cls_noisy[i], replace=False)
        noisy_idx.extend(noisy_cls_idx)

    new_y = y.copy()
    for i in noisy_idx:
        new_y[i] = other_class(num_classes=num_classes, current_class=y[i])

    print("Print symmetric noisy label generation statistics:")
    for i in range(num_classes):
        n_noisy = np.sum(new_y == i)
        print(f"Noisy class {i}, has {n_noisy} samples.")
        if i > 15:
            print("...")
            break

    actual_noise = (new_y != y).mean()
    assert actual_noise > 0.
    print(f'Symmetric actual noise rate: {actual_noise:.3f}')
    return new_y, actual_noise


def noisify_cifar10_asymmetric(y, noise_rate: float, random_seed: int):
    np.random.seed(random_seed)

    # automobile < - truck, bird -> airplane, cat <-> dog, deer -> horse
    source_class = [9, 2, 3, 5, 4]
    target_class = [1, 0, 5, 3, 7]
    y = np.array(y)
    new_y = y.copy()
    for s, t in zip(source_class, target_class):
        cls_idx = np.where(y == s)[0]
        n_noisy = int(noise_rate * len(cls_idx))
        noisy_sample_index = np.random.choice(cls_idx, n_noisy, replace=False)
        for i in noisy_sample_index:
            new_y[i] = t

    actual_noise = (new_y != y).mean()
    assert actual_noise > 0.
    print(f'Asymmetric actual noise rate: {actual_noise:.3f}(Only half of the classes are affected by noise)')
    return new_y, actual_noise


def noisify_cifar100_asymmetric(y, noise_rate: float, random_seed: int):
    """mistakes are inside the same superclass of 5 classes, e.g. 'fish' """
    num_classes = 100
    num_superclasses = 20
    num_subclasses = 5
    prob = np.eye(num_classes)
    for i in range(num_superclasses):
        init, end = i * num_subclasses, (i + 1) * num_subclasses
        prob[init:end, init:end] = build_for_cifar100(num_subclasses, noise_rate)

    y = np.array(y)
    new_y = multiclass_noisify(y, prob=prob, random_seed=random_seed)
    actual_noise = (new_y != y).mean()
    assert actual_noise > 0.0
    print(f'Asymmetric noise rate: {actual_noise:.3f}')
    return new_y, actual_noise
