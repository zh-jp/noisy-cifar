



# Examples
## Noisy from Human
| Noise Type    | aggre | rand1 | rand2 | rand3 | worst | noisy100 |
|---------------|-------|-------|-------|-------|-------|----------|
| Noise Rate(%) | 9.03  | 17.23 | 18.12 | 17.64 | 40.21 | 40.2     |

Type noisy100 belongs to cifar100, and other types belong to cifar10.
```bash
python main.py --dataset cifar10 --human --noise_type aggre
```
```bash
python main.py --dataset cifar100 --human --noise_type noisy100
```
## Noisy from Synthesis
- `sym` to generate symmetric noise, otherwise asymmetric noise.
- `noise_rate` is the noise rate of the dataset.
```bash
python main.py --dataset cifar10 --sym --noise_rate 0.5
```
```bash
python main.py --dataset cifar100 --sym --noise_rate 0.5
```

For asymmetric noise:
- cifar10: the noisy samples are flipped in some classes i.e. automobile < - truck, bird -> airplane, cat <-> dog, deer -> horse
- cifar100: the noisy samples are flipped in same superclass.

```bash
python main.py --dataset cifar10 --noise_rate 0.5
```

```bash
python main.py --dataset cifar100 --noise_rate 0.5
```


# Others
- Under the same resnet structure, the resnet provided by torchvision will reduce performance compared to the custom resnet. ~~(Why?)~~
- Issue is welcomed.

# Reference
1. [cifar-10-100n](https://github.com/UCSC-REAL/cifar-10-100n)
2. [SCELoss-Reproduce](https://github.com/HanxunH/SCELoss-Reproduce/)