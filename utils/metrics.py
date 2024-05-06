from torch import Tensor


class AverageMeter:

    def __init__(self, name, fmt=':f'):
        self.count, self.sum, self.val, self.avg = 0, 0, 0, 0
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: Tensor, target: Tensor, topk=(1,)) -> list:
    maxk = max(topk)
    batch_size = target.size(0)

    # Get the descending order of the top k probabilities
    _, pred = output.topk(maxk, dim=1)  # Shape: [batch_size, maxk]
    pred = pred.t()  # Shape: [maxk, batch_size]
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # Shape: [maxk, batch_size]

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
