import time
import os
from torch.cuda.amp import GradScaler
import torch

from utils.logger import get_logger
from data import cifar_dataloader
import models
import torchvision
from utils.metrics import accuracy, AverageMeter
from utils.args import get_args
from torch import nn


# Adjust learning rate and for SGD Optimizer
def adjust_learning_rate(optimizer, epoch, alpha_plan):
    for param_group in optimizer.param_groups:
        param_group['lr'] = alpha_plan[epoch]


# Train the Model
def train(epoch, dataloader, model, optimizer, criterion, scaler, logger):
    batch_cal_time = AverageMeter('Time', ':6.2f')
    data_load_time = AverageMeter('Data', ':6.2f')
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    end = time.time()
    model.train()
    for i, (images, labels) in enumerate(dataloader):
        data_load_time.update(time.time() - end)
        end = time.time()

        images = images.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        # Forward + Backward + Optimize
        with torch.autocast(device_type="cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        acc1, acc5 = accuracy(logits.cpu(), labels.cpu(), topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        losses.update(loss.item(), images.size(0))
        batch_cal_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            logger.info(f"Epoch[{epoch}][{i}/{len(dataloader)}] "
                        f"lr: {optimizer.param_groups[-1]['lr']: .5f}\t"
                        f"Data load time(s) {data_load_time.val:.3f} ({data_load_time.avg:.3f})\t"
                        f"Batch cal time(s) {batch_cal_time.val:.3f} ({batch_cal_time.avg:.3f})\t"
                        f"Loss {losses.val:.3f} ({losses.avg:.3f})\t"
                        f"Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                        f"Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t")
    logger.info(f'Train Summary\t'
                f'Time cost {batch_cal_time.sum:.3f}s\t'
                f'Loss {losses.avg:.3f}\t'
                f'Acc@1 {top1.avg:.3f}\t'
                f'Acc@5 {top5.avg:.3f}\t')


best_acc1 = 0


# Evaluate the Model
def evaluate(dataloader, model, logger):
    model.eval()
    global best_acc1
    end = time.time()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    for images, labels in dataloader:
        images = images.cuda()
        with torch.no_grad():
            logits = model(images)
        acc1, acc5 = accuracy(logits.cpu(), labels, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

    if top1.avg > best_acc1:
        best_acc1 = top1.avg
    end = time.time() - end
    logger.info(f'Evaluate Time cost {end:.3f}s\tAcc@1 {top1.avg:.3f}(Best: {best_acc1:.3f})')


def main_workers(args):
    time_cost = time.time()
    logger = get_logger(os.path.join('./log', name))
    for k, v in vars(args).items():
        logger.info(f'{k}: {v}')

    train_loader, test_loader, num_classes = cifar_dataloader.get_dataloader(args)

    # load model
    logger.info(f'building model({args.backbone})...')
    if args.torchvision:
        model = getattr(torchvision.models.resnet, args.backbone)(num_classes=num_classes)
        logger.warning('using torchvision model')
    else:
        model = getattr(models.resnet, args.backbone)(num_classes=num_classes)
        logger.warning('using custom model')
    logger.info('building model done')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    alpha_plan = [0.1] * 60 + [0.01] * (args.n_epoch - 60)
    model.cuda()

    # training
    for epoch in range(args.n_epoch):
        epoch_end = time.time()
        # train models
        adjust_learning_rate(optimizer, epoch, alpha_plan)
        train(epoch, train_loader, model, optimizer, criterion, scaler, logger)
        # evaluate models
        evaluate(test_loader, model, logger)

        logger.info(f'Epoch {epoch} Time cost {time.time() - epoch_end:.3f}s')
    time_cost = time.time() - time_cost
    logger.info(f'Time cost: {time_cost / 3600:.2f}h')


if __name__ == '__main__':

    args = get_args()
    # Seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    if os.path.exists('./log') is False:
        os.makedirs('./log')

    if args.human:
        name = f'{args.dataset}-human-{args.noise_type}'
    else:
        name = f'{args.dataset}-synthetic-{args.noise_rate}'
        if args.sym:
            name += '-sym'
        else:
            name += '-asym'
    if os.path.exists(os.path.join('./log', name)) is False:
        os.makedirs(os.path.join('./log', name))

    main_workers(args)
