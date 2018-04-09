from __future__ import print_function

from loaders.dataloader import FolderLoader as DataLoader
from model.mobilenetv2 import MobileNetV2
from model.criterion import Criterion
from model.optim import Optimizer
from train.trainer import Trainer
from manage.manager import Manager
import torch

def main():
    manager = Manager.init()

    models = [
        ["model", MobileNetV2(**manager.args.model)]
    ]

    manager.init_model(models)
    args = manager.args
    criterion = Criterion()
    optimizer, scheduler = Optimizer(models, args.optim).init()

    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda:
        for item in models:
            item[1].cuda()
        criterion.cuda()

    dataloader = DataLoader(args.dataloader, args.cuda)

    summary = manager.init_summary()
    trainer = Trainer(models, criterion, optimizer, scheduler, dataloader, summary, args.cuda)

    for epoch in range(args.runtime.start_epoch, args.runtime.num_epochs+args.runtime.start_epoch):
        try:
            print("epoch {}...".format(epoch))
            trainer.train(epoch)
            manager.save_checkpoint(models, epoch)

            if (epoch + 1) % args.runtime.test_every == 0:
                trainer.validate()
        except KeyboardInterrupt:
            print("Training had been Interrupted\n")
            break
    trainer.test()

if __name__ == "__main__":
    main()
