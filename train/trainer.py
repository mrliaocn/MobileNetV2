from tqdm import tqdm

import torch
from torch.autograd import Variable

class Trainer:
    def __init__(self, models, criterion, optimizer, scheduler, dataloader, summary, cuda):
        print("Training...")
        self.models = models
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = dataloader.train_loader
        self.test_loader = dataloader.test_loader
        self.summary = summary
        self.cuda = cuda
        self.epoch = 0

    def train(self, epoch):
        self.epoch = epoch
        loss_list = []

        self.scheduler.step()
        for _, (data, label) in enumerate(tqdm(self.train_loader)):
            if self.cuda:
                data = data.cuda()
                label = label.cuda()
            data = Variable(data)
            label = Variable(label)

            self.optimizer.zero_grad()
            output = data
            for item in self.models:
                model = item[1]
                output = model(output)

            loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()
            loss_list.append(loss.data[0])

        ave_loss = sum(loss_list)/len(loss_list)
        cur_lr = self.scheduler.get_lr()[0]
        print("epoch {}: - loss: {}".format(epoch, ave_loss))
        print('learning rate:', cur_lr)

        self.summary.add_scalar('training/loss', ave_loss, epoch)
        self.summary.add_scalar('training/learning_rate', cur_lr, epoch)

    def validate(self):
        print('Validating...')
        for item in self.models:
            item[1].eval()
        pred = torch.ByteTensor([])
        for i, (data, target) in enumerate(tqdm(self.test_loader)):
            if self.cuda:
                data = data.cuda()
                target = target.cuda()
            data = Variable(data, volatile=True)

            output = data
            for item in self.models:
                model = item[1]
                output = model(output)

            temp = self.accuracy(output.data, target)
            pred = torch.cat((pred, temp), 0)

        total = pred.view(-1).size()[0]
        acc = pred.sum() / total

        print('\t\t====> Accuracy on test set : {:.4f}'.format(acc))
        self.summary.add_scalar('validate/accuracy', acc, self.epoch)
        for item in self.models:
            item[1].train()

    def test(self):
        print('Testing...')
        for item in self.models:
            item[1].eval()
        pred = torch.ByteTensor([])
        for i, (data, target) in enumerate(tqdm(self.train_loader)):
            if self.cuda:
                data = data.cuda()
                target = target.cuda()
            data = Variable(data, volatile=True)

            output = data
            for item in self.models:
                model = item[1]
                output = model(output)

            temp = self.accuracy(output.data, target)
            pred = torch.cat((pred, temp), 0)

        total = pred.view(-1).size()[0]
        acc = pred.sum() / total

        print('\t\t====> Accuracy on train set : {:.4f}'.format(acc))
        for item in self.models:
            item[1].train()

    def accuracy(self, output, target):
        _, pred = output.topk(1, 1)
        correct = pred.float().view(-1).eq(target.float().view(-1)).view(-1)
        return correct.cpu()
