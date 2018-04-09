from torch import optim
from torch.optim import lr_scheduler

class Optimizer(object):
    def __init__(self, models, optim_args):
        super(Optimizer, self).__init__()
        self.params = []
        for item in models:
            self.params.append({
                'params': item[1].parameters()
            })
        self.args = optim_args

    def __adam(self, lr, weight_decay=0.00004, **kw):
        return optim.Adam(self.params, lr, weight_decay=weight_decay)


    def __rms_prop(self, lr, alpha=0.9, weight_decay=0.00004, momentum=0.9, **kw):
        return optim.RMSprop(self.params, lr, alpha, weight_decay=weight_decay, momentum=momentum)

    def __scheduler(self, optimizer, last_epoch=0, lr_decay=0.99, **kw):
        last_epoch = last_epoch - 1
        return lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay, last_epoch=last_epoch)

    def init(self):
        if self.args.optim == 'adam':
            optimizer = self.__adam(**self.args)
        elif self.args.optim == 'rms_prop':
            optimizer = self.__rms_prop(**self.args)
        else:
            raise ValueError
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

        scheduler = self.__scheduler(optimizer, **self.args)
        return optimizer, scheduler
