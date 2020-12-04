from torch.optim.lr_scheduler import _LRScheduler, StepLR

class MultiPolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=[0.9,0.9,0.95], last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        super(MultiPolyLR, self).__init__(optimizer, last_epoch)
        print("base lrs : ",self.base_lrs)
    
    def get_lr(self):
        return [ max( self.base_lrs[i] * ( 1 - self.last_epoch/self.max_iters )**self.power[i], self.min_lr)
                for i in range(len(self.base_lrs))]
