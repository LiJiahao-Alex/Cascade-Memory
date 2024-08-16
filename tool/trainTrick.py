import numpy
import torch


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='data/checkpoint.pt', trace_func=print, rho=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.min_val = None
        self.stopFlag = False
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.val_loss_min = numpy.Inf
        self.rho = rho

    def __call__(self, val_loss, model):
        if self.min_val is None:
            self.min_val = val_loss
            self.save_checkpoint(val_loss, model)

        elif val_loss > self.min_val + self.delta:
            self.counter += 1
            self.counter_counter = 0
            self.trace_func(f'[EarlyStop] EarlyStopping counter: ({self.counter}/{self.patience})')
            if self.counter >= self.patience:
                self.stopFlag = True

        else:
            if self.min_val - val_loss < self.rho:
                self.counter += 1
                self.counter_counter = 0
                self.trace_func(f'[EarlyStop] EarlyStopping counter: ({self.counter}/{self.patience})')
                if self.counter >= self.patience:
                    self.stopFlag = True
            else:
                self.save_checkpoint(val_loss, model)
                self.min_val = val_loss
                self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(
                f'[EarlyStop] Val loss decreased ({self.val_loss_min:.9f} --> {val_loss:.9f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


if __name__ == '__main__':
    early_stopper = EarlyStopping()
