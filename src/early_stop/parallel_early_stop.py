import numpy as np
from accelerate import Accelerator


class ParallelEarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, out_dir='./', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.out_dir = out_dir
        self.trace_func = trace_func

    def __call__(self, val_loss, accelerator: Accelerator):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, accelerator)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, accelerator)
            self.counter = 0

    def save_checkpoint(self, val_loss, accelerator: Accelerator):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            accelerator.print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss
        accelerator.save_state(self.out_dir)