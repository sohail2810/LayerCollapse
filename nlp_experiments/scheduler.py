from typing import Any, Dict
import torch
from torch.optim.lr_scheduler import LRScheduler
import warnings
import LiveTune as lt

class LiveTuneGammaLR(LRScheduler):
    """
    Implements LiveTune learning rate scheduler.
    """

    def __init__(self, optimizer, live_gamma, last_epoch=-1, verbose=False):
        self.live_gamma = live_gamma 
        self.gamma = live_gamma()
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        if self.live_gamma.changed():
            old_gamma = self.gamma
            self.gamma = self.live_gamma()
            return [group['lr'] / old_gamma * self.gamma
                    for group in self.optimizer.param_groups]
        return [group['lr'] for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        return [base_lr * self.gamma
                for base_lr in self.base_lrs]
    
    def state_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer' and key != 'live_gamma'}
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)
        
