import math
import torch
from torch.optim.optimizer import Optimizer, required

class CustomAdamW(Optimizer):
    def __init__(self, model, lr=required, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, gp_weight=0.0):
        params = model.parameters()
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, gp_weight=gp_weight)
        super(CustomAdamW, self).__init__(params, defaults)
        self.param_to_name = {param: name for name, param in model.named_parameters()}
        self.gp_weight = gp_weight

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                
                p_name = self.param_to_name.get(p)

                # if p_name includes the word bypass, then we want to bypass the gradient
                use_gp = "act" in p_name

                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')

                amsgrad = group['amsgrad']
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0 and not use_gp:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                    # grad.add_(group['weight_decay'], p.data)

                if use_gp:
                    p.data.add_(-2 * (p.data-1), alpha=group['lr'] * group['gp_weight'])
                # Decay the first and second moment running average coefficient
                # exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)


                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                if amsgrad:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                # p.data.addcdiv_(-step_size, exp_avg, denom)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)


        return loss

class CustomAdamW_GPT2(Optimizer):
    def __init__(self, model, lr=required, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, gp_weight=0.0):
        params = model.parameters()
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, gp_weight=gp_weight)
        super(CustomAdamW_GPT2, self).__init__(params, defaults)
        self.param_to_name = {param: name for name, param in model.named_parameters()}
        self.gp_weight = gp_weight

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                
                p_name = self.param_to_name.get(p)

                # if p_name includes the word bypass, then we want to bypass the gradient
                use_gp = "act_LC" in p_name

                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')

                amsgrad = group['amsgrad']
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0 and not use_gp:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                    # grad.add_(group['weight_decay'], p.data)

                if use_gp:
                    p.data.add_(-2 * (p.data-1), alpha=group['lr'] * group['gp_weight'])
                # Decay the first and second moment running average coefficient
                # exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)


                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                if amsgrad:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                # p.data.addcdiv_(-step_size, exp_avg, denom)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)


        return loss
