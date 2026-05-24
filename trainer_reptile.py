import numpy as np
import torch
import torch.nn.functional as F

from attack.pgd_attack import PgdAttack
from attack.pgd_attack_restart import attack_pgd_restart
from utils.context import ctx_noparamgrad
from utils.math_utils import l2_norm_batch as l2b
from collections import deque
import copy

# Legacy mode names from the original experiment logs:
# cfast_at      = RPG + Fast-AT
# cfast_at_ga   = RPG + Fast-AT-GA
# cfast_bat     = RPG + Fast-BAT
# cpgd          = RPG + PGD-AT

def _attack_loss(predictions, labels):
    return -torch.nn.CrossEntropyLoss(reduction='sum')(predictions, labels)

def load(self, backup):
    for p_from, p_to in zip(backup.parameters(), self.parameters()):
        p_to.data = p_from.data.clone()

def clamp(input, min=None, max=None):
    ndim = input.ndimension()
    if min is None:
        pass
    elif isinstance(min, (float, int)):
        input = torch.clamp(input, min=min)
    elif isinstance(min, torch.Tensor):
        if min.ndimension() == ndim - 1 and min.shape == input.shape[1:]:
            input = torch.max(input, min.view(1, *min.shape))
        else:
            assert min.shape == input.shape
            input = torch.max(input, min)
    else:
        raise ValueError("min can only be None | float | torch.Tensor")

    if max is None:
        pass
    elif isinstance(max, (float, int)):
        input = torch.clamp(input, max=max)
    elif isinstance(max, torch.Tensor):
        if max.ndimension() == ndim - 1 and max.shape == input.shape[1:]:
            input = torch.min(input, max.view(1, *max.shape))
        else:
            assert max.shape == input.shape
            input = torch.min(input, max)
    else:
        raise ValueError("max can only be None | float | torch.Tensor")
    return input

class BatTrainer:
    def __init__(self, args, log):
        self.args = args
        self.steps = args.attack_step
        self.eps = args.attack_eps
        self.attack_lr = args.attack_lr
        self.attack_rs = args.attack_rs
        if self.args.lmbda != 0.0:
            self.lmbda = self.args.lmbda
        else:
            self.lmbda = 1. / self.attack_lr

        self.constraint_type = np.inf
        self.log = log
        self.mode = args.mode
        self.z_init_non_sign_attack_lr = 5000. / 255

    def test_sa(self, model, data, labels):
        model.eval()
        with torch.no_grad():
            predictions_sa = model(data)
            correct = (torch.argmax(predictions_sa.data, 1) == labels).sum().cpu().item()
        return correct

    def get_input_grad(self, model, X, y, eps, delta_init='none', backprop=True):

        if delta_init == 'none':
            delta = torch.zeros_like(X, requires_grad=True)
        elif delta_init == 'random_uniform':
            delta = torch.zeros_like(X).uniform_(-eps, eps).requires_grad_(True)
        elif delta_init == 'random_corner':
            delta = torch.zeros_like(X).uniform_(-eps, eps).requires_grad_(True)
            delta = eps * torch.sign(delta)
        else:
            raise ValueError('wrong delta init')

        output = model(X + delta)
        loss = F.cross_entropy(output, y)
        grad = torch.autograd.grad(loss, delta, create_graph=True if backprop else False)[0]
        if not backprop:
            grad, delta = grad.detach(), delta.detach()
        return grad

    def get_perturbation_init(self, model, x, y, eps, device, method, z_init_detach=True):
        z_init = torch.clamp(
            x + torch.FloatTensor(x.shape).uniform_(-eps, eps).to(device),
            min=0, max=1
        ) - x
        z_init.requires_grad_(True)

        retain_graph = not z_init_detach
        pgd_attack_lr = 1.25 * eps
        fgsm_attack_lr = eps

        if method == "random":
            z = z_init

        elif method == "fgsm":
            model.clear_grad()
            model.with_grad()
            z_init = torch.zeros_like(x).requires_grad_(True)
            attack_loss_first = _attack_loss(model(x + z_init), y)
            grad_attack_loss_delta_first = \
                torch.autograd.grad(attack_loss_first, z_init, retain_graph=retain_graph, create_graph=retain_graph)[0]
            z = z_init - fgsm_attack_lr * torch.sign(grad_attack_loss_delta_first)
            z = torch.clamp(x + z, min=0, max=1) - x

        elif method == "pgd":
            model.clear_grad()
            model.with_grad()
            attack_loss_first = _attack_loss(model(x + z_init), y)
            grad_attack_loss_delta_first = \
                torch.autograd.grad(attack_loss_first, z_init, retain_graph=retain_graph, create_graph=retain_graph)[0]
            z = z_init - pgd_attack_lr * torch.sign(grad_attack_loss_delta_first)
            z = torch.clamp(z, min=-eps, max=eps)
            z = torch.clamp(x + z, min=0, max=1) - x

        elif method == "ns-pgd":
            model.clear_grad()
            model.with_grad()
            attack_loss_first = _attack_loss(model(x + z_init), y)
            grad_attack_loss_delta_first = \
                torch.autograd.grad(attack_loss_first, z_init, retain_graph=retain_graph, create_graph=retain_graph)[0]
            z = z_init - self.z_init_non_sign_attack_lr * grad_attack_loss_delta_first
            z = torch.clamp(z, min=-eps, max=eps)
            z = torch.clamp(x + z, min=0, max=1) - x

        elif method == "ns-gd":
            model.clear_grad()
            model.with_grad()
            attack_loss_first = _attack_loss(model(x + z_init), y)
            grad_attack_loss_delta_first = \
                torch.autograd.grad(attack_loss_first, z_init, retain_graph=retain_graph, create_graph=retain_graph)[0]
            z = z_init - self.z_init_non_sign_attack_lr * grad_attack_loss_delta_first

        elif method == "ns-pgd-zero":
            z_init = torch.zeros_like(x).requires_grad_(True)
            model.clear_grad()
            model.with_grad()
            attack_loss_first = _attack_loss(model(x + z_init), y)
            grad_attack_loss_delta_first = \
                torch.autograd.grad(attack_loss_first, z_init, retain_graph=retain_graph, create_graph=retain_graph)[0]
            z = z_init - self.z_init_non_sign_attack_lr * grad_attack_loss_delta_first
            z = torch.clamp(z, min=-eps, max=eps)
            z = torch.clamp(x + z, min=0, max=1) - x

        else:
            raise NotImplementedError

        if z_init_detach:
            return z.detach()
        else:
            return z
        
    

    def train(self, model, train_dl, opt,model_proxy,inner_opt, loss_func, device,inner_scheduler=None, scheduler=None,epoch=0):

        adversary_train = PgdAttack(
            model, loss_fn=loss_func, eps=self.eps, steps=self.steps,
            eps_lr=self.attack_lr, ord=self.constraint_type,
            rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False,
            regular=0, sign=True
        )

        model.train()
        training_loss = torch.tensor([0.])
        train_sa = torch.tensor([0.])
        train_ra = torch.tensor([0.])

        total = 0

        for i, (data, labels) in enumerate(train_dl):
            data = data.type(torch.FloatTensor)
            data = data.to(device)
            labels = labels.to(device)
            real_batch = data.shape[0]
            channels = data.shape[1]
            image_size = data.shape[2]
            total += real_batch

            # Record SA along with each batch
            train_sa += self.test_sa(model, data, labels)

            model.train()

            if self.mode == "fast_at":
                if self.steps == 0:
                    delta_star = torch.zeros_like(data).to(data)
                else:

                    model.train()
                    opt.zero_grad()

                    delta_init = self.get_perturbation_init(model, data, labels, self.eps, device, "random")

                    with ctx_noparamgrad(model):
                        delta_star = adversary_train.perturb(data, labels, delta_init=delta_init) - data

                delta_star.requires_grad = False

                # Update model with perturbed data
                model.clear_grad()
                model.with_grad()
                predictions = model(data + delta_star)
                train_loss = loss_func(predictions, labels) / real_batch
                train_loss.backward()
                opt.step()
            elif self.mode == "cfast_at":
                # RPG + Fast-AT. The adversarial example is generated by the
                # target model; the outer update uses a proxy fast response.
                delta_history = deque(maxlen = self.args.max_len)
                # delta_history.append(data)
                if self.steps == 0:
                    delta_star = torch.zeros_like(data).to(data)
                    delta_history.append(delta_star)
                else:

                    model.train()
                    model_proxy.train()
                    opt.zero_grad()

                    delta_init = self.get_perturbation_init(model, data, labels, self.eps, device, "random")

                    with ctx_noparamgrad(model):
                        x_star,history = adversary_train.perturb_history(data, labels, delta_init=delta_init)
                        delta_star = x_star-data
                        # for _ in range(len(history)):s
                        # #     delta_history.append(history[_])
                        # delta_history.append(history[-1])
                        for _ in range(self.args.max_len):
                            delta_history.append(history[_])
                            # delta_history.append(history[-(_+1)])
                delta_star.requires_grad = False

                model.with_grad()
                opt.zero_grad(set_to_none=False)
                backup_proxy=copy.deepcopy(model_proxy)
                for y_idx in range(-self.args.inner_loop-1, -1, 1):
                    load(model_proxy,backup_proxy)
                    assert len(delta_history)>=2, 'The length of delta_history is invalid!'
                    predictions = model_proxy(clamp(data + delta_history[y_idx+1], 0.0,1.0 ))

                    train_loss = loss_func(predictions, labels) / real_batch
                    alpha = self.args.alpha
                    T = self.args.temperature
                    model_predictions = model_proxy(data)#clamp(data + delta_history[y_idx + 1], 0.0, 1.0))
                    # Teacher-free self-distillation: align the proxy response
                    # on adversarial and clean inputs while keeping CE loss.
                    KD_loss = torch.nn.KLDivLoss()(F.log_softmax(predictions / T, dim=1),
                                             F.softmax(model_predictions / T, dim=1)) * (alpha * T * T) + \
                              train_loss * (1. - alpha)
                    inner_opt.zero_grad()
                    KD_loss.backward()
                    inner_opt.step()

                    # Reweighted Differential Unit (RDU): use the parameter
                    # gap between target theta and proxy fast weights w_tilde
                    # as the target-model update direction.
                    for p, x in zip(model_proxy.parameters(), model.parameters()):
                        if x.grad == None:
                            x.grad = (x.data - p.data)
                        else:
                            x.grad += (x.data - p.data)
                # Move the proxy to the current target before the next batch;
                # this implements "past as prior" with one-step history.
                load(model_proxy,model)
                opt.step()
            elif self.mode == "cpgd":
                # RPG + PGD-AT. Uses multi-step PGD history, then applies the
                # same proxy fast-update and RDU outer update.
                delta_history = deque(maxlen = self.args.max_len)
                if self.steps == 0:
                    delta_star = torch.zeros_like(data).to(data)
                    delta_history.append(delta_star)
                else:

                    model.train()
                    model_proxy.train()
                    opt.zero_grad()

                    delta_init = self.get_perturbation_init(model, data, labels, self.eps, device, "random")

                    with ctx_noparamgrad(model):
                        x_star,history = adversary_train.perturb_max_history(data, labels, delta_init=delta_init)
                        delta_star = x_star-data
                        for _ in range(len(history)):
                            delta_history.append(history[_])
                delta_star.requires_grad = False

                model.with_grad()
                opt.zero_grad(set_to_none=False)
                backup_proxy=copy.deepcopy(model_proxy)
                for y_idx in range(-self.args.inner_loop-1, -1, 1):
                    load(model_proxy,backup_proxy)
                    assert len(delta_history)>=2, 'The length of delta_history is invalid!'
                    predictions = model_proxy(clamp(data + delta_history[y_idx+1], 0.0,1.0 ))
                    train_loss = loss_func(predictions, labels) / real_batch
                    inner_opt.zero_grad()
                    train_loss.backward()
                    inner_opt.step()

                    for p, x in zip(model_proxy.parameters(), model.parameters()):
                        if x.grad == None:
                            x.grad = (x.data - p.data)
                        else:
                            x.grad += (x.data - p.data)
                load(model_proxy,model)
                opt.step()
            elif self.mode == "pgd":
                if self.steps == 0:
                    delta_star = torch.zeros_like(data).to(data)
                else:
                    model.train()
                    opt.zero_grad()

                    delta_init = self.get_perturbation_init(model=model, x=data, y=labels, eps=self.eps, device=device,
                                                            method="random")

                    with ctx_noparamgrad(model):
                        delta_star = adversary_train.perturb(data, labels, delta_init=delta_init) - data

                delta_star.requires_grad = False

                # Update model with perturbed data
                model.clear_grad()
                model.with_grad()
                predictions = model(data + delta_star)
                train_loss = loss_func(predictions, labels) / real_batch
                train_loss.backward()
                opt.step()
            elif self.mode == "fast_at_ga":

                double_bp = True if self.args.ga_coef > 0 else False

                X, y = data.to(device), labels.to(device)
                delta = torch.zeros_like(X, requires_grad=True)

                X_adv = torch.clamp(X + delta, 0, 1)
                output = model(X_adv)
                loss = F.cross_entropy(output, y)
                grad = torch.autograd.grad(loss, delta, create_graph=True if double_bp else False)[0]
                grad = grad.detach()

                argmax_delta = self.eps * torch.sign(grad)

                fgsm_alpha = 1.25
                delta.data = torch.clamp(delta.data + fgsm_alpha * argmax_delta, -self.eps, self.eps)
                delta.data = torch.clamp(X + delta.data, 0, 1) - X
                delta = delta.detach()

                predictions = model(X + delta)
                loss_function = torch.nn.CrossEntropyLoss()
                train_loss = loss_function(predictions, y)
                reg = self.get_ga_reg(model, data, labels, device, double_bp)
                train_loss += reg

                opt.zero_grad()
                train_loss.backward()
                opt.step()
            elif self.mode == "cfast_at_ga":
                # RPG + Fast-AT-GA. The proxy receives the GA-regularized fast
                # update; the target follows the resulting RDU direction.

                double_bp = True if self.args.ga_coef > 0 else False

                X, y = data.to(device), labels.to(device)
                delta = torch.zeros_like(X, requires_grad=True)

                X_adv = torch.clamp(X + delta, 0, 1)
                output = model(X_adv)
                loss = F.cross_entropy(output, y)
                grad = torch.autograd.grad(loss, delta, create_graph=True if double_bp else False)[0]
                grad = grad.detach()

                argmax_delta = self.eps * torch.sign(grad)

                fgsm_alpha = 1.25
                delta.data = torch.clamp(delta.data + fgsm_alpha * argmax_delta, -self.eps, self.eps)
                delta.data = torch.clamp(X + delta.data, 0, 1) - X
                delta = delta.detach()
                predictions = model_proxy(X + delta)
                loss_function = torch.nn.CrossEntropyLoss()
                train_loss = loss_function(predictions, y)
                reg = self.get_ga_reg(model_proxy, data, labels, device, double_bp)
                train_loss += reg
                inner_opt.zero_grad()
                train_loss.backward()
                inner_opt.step()
                model.with_grad()
                opt.zero_grad(set_to_none=False)
                for p, x in zip(model_proxy.parameters(), model.parameters()):
                    if x.grad == None:
                        x.grad = (x.data - p.data)
                    else:
                        x.grad += (x.data - p.data)
                load(model_proxy,model)
                opt.step()  


                # predictions = model(X + delta)
                # loss_function = torch.nn.CrossEntropyLoss()
                # train_loss = loss_function(predictions, y)
                # reg = self.get_ga_reg(model, data, labels, device, double_bp)
                # train_loss += reg

                # opt.zero_grad()
                # train_loss.backward()
                # opt.step()


            elif self.mode == "cfast_bat":
                # RPG + Fast-BAT. The Fast-BAT implicit-gradient correction is
                # computed on the proxy; RDU transfers the proxy response to
                # the target model update.
                z_init = torch.clamp(
                    data + torch.FloatTensor(data.shape).uniform_(-self.eps, self.eps).to(device),
                    min=0, max=1
                ) - data
                z_init.requires_grad_(True)

                model.clear_grad()
                model.with_grad()
                attack_loss = _attack_loss(model(data + z_init), labels)
                grad_attack_loss_delta = torch.autograd.grad(attack_loss, z_init, retain_graph=True, create_graph=True)[
                    0]
                delta = z_init - self.attack_lr * grad_attack_loss_delta
                delta = torch.clamp(delta, min=-self.eps, max=self.eps)
                delta = torch.clamp(data + delta, min=0, max=1) - data

                delta = delta.detach().requires_grad_(True)
                attack_loss_second = _attack_loss(model(data + delta), labels)
                grad_attack_loss_delta_second = \
                    torch.autograd.grad(attack_loss_second, delta, retain_graph=True, create_graph=True)[0] \
                        .view(real_batch, 1, channels * image_size * image_size)
                delta_star = delta - self.attack_lr * grad_attack_loss_delta_second.detach().view(data.shape)
                delta_star = torch.clamp(delta_star, min=-self.eps, max=self.eps)
                delta_star = torch.clamp(data + delta_star, min=0, max=1) - data
                z = delta_star.clone().detach().view(real_batch, -1)

                if self.constraint_type == np.inf:
                    # H: (batch, channel * image_size * image_size)
                    z_min = torch.max(-data.view(real_batch, -1),
                                      -self.eps * torch.ones_like(data.view(real_batch, -1)))
                    z_max = torch.min(1 - data.view(real_batch, -1),
                                      self.eps * torch.ones_like(data.view(real_batch, -1)))
                    H = ((z > z_min + 1e-7) & (z < z_max - 1e-7)).to(torch.float32)
                else:
                    raise NotImplementedError

                delta_cur = delta_star.detach().requires_grad_(True)

                model_proxy.no_grad()
                lgt = model_proxy(data + delta_cur)
                delta_star_loss = loss_func(lgt, labels)
                delta_star_loss.backward()
                delta_outer_grad = delta_cur.grad.view(real_batch, -1)

                hessian_inv_prod = delta_outer_grad / self.lmbda
                bU = (H * hessian_inv_prod).unsqueeze(-1)

                model.with_grad()
                model.clear_grad()
                b_dot_product = grad_attack_loss_delta_second.bmm(bU).view(-1).sum(dim=0)
                b_dot_product.backward()
                cross_term = [-param.grad / real_batch for param in model.parameters()]

                model_proxy.clear_grad()
                model_proxy.with_grad()
                predictions = model_proxy(data + delta_star)
                train_loss = loss_func(predictions, labels) / real_batch
                inner_opt.zero_grad()
                train_loss.backward()

                with torch.no_grad():
                    for p, cross in zip(model_proxy.parameters(), cross_term):
                        new_grad = p.grad + cross
                        p.grad.copy_(new_grad)

                del cross_term, H, grad_attack_loss_delta_second
                inner_opt.step()
                opt.zero_grad(set_to_none=False)

                for p, x in zip(model_proxy.parameters(), model.parameters()):
                    if x.grad == None:
                        x.grad = (x.data - p.data)
                    else:
                        x.grad += (x.data - p.data)

                load(model_proxy,model)
                opt.step()
            elif self.mode == "fast_bat":
                z_init = torch.clamp(
                    data + torch.FloatTensor(data.shape).uniform_(-self.eps, self.eps).to(device),
                    min=0, max=1
                ) - data
                z_init.requires_grad_(True)

                model.clear_grad()
                model.with_grad()
                attack_loss = _attack_loss(model(data + z_init), labels)
                grad_attack_loss_delta = torch.autograd.grad(attack_loss, z_init, retain_graph=True, create_graph=True)[
                    0]
                delta = z_init - self.attack_lr * grad_attack_loss_delta
                delta = torch.clamp(delta, min=-self.eps, max=self.eps)
                delta = torch.clamp(data + delta, min=0, max=1) - data

                delta = delta.detach().requires_grad_(True)
                attack_loss_second = _attack_loss(model(data + delta), labels)
                grad_attack_loss_delta_second = \
                    torch.autograd.grad(attack_loss_second, delta, retain_graph=True, create_graph=True)[0] \
                        .view(real_batch, 1, channels * image_size * image_size)
                delta_star = delta - self.attack_lr * grad_attack_loss_delta_second.detach().view(data.shape)
                delta_star = torch.clamp(delta_star, min=-self.eps, max=self.eps)
                delta_star = torch.clamp(data + delta_star, min=0, max=1) - data
                z = delta_star.clone().detach().view(real_batch, -1)

                if self.constraint_type == np.inf:
                    # H: (batch, channel * image_size * image_size)
                    z_min = torch.max(-data.view(real_batch, -1),
                                      -self.eps * torch.ones_like(data.view(real_batch, -1)))
                    z_max = torch.min(1 - data.view(real_batch, -1),
                                      self.eps * torch.ones_like(data.view(real_batch, -1)))
                    H = ((z > z_min + 1e-7) & (z < z_max - 1e-7)).to(torch.float32)
                else:
                    raise NotImplementedError

                delta_cur = delta_star.detach().requires_grad_(True)

                model.no_grad()
                lgt = model(data + delta_cur)
                delta_star_loss = loss_func(lgt, labels)
                delta_star_loss.backward()
                delta_outer_grad = delta_cur.grad.view(real_batch, -1)

                hessian_inv_prod = delta_outer_grad / self.lmbda
                bU = (H * hessian_inv_prod).unsqueeze(-1)

                model.with_grad()
                model.clear_grad()
                b_dot_product = grad_attack_loss_delta_second.bmm(bU).view(-1).sum(dim=0)
                b_dot_product.backward()
                cross_term = [-param.grad / real_batch for param in model.parameters()]

                model.clear_grad()
                model.with_grad()
                predictions = model(data + delta_star)
                train_loss = loss_func(predictions, labels) / real_batch
                opt.zero_grad()
                train_loss.backward()

                with torch.no_grad():
                    for p, cross in zip(model.parameters(), cross_term):
                        new_grad = p.grad + cross
                        p.grad.copy_(new_grad)

                del cross_term, H, grad_attack_loss_delta_second
                opt.step()

            else:
                raise NotImplementedError()

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == labels
                if self.log is not None:
                    self.log(model,
                             loss=train_loss.cpu(),
                             accuracy=correct.cpu(),
                             learning_rate=scheduler.get_last_lr()[0] if self.args.lr_scheduler !='piecewise' else scheduler(epoch),
                             batch_size=real_batch)
            if self.args.lr_scheduler !='piecewise':
                    scheduler.step()
                    if self.mode[0]=='c':
                        inner_scheduler.step()
            else:
                lr = scheduler(epoch)
                opt.param_groups[0].update(lr=lr)
                if self.mode[0]=='c':
                    lr_inner = inner_scheduler(epoch)
                    inner_opt.param_groups[0].update(lr=lr_inner)

            training_loss += train_loss.cpu().sum().item()
            train_ra += correct.cpu().sum().item()
        return model

    def get_ga_reg(self, model, data, labels, device, double_bp):
        # Regularization for Gradient Alignment
        reg = torch.zeros(1).to(device)[0]
        delta = torch.zeros_like(data, requires_grad=True)
        output = model(torch.clamp(data + delta, 0, 1))
        clean_train_loss = F.cross_entropy(output, labels)
        grad = torch.autograd.grad(clean_train_loss, delta, create_graph=True if double_bp else False)[0]
        grad = grad.detach()

        if self.args.ga_coef != 0.0:
            grad_random_perturb = self.get_input_grad(model, data, labels, self.eps,
                                                      delta_init='random_uniform',
                                                      backprop=True)
            grads_nnz_idx = ((grad ** 2).sum([1, 2, 3]) ** 0.5 != 0) * (
                    (grad_random_perturb ** 2).sum([1, 2, 3]) ** 0.5 != 0)
            grad_clean_data, grad_random_perturb = grad[grads_nnz_idx], grad_random_perturb[grads_nnz_idx]
            grad_clean_data_norms, grad_random_perturb_norms = l2b(grad_clean_data), l2b(
                grad_random_perturb)
            grad_clean_data_normalized = grad_clean_data / grad_clean_data_norms[:, None, None, None]
            grad_random_perturb_normalized = grad_random_perturb / grad_random_perturb_norms[:, None, None,
                                                                   None]
            cos = torch.sum(grad_clean_data_normalized * grad_random_perturb_normalized, (1, 2, 3))
            reg += self.args.ga_coef * (1.0 - cos.mean())

        return reg

    def eval(self, model, test_dl, attack_eps, attack_steps, attack_lr, attack_rs, device):
        total = 0
        robust_total = 0
        correct_total = 0
        test_loss = 0

        for ii, (data, labels) in enumerate(test_dl):
            data = data.type(torch.FloatTensor)
            data = data.to(device)
            labels = labels.to(device)
            real_batch = data.shape[0]
            total += real_batch

            with ctx_noparamgrad(model):
                perturbed_data = attack_pgd_restart(
                    model=model,
                    X=data,
                    y=labels,
                    eps=attack_eps,
                    alpha=attack_lr,
                    attack_iters=attack_steps,
                    n_restarts=attack_rs,
                    rs=(attack_rs > 1),
                    verbose=False,
                    linf_proj=True,
                    l2_proj=False,
                    l2_grad_update=False,
                    cuda=True
                ) + data

            if attack_steps == 0:
                perturbed_data = data

            predictions = model(data)
            correct = torch.argmax(predictions, 1) == labels
            correct_total += correct.sum().cpu().item()

            predictions = model(perturbed_data)
            robust = torch.argmax(predictions, 1) == labels
            robust_total += robust.sum().cpu().item()

            robust_loss = torch.nn.CrossEntropyLoss()(predictions, labels)
            test_loss += robust_loss.cpu().sum().item()

            if self.log:
                self.log(model=model,
                         accuracy=correct.cpu(),
                         robustness=robust.cpu(),
                         batch_size=real_batch)

        return correct_total, robust_total, total, test_loss / total


def norm(x):
    return torch.sqrt(torch.sum(x * x))
