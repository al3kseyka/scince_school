import torch
import torch.nn as nn

from torchdiffeq import odeint

class Cpnf(nn.Module):

    def __init__(self, n: int, h = torch.nn.Tanh()):
        super().__init__()
        self.h = h
        self.w = nn.Parameter(torch.randn(n))
        self.b = nn.Parameter(torch.randn(1))
        self.u = nn.Parameter(torch.randn(n))
        self.n = n

    def forward(self, t, states, log_jac=False, grad=False):
        X: torch.Tensor = states[0] if log_jac else states
        batch_size = X.shape[0]

        if grad:
            pass
            w = self.w
            u = self.u
            b = self.b
        else:
            w = self.w.detach()
            u = self.u.detach()
            b = self.b.detach()

        if log_jac:
            X = X.detach()
            preval = torch.matmul(X, w) + b
            preval.requires_grad_()
            val: torch.Tensor = self.h(preval)
            assert preval.is_leaf
            dhdv, = torch.autograd.grad(outputs=val, inputs=preval, grad_outputs=torch.ones_like(val))
            val = val.detach()

            dhdx = torch.matmul(dhdv.view(batch_size, 1), w.view(1, self.n))
            dlog = -torch.matmul(dhdx, u.view(self.n, 1))
            
            assert not dlog.requires_grad
            return torch.matmul(val.view(batch_size, 1), u.view(1, self.n)), dlog

        
        res: torch.Tensor = torch.matmul(self.h(torch.matmul(X, w) + b).view(batch_size, 1),
                                            u.view(1, self.n))
        return res

def detach(a: tuple):
    return [x.detach() for x in a]

class CNF(nn.Module):
    def __init__(self, activations: nn.ModuleList, fields: nn.ModuleList):
        super().__init__()
        self.activations = activations
        self.fields = fields

    def forward(self, t, states, log_jac=False, grad=False):
        device = next(self.parameters()).device
        if log_jac:
            X = states[0]
        else:
            X = states
        batch_size = X.shape[0]

        res = torch.zeros_like(X)

        if log_jac:
            resultjacoblog = torch.zeros(batch_size, 1, device=device)
            
            for field, activation in zip(self.fields, self.activations):
                delta, jacobianlog = field(t, detach(states), log_jac=True, grad=False)
                with torch.no_grad():
                    alpha = activation(t.view(1, 1)).view(1)
                res = res + alpha * delta
                resultjacoblog = alpha * jacobianlog + resultjacoblog
            return res, resultjacoblog
        for field, activation in zip(self.fields, self.activations):
            delta = field(t, states, grad=grad)
            with torch.set_grad_enabled(grad):
                alpha = activation(t.view(1, 1))
            res = res + alpha * delta
        assert grad == res.requires_grad
        return res

class Wraper:
    def __init__(self, field):
        self.field: nn.Module = field
    def forward(self, X: torch.Tensor, log_jac = False):
        device = next(self.field.parameters()).device
        if log_jac:
            x_t, jac_t = odeint(func=(lambda t, state: self.field(t, detach(state), log_jac=True)),
                          y0=(X, torch.zeros(X.shape[0], device=device)),
                          t=torch.tensor([0., 1], device=device))
            return x_t[-1], jac_t[-1]
        return odeint(func=(lambda t, state: self.field(t, state.detach())),
                          y0=X,
                          t=torch.tensor([0., 1], device=device))[-1]
    def inverse(self, X: torch.Tensor, log_jac = False):
        device = next(self.field.parameters()).device
        if log_jac:
            return odeint(func=(lambda t, state: self.field(t, detach(state), log_jac=True)),
                            y0=(X, torch.zeros(X.shape[0], device=device)),
                            t=torch.tensor([1., 0], device=device))[-1]
        return odeint(func=(lambda t, state: self.field(t, state.detach())),
                            y0=X,
                            t=torch.tensor([1., 0], device=device))[-1]
    def backward(self, X: torch.Tensor, loss: torch.Tensor):
        device = next(self.field.parameters()).device
        self.field.zero_grad()
        #state consist of X, dL/dx, dL/d\theta
        def func(t, state):
            X = state[0]
            #self.field.zero_grad()
            dx = self.field(t, X, grad=True)
            minusA = state[1]
            minusA = -1 * minusA
            # print(minusA.shape, type(minusA), minusA[0].shape)
            da, = torch.autograd.grad(outputs=dx, inputs=X, grad_outputs=minusA, retain_graph=True)
            # print(da)
            dth = torch.autograd.grad(outputs=dx,
                                      inputs=self.field.parameters(),
                                      grad_outputs=[minusA])
            dth = torch.nn.utils.parameters_to_vector(dth)
            # print(dth)
            dx = dx.detach()
            assert not dx.requires_grad and not da.requires_grad and not dth.requires_grad
            # print(dx.shape, da.shape, dth.shape, state[0].shape, state[1].shape, state[2].shape)
            return (dx, da, dth)
        parameters = torch.nn.utils.parameters_to_vector(self.field.parameters()).shape[0]
        dLdx, = torch.autograd.grad(loss, X)

        # print(dLdx.shape, X, torch.zeros(X.shape[0], parameters).shape)
        y0 = (X, dLdx, torch.zeros(parameters, device=device))
        _, _, dLdtheta = odeint(func, y0, torch.tensor([1.,0], device=device))
        # print(dLdtheta)
        vecGrad = dLdtheta[1]
        grads = torch.split(vecGrad, [p.numel() for p in self.field.parameters()])
        for p, grad in zip(self.field.parameters(), grads):
            p.grad = grad.reshape_as(p)

def loss_function(n: int, X: torch.Tensor, logprop: torch.Tensor):
    device = X.device
    normal = torch.distributions.MultivariateNormal(torch.zeros(n, device=device), torch.eye(n, device=device))
    mle = normal.log_prob(X) - logprop
    #may change
    return -mle.mean()
