__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@foxmail.com"

__all__ = ["Manifold", "Euclidean", "PoincareBall", "get_manifold"]

import torch
from torch import Tensor
from torch import nn
# from util.pytorch import euclidean_distance, inner_product
# import torch.nn.functional as F
# import torch.jit
# from torch import jit

eps = 1e-8
MAX_NORM = 85
MIN_NORM = 1e-15


class LeakyClamp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor, min: float, max: float):
        ctx.save_for_backward(x.ge(min) * x.le(max))
        return torch.clamp(x, min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        mask, = ctx.saved_tensors
        mask = mask.type_as(grad_output)
        return grad_output * mask + grad_output * (1 - mask) * eps, None, None


def clamp(x: Tensor, min: float = float("-inf"), max: float = float("+inf")) -> Tensor:
    return LeakyClamp.apply(x, min, max)


def sqrt(x: Tensor) -> Tensor:
    x = clamp(x, min=1e-9)  # Smaller epsilon due to precision around x=0.
    return torch.sqrt(x)


def sinh(x: Tensor):
    # TODO min和max应该再小很多, 才能有效的约束值域
    x = clamp(x, min=-3, max=3)
    return torch.sinh(x)


class Atanh(torch.autograd.Function):
    """
    Numerically stable arctanh that never returns NaNs.
    x = clamp(x, min=-1+eps, max=1-eps)
    Returns atanh(x) = arctanh(x) = 0.5*(log(1+x)-log(1-x)).
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        x = clamp(x, min=-1. + 4 * eps, max=1. - 4 * eps)
        ctx.save_for_backward(x)
        res = (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5)
        return res

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        x, = ctx.saved_tensors
        return grad_output / (1 - x**2)


def atanh(x: Tensor):
    # TODO 定义域为(-1, 1)
    # Numerically stable arctanh that never returns NaNs.
    return Atanh.apply(x)


def tanh(x: Tensor):
    return torch.tanh(x)


class Manifold(object):

    @property
    def radius(self) -> Tensor:
        raise NotImplementedError

    @property
    def curvature(self) -> Tensor:
        raise NotImplementedError

    def exp_map(self, v: Tensor, x: Tensor) -> Tensor:
        # map vector v from the tangent space of x to manifold
        raise NotImplementedError

    def exp_map0(self, v: Tensor) -> Tensor:
        # map vector v from the tangent space of mu0 to manifold
        raise NotImplementedError

    def log_map(self, y: Tensor, x: Tensor) -> Tensor:
        # map point y from manifold to the tangent space of x
        raise NotImplementedError

    def log_map0(self, y: Tensor) -> Tensor:
        # map point v from manifold to the tangent space of mu0
        raise NotImplementedError

    def parallel_transport_from_mu0(self, v: Tensor, dst: Tensor) -> Tensor:
        # parallel transport vector v from origin to x
        raise NotImplementedError

    def parallel_transport_to_mu0(self, v: Tensor, src: Tensor) -> Tensor:
        raise NotImplementedError

    def distance(self, x: Tensor, y: Tensor, keepdim: bool=False) -> Tensor:
        raise NotImplementedError

    def add(self, x: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError

    def scalar_mul(self, scalar, x: Tensor) -> Tensor:
        raise NotImplementedError

    def hadamard_mul(self, w: Tensor, x: Tensor) -> Tensor:
        raise NotImplementedError

    def mat_mul(self, weight: Tensor, x: Tensor) -> Tensor:
        raise NotImplementedError

    def logdetexp(self, x: Tensor, y: Tensor, keepdim: bool=False):
        raise NotImplementedError


class Euclidean(Manifold):
    def __init__(self, radius: torch.Tensor):
        pass

    @property
    def radius(self) -> Tensor:
        return torch.scalar_tensor(0.0)

    @property
    def curvature(self) -> Tensor:
        return torch.scalar_tensor(0.0)

    def exp_map(self, v: Tensor, x: Tensor) -> Tensor:
        # map vector v from the tangent space of x to manifold
        return x + v

    def exp_map0(self, v: Tensor) -> Tensor:
        # map vector v from the tangent space of mu0 to manifold
        return v

    def log_map(self, y: Tensor, x: Tensor) -> Tensor:
        # map point y from manifold to the tangent space of x
        return y - x

    def log_map0(self, y: Tensor) -> Tensor:
        # map point v from manifold to the tangent space of mu0
        return y

    def parallel_transport_from_mu0(self, v: Tensor, dst: Tensor) -> Tensor:
        # parallel transport vector v from origin to x
        return v

    def parallel_transport_to_mu0(self, v: Tensor, src: Tensor) -> Tensor:
        return v

    def distance(self, x: Tensor, y: Tensor, keepdim: bool=False) -> Tensor:
        return torch.norm(x-y, p=None, dim=-1, keepdim=keepdim)

    def add(self, x: Tensor, y: Tensor) -> Tensor:
        return x + y

    def scalar_mul(self, scalar, x: Tensor) -> Tensor:
        return torch.mul(x, scalar)

    def hadamard_mul(self, w: Tensor, x: Tensor) -> Tensor:
        return torch.mul(w, x)

    def mat_mul(self, x: Tensor, weight: Tensor) -> Tensor:
        return torch.matmul(x, weight)

    def logdetexp(self, x: Tensor, y: Tensor, keepdim: bool=False):
        result = torch.zeros(x.shape[:-1], device=x.device)
        if keepdim:
            result = result.unsqueeze(-1)
        return result


class PoincareBall(Manifold):
    def __init__(self, radius: torch.Tensor) -> None:
        self._radius = radius if isinstance(radius, torch.Tensor) else torch.tensor(radius)

    @property
    def radius(self) -> Tensor:
        return self._radius

    @property
    def curvature(self) -> Tensor:
        return -self.radius.pow(-2)

    def exp_map(self, v: Tensor, x: Tensor=None) -> Tensor:
        # map vector v from the tangent space of x to manifold
        c = -self.curvature
        sqrt_c = c ** 0.5
        v_norm = v.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
        second_term = tanh(sqrt_c / 2 * self._lambda_x(x, keepdim=True, dim=-1) * v_norm) * v / (sqrt_c * v_norm)
        ret = self.add(x, second_term)

        assert torch.isfinite(ret).all()
        return ret

    def exp_map0(self, v: Tensor) -> Tensor:
        # map vector v from the tangent space of mu0 to manifold
        c = -self.curvature
        sqrt_c = c ** 0.5
        u_norm = v.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
        ret = tanh(sqrt_c * u_norm) * v / (sqrt_c * u_norm)

        assert torch.isfinite(ret).all()
        ret = self.project(ret)
        return ret

    def log_map(self, y: Tensor, x: Tensor=None) -> Tensor:
        # map point y from manifold to the tangent space of x
        c = -self.curvature
        sub = self.add(-x, y)
        sub_norm = sub.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
        lam = self._lambda_x(x, keepdim=True, dim=-1)
        sqrt_c = c ** 0.5
        ret = 2 / sqrt_c / lam * atanh(sqrt_c * sub_norm) * sub / sub_norm

        assert torch.isfinite(ret).all()
        return ret

    def log_map0(self, y: Tensor) -> Tensor:
        # map point v from manifold to the tangent space of mu0
        c = -self.curvature
        sqrt_c = c ** 0.5
        y_norm = y.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
        ret = y / y_norm / sqrt_c * atanh(sqrt_c * y_norm)

        assert torch.isfinite(ret).all()
        return ret

    def parallel_transport_from_mu0(self, v: Tensor, dst: Tensor) -> Tensor:
        # parallel transport vector v from origin to x
        c = -self.curvature
        ret = v * (1 - c * dst.pow(2).sum(dim=-1, keepdim=True)).clamp_min(MIN_NORM)
        assert torch.isfinite(ret).all()
        return ret

    def parallel_transport_to_mu0(self, v: Tensor, src: Tensor) -> Tensor:
        c = -self.curvature
        ret = v / (1 - c * src.pow(2).sum(dim=-1, keepdim=True)).clamp_min(MIN_NORM)
        assert torch.isfinite(ret).all()
        return ret

    def distance(self, x: Tensor, y: Tensor, keepdim: bool=False):
        c = -self.curvature
        sqrt_c = sqrt(c)
        mob = self.add(-x, y).norm(dim=-1, p=2, keepdim=keepdim)
        arg = sqrt_c * mob
        dist_c = atanh(arg)
        ret = 2 * (dist_c / sqrt_c)
        assert torch.isfinite(ret).all()
        return ret

    def add(self, x: Tensor, y: Tensor) -> Tensor:
        c = -self.curvature
        x2 = x.pow(2).sum(dim=-1, keepdim=True)
        y2 = y.pow(2).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
        ret = num / denom.clamp_min(MIN_NORM)

        ret = self.project(ret)
        return ret

    def scalar_mul(self, scalar, x: Tensor) -> Tensor:
        c = -self.curvature
        x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(MIN_NORM)
        sqrt_c = c ** 0.5
        ret = tanh(scalar * atanh(sqrt_c * x_norm)) * x / (x_norm * sqrt_c)

        ret = self.project(ret)
        return ret

    def hadamard_mul(self, w: Tensor, x: Tensor) -> Tensor:
        c = -self.curvature
        x_norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(MIN_NORM)
        sqrt_c = c ** 0.5
        wx = w * x
        wx_norm = wx.norm(dim=-1, keepdim=True, p=2).clamp_min(MIN_NORM)
        res_c = tanh(wx_norm / x_norm * atanh(sqrt_c * x_norm)) * wx / (wx_norm * sqrt_c)
        cond = (wx == 0).prod(dim=-1, keepdim=True, dtype=torch.uint8)
        res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
        ret = torch.where(cond, res_0, res_c)

        ret = self.project(ret)
        return ret

    def mat_mul(self, x: Tensor, weight: Tensor) -> Tensor:
        c = -self.curvature
        x = x + eps
        wx = torch.matmul(x, weight) + eps
        wx_norm = torch.norm(wx, dim=-1, keepdim=True)
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        ret_c = c.pow(-0.5) * tanh(wx_norm/x_norm*atanh(c.sqrt()*x_norm))/wx_norm*wx

        cond = (wx == 0).prod(dim=-1, keepdim=True, dtype=torch.uint8)
        ret_0 = ret_c.new_zeros(1)
        ret = torch.where(cond, ret_0, ret_c)
        assert torch.isfinite(ret).all()
        ret = self.project(ret)
        return ret

    def logdetexp(self, x: Tensor, y: Tensor, keepdim: bool=False):
        dim = x.shape[-1]
        c = -self.curvature

        d = self.distance(x, y, keepdim=keepdim)
        dxc = c.sqrt() * d.clamp_min(eps)
        ret = (dim - 1) * (sinh(dxc) / dxc).log()
        if not torch.isfinite(ret).all():
            print()
        assert torch.isfinite(ret).all()
        return ret

    def project(self, x: Tensor) -> Tensor:
        c = -self.curvature
        norm = x.norm(dim=-1, keepdim=True, p=2).clamp_min(MIN_NORM)
        eps = 1e-5
        maxnorm = (1 - eps) / c.pow(0.5)
        cond = norm > maxnorm
        projected = x / norm * maxnorm
        return torch.where(cond, projected, x)

    def _lambda_x(self, x, keepdim: bool=False, dim: int=-1):
        c = -self.curvature
        return 2 / (1 - c * x.pow(2).sum(dim=dim, keepdim=keepdim).clamp_min(MIN_NORM))


manifold_dict = {"Euclidean": Euclidean,
                 "PoincareBall": PoincareBall,
                 "e": Euclidean,
                 "p": PoincareBall}


def get_manifold(name):
    if name in manifold_dict:
        return manifold_dict[name]
    else:
        raise ValueError(f"{name}' is an invalid manifold.'")


class ManifoldLinear(nn.Module):
    def __init__(self, manifold: Manifold, in_dim: int, out_dim: int, bias: bool=True):
        super(ManifoldLinear, self).__init__()
        self._manifold = manifold
        self._e_linear = nn.Linear(in_dim, out_dim, bias=bias)
        # self.reset_parameters()

    @property
    def weight(self):
        return self._e_linear.weight

    @property
    def bias(self):
        return self._e_linear.bias

    def forward(self, m_x):
        output = self._manifold.mat_mul(x=m_x, weight=self.weight.t())
        if self.bias is not None:
            m_bias = self._manifold.exp_map0(self.bias)
            output = self._manifold.add(output, m_bias)
        return output
