import pymoto as pym
import numpy as np
import scipy.special as spsp


class SumExpMaxMin(pym.Module):
    """ Soft maximum/minimum function
    S_a(x1, ..., xn) = sum_i(xi exp(a*xi)) / sum_i(exp(a*xi))
    When using as maximum, it underestimates the maximum
    It is exact however when x1=x2=...=xn
    """
    def _prepare(self, alpha=1.0, multi=None):
        self.alpha = alpha
        if multi is None:
            self.multi = len(self.sig_in) > 1
        else:
            self.multi = multi

    def _response(self, *args):
        if self.multi:
            v = np.stack(args)
            return np.sum(v*spsp.softmax(self.alpha*v, axis=0), axis=0)
        else:
            v = args[0]
            return np.sum(v*spsp.softmax(self.alpha*v))

    def _sensitivity(self, dfds):
        if self.multi:
            v = np.stack([s.state for s in self.sig_in])
            dv = spsp.softmax(self.alpha*v, axis=0)*(1+self.alpha*(v-self.sig_out[0].state)) * dfds
            return [dv[i, ...] for i in range(len(self.sig_in))]
        else:
            v = self.sig_in[0].state
            return spsp.softmax(self.alpha*v)*(1+self.alpha*(v-self.sig_out[0].state)) * dfds


class ClipZero(pym.Module):
    def _prepare(self, tol=1e-2):
        self.tol = tol

    def _response(self, x):
        self.iszero = np.abs(x) < self.tol
        y = x.copy()
        y[self.iszero] = 0
        return y

    def _sensitivity(self, dy):
        if dy is None:
            return None
        else:
            dx = dy.copy()
            dx[self.iszero] = 0
            return dx


class HarmonicMean(pym.Module):
    """ Calculates the harmonic mean of its inputs """
    def _prepare(self, inverse=True):
        self.inverse = inverse

    def _response(self, x):
        self.sump = np.sum(np.power(x, -1.0))
        if self.inverse:
            return 1/self.sump
        else:
            return self.sump

    def _sensitivity(self, dm):
        x = self.sig_in[0].state
        dx = np.zeros_like(x)
        if self.inverse:
            dx += dm * np.power(self.sump, -2.0) * np.power(x, -2.0)
        else:
            dx -= dm * np.power(x, -2.0)
        return dx


class IterScale(pym.Module):
    def _prepare(self, scale=100.0):
        self.rescale(scale)

    def rescale(self, scale=None):
        if scale is not None:
            self.value = scale
        self.sf = None

    def _response(self, x):
        if self.sf is None:
            self.sf = abs(self.value / x)
        assert self.sf > 0, "Scaling factor must be positive!"
        return self.sf*x

    def _sensitivity(self, dfdy):
        return self.sf*dfdy


class Symmetry(pym.Module):
    def _prepare(self, domain: pym.DomainDefinition, direction=0):
        self.dom = domain
        assert self.dom.nz == 0, "Only implemented for 2D"

        if isinstance(direction, str):
            keys = {'x': 0, 'y': 1}
            direction = keys[direction]
        self.direc = 1-direction

    def _response(self, x):
        y = x.reshape((self.dom.ny, self.dom.nx))
        return (np.flip(y, axis=self.direc).flatten() + x) / 2

    def _sensitivity(self, dy):
        dx = dy.reshape((self.dom.ny, self.dom.nx))
        return (np.flip(dx, axis=self.direc).flatten() + dy) / 2
