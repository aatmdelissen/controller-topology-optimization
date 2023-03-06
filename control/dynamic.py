import pymoto as pym
import scipy.linalg as spla
import numpy as np
import scipy.sparse as spsp


class MassInterpolation(pym.Module):
    """ Two-range material interpolation
    For x >= threshold:
        y = rho * x^p1
    For x < threshold:
        y = rho * x^p0 / (t^(p0-p1))
    """
    def _prepare(self, rhoval=1.0, threshold=0.1, p0=6.0, p1=1.0):
        self.rhoval = rhoval
        self.threshold = threshold
        self.p0, self.p1 = p0, p1

    def _response(self, x):
        xx = x**self.p1
        xx[x < self.threshold] = x[x < self.threshold]**self.p0 / (self.threshold**(self.p0-self.p1))
        return self.rhoval * xx

    def _sensitivity(self, drho):
        x = self.sig_in[0].state
        dx = self.p1*x**(self.p1-1)
        dx[x < self.threshold] = self.p0*x[x < self.threshold]**(self.p0-1) / (self.threshold**(self.p0-self.p1))
        return self.rhoval*dx*drho


class AssembleLumpedMass(pym.Module):
    """ Assemble the (Diagonal) lumped mass matrix """
    def _prepare(self, domain, rho=1.0, bc=None, bcdiagval=None, asvector=False):
        # Element mass matrix
        # 1/4 Mass of one element acts on one node
        self.mel = rho * np.prod(domain.element_size) / domain.elemnodes
        self.domain = domain
        self.bc = bc
        self.bcdiagval = self.mel if bcdiagval is None else bcdiagval
        self.asvector = asvector

    def _response(self, scale_m):
        sM = np.kron(scale_m, np.ones(4)*self.mel)
        xdiag = np.zeros(self.domain.nnodes*2)
        np.add.at(xdiag, self.domain.conn.flatten()*2, sM)  # Assemble the diagonal (x-directions)
        np.add.at(xdiag, self.domain.conn.flatten()*2+1, sM)  # And y-directions
        if self.bc is not None:
            xdiag[self.bc] = self.bcdiagval
        return xdiag if self.asvector else spsp.diags(xdiag)

    def _sensitivity(self, dM):
        # DM is a dyad carrier
        if dM.size <= 0:
            return [None]

        dMdiag = dM.copy() if self.asvector else dM.diagonal().copy()
        if self.bc is not None:
            dMdiag[self.bc] = 0.0

        dx = np.zeros_like(self.sig_in[0].state)
        dx += np.sum(dMdiag[self.domain.conn*2],   axis=1)
        dx += np.sum(dMdiag[self.domain.conn*2+1], axis=1)
        dx *= self.mel
        return dx


class Rayleigh(pym.Module):
    """ Rayleigh Damping: C = aM + bK """
    def _prepare(self, alpha=1e-3, beta=1e-3):
        self.alpha, self.beta = alpha, beta

    def _response(self, K, M):
        return self.alpha*M + self.beta*K

    def _sensitivity(self, dC):
        return dC*self.beta, dC*self.alpha


class ModalDamping(pym.Module):
    """ Creates modal damping: C = 2ζω"""
    def _prepare(self, zeta=1e-3):
        self.zeta = zeta

    def _response(self, eigvals):
        ev0 = np.maximum(eigvals, 0.0)  # Prevent negative damping
        self.freqs = np.sqrt(ev0)
        return np.diag(2*self.zeta*self.freqs)

    def _sensitivity(self, dC):
        nonzero = np.abs(self.freqs) > 1e-50
        if np.linalg.norm(dC) == 0:
            return [None]

        deigvals = np.zeros_like(self.sig_in[0].state)
        deigvals[nonzero] += self.zeta / (self.freqs[nonzero]) * np.diag(dC)[nonzero]
        return deigvals
