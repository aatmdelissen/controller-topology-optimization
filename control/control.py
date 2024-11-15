import pymoto as pym
import numpy as np
import scipy.linalg as spla
import scipy.sparse as spspa
import scipy.sparse.linalg as spspla


class StateSpace:
    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, system_label=None, input_labels=None, output_labels=None, copy=True):
        if A.ndim == 0:
            self.n_state = B.shape[0]
        else:
            self.n_state = A.shape[0]
            assert A.shape[0] == A.shape[1], "A must be square"
        self.A = A.copy() if copy else A

        self.n_in = B.shape[1]
        assert B.shape[0] == self.n_state, "B must have first size equal to number of states"
        self.B = B.copy() if copy else B

        self.n_out = C.shape[0]
        assert C.shape[1] == self.n_state, "C must have second size equal to number of states"
        self.C = C.copy() if copy else C

        assert D.shape == (self.n_out, self.n_in), "D must be of size n_out x n_in"
        self.D = D.copy() if copy else D

        if system_label is None:
            system_label = "Unnamed"
        self.system_label = system_label

        if input_labels is None:
            input_labels = [f"input_{i}" for i in range(self.n_in)]
        self.input_labels = np.asarray(input_labels, dtype=object).copy()
        assert self.input_labels.ndim == 1, "Input labels must be one dimensional"
        assert self.input_labels.size == self.n_in, "Input labels must be of length n_in"

        if output_labels is None:
            output_labels = [f"output_{i}" for i in range(self.n_out)]
        self.output_labels = np.asarray(output_labels, dtype=object).copy()
        assert self.output_labels.ndim == 1, "Input labels must be one dimensional"
        assert self.output_labels.size == self.n_out, "Output labels must be of length n_out"

    def __iadd__(self, other):
        assert self.n_state == other.n_state, "Number of states must be equal"
        assert self.n_in == other.n_in, "Number of inputs must be equal"
        assert self.n_out == other.n_out, "Number of outputs must be equal"
        self.A += other.A
        self.B += other.B
        self.C += other.C
        self.D += other.D
        return self

    def zeros_like(self):
        """ Makes a copy of this state space model (of same size), filled with zeroes """
        return StateSpace(np.zeros_like(self.A), np.zeros_like(self.B), np.zeros_like(self.C), np.zeros_like(self.D),
                   system_label=self.system_label, input_labels=self.input_labels, output_labels=self.output_labels)

    @staticmethod
    def zeros(nstate, nin, nout, system_label=None, input_labels=None, output_labels=None, dtype=np.float64):
        """ Makes a zero StateSpace model with the given settings

        :param nstate: Number of states
        :param nin: Number of inputs
        :param nout: Number of outputs
        :param system_label: System label
        :param input_labels: Input labels (size must correspond to number of inputs)
        :param output_labels: Output labels (size must correspond to number of outputs)
        :param dtype: Data type of the model (usually real valued)
        :return: New zeroed-out StateSpace model
        """
        A = np.zeros((nstate, nstate), dtype=dtype)
        B = np.zeros((nstate, nin), dtype=dtype)
        C = np.zeros((nout, nstate), dtype=dtype)
        D = np.zeros((nout, nin), dtype=dtype)
        return StateSpace(A, B, C, D, system_label=system_label, input_labels=input_labels, output_labels=output_labels)

    def similarity_transform(self, T, Tinv=None):
        """ Performs a similarity transformation using the matrix T.
        A new system represents the dynamics, which is similar tot he original. The dynamic properties do not change.
        :param T: Transformation matrix
        :param Tinv: Inverse of the transformation matrix (optional)
        :return: New StateSpace object
        """
        if Tinv is None:
            Tinv = np.linalg.inv(T)
        else:
            assert np.linalg.norm(Tinv.dot(T) - np.eye(*T.shape)) < 1e-10, "Not a correct inverse is given!"

        TiAT = Tinv.dot(self.A).dot(T)
        TiB = Tinv.dot(self.B)
        CT = self.C.dot(T)
        D = self.D.copy()
        return StateSpace(TiAT, TiB, CT, D, system_label=self.system_label+"_T", input_labels=self.input_labels, output_labels=self.output_labels, copy=False)

    def eval(self, s):
        if spspa.issparse(self.A):
            return self.eval_scipy(np.asarray(s))
        else:
            return self.eval_numpy(np.asarray(s))

    def eval_numpy(self, srange):
        assert srange.ndim == 1, "Only for 1-dimensional array"
        G = np.zeros([len(srange), self.n_out, self.n_in], dtype=np.complex128)
        Z = np.zeros(self.A.shape, dtype=np.complex128)
        Z -= self.A
        inds = np.arange(self.A.shape[0])
        Adiag = self.A[inds, inds]
        for i, s in enumerate(srange):
            Z[inds, inds] = s - Adiag
            G[i, ...] = self.C.dot(np.linalg.solve(Z, self.B)) + self.D
        return G
    
    def eval_modal(self, srange):
        assert srange.ndim == 1, "Only for 1-dimensional array"
        l, vl, vr = spla.eig(self.A, left=True, right=True)
        T = vr
        Tinv = np.linalg.inv(T) # np.diag(1/np.diag(vl.conj().T.dot(vr))).dot(vl.conj().T)
        TiAT = l
        TiB = Tinv.dot(self.B)
        CT = self.C.dot(T)
        D = self.D.copy()
        ss, ll = np.meshgrid(srange, l, indexing='ij')

        BC = np.einsum("iw,wj->wij", CT, TiB)
        return np.einsum("wm,mij->wij", 1/(ss-ll), BC) + D


    def eval_scipy(self, srange):
        assert srange.ndim == 1, "Only for 1-dimensional array"
        G = np.zeros([len(srange), self.n_out, self.n_in], dtype=np.complex128)

        I = spspa.eye(*self.A.shape)
        for i, s in enumerate(srange):
            u = spspla.spsolve(s*I - self.A, self.B)
            if u.ndim == 1:
                u = u[:, np.newaxis]
            G[i, ...] = self.C@u + self.D
        return G

    def get_io(self, inputs=None, outputs=None):
        if inputs is None:
            inputs = np.arange(self.n_in)

        if outputs is None:
            outputs = np.arange(self.n_out)

        inputs = np.asarray(inputs).flatten()
        outputs = np.asarray(outputs).flatten()
        assert max(inputs) < self.n_in, "Selected inputs cannot be larger then number of inputs"
        assert max(outputs) < self.n_out, "Selected outputs cannot be larger then number of outputs"

        return StateSpace(self.A, self.B[:, inputs], self.C[outputs, :], self.D[outputs, :][:, inputs],
                          system_label=self.system_label, input_labels=self.input_labels[inputs], output_labels=self.output_labels[outputs])


class StateSpaceGetA(pym.Module):
    """ Gets the A matrix of a StateSpace object """
    def _response(self, ss):
        return ss.A

    def _sensitivity(self, dA):
        dss = self.sig_in[0].state.zeros_like()
        dss.A += dA
        return dss


class FitCirclePureParticipation(pym.Module):
    def _prepare(self, plot=False, which=None, s_freq=None, min_diameter=None, max_diameter=None):
        self.do_plot = plot
        self.which = which
        self.s_freq = s_freq  # Only used to find the corresponding poles

        if min_diameter is None:
            self.smooth_lower = False
        else:
            self.smooth_lower = True
            self.min_radius = (min_diameter/2)

        if max_diameter is None:
            self.smooth_upper = False
        else:
            self.smooth_upper = True
            self.max_radius = (max_diameter/2)

    def _response(self, G, ss, poles, vl, vr):
        radius = np.zeros_like(G, dtype=float)
        midpts = np.zeros_like(G)

        self.freqlist = np.asarray(self.s_freq.state)
        if self.freqlist.ndim == 0:
            self.freqlist = self.freqlist[np.newaxis]

        self.fac = [None for _ in self.freqlist]
        self.part = [None for _ in self.freqlist]
        self.part_D = [None for _ in self.freqlist]
        self.part_N = [None for _ in self.freqlist]
        self.corresponding_i = [None for _ in self.freqlist]
        self.radius_A = np.zeros_like(radius)
        self.radius_U = np.zeros_like(radius)
        self.participations = np.zeros_like(midpts)
        for i, w_i in enumerate(self.freqlist):
            icur = np.argmin(np.abs(np.imag(poles) - w_i))
            self.corresponding_i[i] = icur
            self.part_D[i] = - np.real(poles[icur])

            # Numerator
            vli = vl[:, icur]
            vri = vr[:, icur]

            TiB = vli.conj()@ss.B
            CT = ss.C@vri
            norm = vli.conj()@vri
            self.part_N[i] = np.outer(CT, TiB) / norm

            # Participation factor
            self.part[i] = self.part_N[i] / self.part_D[i]
            self.participations[..., i] = self.part[i]

            self.radius_A[..., i] = np.absolute(self.part[i])/2

            if self.smooth_upper:
                self.radius_U[..., i] = (self.radius_A[..., i]**-2 + self.max_radius**-2)**(-1/2)
                self.fac[i] = self.radius_U[..., i] / self.radius_A[..., i]
            else:
                self.fac[i] = 1.0
                self.radius_U[..., i] = self.radius_A[..., i]

            if self.smooth_lower:
                radius[..., i] = np.sqrt(self.radius_U[..., i]**2 + self.min_radius**2)
            else:
                radius[..., i] = self.radius_U[..., i]

            # Get direction from participation
            midpts[..., i] = G[..., i] - 0.5 * self.fac[i] * self.part[i]

        self.midpts, self.radius = midpts, radius
        if self.which is None:
            return midpts, radius, self.participations
        else:
            return midpts[self.which[1], self.which[0], ...], radius[self.which[1], self.which[0], ...], self.participations

    def _sensitivity(self, dmidpts, dradius, dparts):

        dmid = np.zeros_like(self.sig_in[0].state)
        if dmidpts is not None:
            if self.which is None:
                dmid[...] = dmidpts
            else:
                dmid[self.which[1], self.which[0], ...] = dmidpts
        dmidpts = dmid

        drad = np.zeros_like(self.sig_in[0].state, dtype=float)
        if dradius is not None:
            if self.which is None:
                drad[...] = dradius
            else:
                drad[self.which[1], self.which[0], ...] = dradius
        dradius = drad

        G, ss, poles, vl, vr = [s.state for s in self.sig_in]
        adj_G = np.zeros_like(self.sig_in[0].state)
        adj_ss = self.sig_in[1].state.zeros_like()
        adj_poles = np.zeros_like(self.sig_in[2].state)
        adj_vl = np.zeros_like(self.sig_in[3].state)
        adj_vr = np.zeros_like(self.sig_in[4].state)

        for i, w_i in enumerate(self.freqlist):
            icur = self.corresponding_i[i]

            d_part = np.zeros_like(self.part[i]) if dparts is None else dparts[..., i]

            # midpts[..., i] = G[..., i] - self.flip[i]*direction
            adj_G[..., i] += dmidpts[..., i]

            # Get direction from participation
            # midpts[..., i] = G[..., i] - 0.5 * self.fac[i] * self.part[i]
            d_part -= 0.5 * self.fac[i] * dmidpts[..., i]
            dfac = np.real(-0.5 * np.conj(self.part[i]) * dmidpts[..., i])

            d_radius_U = np.zeros_like(self.radius_U[..., i])
            if self.smooth_lower:
                # radius[..., i] = np.sqrt(self.radius_U[..., i]**2 + self.min_radius**2)
                d_radius_U += dradius[..., i] * self.radius_U[..., i] / self.radius[..., i]
            else:
                # radius[..., i] = self.radius_U[...,i]
                d_radius_U += dradius[..., i]

            d_radius_A = np.zeros_like(self.radius_A[..., i])
            if self.smooth_upper:
                if dfac is not None:
                    # self.fac[i] = self.radius_U[..., i] / self.radius_A[..., i]
                    d_radius_A -= dfac * self.radius_U[..., i] / (self.radius_A[..., i] * self.radius_A[..., i])
                    d_radius_U += dfac / self.radius_A[..., i]

                # self.radius_U[..., i] = (self.radius_A[..., i]**-2 + self.max_radius**-2)**(-1/2)
                d_radius_A += d_radius_U * self.radius_A[..., i]**-3 * (self.radius_A[..., i]**-2 + self.max_radius**-2)**(-3/2)
            else:
                # self.radius_U[..., i] = self.radius_A[...,i]
                d_radius_A += d_radius_U

            # self.radius_A[...,i] = np.absolute(self.part[i])/2
            d_part += d_radius_A * self.part[i]/(2*np.absolute(self.part[i]))

            #  self.part[i] = self.part_N[i] / self.part_D[i]
            d_partN = d_part / self.part_D[i]
            d_partD = -np.conj(d_part) * self.part_N[i] / (self.part_D[i]**2)

            # Recalculate some from response
            vli = vl[:, icur]
            vri = vr[:, icur]
            TiB = vli.conj()@ss.B
            CT = ss.C@vri
            norm = vli.conj()@vri

            # self.part_N[i] = np.outer(CT, TiB) / norm
            d_CT = np.conj(d_partN) @ TiB / norm
            d_TiB = CT @ np.conj(d_partN) / norm
            d_norm = - np.einsum("ij,ij->", np.conj(d_partN), np.outer(CT, TiB)) / (norm**2)

            # norm = vli.conj()@vri
            dvli = d_norm*vri
            dvri = np.conj(d_norm)*vli

            # CT = ss.C@vri
            adj_ss.C += np.real(np.outer(d_CT, vri))
            dvri += np.conj(d_CT)@ss.C

            # TiB = vli.conj()@ss.B
            adj_ss.B += np.real(np.outer(vli.conj(), d_TiB))
            dvli += ss.B@d_TiB

            # vli = vl[:, icur]
            # vri = vr[:, icur]
            adj_vl[:, icur] += dvli
            adj_vr[:, icur] += dvri

            # self.part_D[i] = - np.real(poles[icur])
            adj_poles[icur] -= np.sum(np.real(d_partD))

        return adj_G, adj_ss, adj_poles, adj_vl, adj_vr


class DistToLine(pym.Module):
    """ Distance to a line, defined by normal (positive direction) and origin
    Im(G) > alpha * Re(G) + c
    """
    def _prepare(self, normal=1.0+0.2*1j, origin=0+0*1j):
        self.normal = normal / np.absolute(normal)
        self.origin = origin

    def _response(self, midp, radius):
        self.dx = midp - self.origin
        return np.real(self.dx)*np.real(self.normal) + np.imag(self.dx)*np.imag(self.normal) - radius

    def _sensitivity(self, ddist):
        midp, radius = [s.state for s in self.sig_in]
        dradius = -ddist

        dmid = np.zeros_like(midp)
        dmid += ddist * np.real(self.normal)
        dmid += 1j * ddist * np.imag(self.normal)
        return dmid, dradius


class DistToArea2(pym.Module):
    """Distance to area defined by corner point 'target' and two normals n1, n2 """
    def _prepare(self, target, n1, n2, n1dir=1, n2dir=-1):
        """

        :param target:
        :param n1:
        :param n2:
        :param n1dir: Direction using righthand rule -- 1 is to the right, -1 is to the left from the normal
        :param n2dir: Active direction for line 2
        :return:
        """
        self.target = target
        self.n1 = n1 / np.absolute(n1)
        self.n2 = n2 / np.absolute(n2)
        self.n1dir = n1dir
        self.n2dir = n2dir

    def _response(self, midp, radius):
        dx = midp - self.target
        dist_point = np.absolute(dx)
        dist_line1 = np.real(dx * np.conj(self.n1))
        dist_line2 = np.real(dx * np.conj(self.n2))
        dists = np.array([dist_point, dist_line1, dist_line2])

        # Determine which section is active
        sel_line1 = np.real(dx * np.conj(self.n1dir*1j*self.n1))
        sel_line2 = np.real(dx * np.conj(self.n2dir*1j*self.n2))

        self.isel = np.zeros_like(radius, dtype=int)-1
        self.isel[np.logical_and(sel_line1 < 0, sel_line2 < 0)] = 0  # Condition for point
        self.isel[np.logical_and(self.isel < 0, dist_line1 > dist_line2)] = 1  # Condition for line 1
        self.isel[np.logical_and(self.isel < 0, dist_line1 <= dist_line2)] = 2  # Condition for line 2
        assert np.all(self.isel >= 0)

        # Get the correct distance
        dist = np.zeros_like(radius)
        for i in range(dists.shape[0]):
            dist[self.isel == i] = dists[i, self.isel == i]
        return dist - radius

    def _sensitivity(self, ddist):
        midp, radius = [s.state for s in self.sig_in]
        dradius = -ddist

        dmid_point = (midp - self.target) * ddist / np.absolute(midp - self.target)
        dmid_line1 = ddist * self.n1
        dmid_line2 = ddist * self.n2
        dmids = np.array([dmid_point, dmid_line1, dmid_line2])
        dmid = np.zeros_like(midp)
        for i in range(dmids.shape[0]):
            dmid[self.isel == i] = dmids[i, self.isel == i]

        return dmid, dradius


class DistFromTarget(pym.Module):
    def _prepare(self, target=0.0, where='furthest'):
        self.target = target
        assert where.lower() in ['furthest', 'closest']
        self.factor = 1.0 if where.lower() == 'furthest' else -1.0

    def _response(self, midp, radius):
        return np.absolute(midp - self.target) + self.factor*radius

    def _sensitivity(self, ddist):
        midp, radius = [s.state for s in self.sig_in]
        dmid = (midp - self.target) * ddist / np.absolute(midp - self.target)
        dradius = ddist*self.factor
        return dmid, dradius


# TODO Flip imag part and use pym version
class ImagPart(pym.Module):
    """ y_i = Im(u_i) """
    def _response(self, u):
        return np.imag(u)

    def _sensitivity(self, dy):
        return 0 + dy*1j


class SOToStateSpace(pym.Module):
    r""" Converts a second order system to state space
    Input: Second order system for the dynamical system
      K x + C xdot + M xddot = b
      y = c . x

    Output: First order state space system
      udot = A u + B f
      y = C u + D f

      with u = [  x   ]
               [ xdot ]
    """
    def _prepare(self, input_labels=None, output_labels=None):
        self.input_labels = input_labels
        self.output_labels = output_labels

    def _response(self, K, Cd, M, b, c):
        self.K = K
        self.Cd = Cd
        self.M = M
        self.b = b
        self.c = c
        self.N = self.K.shape[0]

        if spspa.issparse(K) or spspa.issparse(Cd) or spspa.issparse(M):
            assert spspa.isspmatrix_dia(self.M), "Only works for diagonal mass matrix"
            self.Minv = spspa.diags(1/self.M.diagonal())
            ssA = spspa.bmat([[None, spspa.eye(self.N)], [-self.Minv.dot(self.K), -self.Minv.dot(self.Cd)]]).tocsr()
        else:
            # ### A
            A00 = np.zeros_like(self.K)
            A01 = np.eye(*self.K.shape)
            self.Minv = np.linalg.inv(self.M)
            A10 = -np.dot(self.Minv, self.K)
            A11 = -np.dot(self.Minv, self.Cd)

            ssA = np.block([[A00, A01], [A10, A11]])

        # ### B
        self.nin = self.b.size // self.N  # Number of inputs
        if self.b.size % self.N != 0:
            raise RuntimeError("Size of b vector incorrect: should be multiple of number of dofs")

        ssB = np.zeros((2*self.N, self.nin))

        if self.b.ndim == 1:
            ssB[self.N:, ...] = self.Minv.dot(self.b)[:, np.newaxis]
        elif self.b.ndim == 2:
            ssB[self.N:, ...] = self.Minv.dot(self.b.T)
        else:
            raise RuntimeError("Only b vector of dimension 1 (Ndof) or 2 (Nin, Ndof) supported")

        # ### C
        self.nout = self.c.size // self.N  # Number of outputs
        if self.c.size % self.N != 0:
            raise RuntimeError("Size of c vector incorrect: should be multiple of number of dofs")

        ssC = np.zeros((self.nout, 2*self.N))

        if self.c.ndim < 1 or self.c.ndim > 2:
            raise RuntimeError("Only c vector of dimension 1 (Ndof) or 2 (Nin, Ndof) supported")
        ssC[..., :self.N] = self.c

        # ### D
        ssD = np.zeros((self.nout, self.nin))

        ss = StateSpace(ssA, ssB, ssC, ssD, system_label="plant", input_labels=self.input_labels, output_labels=self.output_labels)
        return ss

    def _sensitivity(self, dss):
        dssA, dssB, dssC, dssD = dss.A, dss.B, dss.C, dss.D
        dK = pym.DyadCarrier() if spspa.issparse(self.K) else np.zeros_like(self.K)
        dC = pym.DyadCarrier() if spspa.issparse(self.Cd) else np.zeros_like(self.Cd)
        dM = pym.DyadCarrier() if spspa.issparse(self.M) else np.zeros_like(self.M)

        if dssC is not None:
            dc = np.reshape(dssC[..., :self.N], self.c.shape)
        else:
            dc = None

        if dssA is not None and dssA.ndim == 2:
            dssA10 = dssA[self.N:, :self.N]
            dssA11 = dssA[self.N:, self.N:]
            dK -= self.Minv.T@dssA10
            dC -= self.Minv.T@dssA11
            dM += dssA10@(self.K.T@(self.Minv.T))
            dM += dssA11@(self.Cd.T@(self.Minv.T))

        if dssB is not None:
            dssB1 = dssB[self.N:, ...]
            db = np.zeros_like(self.b)
            if self.b.ndim == 1:
                db += (self.Minv.T@dssB1).flatten()
                if pym.isdyad(dM):
                    dM -= pym.DyadCarrier(dssB1.flatten(), self.Minv@self.b)
                else:
                    dM -= np.outer(dssB1, self.Minv@self.b)
            elif self.b.ndim == 2:
                db += np.dot(self.Minv.T, dssB1).T
                if isinstance(dM, pym.DyadCarrier):
                    dM -= pym.DyadCarrier(dssB1, self.Minv@(self.b.T))  # TODO TEST
                else:
                    dM -= np.einsum("ik,jk->ij", dssB1, self.Minv@(self.b.T))
        else:
            db = None

        dM = self.Minv.T@dM

        return dK, dC, dM, db, dc


class SeriesSS(pym.Module):
    """     -----   x    -----
    u ---> | S_1 | ---> | S_2 | ---> y
            -----        -----
    Output x of S1 gets replaced by output y of S2
    Input x of S2 gets replaced by input u of S1

    Number of inputs to link: N
    New inputs:  [inputs of S1 (n1), inputs of S2 (n2)] -> n1 + n2
    New outputs: [outputs of S1 (m1), outputs of S2 (m2)] -> m1 + m2
    """
    def _prepare(self, S1_out=0, S2_in=0, remove_connected=False):
        """
        :param S1_out: Output number to connect
        :param S2_in: Input number to connect
        :param remove_connected: Remove the connected signals as in- and outputs of the system
        """
        self.remove_connected = remove_connected
        self.S1_out = np.asarray(S1_out)
        self.S2_in = np.asarray(S2_in)
        if self.S1_out.ndim < 1:
            self.S1_out = self.S1_out[np.newaxis]
        if self.S2_in.ndim < 1:
            self.S2_in = self.S2_in[np.newaxis]

        assert self.S1_out.ndim == 1 and self.S2_in.ndim == 1, "S1_out and S2_in should be given as integer, or as 1-dimensional array"
        assert self.S1_out.size == self.S2_in.size, "Input and outputs linked should be of equal length"
        self.nlink = self.S1_out.size

    def _response(self, ss1, ss2):
        Nout1, Nin1 = ss1.n_out, ss1.n_in
        Nout2, Nin2 = ss2.n_out, ss2.n_in
        N1, N2 = ss1.n_state, ss2.n_state

        out1_remain = np.arange(Nout1)
        in2_remain = np.arange(Nin2)
        if self.remove_connected:
            out1_remain = np.setdiff1d(out1_remain, self.S1_out)
            in2_remain = np.setdiff1d(in2_remain, self.S2_in)

        BTC = ss2.B[:, self.S2_in].dot(ss1.C[self.S1_out, :])
        if spspa.issparse(ss1.A) or spspa.issparse(ss2.A):
            A = spspa.bmat([[ss1.A, None], [BTC, ss2.A]])
        else:
            A = np.zeros((N1+N2, N1+N2))
            A[:N1, :N1] = ss1.A
            A[N1:, N1:] = ss2.A
            A[N1:, :N1] = BTC

        B = np.zeros((N1+N2, Nin1+len(in2_remain)))
        B[:N1, :Nin1] = ss1.B
        B[N1:, Nin1:] = ss2.B[:, in2_remain]
        B[N1:, :Nin1] = ss2.B[:, self.S2_in].dot(ss1.D[self.S1_out, :])

        C = np.zeros((len(out1_remain)+Nout2, N1+N2))
        C[:len(out1_remain), :N1] = ss1.C[out1_remain, :]
        C[len(out1_remain):, :N1] = ss2.D[:, self.S2_in].dot(ss1.C[self.S1_out, :])
        C[len(out1_remain):, N1:] = ss2.C

        D = np.zeros((len(out1_remain)+Nout2, Nin1+len(in2_remain)))
        D[len(out1_remain):, :Nin1] = ss2.D[:, self.S2_in].dot(ss1.D[self.S1_out, :])
        D[out1_remain, :Nin1] = ss1.D[out1_remain, :]
        D[len(out1_remain):, in2_remain] = ss2.D[:, in2_remain]
        return StateSpace(A, B, C, D, 
                          system_label=ss1.system_label+" + "+ss2.system_label, 
                          input_labels=[*ss1.input_labels, *ss2.input_labels[in2_remain]],
                          output_labels=[*ss1.output_labels[out1_remain], *ss2.output_labels], copy=False)

    def _sensitivity(self, dss):
        dA, dB, dC, dD = dss.A, dss.B, dss.C, dss.D
        dss1, dss2 = [s.state.zeros_like() for s in self.sig_in]

        ss1, ss2 = [s.state for s in self.sig_in]
        A1, B1, C1, D1 = ss1.A, ss1.B, ss1.C, ss1.D
        A2, B2, C2, D2 = ss2.A, ss2.B, ss2.C, ss2.D
        
        Nout1, Nin1 = ss1.n_out, ss1.n_in
        Nout2, Nin2 = ss2.n_out, ss2.n_in
        N1, N2 = ss1.n_state, ss2.n_state

        out1_remain = np.arange(Nout1)
        in2_remain = np.arange(Nin2)
        if self.remove_connected:
            out1_remain = np.setdiff1d(out1_remain, self.S1_out)
            in2_remain = np.setdiff1d(in2_remain, self.S2_in)

        if dA is not None and dA.ndim == 2:
            # A[:N1, :N1] = A1
            dss1.A += dA[:N1, :N1]

            # A[N1:, N1:] = A2
            dss2.A += dA[N1:, N1:]

            # A[N1:, :N1] = B2[:, self.S2_in].dot(C1[self.S1_out, :])
            dss2.B[:, self.S2_in] += dA[N1:, :N1]@(C1[self.S1_out, :].T)
            dss1.C[self.S1_out, :] += (B2[:, self.S2_in].T)@(dA[N1:, :N1])

        if dB is not None:
            # B[:N1, :Nin1] = B1
            dss1.B += dB[:N1, :Nin1]

            # B[N1:, Nin1:] = B2
            dss2.B[:, in2_remain] += dB[N1:, Nin1:]

            # B[N1:, :Nin1] = B2[:, self.S2_in].dot(D1[self.S1_out, :])
            dss2.B[:, self.S2_in] += dB[N1:, :Nin1].dot(D1[self.S1_out, :].T)
            dss1.D[self.S1_out, :] += B2[:, self.S2_in].T.dot(dB[N1:, :Nin1])

        if dC is not None:
            # C[:Nout1, :N1] = C1
            dss1.C[out1_remain, :] += dC[:len(out1_remain), :N1]

            # C[Nout1:, :N1] = D2[:, self.S2_in].dot(C1[self.S1_out, :])
            dss2.D[:, self.S2_in] += dC[len(out1_remain):, :N1].dot(C1[self.S1_out, :].T)
            dss1.C[self.S1_out, :] += D2[:, self.S2_in].T.dot(dC[len(out1_remain):, :N1])

            # C[Nout1:, N1:] = C2
            dss2.C += dC[len(out1_remain):, N1:]

        if dD is not None:
            # D[Nout1:, :Nin1] = D2[:, self.S2_in].dot(D1[self.S1_out, :])
            dss2.D[:, self.S2_in] += dD[len(out1_remain):, :Nin1].dot(D1[self.S1_out, :].T)
            dss1.D[self.S1_out, :] += D2[:, self.S2_in].T.dot(dD[len(out1_remain):, :Nin1])

            # D[out1_remain, :Nin1] = D1[out1_remain, :]
            dss1.D[out1_remain, :] += dD[out1_remain, :Nin1]

            # D[Nout1:, in2_remain] = D2[:, in2_remain]
            dss2.D[:, in2_remain] += dD[len(out1_remain):, in2_remain]

        return dss1, dss2


class SSGetIO(pym.Module):
    """ Select certain in-/output port(s) of a state-space model """
    def _prepare(self, inputs=None, outputs=None):
        self.inp, self.out = Ellipsis, Ellipsis
        if inputs is None and outputs is None:
            raise RuntimeError("Both inputs and outputs are unspecified")

        if inputs is not None:
            self.inp = np.asarray(inputs)
            if self.inp.ndim < 1:
                self.inp = self.inp[np.newaxis]
            assert self.inp.ndim == 1, "Inputs should be given as integer, or as 1-dimensional array"

        if outputs is not None:
            self.out = np.asarray(outputs)
            if self.out.ndim < 1:
                self.out = self.out[np.newaxis]
            assert self.out.ndim == 1, "Outputs should be given as integer, or as 1-dimensional array"

    def _response(self, ss):
        A, B, C, D = ss.A, ss.B, ss.C, ss.D
        return StateSpace(A, B[:, self.inp], C[self.out, :], D[self.out, :][:, self.inp],
                          system_label=ss.system_label, input_labels=ss.input_labels[self.inp], output_labels=ss.output_labels[self.out], copy=False)

    def _sensitivity(self, dss_io):
        dA1, dB1, dC1, dD1 = dss_io.A, dss_io.B, dss_io.C, dss_io.D
        dss = self.sig_in[0].state.zeros_like()

        if dA1 is not None:
            dss.A += dA1

        if dB1 is not None:
            dss.B[:, self.inp] = dB1

        if dC1 is not None:
            dss.C[self.out, :] = dC1

        if dD1 is not None:
            if isinstance(self.out, type(Ellipsis)) and isinstance(self.inp, type(Ellipsis)):
                dss.D[:, :] = dD1
            elif isinstance(self.out, type(Ellipsis)) or isinstance(self.inp, type(Ellipsis)):
                dss.D[self.out, self.inp] = dD1
            else:
                dss.D[tuple(np.meshgrid(self.out, self.inp, indexing='ij'))] = dD1

        return dss


# TODO Flip imag
class TransferFunctionSS(pym.Module):
    """
    Calculates the tranfer function from state space (A,B,C,D)
    G(s) = C (iw I - A)^-1 B + D
    """
    def _prepare(self, omegas=None, which=None):
        self.w = omegas
        self.variable_freq = False
        if len(self.sig_in) == 2:
            self.variable_freq = True
        elif self.w is None:
            raise RuntimeError("No frequency range given")

        self.which = which
        if self.which is None:
            self.which = (Ellipsis, )
        if len(self.which) == 2:
            self.which += (Ellipsis, )


    def _response(self, ss, *args):
        self.A, self.B, self.C, self.D = ss.A, ss.B, ss.C, ss.D
        if self.variable_freq:
            self.w = args[0]

        self.nin = self.B.shape[1]
        self.nout = self.C.shape[0]
        # G = np.zeros((*self.D.shape, self.w.size), dtype=np.complex128)
        G = np.moveaxis(ss.eval(self.w*1j), 0, -1) # shape [nout, nin, nw]

        return G[self.which]

    def _sensitivity(self, dG):
        dss = self.sig_in[0].state.zeros_like()

        dw = np.zeros_like(self.w) if self.variable_freq else None

        for i, w in enumerate(self.w):
            dGG = np.zeros((self.nout, self.nin), dtype=np.complex128)
            dGG[self.which] = dG[..., i]
            if np.linalg.norm(dGG) == 0.0:
                continue
            U = np.linalg.solve(1j*w*np.eye(*self.A.shape) - self.A, self.B)  # Solution
            L = np.linalg.solve((1j*w*np.eye(*self.A.shape)-self.A).T, self.C.T).T  # Adjoint solution

            dss.A += np.real(np.einsum("ij,ik,lj->kl", np.conj(dGG), L, U))
            dss.B += np.real(np.dot(L.T, np.conj(dGG)))
            dss.C += np.real(np.dot(np.conj(dGG), U.T))
            if dw is not None:
                dw[i] += np.real(np.sum(-1j*np.einsum("ij,jk,ik->", L, U, np.conj(dGG))))
                # dw[i] += np.imag(np.dot(L.T, np.conj(dGG)).T.dot(U))[0][0]

        dss.D[:] = np.real(np.sum(np.conj(dG), axis=-1))

        if self.variable_freq:
            return dss, dw
        else:
            return dss


class DiagController(pym.Module):
    """ Constructs a diagonal controller (with given bandwidths) in state-space form """
    def _prepare(self, val1=3.0, val2=5.0):
        """
        :param val1: First control constant
        :param val2: Second controller constant
        """

        self.val1 = val1
        self.val2 = val2

    def _response(self, w_b, k):
        if hasattr(w_b, "shape"):
            self.n = len(w_b)
        elif hasattr(k, "shape"):
            self.n = len(k)
        else:
            self.n = 1

        if not hasattr(w_b, "shape"):
            w_b = w_b * np.ones(self.n)
        if not hasattr(k, "shape"):
            k = k * np.ones(self.n)

        self.w_b, self.k = w_b, k
        n0 = k * w_b * w_b * w_b
        n1 = (self.val1 + self.val2) * k * w_b * w_b
        n2 = self.val1 * self.val2 * k * w_b

        d1 = - self.val1 * self.val2 * w_b * w_b
        d2 = - (self.val1 + self.val2) * w_b

        dtype = d1.dtype
        Ntot = 3*self.n
        A = np.zeros((Ntot, Ntot), dtype=dtype)
        B = np.zeros((Ntot, self.n), dtype=dtype)
        C = np.zeros((self.n, Ntot), dtype=dtype)
        D = np.zeros((self.n, self.n), dtype=dtype)

        for i in range(self.n):
            A[i*3, i*3+1] = 1.0
            A[i*3+1, i*3+2] = 1.0
            A[i*3+2, i*3+1] = d1[i]
            A[i*3+2, i*3+2] = d2[i]

            B[i*3+2, i] = 1.0

            C[i, i*3] = n0[i]
            C[i, i*3+1] = n1[i]
            C[i, i*3+2] = n2[i]

        return StateSpace(A, B, C, D, input_labels=[f"ctr{i}_in" for i in range(self.n)],
                          output_labels=[f"ctr{i}_out" for i in range(self.n)], system_label="Controller")

    def _sensitivity(self, dSS):
        dd1, dd2 = np.zeros(self.n), np.zeros(self.n)
        dn0, dn1, dn2 = np.zeros(self.n), np.zeros(self.n), np.zeros(self.n)

        for i in range(self.n):
            dd1[i] = dSS.A[i*3+2, i*3+1]
            dd2[i] = dSS.A[i*3+2, i*3+2]

            dn0[i] = dSS.C[i, i*3]
            dn1[i] = dSS.C[i, i*3+1]
            dn2[i] = dSS.C[i, i*3+2]

        dk, dw_b = np.zeros(self.n), np.zeros(self.n)
        w_b, k = self.w_b, self.k

        # n0 = k * w_b * w_b * w_b
        dk += w_b * w_b * w_b * dn0
        dw_b += k * 3 * w_b * w_b * dn0

        # n1 = (self.val1 + self.val2) * k * w_b * w_b
        dk += (self.val1 + self.val2) * w_b * w_b * dn1
        dw_b += (self.val1 + self.val2) * k * 2 * w_b * dn1

        # n2 = self.val1 * self.val2 * k * w_b
        dk += self.val1 * self.val2 * w_b * dn2
        dw_b += self.val1 * self.val2 * k * dn2

        # d1 = - self.val1 * self.val2 * w_b * w_b
        dw_b -= self.val1 * self.val2 * 2 * w_b * dd1

        # d2 = - (self.val1 + self.val2) * w_b
        dw_b -= (self.val1 + self.val2) * dd2

        if not hasattr(self.sig_in[0].state, "shape"):
            dw_b = sum(dw_b)
        if not hasattr(self.sig_in[1].state, "shape"):
            dk = sum(dk)
        return dw_b, dk


class SelectPoleSet(pym.Module):
    """
    Removes:
     o Poles with large real part wrt imaginary part (controller-modes)
    """
    def _prepare(self, N_modes, rbm_tol=1e-1):
        self.nmodes = N_modes
        self.rbm_tol = rbm_tol

    def _response(self, p, *args):
        p_all = np.arange(len(p))

        # Remove rigid body modes
        p_sel_2 = p_all[np.absolute(p[p_all]) > self.rbm_tol]

        # Controller modes (and rigid body modes) are the inner ones, close to zero, with large real part
        len_remaining = len(p_sel_2)
        to_remove = len_remaining - 2*self.nmodes
        imre_ratio = np.abs(np.imag(p[p_sel_2])/np.real(p[p_sel_2]))

        p_sel_3 = np.setdiff1d(p_sel_2, p_sel_2[np.argsort(imre_ratio)[:to_remove]])

        # Only take the poles with positive imaginary part
        p_sel_4 = p_sel_3[np.imag(p[p_sel_3]) > 0]
        self.selected = p_sel_4

        return (p[self.selected], *[a[..., self.selected] for a in args])

    def _sensitivity(self, dp, *dargs):
        if dp is None:
            dp_out = None
        else:
            dp_out = np.zeros_like(self.sig_in[0].state)
            dp_out[self.selected] += dp

        dargs_out = [None for _ in self.sig_in[1:]]
        for i, (da, da_out) in enumerate(zip(dargs, dargs_out)):
            if da is None:
                dargs_out[i] = None
            else:
                dargs_out[i] = np.zeros_like(self.sig_in[1+i].state)
                dargs_out[i][..., self.selected] += da

        return [dp_out, *dargs_out]


def floodfill_py(domain: pym.DomainDefinition, x: np.ndarray, istart: np.ndarray, threshold=0.3):
    """ Floodfill the array x, for any values below threshold """
    is_processed = np.zeros_like(x, dtype=bool)
    active_list = np.unique(istart.copy().flatten())
    is_processed[active_list] = True

    di = np.array([-1, 0, 1, 0])
    dj = np.array([0, -1, 0, 1])
    dk = np.array([0, 0, 0, 0])

    while len(active_list) > 0:
        el_cur = active_list[np.argmax(x[active_list])]
        i_cur = el_cur % domain.nelx
        j_cur = (el_cur // domain.nelx) % domain.nely
        k_cur = el_cur // (domain.nelx * domain.nely)
        # assert domain.get_elemnumber(i_cur, j_cur, k_cur) == el_cur
        # assert k_cur == 0, "Only for 2D"

        # Get the element numbers for the offsets
        next_els = domain.get_elemnumber(i_cur+di, j_cur+dj, k_cur+dk)

        # Remove elements that are not in the domain
        in_domain = np.logical_and(np.logical_and(i_cur+di >= 0, i_cur+di < domain.nelx),
                                   np.logical_and(j_cur+dj >= 0, j_cur+dj < domain.nely))
        next_els = next_els[in_domain]

        # Remove elements that are already processed
        next_els = next_els[np.logical_not(is_processed[next_els])]

        # Change the density
        x_cur = x[el_cur]
        if next_els.size > 0 and x_cur < threshold:
            x[next_els] = np.minimum(x[next_els], x_cur)

        # Remove the current element
        active_list = active_list[active_list != el_cur]

        # Add the elements that are added
        active_list = np.concatenate((active_list, next_els))
        is_processed[next_els] = True

    assert np.all(is_processed)
