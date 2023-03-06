import warnings
import pymoto as pym
import numpy as np
import scipy.linalg as spla
import scipy.sparse as spsp
import scipy.sparse.linalg as spspla


class MatrixProjection(pym.Module):
    """ Projection of V.T K V """
    def _response(self, K, V):
        return V.T@K@V
    def _sensitivity(self, dKr):
        K, V = [s.state for s in self.sig_in]
        dV = (V.T@K).T@dKr + (K@V)@(dKr.T)
        dK = pym.DyadCarrier()
        for i in range(dKr.shape[0]):
            rr = dKr[i, :]
            dK += pym.DyadCarrier(V[:, i], V@rr)
        return dK, dV


class VectorSet(pym.Module):
    def _prepare(self, indices, value):
        self.inds = indices
        self.val = value

    def _response(self, x):
        y = x.copy()
        y[self.inds] = self.val
        return y

    def _sensitivity(self, dy):
        dx = dy.copy()
        dx[self.inds] = 0.0
        return dx


class MatrixAddDiagonal(pym.Module):
    """ Add a vector to a sparse (square) matrix """
    def _prepare(self, diag):
        self.diag = diag

    def _response(self, A):
        return A + spsp.diags(self.diag, shape=A.shape)

    def _sensitivity(self, dB):
        return dB.copy()


class Diag(pym.Module):
    """ Creates a diagonal matrix from a vector """
    def _response(self, x):
        return np.diag(x)

    def _sensitivity(self, dy):
        return np.diag(dy)


class SplitToScalar(pym.Module):
    """ Splits a vector up to a number of separate scalar Signals """
    def _prepare(self, pad_value=None):
        self.pad_value = pad_value
        self.Nsplit = len(self.sig_out)

    def _response(self, x):
        assert x.ndim == 1, "Must be an 1-dimensional array"
        if self.pad_value is None and len(x) < self.Nsplit:
            raise ValueError("Not enough values in the vector to split. Try to use padding...")
        if len(x) > self.Nsplit:
            warnings.warn(f"SplitToScalar: Too many values to split in <{self.sig_in[0].tag}>. Only taking the first {self.Nsplit} instead of {len(x)}...")
        return [x[i] if i < len(x) else self.pad_value for i in range(self.Nsplit)]

    def _sensitivity(self, *dxi):
        x = self.sig_in[0].state
        dx = np.zeros_like(self.sig_in[0].state)
        for i, dxii in enumerate(dxi):
            if dxii is None or i >= len(x):
                continue
            dx[i] += dxii
        return dx


def solve_traveling_salesman(mac, initial=None, max_swap=-1, threshold=1e-5):
    """ Solves an adapted travelling salesman problem to maximize the total MAC value
    :param mac: Indexed as MAC[ref_ind, cur_ind]
    :param initial: Initial guess for the ordering
    :param max_swap: Maximum swapping distance
    :param threshold: Stopping criteria (minimum improvement)
    :return: Sorted list for which V_cur[order] is optimal wrt V_ref
    """
    assert mac.shape[0] == mac.shape[1], "MAC matrix must be square"
    n = mac.shape[0]

    # Calculate the euclidian distance in n-space of the route r traversing cities c, ending at the path start.
    def sum_mac(order):
        return np.sum(mac[np.arange(n), order])

    # 2-opt Algorithm adapted from https://en.wikipedia.org/wiki/2-opt
    route = np.arange(n) if initial is None else initial  # Make an array of row numbers corresponding to cities.
    improvement_factor = 1.0  # Initialize the improvement factor.
    best_value = sum_mac(route)  # Calculate the distance of the initial path.
    while improvement_factor > threshold:  # If the route is still improving, keep going!
        value_to_beat = best_value  # Record the distance at the beginning of the loop.
        for swap_first in range(n-1):
            for swap_last in range(swap_first+1, n if max_swap < 1 else min(swap_first+max_swap+1, n)):
                new_route = route.copy()  # Swap two entries
                new_route[swap_first], new_route[swap_last] = new_route[swap_last], new_route[swap_first]
                new_value = sum_mac(new_route)  # and check the total distance with this modification.
                if new_value > best_value:  # If the path distance is an improvement,
                    route = new_route  # make this the accepted best route
                    best_value = new_value  # and update the distance corresponding to this route.
        improvement_factor = best_value/value_to_beat - 1  # Calculate how much the route has improved.

    return route  # When the route is no longer improving substantially, stop searching and return the route.


class MAC:
    """ Calculates the MAC values of a set of eigenvectors with respect to a reference set"""
    def __init__(self, s_vec, s_vec_ref=None, which=None):
        self.s_vec = s_vec
        self.s_vec_ref = s_vec_ref
        self.store_hist = self.s_vec_ref is None
        self.history = None
        self.which = which

    def calculate(self):
        # Get current vectors
        V = self.get_vecs(self.s_vec)

        if self.store_hist:
            # Load history
            if self.history is None:
                return np.eye(V.shape[-1])
            else:
                Vref = self.history
        else:
            Vref = self.get_vecs(self.s_vec_ref)

        normc = np.linalg.norm(V, axis=0)**2
        normr = np.linalg.norm(Vref, axis=0)**2

        val = np.absolute(Vref.conj().T @ V) ** 2 / np.outer(normr, normc)

        return val

    def get_vecs(self, signal):
        if self.which is None:
            return signal.state
        else:
            return signal.state[..., self.which]

    def store_history(self, order):
        if self.store_hist:
            self.history = self.get_vecs(self.s_vec)[..., order].copy()


class Eigensolve(pym.Module):
    def _prepare(self, ishermitian=False, trackmodes=False, nmodes=6, sigma=0.0, Kinv=None):
        self.isgeneralized = len(self.sig_in) == 2
        self.iseigenvalueonly = len(self.sig_out) == 1
        self.ishermitian = ishermitian
        self.trackmodes = trackmodes
        self.mpc_tol = 1e-10
        self.nmodes = nmodes  # Only for the sparse case
        self.sigma = sigma  # Only for the sparse case
        self.Kinv = Kinv # Linear solver used in the iterative eigensolver
        if self.trackmodes:
            if self.iseigenvalueonly:
                self.s_eigvecR = pym.Signal("eigvecR")
                self.mactracker = MAC(self.s_eigvecR)
            else:
                self.mactracker = MAC(self.sig_out[-1])

    def left_residual(self, eigval, left_vec):
        if spsp.issparse(self.K):
            normfn = spspla.norm
        else:
            normfn = np.linalg.norm
        norm = normfn(self.K) + np.absolute(eigval) * normfn(self.M)
        return np.linalg.norm((self.K - eigval * self.M).T.dot(left_vec.conj())) / norm

    def right_residual(self, eigval, right_vec):
        if spsp.issparse(self.K):
            normfn = spspla.norm
        else:
            normfn = np.linalg.norm
        norm = normfn(self.K) + np.absolute(eigval) * normfn(self.M)
        return np.linalg.norm((self.K - eigval * self.M).dot(right_vec)) / norm

    def detect_multiplicities(self, eigvals, eigvecsL, eigvecsR):
        N = len(eigvals)
        bins = []

        # Which eigenvalues are finite? Do not consider inf or nan values
        whichfinite = np.argwhere(np.isfinite(eigvals)).flatten()
        remaining = set(whichfinite)

        # Look only at sets with nearly-equal eigenvalue
        tol_eigval = self.mpc_tol * np.average(np.absolute(eigvals[whichfinite]))

        errors_eigval = np.ones((N, N))*10*tol_eigval
        Is, Js = np.meshgrid(whichfinite, whichfinite, indexing='ij')
        errors_eigval[Is, Js] = np.absolute(eigvals[whichfinite][np.newaxis] - eigvals[whichfinite][np.newaxis].T)
        bool_eigval = errors_eigval < tol_eigval

        while True:
            i = list(remaining)[0]
            inds = set(list(np.argwhere(bool_eigval[:, i]).flatten()))
            inds.union(list(np.argwhere(bool_eigval[i, :]).flatten()))

            if len(inds) > 1 and np.average(np.absolute(eigvals[list(inds)])) > 1e-10:
                print("Multiplicit eigenvalues detected: {0}, ({1}) with errors < {2:.3e}".format(inds, eigvals[list(inds)], tol_eigval))
            #     print(errors_eigval[:, list(inds)][list(inds), :])

            bins.append(list(inds))
            remaining -= inds
            if len(remaining) == 0:
                break


        # errors = np.ones((N, N))
        # for i in range(N):
        #     for j in range(N):
        #         errors[i, j] = self.left_residual(eigvals[i], eigvecsL[:, j])
        #         if not self.ishermitian:
        #             errors[i, j] += self.right_residual(eigvals[i], eigvecsR[:, j])
        #
        # bool_error = errors < self.mpc_tol
        # inds_notfinite = np.argwhere(np.logical_or(np.isinf(eigvals), np.isnan(eigvals)))
        # bool_error[:, inds_notfinite] = False
        # bool_error[inds_notfinite, :] = False
        # bool_error[inds_notfinite, inds_notfinite] = True

        return bins

    @staticmethod
    def calc_eigs(A, OPinv=None, vr0=None, vl0=None, **kwargs):
        lams, vr = spspla.eigs(A, OPinv=OPinv, v0=vr0, **kwargs)
        lams1, vl = spspla.eigs(A.conj().T, OPinv=OPinv.adjoint(), v0=vl0, **kwargs)
        vl = vl.conj()
        if not np.linalg.norm(lams - lams1) / np.linalg.norm(lams) < 1e-3:
            print("foo")
        if not np.allclose(lams, lams1, rtol=1e-3, atol=1e-8*np.linalg.norm(lams)):
            print("bar")

        # lams, vl, vr = spla.eig(A.toarray(), left=True, right=True)
        # isparse = 3

        # lams2, vl2, vr2 = spla.eig(A.toarray(), left=True, right=True)


        # for i in range(len(lams)):
        #     idense = np.argmin(np.absolute(lams2 - lams[i]))
        #     valdense = vl2[:, idense].conj().T@vr2[:, idense]
        #     val = (vl[:, isparse].conj().T)@(vr[:, isparse])
        #
        #     if np.abs(np.imag(val)) < np.abs(np.real(val)):
        #         vl[:, i] *= 1j
        #
        #     valdense = vl2[:, idense].conj().T@vr2[:, idense]
        #     val = (vl[:, isparse].conj().T)@(vr[:, isparse])
        #     if np.sign(np.imag(valdense)) != np.sign(np.imag(val)):
        #         vl[:, i] *= -1
        #
        #
        # idense = np.argmin(np.absolute(lams2 - lams[isparse]))
        #
        # print((vl[:, isparse].conj().T)@(vr[:, isparse]))
        # print(vl2[:, idense].conj().T@vr2[:, idense])
        # -vr[:, isparse]*1j - vr2[:, idense]
        # vl[:, isparse].conj() - vl2[:, idense]

        # return lams2, vl2, vr2
        # vl.T.conj().dot(vr)
        #

        # assert np.linalg.norm(lams - lams1) / np.linalg.norm(lams) < 1e-3, "Couldn't compute left eigenvectors"
        # assert np.allclose(lams, lams1, rtol=1e-3, atol=1e-8*np.linalg.norm(lams)), "Couldn't compute left eigenvectors"

        return lams, vl, vr

    def _response(self, *args):
        """
        Matrix  Hermitian  Eigenvalues  Eigenvectors
         real     yes   ->  real         real, and left == right
         imag     yes   ->  real         imag, and left == right
         real     no    ->  imag         imag (possibly real)
         imag     no    ->  imag         imag

        Uses MATLAB's definition instead of scipy's definition:

        Right eigenvalue problem:      A r = λ     M r
        Left eigenvalue problem:   l.H A   = λ l^H M

        or in decomposed form
              A @ R = R @ D         or         A @ R = M @ R @ D
        L.H @ A     =     D @ L.H   or   L.H @ A     =         D @ L.H @ M

        Orthogonalisation
        L[i].H @ R[j] = 0   or  .... ? ...  for i ≠ j
        L[i].T @ L[i] = 1   or   L[i].T @ M @ L[i]
        R[i].T @ R[i] = 1   or   R[i].T @ M @ R[i]

        (K - λ[i] M)^H vL[:,i] = 0
        (K - λ[i] M) vR[:,i] = 0

        :param x: [$$\mat{K}$$: Real/Complex valued stiffness matrix, $$\mat{M}$$: Real/Complex valued mass matrix (optional)]
        :return: [lam, vLeft, vRight]
        """

        # Parse inputs
        # Set the M matrix to I for normal eigenvalue problems
        self.K = args[0]

        # Get the eigensolver function operator
        self.issparse = spsp.issparse(self.K)
        if self.issparse:
            if self.nmodes is None:
                self.nmodes = 6

            self.M = args[1] if self.isgeneralized else spsp.eye(*self.K.shape)
            if self.sigma is None:
                self.sigma = 0.0

            if self.sigma is None or self.sigma == 0.0:
                A = self.K.tocsc()
            else:
                A = (self.K - self.sigma*self.M).tocsc()

            if not hasattr(self, 'Kinv') or self.Kinv is None:
                self.Kinv = pym.SolverSparseLU()

            self.Kinv.update(A)
            KinvOp = spspla.LinearOperator(A.shape, matvec=self.Kinv.solve, rmatvec=self.Kinv.adjoint)

            # Solve the eigenvalue problem
            if self.ishermitian:
                # v0 = np.sum(self.vecR, axis=1) if hasattr(self, 'vecR') else None
                eigOpG = lambda a, b: spspla.eigsh(a, M=b, k=self.nmodes, OPinv=KinvOp, sigma=self.sigma)
            else:
                # vr0 = np.sum(self.vecR, axis=1) if hasattr(self, 'vecR') else None
                # vl0 = np.sum(self.vecL, axis=1) if hasattr(self, 'vecL') else None
                eigOpG = lambda a, b: self.calc_eigs(a, M=b, k=self.nmodes, OPinv=KinvOp, sigma=self.sigma)
        else:
            self.M = args[1] if self.isgeneralized else np.eye(*self.K.shape)
            if self.ishermitian:
                eigOpG = lambda a, b: spla.eigh(a, b=b)
            else:
                eigOpG = lambda a, b: spla.eig(a, b=b, left=True, right=True)

        if self.isgeneralized:
            eigOp = eigOpG
        else:
            eigOp = lambda a, b: eigOpG(a, None)

        # Execute eigenvalue analysis
        if self.ishermitian:
            [self.lams, self.vecR] = eigOp(self.K, self.M)
            self.vecL = None
        else:
            [self.lams, self.vecL, self.vecR] = eigOp(self.K, self.M)
            # self.vecL = self.vecL.conj()

        # Normalize
        for i in range(self.vecR.shape[1]):
            self.vecR[:, i] /= np.sqrt(self.vecR[:, i]@self.M@self.vecR[:, i])
            if self.vecL is not None:
                self.vecL[:, i] /= np.sqrt(self.vecL[:, i]@self.M@self.vecL[:, i])

        # Condition number https://www-sciencedirect-com.tudelft.idm.oclc.org/science/article/pii/S0377042705000567
        # print("Eigenvalue condition = {}".format(np.linalg.norm(self.vecR)*np.linalg.norm(np.linalg.inv(self.vecR))))

        # Switch positive/negative direction to first element that is larger than average value
        for i in range(self.vecR.shape[1]):
            avgvalR = np.average(np.abs(self.vecR[:, i]))
            refvalR = 1.0

            for j in range(len(self.vecR[:, i])):
                if np.abs(self.vecR[j, i]) > avgvalR:
                    refvalR = self.vecR[j, i]
                    break

            self.vecR[:, i] *= np.sign(np.real(refvalR))

        if self.vecL is not None:
            for i in range(self.vecL.shape[1]):
                avgvalL = np.average(np.abs(self.vecL[:, i]))
                refvalL = 1.0

                for j in range(len(self.vecL[:, i])):
                    if np.abs(self.vecL[j, i]) > avgvalL:
                        refvalL = self.vecL[j, i]
                        break

                self.vecL[:, i] *= np.sign(np.real(refvalL))

        # Sort values
        if self.trackmodes:
            if self.iseigenvalueonly:
                self.s_eigvecR.state = self.vecR
            # Renumber based on mode tracking
            macval = self.mactracker.calculate()
            sorted_inds = solve_traveling_salesman(macval)
            self.mactracker.store_history(sorted_inds)
        else:
            # Renumber based on magnitude
            sorted_inds = np.lexsort((np.real(self.lams), np.imag(self.lams)))
        self.lams = self.lams[sorted_inds]
        self.vecR = self.vecR[:, sorted_inds]
        if self.vecL is not None:
            self.vecL = self.vecL[:, sorted_inds]

        # Create sets for multiplicity
        self.eigval_sets = self.detect_multiplicities(self.lams, self.vecL, self.vecR)

        # Check residuals
        tol = 1e-5
        for i in range(len(self.lams)):
            if not np.isfinite(self.lams[i]):
                continue
            right_res = self.right_residual(self.lams[i], self.vecR[:, i])
            left_res = self.left_residual(self.lams[i], self.vecL[:, i]) if self.vecL is not None else 0.0

            if right_res > tol or left_res > tol:
                if self.ishermitian:
                    print("Hermitian eigenvalue residual large! i = {}, res = {}".format(i, right_res))
                else:
                    print("Eigenvalue residual large! i = {}, left res = {}, right res = {}".format(i, left_res, right_res))

        self.reset_adjoint_solvers()
        if len(self.sig_out) == 1:
            return self.lams
        elif len(self.sig_out) == 2:
            return self.lams, self.vecR
        elif len(self.sig_out) == 3:
            return self.lams, self.vecL, self.vecR
        else:
            raise RuntimeError("Maximum of 3 outputs are available")

    def calculate_sens_eigval_simple(self, lam, vecl, vecr, dlam, dk, dm=None):
        """ Eigenvalue sensitivities for simple eigenvalues
            dlam_i/dk = dyad(vecL_i^H, vecR_i) / (vecL^H M vecR)
            dlam_i/dm = -lam_i*dyad(vecL_i^H, vecR_i) / (vecL^H M vecR)
            :param dlams:
            :return:
        """
        assert np.isfinite(dlam), "dlam should be a finite value"
        if dlam == 0:
            return

        vlefth = np.conj(vecl)

        vl_m_vr = np.dot(vlefth, self.M.dot(vecr))  # np.einsum('i,ij,j->', vlefth, self.M, vecr)

        # Store sensitivities
        if self.issparse:
            dki_u = dlam * (vlefth/vl_m_vr).conj()
            dki_v = vecr.conj()
            if np.isrealobj(self.K):
                dk += pym.DyadCarrier([np.real(dki_u), -np.imag(dki_u)], [np.real(dki_v), np.imag(dki_v)])
            else:
                dk += pym.DyadCarrier(dki_u, dki_v)
        else:
            dki = dlam * np.conj(np.outer(vlefth, vecr) / vl_m_vr)
            dk += np.real(dki) if np.isrealobj(self.K) else dki

        if dm is not None:
            if self.issparse:
                dmi_u = dlam*np.conj(lam*vlefth/vl_m_vr)
                dmi_v = vecr.conj()
                if np.isrealobj(self.M):
                    dm -= pym.DyadCarrier([np.real(dmi_u), -np.imag(dmi_u)], [np.real(dmi_v), np.imag(dmi_v)])
                else:
                    dm -= pym.DyadCarrier(dmi_u, dmi_v)
            else:
                dmi = dlam * np.conj(lam * np.outer(vlefth, vecr) / vl_m_vr)
                dm -= np.real(dmi) if np.isrealobj(self.M) else dmi

    def calculate_sens_eigval_multi(self, lam, vecl, vecr, dlam, dk, dm=None, solver=None):
        """ Eigenvalue sensitivities for multiplicit eigenvalues
            dlam_i/dk = dyad(vecL_i^H, vecR_i) / (vecL^H M vecR)
            dlam_i/dm = -lam_i*dyad(vecL_i^H, vecR_i) / (vecL^H M vecR)
            :param dlams:
            :return:
        """

        print(f"Multi-sensitivity calculation [ NOT IMPLEMENTED !!!!! ] -> Calculate as if they are simple (dlam = {dlam})")
        for i in range(len(lam)):
            self.calculate_sens_eigval_simple(lam[i], vecl[:, i], vecr[:, i], dlam[i], dk, dm)
        return
        n_mult = len(lam)

        b = np.vstack((vecr*0, np.diag(np.conj(dlam))))
        adj = solver.solve(b)


        # Get adjoints
        nu = adj[:-n_mult]
        alpha = adj[-n_mult:]

        # Store sensitivities
        if self.issparse:
            dk -= pym.DyadCarrier([np.real(nu), -np.imag(nu)], [np.real(vecr), np.imag(vecr)])
        else:
            dk -= np.real(nu @ (vecr.T))
            # dk += np.real(nu.T @ (vecr))

        if self.isgeneralized:
            if self.issparse:
                raise RuntimeError("not impelemented")
            else:
                dm += lam[0] * nu @ (vecr.T)
                dm += vecr @ ((alpha/2) @ (vecr.T))
            # for i in range(n_mult):
            #
            #     for j in range(n_mult):
            #         vc = alpha[j, i]/2 * vecr[...,j] + lam[0]*nu[...,i]
            #         if self.issparse:
            #             dm += pym.DyadCarrier([np.real(vc), -np.imag(vc)], [np.real(vecr), np.imag(vecr)])
            #         else:
            #             dm += np.real(np.outer(vc, vecr[..., i]))


    def sens_hermitian(self, dlams, dvec):
        """ Eigenvector sensitivities:
        Obtain lagrange multiplier values from adjoint saddlepoint problem:
        [ K - lam M  M phi ][  nu   ] = [ -u ]
        [  phi^T M     0   ][ alpha ] = [  0 ]
        :param dfdv:
        :return:
        """
        pass

    def sens_nonhermitian(self, lam, vl, vr, dlam, dvl, dvr):
        pass

    def reset_adjoint_solvers(self):
        self.adj_solvers = [None for _ in self.eigval_sets]

    def get_adjoint_solver(self, iset, which='right'):
        inds = self.eigval_sets[iset]
        isleft = which.lower() == 'left'
        islv = 0 if isleft else 1
        n_multi = len(inds)
        if self.adj_solvers[iset] is None:
            # default_solver = pym.SolverSparseLU if self.issparse else pym.SolverDenseLU
            #                          Left adjoint             Right adjoint (updated, solver)
            # self.adj_solvers[iset] = [[False, default_solver()], [False, default_solver()]]

            self.adj_solvers[iset] = [[False, pym.SolverSparsePardiso(symmetric=True) if self.issparse else pym.SolverDenseLU()],
                                      [False, pym.SolverSparsePardiso(symmetric=True) if self.issparse else pym.SolverDenseLU()]]

        #
        solver = self.adj_solvers[iset][islv][1]
        if not self.adj_solvers[iset][islv][0]:
            # Requires updating
            lam = self.lams[inds[0]]
            eigvec = self.vecL[:, inds] if isleft else self.vecR[:, inds]
            pencil = self.K - lam*self.M
            pencil = pencil.conj() if isleft else pencil.T
            Mv = self.M@eigvec

            if self.issparse:
                mat = spsp.bmat([[pencil, -Mv],
                                 [-Mv.T, None]]).tocoo()
            else:
                mat = np.vstack((np.hstack((pencil, -Mv)),
                                 np.hstack((-Mv.T, np.zeros((n_multi, n_multi))))))
            solver.update(mat)
            self.adj_solvers[iset][islv][0] = True

        return solver

    def _sensitivity(self, dlams, *args):
        # Assumes no multiplicit eigenvalues
        if len(args) == 0:
            dvecL = None
            dvecR = None
        elif len(args) == 1:
            dvecR, = args
            dvecL = None
        elif len(args) == 2:
            dvecL, dvecR = args
        else:
            raise RuntimeError("Obtained too many sensitivities")

        dlams_is0 = dlams is None
        dvecL_is0 = dvecL is None
        dvecR_is0 = dvecR is None
        dvecs_is0 = dvecL_is0 and dvecR_is0

        if dlams is not None:
            dlams_new = dlams.copy()
            dlams_new[dlams_new == None] = 0.0
            dlams = dlams_new.astype(np.double if self.ishermitian else np.complex128)

        # Start building sensitivities
        dk = pym.DyadCarrier() if self.issparse else np.zeros_like(self.K)
        dm = None
        if self.isgeneralized:
            dm = pym.DyadCarrier() if self.issparse else np.zeros_like(self.M)

        # Sensitivities for eigenvalues only
        if not dlams_is0:
            for iset, inds in enumerate(self.eigval_sets):
                if np.all(np.logical_or(np.logical_not(np.isfinite(dlams[inds])), dlams[inds] == 0.0)):
                    continue  # If all the eigenvalue sensitivities are None or 0.0, there is nothing to be done here
                ii = inds if len(inds) > 1 else inds[0]
                lam = self.lams[ii]
                vecl = self.vecR[:, ii] if self.vecL is None else self.vecL[:, ii]
                vecr = self.vecR[:, ii]
                if len(inds) > 1:
                    # At multiplicity, we need to solve a system
                    solver = self.get_adjoint_solver(iset, 'right')
                    self.calculate_sens_eigval_multi(lam, vecl, vecr, dlams[ii], dk, dm, solver=solver)
                else:
                    self.calculate_sens_eigval_simple(lam, vecl, vecr, dlams[ii], dk, dm)

        if not dvecs_is0:
            # Loop over all eigenpairs
            for i, lam in enumerate(self.lams):
                if abs(lam)<1e-5:
                    continue
                vl = self.vecR[:, i] if self.vecL is None else self.vecL[:, i]
                vr = self.vecR[:, i]

                dvr = None if dvecR is None else dvecR[:, i]
                dvl = None if dvecL is None else dvecL[:, i]
                dlam = None if dlams is None else dlams[i]

                if dlam is None or not np.isfinite(dlam):
                    dlam = 0.0

                # In which iset is this eigenvalue?
                for iset, inds in enumerate(self.eigval_sets):
                    if np.any(np.equal(i, inds)):
                        break

                # Right eigenvector sensitivities
                if dvr is not None and np.linalg.norm(dvr) != 0:
                    # Set dlam to zero, because it is already added. For the asymetric eigenvalue problem there is some
                    # bug if this is nonzero, making sensitivities on the diagonal wrong
                    solver = self.get_adjoint_solver(iset, 'right')
                    b = np.hstack((np.conj(dvr), np.conj(dlam)*np.zeros_like(inds)))
                    mask = np.ones(len(b), dtype=bool)
                    mask[-len(inds):] = [mi == i for mi in inds]
                    adj = solver.solve(b)[mask]  # Calculate adjoint vector and remove 'added' entries
                    assert adj.shape[0] == dvr.shape[0] + 1

                    # Get adjoints
                    nu = adj[:-1]
                    alpha = adj[-1]

                    # Store sensitivities
                    if self.issparse:
                        dk -= pym.DyadCarrier([np.real(nu), -np.imag(nu)], [np.real(vr), np.imag(vr)])
                    else:
                        dk -= np.real(np.outer(nu, vr)) if np.isrealobj(self.K) else np.outer(nu, vr).conj()

                    if self.isgeneralized:
                        vc = alpha/2 * vr + lam*nu
                        if self.issparse:
                            dm += pym.DyadCarrier([np.real(vc), -np.imag(vc)], [np.real(vr), np.imag(vr)])
                        else:
                            dm += np.real(np.outer(vc, vr)) # TODO What if M is complex?

                # Left eigenvector sensitivities
                if dvl is not None and np.linalg.norm(dvl) != 0:
                    solver = self.get_adjoint_solver(iset, 'left')
                    b = np.hstack((np.conj(dvl), np.conj(dlam)*np.zeros_like(inds)))
                    mask = np.ones(len(b), dtype=bool)
                    mask[-len(inds):] = [mi == i for mi in inds]
                    adj = solver.solve(b)[mask]  # Calculate adjoint vector and remove 'added' entries

                    # Get adjoints
                    nu = adj[:-1]
                    alpha = adj[-1]

                    # Store sensitivities
                    if self.issparse:
                        dk -= pym.DyadCarrier([np.real(vl), -np.imag(vl)], [np.real(nu), np.imag(nu)])
                    else:
                        dk -= np.real(np.outer(vl, nu)) if np.isrealobj(self.K) else np.outer(vl, nu)

                    if self.isgeneralized:
                        vc = alpha/2 * vl + lam*nu
                        if self.issparse:
                            dm += pym.DyadCarrier([np.real(vl), -np.imag(vl)], [np.real(vc), np.imag(vc)])
                        else:
                            dm += np.real(np.outer(vl, vc)) # TODO What if M is complex?
        if dk.shape == (-1, -1):
            dk = None
        if dm is not None and dm.shape == (-1, -1):
            dm = None
        # Finally return the sensitivities
        if self.isgeneralized:
            return dk, dm
        else:
            return dk
