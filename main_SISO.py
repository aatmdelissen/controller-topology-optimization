import pymoto as pym
import control as ct
import numpy as np
import os.path
from pathlib import Path
import sys

import colorsys

def getDistinctColors(ncol):
    huePartition = 1.0 / (ncol)
    return ([colorsys.hsv_to_rgb(huePartition * value, 1.0, 1.0) for value in np.arange(0, ncol)])


def flatten_list(x, output=None):
    if output is None:
        output = []
    for l in x:
        if isinstance(l, list):
            flatten_list(l, output)
        else:
            output.append(l)
    return output

# Some general settings
lenscale = 100.0
rho = 2.7e-6  # kg/mm3 Aluminium
E = 65.0  # GPa (kN/mm2)
# n = 70
n = 40

rayleigh_alpha = 2e-3
rayleigh_beta = 2e-3

np.set_printoptions(precision=3, linewidth=np.inf)


def setup_model_siso(nn, npos):
    """
    :param nn: Number of elements
    :param npositions: Number of positions
    :return: Model
    """
    aspect = 4
    nx = aspect*nn
    ny = nn

    dom = pym.DomainDefinition(nx, ny, unitx=lenscale/nn, unity=lenscale/nn, unitz=aspect*lenscale)
    print(f"Total number of dofs: {2*dom.nnodes}")

    # Rigid body modes
    rigid_body = [np.zeros(dom.nnodes*2)]
    rigid_body[0][1::2] = 1.0
    rigid_body[0] /= np.linalg.norm(rigid_body[0])

    # Actuator and sensor sizes
    sens_point = True
    dx_sens, dy_sens = max(nn // 8, 1), max(nn // 8, 1)
    dx_act, dy_act = max(nn // 8, 1), max(nn//8, 1)

    # Actuator position
    # offset = nx//2-dx_act
    # act_pos = nx//2 + offset  # Middle
    act_pos = nx//4

    # Actuator
    i, j = np.meshgrid(np.arange(act_pos-dx_act, act_pos+dx_act+1), np.arange(0, dy_act+1), indexing='ij')
    indsN = dom.get_nodenumber(i, j)
    b = np.zeros(dom.nnodes*2)
    b[indsN*2+1] = 1.0
    b /= sum(b)

    # Sensor positions
    sens_pos = np.round(np.linspace(0, nx, npos+2)[1:-1]).astype(int)
    c_positions = sens_pos / nx

    # Sensor
    c_array = []
    for px in sens_pos:
        indsN = dom.get_nodenumber(px, ny)

        c_array.append(np.zeros(dom.nnodes*2))
        c_array[-1][indsN*2+1] = 1.0

    # Nondesign domains
    i, j = np.meshgrid(np.arange(act_pos-dx_act, act_pos+dx_act), np.arange(0, dy_act), indexing='ij')
    indsEa = dom.get_elemnumber(i, j)

    i, j = np.meshgrid(np.arange(0, nx), np.arange(ny-dy_sens, ny), indexing='ij')
    indsEs = dom.get_elemnumber(i, j)

    nondes = np.concatenate((indsEa.flatten(), indsEs.flatten()))

    # Boundary conditions
    i, j = np.meshgrid(np.arange(0, 1), np.arange(0, ny+1), indexing='ij')
    indsN1 = dom.get_nodenumber(i, j)*2

    i, j = np.meshgrid(np.arange(nx, nx+1), np.arange(0, ny+1), indexing='ij')
    indsN2 = dom.get_nodenumber(i, j)*2

    bcc = np.concatenate((indsN1.flatten(), indsN2.flatten()))

    # Concentrated masses
    i, j = np.meshgrid(np.arange(nx//2, nx//2+1), np.arange(ny, ny+1), indexing='ij')
    inds_mass = (np.repeat(dom.get_nodenumber(i, j).flatten()[np.newaxis]*2, 2, axis=0)+np.arange(2)[..., np.newaxis]).flatten()
    concentrated_mass = np.zeros(dom.nnodes*2)
    # total_mass = volfrac*rho*np.prod(dom.element_size)*dom.nel
    concentrated_mass[inds_mass] = 5
    concentrated_mass = None

    return (dom, b, c_array, nondes, bcc, rigid_body, c_positions, concentrated_mass)


def setup_model_siso1(nn, npos, tip_mass=1.0):
    """
    :param nn: Number of elements
    :param npositions: Number of positions
    :return: Model
    """
    nx = 3*nn
    ny = nn

    dom = pym.DomainDefinition(nx, ny, unitx=lenscale/nn, unity=lenscale/nn, unitz=lenscale*nx/nn)
    print(f"Total number of dofs: {2*dom.nnodes}")

    # Rigid body modes
    rigid_body = [np.zeros(dom.nnodes*2)]
    rigid_body[0][1::2] = 1.0
    rigid_body[0] /= np.linalg.norm(rigid_body[0])

    # Actuator and sensor sizes
    # dx_sens, dy_sens = max(nn // 16, 1), max(nn//8, 1)
    # dx_sens, dy_sens = max(nn // 8, 1), max(nn//4, 1)  # Large sensor which converges nicely
    dx_sens, dy_sens = max(nn // 8, 1), max(nn//8, 1)  # Used for paper cases
    dx_act, dy_act = max(nn // 8, 1), max(nn//16, 1)

    # Actuator position
    i, j = np.meshgrid(np.arange(0, dx_act+1), np.arange(0, ny+1), indexing='ij')
    indsN = dom.get_nodenumber(i, j)
    b = np.zeros(dom.nnodes*2)
    b[indsN*2+1] = 1.0
    b /= sum(b)

    # Sensor positions
    sens_pos = np.round(np.linspace(nx//2, nx, npos)).astype(int)[::-1] if npos > 1 else np.array([nx])
    c_positions = sens_pos / nx

    # Sensor
    c_array = []
    for px in sens_pos:
        indsN = dom.get_nodenumber(px, ny)

        c_array.append(np.zeros(dom.nnodes*2))
        c_array[-1][indsN*2+1] = 1.0

    # Nondesign domains
    i, j = np.meshgrid(np.arange(0, dx_act),#nx-dx_act, nx),
                       np.arange(0, ny), indexing='ij')
    indsE1 = dom.get_elemnumber(i, j)

    i, j = np.meshgrid(np.arange(nx//2, nx), np.arange(ny-dy_sens, ny), indexing='ij')
    indsE2 = dom.get_elemnumber(i, j)

    nondes = np.concatenate((indsE1.flatten(), indsE2.flatten()))

    # Boundary conditions
    i, j = np.meshgrid(np.arange(0, 1), np.arange(0, ny+1), indexing='ij')
    indsN1 = dom.get_nodenumber(i, j)*2

    i, j = np.meshgrid(np.arange(nx, nx+1), np.arange(0, ny+1), indexing='ij')
    indsN2 = dom.get_nodenumber(i, j)*2

    bcc = indsN1.flatten()#, indsN2.flatten()))

    # Concentrated masses
    if tip_mass is not None:
        i, j = np.meshgrid(np.arange(nx, nx+1), np.arange(ny, ny+1), indexing='ij')
        inds_mass = (np.repeat(dom.get_nodenumber(i, j).flatten()[np.newaxis]*2, 2, axis=0)+np.arange(2)[..., np.newaxis]).flatten()
        concentrated_mass = np.zeros(dom.nnodes*2)
        # total_mass = volfrac*rho*np.prod(dom.element_size)*dom.nel
        concentrated_mass[inds_mass] = tip_mass
    else:
        concentrated_mass =None

    return (dom, b, c_array, nondes, bcc, rigid_body, c_positions, concentrated_mass)


def run_problem(
        x0=None,
        nmodes=10,  # Number of eigenmodes
        n_positions=1,  # Number of positions
        save_dir=Path('out'),
        setup_case=setup_model_siso1,
        robust=False,
        symmetry=False,
        n_pert=1,  # Number of robust perturbations (in each direction)
        max_eta_offset=0.05,  # 0.5 +- eta
        beta_start=0.1,
        beta_end=20.0,
        beta_firstit=15,  # Iteration where beta starts increasing
        beta_lastit=180,  # Iteration where beta reaches its maximum value
        xmin=1e-7,
        volfrac=0.3,
        simp_p=3,  # SIMP power
        modal_zeta=0.01,  # Modal damping factor
        filter_radius=max(10*n/lenscale, 2),  # Density filter radius
        scale_with_freq=False,  # Scale objective relative to eigenfrequencies
        initial_wb=None,
        rho_interpolation_t=0.3,
        wbrange=(0.1, 10.0),
        optimize_integrated=True,
        optimize_eigenfrequencies=False,
        do_floodfill=True,
        each_robust_explicit=False,
        integrated_maxit=200,
        eigfreq_maxit=100,
        plot=True  # Show bode and nyquist plots etc.
    ):
    Path(save_dir).mkdir(parents=True, exist_ok=True)  # Make output directory

    ct.Iteration('mma').value = 0

    # Print all variables of the run
    for k in list(locals()):
        print(f"{k}: {locals()[k]}")

    n_controllers = 1

    print("\n\n1) Model setup...\n")
    model, f_in, f_out, ndd, bc, rbms, c_positions, concentrated_mass = setup_case(n, n_positions)
    input_labels_plant = ["b"]
    output_labels_plant = [f"c{i}@({np.round(c_positions[i], decimals=3)})" for i in range(len(f_out))]
    assert len(output_labels_plant) == n_positions
    assert len(f_out) == n_positions

    # Signals for the forcing vectors
    s_B = pym.Signal("inputs", state=f_in[np.newaxis])
    s_C = pym.Signal("outputs", state=np.stack(f_out))

    # Signal for the scaling array (density variables)
    if x0 is None:
        x0 = np.ones(model.nel)*volfrac
    sx = pym.Signal("design", state=x0)

    # Signal for the PID variables (bandwidth)
    sx_wb = pym.Signal("xwb", state=np.zeros(n_controllers))
    sx_k = pym.Signal("xk", state=np.zeros(n_controllers))

    # ##################################################################################################################
    # ################# GEOMETRY #######################################################################################
    # ##################################################################################################################
    print("\n\n2) Problem setup...\n")
    fn_plant = pym.Network()

    if symmetry:  # Force a symmetric design
        sx_sym = fn_plant.append(ct.Symmetry(sx, domain=model, direction=0))
    else:
        sx_sym = sx

    # Set the nondesign domain
    sx_nondes0 = fn_plant.append(ct.VectorSet(sx_sym, indices=ndd, value=1.0))

    # Density filter
    sx_filt = fn_plant.append(pym.DensityFilter(sx_nondes0, domain=model, radius=filter_radius))
    sx_filt.tag = "xfilt"

    x_designs = []
    if robust:  # Robust projections
        # Setup beta continuation
        cnt = ct.Linear(beta_start, beta_end, beta_firstit, beta_lastit, ct.Iteration('mma'))
        s_beta = pym.Signal("bt", state=cnt)

        # Number of eta perturbations
        perturbations = 0.5+np.linspace(-max_eta_offset, max_eta_offset, 2*n_pert+1)
        for eta in perturbations:
            s_xprojected = fn_plant.append(pym.MathGeneral([sx_filt, s_beta], expression=f"(tanh(bt*{eta}) + tanh(bt*(inp0-{eta}))) / (tanh(bt*{eta}) + tanh(bt*(1-{eta})))"))
            s_xprojected.tag = f"xproj,eta={eta}"
            x_designs.append(s_xprojected)

        i_nominal = n_pert
    else:
        x_designs.append(sx_filt)
        i_nominal = 0

    n_models = len(x_designs)
    sx_design = x_designs[i_nominal]  # Nominal design

    # Show the design
    fn_plant.append(pym.PlotDomain(sx_design, domain=model, saveto=Path(save_dir, 'design', 'design.png')))

    # Volume constraints
    s_vol = fn_plant.append(pym.EinSum(sx_design, expression="i->"))

    ndd_vol = len(ndd)
    sg_maxvol = fn_plant.append(pym.MathGeneral(s_vol, expression=f"10 * ((inp0-{ndd_vol})/{volfrac*(model.nel-ndd_vol)} - 1)"))
    sg_maxvol.tag = 'max volume'

    volume_offset = 0.1
    sg_minvol = fn_plant.append(pym.MathGeneral(s_vol, expression=f"10 * (1 - (inp0-{ndd_vol})/{(volfrac-volume_offset)*(model.nel-ndd_vol)})"))
    sg_minvol.tag = 'min volume'

    # ##################################################################################################################
    # ################# DYNAMICS #######################################################################################
    # ##################################################################################################################
    # Assemble system matrices
    K_matrices = []
    M_matrices = []
    rho_fields = []
    for i, sxx in enumerate(x_designs):
        # Stiffness
        sx_E = fn_plant.append(pym.MathGeneral(sxx, expression=f"{E}*({xmin} + {1-xmin}*inp0^{simp_p})"))
        sx_E.tag = f"E-field({i})"

        s_K = fn_plant.append(pym.AssembleStiffness(sx_E, domain=model, bc=bc, e_modulus=1.0, poisson_ratio=0.3, bcdiagval=1.0, plane='strain'))
        s_K.tag = f"K{i}"
        K_matrices.append(s_K)

        # Mass
        sx_rho = fn_plant.append(ct.MassInterpolation(sxx, rhoval=rho, threshold=rho_interpolation_t))
        sx_rho.tag = f"rho-field({i})"
        rho_fields.append(sx_rho)

        s_M = fn_plant.append(ct.AssembleLumpedMass(sx_rho, domain=model, bc=bc, rho=1.0, bcdiagval=0))
        # Optionally add a concentrated mass at specific nodes
        if concentrated_mass is not None:
            s_M1 = fn_plant.append(ct.MatrixAddDiagonal(s_M, diag=concentrated_mass))
        else:
            s_M1 = s_M
        s_M1.tag = f"M{i}"
        M_matrices.append(s_M1)

    # Undamped eigenvalues
    s_eigval0, s_eigvec = pym.Signal("undamped_eigenvalues"), pym.Signal("undamped_eigenvectors")
    if robust and optimize_eigenfrequencies:
        fn_plant.append(ct.Eigensolve([K_matrices[-1], M_matrices[-1]], [s_eigval0, s_eigvec], ishermitian=True, nmodes=nmodes + n_controllers))
    else:
        fn_plant.append(ct.Eigensolve([K_matrices[i_nominal], M_matrices[i_nominal]], [s_eigval0, s_eigvec], ishermitian=True, nmodes=nmodes + n_controllers))

    # Write design and eigenmodes to Paraview
    fn_plant.append(pym.WriteToVTI([*x_designs, s_eigvec], domain=model, saveto=Path(save_dir, 'vti', 'design.vti')))

    # Clip the eigenvalues close to zero
    s_eigval = pym.Signal("undamped_eigenvalues")
    fn_plant.append(ct.ClipZero(s_eigval0, s_eigval, tol=1e-5))

    # Get flexible eigenfrequencies
    s_eigval_flex = s_eigval[len(rbms)+np.arange(nmodes)]
    s_eigfreq_flex = fn_plant.append(pym.MathGeneral(s_eigval_flex, expression="sqrt(inp0)"))
    s_eigfreq_flex.tag = "undamped_eigfreqs"

    # Make reduced order model(s)
    K_diags = []
    ss_plants = []
    for i, (s_K, s_M) in enumerate(zip(K_matrices, M_matrices)):
        s_Kr, s_Dr = pym.Signal("Kr"), pym.Signal("Dr")
        s_Br, s_Cr = pym.Signal("Br"), pym.Signal("Cr")
        s_Mr = pym.Signal("Mr", np.eye(nmodes+n_controllers))

        if i == i_nominal:
            s_Kdiag, s_Vproj = s_eigval, s_eigvec
        else:
            if each_robust_explicit:  # Do an extra eigensolve for the perturbed designs
                s_eigval0_proj, s_Vproj = fn_plant.append(ct.Eigensolve([K_matrices[i], M_matrices[i]], ishermitian=True, nmodes=nmodes + n_controllers))
                s_Kdiag = fn_plant.append(ct.ClipZero(s_eigval0_proj, tol=1e-5))
            else:
                # Project matrices
                s_VtKV, s_VtMV = pym.Signal(f"VtKV{i}"), pym.Signal(f"VtMV{i}")
                fn_plant.append(ct.MatrixProjection([s_K, s_eigvec], s_VtKV))
                fn_plant.append(ct.MatrixProjection([s_M, s_eigvec], s_VtMV))

                s_ritzval0, s_redvec = fn_plant.append(ct.Eigensolve([s_VtKV, s_VtMV], ishermitian=True))
                s_Kdiag = fn_plant.append(ct.ClipZero(s_ritzval0, tol=1e-5))
                s_Vproj = fn_plant.append(pym.EinSum([s_eigvec, s_redvec], expression="ij,jk->ik"))

        s_Kdiag.tag = f"Kdiag{i}"
        s_Vproj.tag = f"Vproj{i}"

        # Make reduced system matrices
        K_diags.append(s_Kdiag)
        fn_plant.append(ct.Diag(s_Kdiag, s_Kr))
        modal_damping = True
        if modal_damping:  # Here you can choose between modal damping or Rayleigh damping
            fn_plant.append(ct.ModalDamping(s_Kdiag, s_Dr, zeta=modal_zeta))
        else:
            fn_plant.append(ct.Rayleigh([s_Kr, s_Mr], s_Dr, alpha=rayleigh_alpha, beta=rayleigh_beta))
        fn_plant.append(pym.EinSum([s_Vproj, s_B], s_Br, expression="ji,Bj->Bi"))
        fn_plant.append(pym.EinSum([s_Vproj, s_C], s_Cr, expression="ji,Cj->Ci"))

        # Make statespace model
        s_ssplant = fn_plant.append(ct.SOToStateSpace([s_Kr, s_Dr, s_Mr, s_Br, s_Cr], input_labels=input_labels_plant, output_labels=output_labels_plant))
        s_ssplant.tag = f"ss_plant{i}"
        ss_plants.append(s_ssplant)

    # Eigenfrequency maximization
    if optimize_eigenfrequencies:
        fn_eigfreq = pym.Network(*fn_plant.mods)
        s_eigfreqs_mean = []
        for kdiag in K_diags:  # Take harmonic mean of all robust projected eigenfrequencies
            s_eigfreqs_robust = fn_plant.append(pym.MathGeneral(kdiag[len(rbms)+np.arange(3)], expression="sqrt(inp0)"))
            s_eigfreq_mean = fn_eigfreq.append(ct.HarmonicMean(s_eigfreqs_robust, inverse=False))
            s_eigfreqs_mean.append(s_eigfreq_mean)
        # Take smooth maximum of the harmonic means to optimize
        sg_eigfreq = fn_eigfreq.append(ct.SumExpMaxMin(s_eigfreqs_mean, alpha=1.0))
        sg_eigfreq.tag = "g_eigfreq"

        # Scale the eigenfrequency objective to 100
        sg_frsc = fn_eigfreq.append(ct.IterScale(sg_eigfreq, scale=100.0))

        # Perform the optimization
        ct.MMA(fn_eigfreq, sx, [sg_frsc, sg_minvol, sg_maxvol], maxit=eigfreq_maxit, xmin=0, xmax=1, move=0.1).response()
    else:
        # Calculate response once for the control part
        fn_plant.response()

    # ##################################################################################################################
    # ######################################## CONTROL #################################################################
    # ##################################################################################################################
    rescales = []
    fn_control = pym.Network()

    # Controller scaling parameters
    xwmax = 1.0
    wbmin, wbmax = wbrange
    s_pidwb = fn_control.append(pym.MathGeneral(sx_wb, expression=f"{wbmin}*({wbmax/wbmin}^(inp0/{xwmax}))"))
    s_pidwb.tag = "PIDvalwb"

    # (optional) Save design data for restarting
    fn_control.append(ct.SaveToNumpy([sx, s_pidwb], [], saveto=Path(save_dir, "restart", "restart.npy"), iter=ct.Iteration('mma')))

    # Scale control gain
    s_dens = fn_control.append(pym.EinSum(rho_fields[i_nominal], expression="i->"))  # Mass of the system
    v_el = model.unitx * model.unity * model.unitz  # Volume of one element
    k0 = 1.1
    s_pidk = fn_control.append(pym.MathGeneral([s_dens, s_pidwb], expression=f"{k0}*inp0*{v_el}*inp1^2"))
    s_pidk.tag = "PIDvalk"

    # Make controller
    s_sscontr = pym.Signal("ss_controller")
    fn_control.append(ct.DiagController([s_pidwb, s_pidk], s_sscontr))

    fn_control.append(ct.Print([s_pidwb, s_pidk]))  # TODO Remove print

    # Add the controller in series to all reduced order models
    redmodels = [{} for _ in ss_plants]
    for i, s_ssplant in enumerate(ss_plants):
        redmodel = redmodels[i]
        redmodel.update(ss_plant=s_ssplant)

        # Add the controller
        redmodel.update(ss_loop=pym.Signal(f'ss_loop{i}'))
        fn_control.append(ct.SeriesSS([s_sscontr, s_ssplant], redmodel['ss_loop'], remove_connected=True))

    # ================================= #
    # ## Circle fit onto the loop gains ##
    s_gcirc_cl_der = []

    # Go through all the perturbed models
    s_gcircle_agg = [[] for _ in range(n_positions)]
    for i_model, redmodel in enumerate(redmodels):
        lbl = "" if len(redmodels) == 1 else f"_model{i_model}"

        # Get frequencies
        ssA = fn_control.append(ct.StateSpaceGetA(redmodel['ss_loop']))

        # Solve non-symmetric eigenvalue problem for left & right eigenvectors and eigenvalues
        ssEigvals, ssEigvecL, ssEigvecR = pym.Signal(f"eval{lbl}"), pym.Signal(f"evecL{lbl}"), pym.Signal(f"evecR{lbl}")
        # TODO imag
        fn_control.append(ct.Eigensolve(ssA, [ssEigvals, ssEigvecL, ssEigvecR]))

        # Select correct poles (corresponding to the flexible modes
        ssPoles, ssVecL, ssVecR = pym.Signal(f"poles{lbl}"), pym.Signal(f"vecL{lbl}"), pym.Signal(f"vecR{lbl}")
        fn_control.append(ct.SelectPoleSet([ssEigvals, ssEigvecL, ssEigvecR], [ssPoles, ssVecL, ssVecR], N_modes=nmodes))

        # Get the imaginary part as frequency
        redmodel.update(circ_freqs=pym.Signal(f"circ_freqs{lbl}"))
        fn_control.append(ct.ImagPart(ssPoles, redmodel['circ_freqs']))  # TODO Flip imag

        # Get the transfer function value at frequency
        s_circ_G = pym.Signal(f"circle_G{lbl}")
        # TODO Flip imag
        fn_control.append(ct.TransferFunctionSS([redmodel['ss_loop'], redmodel['circ_freqs']], s_circ_G))
        redmodel['Gvalues'] = s_circ_G

        # Calculate the midpoints, radius, and participation
        s_circ_midpp = pym.Signal(f"midp{lbl}")
        s_circ_radiusp = pym.Signal(f"radius{lbl}")
        s_circ_particip = pym.Signal(f"participation{lbl}")
        # TODO Flip imag
        fn_control.append(ct.FitCirclePureParticipation([s_circ_G, redmodel['ss_loop'], ssPoles, ssVecL, ssVecR],
                                                        [s_circ_midpp, s_circ_radiusp, s_circ_particip],
                                                        s_freq=redmodel['circ_freqs'], min_diameter=5e-2))
        redmodel.update(midpts=s_circ_midpp, radii=s_circ_radiusp, particp=s_circ_particip)

        # Calculate distance to the circle at -1
        circ_target = -1.0
        s_circ_cl_dists = pym.Signal(f"dist{lbl}")
        # TODO Flip imag
        fn_control.append(ct.DistToArea2([s_circ_midpp, s_circ_radiusp], s_circ_cl_dists, target=circ_target, n1=1+0.5*1j, n2=-1j))

        # Convert to constraint
        s_constr = pym.Signal(f"gcirc{lbl}")
        fn_control.append(pym.MathGeneral(s_circ_cl_dists, s_constr, expression="10*(1.0 - inp0/0.5)"))

        # Go through all other positions
        for i_pos in range(n_positions):
            s_gcircle_agg[i_pos].append(s_constr[i_pos, 0])

    for s_gmodels in s_gcircle_agg:
        if len(s_gmodels) > 1:  # Aggregate using smooth maximum
            s_constr = fn_control.append(ct.SumExpMaxMin(s_gmodels, s_constr, alpha=1.0))
            s_constr.tag = s_gmodels[0].tag.replace("_model0", "")
        else:
            s_constr = s_gmodels[0]

        # Separate constraints to scalars
        split_sigs = [pym.Signal(f"{s_constr.tag}_freq{j}") for j in range(nmodes)]
        fn_control.append(ct.SplitToScalar(s_constr, split_sigs))

        s_gcirc_cl_der.extend(split_sigs)

    # ================================= #
    # ## Objective function: bandwidth ##
    s_invbw = pym.Signal("1/bw")
    fn_control.append(ct.HarmonicMean(s_pidwb, s_invbw, inverse=False))

    sg_0 = pym.Signal("objective(bw)")
    fn_control.append(ct.IterScale(s_invbw, sg_0, scale=1.0))
    rescales.append(fn_control[-1])

    # Log some useful information to file
    m_logger = ct.Logger([sx_k, sx_wb, s_pidk, s_pidwb, s_vol, s_eigfreq_flex, redmodels[i_nominal]['circ_freqs'],
                          sg_0, *s_gcirc_cl_der, sg_minvol, sg_maxvol],
                         iteration=ct.Iteration('mma'), saveto=Path(save_dir, 'log.csv'))
    fn_control.append(m_logger)

    m_pickle = ct.SaveToPickle([], [], savedat=redmodels, saveto=Path(save_dir, 'data', 'data.p'), iter=ct.Iteration('mma'))
    fn_control.append(m_pickle)

    # #################### PLOT
    fn_plot = []
    if plot:
        s_pidwb_sep = [s_pidwb[i] for i in range(n_controllers)]
        s_wfracrange = pym.Signal("w_frac_range", state=np.logspace(-1.5, 2, 5000))

        # Find the scaling value
        s_contributions = pym.Signal("wfracs", state=np.ones(n_controllers)/n_controllers)
        s_wrange_scale = fn_control.append(pym.EinSum([s_contributions, s_pidwb], expression="i,i->"))

        # Scale frequency-range
        s_wrange = fn_control.append(pym.MathGeneral([s_wfracrange, s_wrange_scale], expression="inp0*inp1"))

        # Controller TF
        s_TF_contr = fn_control.append(ct.TransferFunctionSS([s_sscontr, s_wrange]))

        i_plot = np.arange(n_models)  # [i_nominal]
        s_TF_plants = [pym.Signal(f"Plant{i_model}") for i_model in i_plot]
        s_TF_loops = [pym.Signal(f"Loop{i_model}") for i_model in i_plot]
        s_TF_Ss = [pym.Signal(f"S{i_model}") for i_model in i_plot]
        s_TF_Ts = [pym.Signal(f"T{i_model}") for i_model in i_plot]
        s_TF_PSs = [pym.Signal(f"PS{i_model}") for i_model in i_plot]
        for i, i_model in enumerate(i_plot):
            # Plant TF
            fn_control.append(ct.TransferFunctionSS([ss_plants[i_model], s_wrange], s_TF_plants[i]))

            # Calculate open loop transfer
            fn_control.append(pym.MathGeneral([s_TF_contr, s_TF_plants[i]], s_TF_loops[i], expression="inp0*inp1"))

            # Calculate closed loop
            s_TF_closedloop = pym.Signal("TF_closedloop")
            fn_control.append(pym.MathGeneral(s_TF_loops[i], s_TF_closedloop, expression="1/(1+inp0)"))
            s_Snrm=pym.Signal(f"sensitivity_norm{i_model}")
            fn_control.append(pym.ComplexNorm(s_TF_closedloop, s_Snrm))
            fn_control.append(pym.MathGeneral(s_Snrm, s_TF_Ss[i], expression='20*log(inp0, 10)'))

        cols = np.array(getDistinctColors(len(i_plot)*n_positions*n_controllers)).reshape((len(i_plot), n_positions, n_controllers, 3))

        # Open-Loop Nyquist plot
        mod_nyq = ct.PlotNyquist(stab_lim=6.0, saveto=Path(save_dir, 'nyquist', 'nyquist.png'), iter=ct.Iteration('mma'))
        for i, s_TF_loop in enumerate(s_TF_loops):
            mod_nyq.add_xyplot(complex_data=s_TF_loop, label=s_TF_loop.tag, color=cols[i, ...])
        for i, i_model in enumerate(i_plot):
            redmodel = redmodels[i_model]
            for i_pos in range(n_positions):
                mod_nyq.add_xyplot('.', complex_data=redmodel['Gvalues'][i_pos,...], color=cols[i, i_pos, 0, ...])
                mod_nyq.add_circles(redmodel['midpts'][i_pos, ...], redmodel['radii'][i_pos, ...], color=cols[i, i_pos, 0, ...], alpha=0.2, linewidth=0.5, linestyle=":", which=([0], [0]))
        fn_control.append(mod_nyq)

        # Open-loop Bode plot
        mod_bode = ct.PlotBode(s_wrange, saveto=Path(save_dir, 'bode', 'bode.png'), iter=ct.Iteration('mma'))
        for s in s_pidwb_sep:  # Bandwidth line
            mod_bode.add_vertical_line(s, 'k:', label=s.tag, which='both')
        for i, (s_TF_plant, s_TF_loop) in enumerate(zip(s_TF_plants, s_TF_loops)):
            mod_bode.add_transfer(s_TF_plant, '--', label=s_TF_plant.tag, color=cols[i, ...])
            mod_bode.add_transfer(s_TF_loop, '-', label=s_TF_loop.tag, color=cols[i, ...])
        fn_control.append(mod_bode)

        # Sensitivity plot
        mod_sens = ct.PlotSensitivity(s_wrange, stab_lim=6.0, saveto=Path(save_dir, 'sensitivity', 'sensitivity.png'), iter=ct.Iteration('mma'), ylim=[-50, 10])
        for s in s_pidwb_sep:
            mod_sens.add_vertical_line(s, 'k:', label=s.tag)
        for i, s_TF_S in enumerate(s_TF_Ss):
            mod_sens.add_xyplot(s_TF_S, label=s_TF_S.tag, color=cols[i, ...])
        fn_control.append(mod_sens)
        # End plot

    # Preoptimize controller
    preopt_controller = initial_wb is None
    if preopt_controller:
        blks_control = pym.Network([*fn_control, *fn_plot])
        responses_ctrl = [sg_0, *s_gcirc_cl_der]

        mma1 = ct.MMA(blks_control, sx_wb, responses_ctrl, maxit=20, xmin=0, xmax=xwmax, move=0.1, epsimin=1e-9, tolx=1e-7)
        mma1.response()
    else:  # Initial bandwidth is given as an argument
        initial_xwb = xwmax*np.log(initial_wb / wbmin) / np.log(wbmax / wbmin)
        sx_wb.state = np.array([initial_xwb])

    [r.rescale() for r in rescales] # Rescale objectives to 100

    # Integrated optimization
    variables = [sx, sx_wb]
    move = [0.05, 0.05]  # Move limit

    blks_total = pym.Network([*fn_plant, *fn_control, *fn_plot])
    blks_total.append(pym.PlotIter(variables[1:]))

    responses = [sg_0, *s_gcirc_cl_der, sg_maxvol, sg_minvol]

    def fn_callback():
        ct.floodfill_py(model, sx.state, ndd, threshold=0.3)

    if optimize_integrated:
        # pym.finite_difference(blks_total, [variables[1], variables[0]], [s_circ_cl_dists], dx=1e-5)
        mma2 = ct.MMA(blks_total, variables, responses, maxit=integrated_maxit, xmin=0, xmax=[1, xwmax], move=move, epsimin=1e-9, tolx=0.0, fn_callback=fn_callback if do_floodfill else None)
        mma2.response()
    elif integrated_maxit == 1:  # Only do one evaluation
        fn_control.response()

    return sx.state, s_pidwb.state[0] if s_pidwb.state is not None else -1


if __name__ == '__main__':
    main_dir = Path('out/paper')

    cases = ['case1', 'case2']
    case_settings_list = [
        {'symmetry': False, 'setup_case': setup_model_siso, 'wbrange': (0.4, 50.0)},  # Stage
        {'symmetry': False, 'setup_case': setup_model_siso1, 'wbrange': (0.1, 10.0)}  # Lift
    ]

    # Robust experiment
    robust_experiment = False
    if robust_experiment:
        save_dir = Path(main_dir, cases[1], 'robust_exp')
        rad_rob = 5
        # Optimize only eigenfrequencies with a 'virtual' tip mas
        case_fn = lambda nn, npos: setup_model_siso1(nn, npos, tip_mass=1.0)
        case_sett = {'symmetry': False, 'setup_case': case_fn, 'wbrange': (0.1, 10.0)}
        x_ef, _ = run_problem(save_dir=Path(save_dir, 'topology'),
                              eigfreq_maxit=100, optimize_eigenfrequencies=True, optimize_integrated=False,
                              robust=True, n_pert=1, max_eta_offset=0.1, beta_lastit=90, modal_zeta=0.01,
                              rho_interpolation_t=0.2, initial_wb=0.01, filter_radius=rad_rob, **case_sett)

        _, wb_ef = run_problem(n_positions=6, save_dir=Path(save_dir, "controller"),
                               robust=True, n_pert=1, max_eta_offset=0.1, beta_start=20.0, modal_zeta=0.01,
                               x0=x_ef, filter_radius=rad_rob, optimize_integrated=False, **case_settings_list[1])

        run_problem(n_positions=6, save_dir=Path(save_dir, 'projected'),
                    x0=x_ef, initial_wb=wb_ef, optimize_integrated=False, integrated_maxit=1, filter_radius=rad_rob,
                    robust=True, n_pert=20, max_eta_offset=0.2, beta_start=20.0, modal_zeta=0.01, plot=False, **case_settings_list[1])
        run_problem(n_positions=6, save_dir=Path(save_dir, 'explicit'),
                    x0=x_ef, initial_wb=wb_ef, optimize_integrated=False, integrated_maxit=1, filter_radius=rad_rob,
                    robust=True, n_pert=20, max_eta_offset=0.2, beta_start=20.0, modal_zeta=0.01, plot=False, each_robust_explicit=True, **case_settings_list[1])
        exit()

    eigenvalue_reference = False
    if eigenvalue_reference:
        rad = 2
        x_ef, _ = run_problem(save_dir=Path(main_dir, cases[1], 'eigenfrequency_ref', 'topology'), rho_interpolation_t=0.2, optimize_eigenfrequencies=True, optimize_integrated=False, initial_wb=0.01, filter_radius=rad, **case_settings_list[1])
        for npos in [1, 3, 6]:
            run_problem(n_positions=npos, save_dir=Path(main_dir, cases[1], 'eigenfrequency_ref', f"{npos}pos"), x0=x_ef, optimize_integrated=False, filter_radius=rad, **case_settings_list[1])
        exit()


    integrated_optimization = True
    if integrated_optimization:
        cur_settings = dict(filter_radius=2, n_positions=1, **case_settings_list[1])
        save_dir = Path(main_dir, cases[1], f'integrated_{cur_settings["n_positions"]}pos')

        # Preoptimize controller
        _, wb0 = run_problem(save_dir=Path(save_dir, 'controller'), optimize_integrated=False, **cur_settings)
        # wb0 = 0.49449227843679755
        run_problem(integrated_maxit=500, save_dir=Path(save_dir, 'circle'), initial_wb=wb0, **cur_settings)
        exit()


    robust_settings = dict(robust=True, filter_radius=8, n_pert=1, beta_start=1.0, beta_firstit=50, beta_end=20.0, max_eta_offset=0.05)
    robust_optimization = False
    if robust_optimization:
        npos = 1
        save_dir = Path(main_dir, cases[1], f'integrated_{npos}pos_robust')

        # Preoptimize controller
        _, wb0 = run_problem(n_positions=npos, save_dir=Path(save_dir, 'controller'), **robust_settings, optimize_integrated=False, plot=False, **case_settings_list[1])
        x, wb = run_problem(n_positions=npos, save_dir=Path(save_dir, 'circle'), **robust_settings, initial_wb=wb0, plot=False, **case_settings_list[1])

        # Validation
        run_problem(n_positions=npos, save_dir=Path(save_dir, 'validation_projected'),
                    x0=x, initial_wb=wb, optimize_integrated=False, integrated_maxit=1, filter_radius=robust_settings['filter_radius'],
                    robust=True, n_pert=15, max_eta_offset=robust_settings['max_eta_offset']*2, beta_start=robust_settings['beta_end'], plot=False, **case_settings_list[1])
        run_problem(n_positions=npos, save_dir=Path(save_dir, 'validation_explicit'),
                    x0=x, initial_wb=wb, optimize_integrated=False, integrated_maxit=1, filter_radius=robust_settings['filter_radius'],
                    robust=True, n_pert=15, max_eta_offset=robust_settings['max_eta_offset']*2, beta_start=robust_settings['beta_end'], plot=False, each_robust_explicit=True, **case_settings_list[1])
        exit()

    robust_eigenfreq_ref = False
    if robust_eigenfreq_ref:
        robust_settings['beta_firstit'] = 15
        robust_settings['beta_lastit'] = 90
        x_ef, _ = run_problem(save_dir=Path(main_dir, cases[1], 'robust_eigenfrequency_ref', 'topology'), **robust_settings, rho_interpolation_t=0.2, optimize_eigenfrequencies=True, optimize_integrated=False, initial_wb=0.01, **case_settings_list[1])
        for npos in [1, 3, 6]:
            run_problem(n_positions=npos, save_dir=Path(main_dir, cases[1], 'robust_eigenfrequency_ref', f"{npos}pos"),
                        x0=x_ef, optimize_integrated=False, filter_radius=robust_settings['filter_radius'], robust=True, n_pert=robust_settings['n_pert'],
                        max_eta_offset=robust_settings['max_eta_offset'], beta_start=robust_settings['beta_end'], **case_settings_list[1])
        exit()

    validate_robust = False
    if validate_robust:
        npos = 6
        save_dir = Path(main_dir, cases[1], f'integrated_{npos}pos_robust')
        load_restart = Path(save_dir, 'circle', 'restart', 'restart.0199.npy')
        with open(load_restart, 'rb') as f:
            x = np.load(f)
            wb = np.load(f)

        run_problem(n_positions=npos, save_dir=Path(save_dir, 'validation_projected'),
                    x0=x, initial_wb=wb, optimize_integrated=False, integrated_maxit=1, filter_radius=robust_settings['filter_radius'],
                    robust=True, n_pert=15, max_eta_offset=robust_settings['max_eta_offset']*2, beta_start=robust_settings['beta_end'], plot=False, **case_settings_list[1])
        run_problem(n_positions=npos, save_dir=Path(save_dir, 'validation_explicit'),
                    x0=x, initial_wb=wb, optimize_integrated=False, integrated_maxit=1, filter_radius=robust_settings['filter_radius'],
                    robust=True, n_pert=15, max_eta_offset=robust_settings['max_eta_offset']*2, beta_start=robust_settings['beta_end'], plot=False, each_robust_explicit=True, **case_settings_list[1])
        exit()

    eigenfrequency = True
    if eigenfrequency:

        run_problem(optimize_eigenfrequencies=True, filter_radius=2, **case_settings_list[1])
        exit()

    ncases = len(cases)
    for i in [1]:#range(ncases):
        case_settings = case_settings_list[i]
        position_list = [1, 3, 6]

        # Eigenfrequency reference design
        x_ef, _ = run_problem(save_dir=Path(main_dir, cases[i], 'eigenfrequency'), optimize_eigenfrequencies=True, optimize_integrated=False, initial_wb=0.01, **case_settings)

        for npos in position_list:
            casedir = Path(main_dir, cases[i], f"{npos}pos")

            # Eigenfrequency reference controller
            run_problem(n_positions=npos, save_dir=Path(casedir, 'eigenfrequency'), x0=x_ef, optimize_integrated=False, **case_settings)


            # Preoptimize controller
            # _, wb0 = run_problem(n_positions=npos, save_dir=Path(casedir, 'controller'), optimize_integrated=False, **case_settings)

            do_settings = [#{'do_inconsistent_mass_int': False, 'do_lowdens_penalty': False, 'do_floodfill': False}]#,  # All off
                           # {'do_inconsistent_mass_int': True, 'do_lowdens_penalty': False, 'do_floodfill': False},  # Inconsistent mass only
                           {'do_inconsistent_mass_int': False, 'do_lowdens_penalty': False, 'do_floodfill': False}]#,  # Penalty only
                           # {'do_inconsistent_mass_int': False, 'do_lowdens_penalty': False, 'do_floodfill': True},  # Floodfill only
                           # {'do_inconsistent_mass_int': False, 'do_lowdens_penalty': True, 'do_floodfill': True},  # No inconsistent mass only
                           # {'do_inconsistent_mass_int': True, 'do_lowdens_penalty': False, 'do_floodfill': True},  # No penalty only
                           # {'do_inconsistent_mass_int': True, 'do_lowdens_penalty': True, 'do_floodfill': False}]  # No floodfill only

            for ds in do_settings:
                do_saveto = f"im{1 if ds['do_inconsistent_mass_int'] else 0}_pn{1 if ds['do_lowdens_penalty'] else 0}_ff{1 if ds['do_floodfill'] else 0}"
                # run_problem(n_positions=npos, save_dir=Path(casedir, 'localmodes', f'{do_saveto}_circle'), initial_wb=wb0, **ds, **case_settings)

            # Run optimization for the two methods
            # run_problem(n_positions=npos, save_dir=Path(casedir, 'circle'), initial_wb=wb0, do_inconsistent_mass_int=False, do_lowdens_penalty=False, do_floodfill=True, **case_settings)

            # Run optimization for the two methods with robust on
            # run_problem(n_positions=npos, save_dir=Path(casedir, 'robust_circle_beta20'), initial_wb=wb0, robust=True, do_inconsistent_mass_int=False, do_lowdens_penalty=False, do_floodfill=True, **case_settings)

    print("Alles goed :)")

