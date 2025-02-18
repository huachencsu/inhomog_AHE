import matplotlib.pyplot as plt
import fenics as fen
import numpy as np
import scipy.special as sp
import math
import time
import os

##########################################
####### Random field generation ##########
##########################################

def grf_gen(_lcorr, _meshN):
    ''' Generate Gaussian random field g(x,y) that has the following properties:
        1. defined on [0,1) x [0,1)
        2. has _meshN discrete points along each direction
        3. has zero mean
        4. has the covariance <g(x,y)*g(x',y')> = exp[-( (x-x')^2 + (y-y')^2 )/(2*_lcorr^2)]
    '''
    fft = np.fft
    fftfreq = fft.fftfreq
    grfshape = (_meshN, _meshN)
    # generate the k grid
    all_k = [fftfreq(s, 1.0/_meshN) for s in grfshape]
    kgrid = np.meshgrid(*all_k, indexing = "ij")
    knorm = np.sqrt(np.sum(np.power(kgrid,2),axis=0))
    def Pk(_k):
        return 2 * np.pi * _meshN**4 * _lcorr**2 * np.exp(-2 * np.pi**2 * _k**2 * _lcorr ** 2 )
    field = np.random.normal(loc=0, scale=1, size=grfshape)
    fftfield = fft.fftn(field) / _meshN * np.sqrt( Pk(knorm) )
    # 1/_meshN is needed so that the real and imaginary parts of fftn(field) at each k
    # are separately uncorrelated random variables with zero mean and variance = 0.5
    return np.real(fft.ifftn(fftfield))

def sigma_gen(_grf,
              _sig0, _sigh, _sigd1, _sigd2,
              _h, _hc, _r,
              _eh, _grfscale, _ew):
    ''' Create the conductivity tensor defined on the same mesh as the given _grf.
        sigma(x,y) = _sig0 * (1   0)
                             (0   1)

                   + _sigh * (1 + _r * _grf(x,y)) * (0  -1) * erf ( (_h-_hc)/_eh + _grfscale *_grf(x,y) )
                                                    (1   0)

                   + f(x,y) * R[theta(x,y)] * (_sigd1     0) * R^(-1)[theta(x,y)]
                                              (0     _sigd2)
        where
                  f(x,y) = 1.0 - erf ( 1.0 / _ew * ((_h-_hc)/_eh + _grfscale * _grf(x,y)) ) ** 2
                  R(t) = (cos(t)  -sin(t))
                         (sin(t)   cos(t))
                  theta(x,y) = np.arctan2( grad_y f(x,y), grad_x f(x,y) )
        '''    
    _meshN = _grf.shape[0]
    sigma_grf = np.zeros((2, 2, _meshN, _meshN))
    grf_M = sp.erf( ((_h-_hc)/_eh + _grfscale * _grf) )   # magnetic domain profile
    grf_sigh = (1.0 + _r*_grf) * grf_M
    f = 1.0 - (sp.erf( 1.0 / _ew * ((_h-_hc)/_eh + _grfscale * _grf ) ))** 2
    gradf = np.gradient(f)
    theta = np.arctan2(gradf[1], gradf[0])  # arctan2(y,x)
    cs = np.cos(theta)
    sn = np.sin(theta)
    grf_sigdxx = (cs**2 * _sigd1 + sn**2 * _sigd2) * f
    grf_sigdyy = (cs**2 * _sigd2 + sn**2 * _sigd1) * f
    grf_sigdxy = cs * sn * (_sigd1 - _sigd2) * f
    sigma_grf[0, 0] = _sig0 + grf_sigdxx
    sigma_grf[0, 1] = -1.0 * _sigh * grf_sigh + grf_sigdxy
    sigma_grf[1, 0] = _sigh * grf_sigh + grf_sigdxy
    sigma_grf[1, 1] = _sig0 + grf_sigdyy
    return sigma_grf, f, grf_M


###########################################
####### Solve transport equation ##########
###########################################


def solve_phi(sigma_grf):
    ''' Solve the electric potential phi(x,y) in a unit square [0, 1] x [0, 1].
        The electric field is defined as E = - \nabla \phi.
        The electric current is defined as J = \sigma \cdot E = - \sigma \cdot \nabla \phi.
        The transport equation is \nabla \cdot J = -\nabla \cdot (\sigma \cdot \nabla \phi) = 0.
        The conductivity tensor is provided as sigma_grf.
        The boundary condition is J\cdot \hat{n}_L = -1, J\cdot \hat{n}_R = +1. Namely, uniform current
        flowing along x.
        The solved phi, FEniCS mesh, sig00, sig01, sig10, sig11, and xtol are returned for further processing.
    '''
    meshN = sigma_grf.shape[2]
    # Initialize FEniCS mesh and function space
    meshN_fen = meshN - 1 # -1 is because fenics mesh includes boundary points
    mesh = fen.UnitSquareMesh(meshN_fen, meshN_fen)    
    V = fen.FunctionSpace(mesh, 'P', 1)
    # ('P', 1) corresponds to the P_1 family of the finite element function space in the
    # "Periodic Table of the Finite Elements". mesh indicates that the P_1 discrete functions are defined
    # only on the vertices of the given mesh.
    coord_list = V.tabulate_dof_coordinates()
    # coord_list includes the coordinates of all vertices in the mesh. For a unit square mesh of size meshN_fen x meshN_fen
    # it has (meshN_fen + 1)**2 vertices. The x and y coordinates of the vertices range from 0 to 1 with step length 1.0/(meshN_fen)
    grf_mat_index = np.array([[x[0]*meshN_fen, x[1]*meshN_fen] for x in coord_list]).astype(int)
    # build the conductivity tensor as a fenics function
    sig00=fen.Function(V)
    sig01=fen.Function(V)
    sig10=fen.Function(V)
    sig11=fen.Function(V)
    sig00.vector()[:] = sigma_grf[0,0][grf_mat_index[:,0],grf_mat_index[:,1]]
    sig01.vector()[:] = sigma_grf[0,1][grf_mat_index[:,0],grf_mat_index[:,1]]
    sig10.vector()[:] = sigma_grf[1,0][grf_mat_index[:,0],grf_mat_index[:,1]]
    sig11.vector()[:] = sigma_grf[1,1][grf_mat_index[:,0],grf_mat_index[:,1]]
    u=fen.TrialFunction(V)
    v=fen.TestFunction(V)
    f=fen.Constant(0)   
    xtol = 0.1/meshN_fen   # tolerance for coordinates
    exp1 = fen.Expression("x[0] < xtol ? -1 : 0", xtol = xtol, degree=4)
    exp2 = fen.Expression("x[0] > 1-xtol ? +1 : 0", xtol = xtol, degree=4)
    g = exp1 + exp2 
    a=(
        ( sig00 * u.dx(0)
        + sig01 * u.dx(1)
        )*v.dx(0)
        +
        ( sig10 * u.dx(0)
        + sig11 * u.dx(1)
        )*v.dx(1)
        )*fen.dx
    L = f*v*fen.dx - g*v*fen.ds
    phi = fen.Function(V)
    fen.set_log_active(False) # this suppresses fenics printing out a bunch of stuff
    fen.solve(a == L, phi)
    return phi, mesh, sig00, sig01, sig10, sig11, xtol

def calc_R(phi, 
           left_bound_list,
           right_bound_list,
           top_bound_list,
           bot_bound_list,
           Ix = 1.0):
    ''' Calculate the longitudinal and Hall resistance from the electric potential phi solved by FEniCS.
        left/right/top/bottom_bound_list includes the coordinates of the respective boundary sites.
        Ix is the total current flowing along x through the system.
        The voltage difference along x is U_x = phi_left_avg - phi_right_avg
                                     y    U_y = phi_bot_avg - phi_top_avg
        The resistance is calculated as
        R_xx = U_x / I_x
        R_yx = U_y / I_x
    '''
    top_volt = 0
    bot_volt = 0
    left_volt = 0
    right_volt = 0
    for j in top_bound_list:
        top_volt += phi(j)
    for j in bot_bound_list:
        bot_volt += phi(j)
    for j in left_bound_list:
        left_volt += phi(j)
    for j in right_bound_list:
        right_volt += phi(j)
    R_yx = bot_volt/len(bot_bound_list) - top_volt/len(top_bound_list)
    R_xx = left_volt/len(left_bound_list) - right_volt/len(right_bound_list)
    R_yx /= Ix
    R_xx /= Ix
    return R_xx, R_yx

def calc_R_all(phi_Ix, phi_Iy, 
           left_bound_list,
           right_bound_list,
           top_bound_list,
           bot_bound_list,
           Ixy = 1.0):
    ''' Calculate the whole resistance tensor from the electric potential phi_Ix and phi_Iy solved by FEniCS.
        phi_Ix is the solution for a given conductivity tensor.
        phi_Iy is the solution for rotating the system clockwise by 90 degrees.
        The calc_R function is called twice for phi_Ix and phi_Iy, respectively.
        For phi_Ix, the returned components are R_xx, R_yx.
            phi_Iy                              R_yy, -R_xy
    '''
    R_xx, R_yx = calc_R(phi_Ix, 
               left_bound_list,
               right_bound_list,
               top_bound_list,
               bot_bound_list,
               Ixy)
    R_yy, R_xy = calc_R(phi_Iy, 
               left_bound_list,
               right_bound_list,
               top_bound_list,
               bot_bound_list,
               Ixy)
    Rtensor = np.array([[R_xx, -R_xy],
                        [R_yx,  R_yy]])
    return Rtensor

def rot_sigma_grf(sigma_grf):
    ''' Rotate the system defined by sigma_grf(x,y) clockwise by 90 degrees.
        A tensor field sigma(r) rotated by a rotation matrix R transforms as
        sigma(r) --> sigma'(r) = R.sigma(R^T.r).R^T
        For R = (0,  1),   sigma(r) = (s11, s12) (x, y)
                (-1, 0)               (s21, s22)
        sigma'(r) = (s22, -s21) (-y, x)
                    (-s12  s11)
        Since sigma_grf is defined in [0,1) x [0,1), the rotated sigma_grf
        will be defined in [0, 1) x [0, -1). It is then shifted to [0, 1) x [0, 1)
        by replacing y with y -> y - 1. The final result is
        sigma'(r) = (s22, -s21) (1-y, x)
                    (-s12  s11)                                                              
    '''
    sigma_grf_rot = np.zeros(sigma_grf.shape)
    sigma_grf_rot[0,0,:,:] = sigma_grf[1,1].T[::-1,:]
    sigma_grf_rot[0,1,:,:] = -1.0*sigma_grf[1,0].T[::-1,:]
    sigma_grf_rot[1,0,:,:] = -1.0*sigma_grf[0,1].T[::-1,:]
    sigma_grf_rot[1,1,:,:] = sigma_grf[0,0].T[::-1,:]
    return sigma_grf_rot 

def calc_J(phi, mesh, sig00, sig01, sig10, sig11):
    ''' Calculate the electric current density using the phi solved by FEniCS.
        The electric current is defined as J = \sigma \cdot E = - \sigma \cdot \nabla \phi
        mesh              :  FEniCS mesh used to obtain phi, returned by solve_phi()
        (sig00  sig01)
        (sig10  sig11)    :  Conductivity tensor in terms of FEniCS functions, returned by solve_phi()   
    '''
    # functionspace for J as a vector function
    W = fen.VectorFunctionSpace(mesh,'P', 1)
    # Combine two scalar functions to a vector, followed by
    # projecting the vector to the vector function space.
    J = fen.project( fen.as_vector(
        [ -sig00*phi.dx(0) - sig01*phi.dx(1),
          -sig10*phi.dx(0) - sig11*phi.dx(1) ]) , W)
    return J

def gen_bound_list(mesh, xtol):
    ''' Generate lists of boundary sites, defined by vertices within xtol by a given boundary.
    '''
    left_bound_list=[]
    right_bound_list=[]
    top_bound_list=[]
    bot_bound_list=[]
    mesh_coord=(mesh.coordinates())   
    for x in mesh_coord:
        if x[0] < xtol:
            left_bound_list.append(x)
        elif x[0] > 1.0 - xtol:
            right_bound_list.append(x)
        if x[1] > 1.0 - xtol:
            top_bound_list.append(x)
        elif x[1] < xtol:
            bot_bound_list.append(x) 
    return left_bound_list, right_bound_list, top_bound_list, bot_bound_list

def calc_hyst(_grf,
              _sig0, _sigh, _sigd1, _sigd2,
              _hc, _r,
              _eh, _grfscale, _ew,
              _hmax, _hnum,
              _loadfile = True):
    ''' Calculate the hysteresis loops of resistance based on a given random field.
        Also return the conductance tensor from directly inverting the resistance,
        and the spatially averaged conductivity tensor.
        See Docstring of sigma_gen() for the definitions of the following arguments:
        _grf, _sig0, _sigh, _sigd1, _sigd2, _hc, _r, _eh, _grfscale, _ew
        _hmax      :    maximum value of the magnetic field
        _hnum      :    number of magnetic field values
        _loadfile  :    whether to load data from files
    '''
    meshN = _grf.shape[0]
    if not _loadfile:
        # Generate list of h values symmetric about 0
        hplus = np.linspace(0.0, _hmax, np.int32(_hnum/2)+1, endpoint = True)
        hminus = -1.0*hplus[-1:0:-1]
        hlist = np.concatenate((hminus,hplus))
        hnumtot = len(hlist)
        # Resistance, conductance, and spatially averaged conductivity
        Rtensorlist = np.zeros((hnumtot, 2, 2, 2))   # (hnumtot, sweep up or down, Rtensor row,  Rtensor column)
        Gtensorlist = np.zeros((hnumtot, 2, 2, 2))
        sigmaavglist = np.zeros((hnumtot, 2, 2, 2))
        # Generate boundary lists only once
        meshN_fen = meshN - 1 # -1 is because fenics mesh includes boundary points
        mesh = fen.UnitSquareMesh(meshN_fen, meshN_fen)
        xtol = 0.1/meshN_fen   # tolerance for coordinates
        left_bound_list, right_bound_list, top_bound_list, bot_bound_list = gen_bound_list(mesh, xtol)
        # Loop through hlist
        starttime = time.time()
        for i in range(hnumtot):
            _h = hlist[i]
            #####################
            ###### up_sweep #####
            #####################
            # Generate sigma_grf for up_sweep
            sigma_grf_u, f_u, grf_sigh_u = sigma_gen(_grf,
                                         _sig0, _sigh, _sigd1, _sigd2,
                                         _h, _hc, _r,
                                         _eh, _grfscale, _ew) 
            # Solve phi_Ix, phi_Iy for up_sweep
            phi_Ix, mesh, sig00, sig01, sig10, sig11, xtol = solve_phi(sigma_grf_u)
            phi_Iy, mesh, sig00, sig01, sig10, sig11, xtol = solve_phi(rot_sigma_grf(sigma_grf_u))
            # Calculate R tensor for up_sweep
            R_u = calc_R_all(phi_Ix, phi_Iy, 
                            left_bound_list,
                            right_bound_list,
                            top_bound_list,
                            bot_bound_list,
                            Ixy = 1.0)
            Rtensorlist[i,0] = R_u
            # Calculate G tensor for up_sweep
            G_u = np.linalg.inv(R_u)
            Gtensorlist[i,0] = G_u
            # Calculate average conductivity for up_sweep
            sigmaavg_u = np.mean(sigma_grf_u, axis = (2, 3))
            sigmaavglist[i,0] = sigmaavg_u
            #####################
            #### down_sweep #####
            #####################        
            # Generate sigma_grf for down_sweep
            # _grf, _r, _hc should change sign
            sigma_grf_d, f_d, grf_sigh_d = sigma_gen(-1.0*_grf,
                                         _sig0, _sigh, _sigd1, _sigd2,
                                         _h, -1.0*_hc, -1.0*_r,
                                         _eh, _grfscale, _ew)          
            # Solve phi_Ix, phi_Iy for down_sweep
            phi_Ix, mesh, sig00, sig01, sig10, sig11, xtol = solve_phi(sigma_grf_d)
            phi_Iy, mesh, sig00, sig01, sig10, sig11, xtol = solve_phi(rot_sigma_grf(sigma_grf_d))
            # Calculate R tensor for down_sweep
            R_d = calc_R_all(phi_Ix, phi_Iy, 
                            left_bound_list,
                            right_bound_list,
                            top_bound_list,
                            bot_bound_list,
                            Ixy = 1.0)
            Rtensorlist[i,1] = R_d
            # Calculate G tensor for down_sweep
            G_d = np.linalg.inv(R_d)
            Gtensorlist[i,1] = G_d
            # Calculate average conductivity for down_sweep
            sigmaavg_d = np.mean(sigma_grf_d, axis = (2, 3))
            sigmaavglist[i,1] = sigmaavg_d
            # Print results for the current h
            print("h = ", _h)
            print("R_u = %12.6f, %12.6f, %12.6f, %12.6f" % tuple(R_u.flatten()))
            print("R_d = %12.6f, %12.6f, %12.6f, %12.6f" % tuple(R_d.flatten()))
            print("G_u = %12.6f, %12.6f, %12.6f, %12.6f" % tuple(G_u.flatten()))
            print("G_d = %12.6f, %12.6f, %12.6f, %12.6f" % tuple(G_d.flatten()))
            print("sig_u = %12.6f, %12.6f, %12.6f, %12.6f" % tuple(sigmaavg_u.flatten()))
            print("sig_d = %12.6f, %12.6f, %12.6f, %12.6f" % tuple(sigmaavg_d.flatten()))
            if i == 0:
                timecheck = time.time()
                print("Estimated total time needed: ", (timecheck - starttime) * hnumtot)
        # Symmetric part
        R_S = 0.5 * (Rtensorlist[:,:,:,:] + Rtensorlist[::-1,::-1,:,:])
        G_S = 0.5 * (Gtensorlist[:,:,:,:] + Gtensorlist[::-1,::-1,:,:])
        sigavg_S = 0.5 * (sigmaavglist[:,:,:,:] + sigmaavglist[::-1,::-1,:,:])
        # Antisymmetric part    
        R_A = 0.5 * (Rtensorlist[:,:,:,:] - Rtensorlist[::-1,::-1,:,:])
        G_A = 0.5 * (Gtensorlist[:,:,:,:] - Gtensorlist[::-1,::-1,:,:])
        sigavg_A = 0.5 * (sigmaavglist[:,:,:,:] - sigmaavglist[::-1,::-1,:,:])
        # Save results
        np.savetxt('hlist_'
                   + '.dat', 
                   hlist, fmt='%16.10f')
        np.savetxt('Rtensor_S_meshN_'
                   + str(meshN) 
                   + '.dat', 
                   R_S.flatten(), fmt='%16.10f')
        np.savetxt('Rtensor_A_meshN_'
                   + str(meshN) 
                   + '.dat', 
                   R_A.flatten(), fmt='%16.10f')
        np.savetxt('Gtensor_S_meshN_'
                   + str(meshN) 
                   + '.dat', 
                   G_S.flatten(), fmt='%16.10f')
        np.savetxt('Gtensor_A_meshN_'
                   + str(meshN) 
                   + '.dat', 
                   G_A.flatten(), fmt='%16.10f')
        np.savetxt('Sigavg_S_meshN_'
                   + str(meshN) 
                   + '.dat', 
                   sigavg_S.flatten(), fmt='%16.10f')
        np.savetxt('Sigavg_A_meshN_'
                   + str(meshN) 
                   + '.dat', 
                   sigavg_A.flatten(), fmt='%16.10f')   
    else:
        # load files
        hlist = np.loadtxt('hlist_'
                   + '.dat')
        hnumtot = len(hlist)
        tensorshape = (hnumtot, 2, 2, 2)
        R_S = np.loadtxt('Rtensor_S_meshN_'
                   + str(meshN) 
                   + '.dat').reshape(tensorshape)
        R_A = np.loadtxt('Rtensor_A_meshN_'
                   + str(meshN) 
                   + '.dat').reshape(tensorshape)
        G_S = np.loadtxt('Gtensor_S_meshN_'
                   + str(meshN) 
                   + '.dat').reshape(tensorshape)
        G_A = np.loadtxt('Gtensor_A_meshN_'
                   + str(meshN) 
                   + '.dat').reshape(tensorshape)
        sigavg_S = np.loadtxt('Sigavg_S_meshN_'
                   + str(meshN) 
                   + '.dat').reshape(tensorshape)
        sigavg_A = np.loadtxt('Sigavg_A_meshN_'
                   + str(meshN) 
                   + '.dat').reshape(tensorshape)
    # Plot the hysteresis curves
    #######################
    # R_S (xx,xy,yx,yy)  ##
    #######################
    hyst_data = R_S[:,:,0,0]
    filename = 'R_S_xx_meshN_'+str(meshN)
    ylabel = r'$R^S_{xx}$ (a.u.)'
    plot_hyst(hlist, hyst_data, filename, ylabel)
    hyst_data = R_S[:,:,0,1]
    filename = 'R_S_xy_meshN_'+str(meshN)
    ylabel = r'$R^S_{xy}$ (a.u.)'
    plot_hyst(hlist, hyst_data, filename, ylabel)
    hyst_data = R_S[:,:,1,0]
    filename = 'R_S_yx_meshN_'+str(meshN)
    ylabel = r'$R^S_{yx}$ (a.u.)'
    plot_hyst(hlist, hyst_data, filename, ylabel)
    hyst_data = R_S[:,:,1,1]
    filename = 'R_S_yy_meshN_'+str(meshN)
    ylabel = r'$R^S_{yy}$ (a.u.)'
    plot_hyst(hlist, hyst_data, filename, ylabel)
    #######################
    # R_A (xx,xy,yx,yy)  ##
    #######################
    hyst_data = R_A[:,:,0,0]
    filename = 'R_A_xx_meshN_'+str(meshN)
    ylabel = r'$R^A_{xx}$ (a.u.)'
    plot_hyst(hlist, hyst_data, filename, ylabel)
    hyst_data = R_A[:,:,0,1]
    filename = 'R_A_xy_meshN_'+str(meshN)
    ylabel = r'$R^A_{xy}$ (a.u.)'
    plot_hyst(hlist, hyst_data, filename, ylabel)
    hyst_data = R_A[:,:,1,0]
    filename = 'R_A_yx_meshN_'+str(meshN)
    ylabel = r'$R_{\rm h}$ (a.u.)'
    plot_hyst(hlist, hyst_data, filename, ylabel)
    hyst_data = R_A[:,:,1,1]
    filename = 'R_A_yy_meshN_'+str(meshN)
    ylabel = r'$R^A_{yy}$ (a.u.)'
    plot_hyst(hlist, hyst_data, filename, ylabel)    
    #######################
    # G_S (xx,xy,yx,yy)  ##
    #######################
    hyst_data = G_S[:,:,0,0]
    filename = 'G_S_xx_meshN_'+str(meshN)
    ylabel = r'$G^S_{xx}$ (a.u.)'
    plot_hyst(hlist, hyst_data, filename, ylabel)
    hyst_data = G_S[:,:,0,1]
    filename = 'G_S_xy_meshN_'+str(meshN)
    ylabel = r'$G^S_{xy}$ (a.u.)'
    plot_hyst(hlist, hyst_data, filename, ylabel)
    hyst_data = G_S[:,:,1,0]
    filename = 'G_S_yx_meshN_'+str(meshN)
    ylabel = r'$G^S_{yx}$ (a.u.)'
    plot_hyst(hlist, hyst_data, filename, ylabel)
    hyst_data = G_S[:,:,1,1]
    filename = 'G_S_yy_meshN_'+str(meshN)
    ylabel = r'$G^S_{yy}$ (a.u.)'
    plot_hyst(hlist, hyst_data, filename, ylabel)
    #######################
    # G_A (xx,xy,yx,yy)  ##
    #######################
    hyst_data = G_A[:,:,0,0]
    filename = 'G_A_xx_meshN_'+str(meshN)
    ylabel = r'$G^A_{xx}$ (a.u.)'
    plot_hyst(hlist, hyst_data, filename, ylabel)
    hyst_data = G_A[:,:,0,1]
    filename = 'G_A_xy_meshN_'+str(meshN)
    ylabel = r'$G^A_{xy}$ (a.u.)'
    plot_hyst(hlist, hyst_data, filename, ylabel)
    hyst_data = G_A[:,:,1,0]
    filename = 'G_A_yx_meshN_'+str(meshN)
    ylabel = r'$G^A_{yx}$ (a.u.)'
    plot_hyst(hlist, hyst_data, filename, ylabel)
    hyst_data = G_A[:,:,1,1]
    filename = 'G_A_yy_meshN_'+str(meshN)
    ylabel = r'$G^A_{yy}$ (a.u.)'
    plot_hyst(hlist, hyst_data, filename, ylabel) 
    ############################
    # sigavg_S (xx,xy,yx,yy)  ##
    ############################
    hyst_data = sigavg_S[:,:,0,0]
    filename = 'sigavg_S_xx_meshN_'+str(meshN)
    ylabel = r'$\langle\sigma\rangle^S_{xx}$ (a.u.)'
    plot_hyst(hlist, hyst_data, filename, ylabel)
    hyst_data = sigavg_S[:,:,0,1]
    filename = 'sigavg_S_xy_meshN_'+str(meshN)
    ylabel = r'$\langle\sigma\rangle^S_{xy}$ (a.u.)'
    plot_hyst(hlist, hyst_data, filename, ylabel)
    hyst_data = sigavg_S[:,:,1,0]
    filename = 'sigavg_S_yx_meshN_'+str(meshN)
    ylabel = r'$\langle\sigma\rangle^S_{yx}$ (a.u.)'
    plot_hyst(hlist, hyst_data, filename, ylabel)
    hyst_data = sigavg_S[:,:,1,1]
    filename = 'sigavg_S_yy_meshN_'+str(meshN)
    ylabel = r'$\langle\sigma\rangle^S_{yy}$ (a.u.)'
    plot_hyst(hlist, hyst_data, filename, ylabel)
    ############################
    # sigavg_A (xx,xy,yx,yy)  ##
    ############################
    hyst_data = sigavg_A[:,:,0,0]
    filename = 'sigavg_A_xx_meshN_'+str(meshN)
    ylabel = r'$\langle\sigma\rangle^A_{xx}$ (a.u.)'
    plot_hyst(hlist, hyst_data, filename, ylabel)
    hyst_data = sigavg_A[:,:,0,1]
    filename = 'sigavg_A_xy_meshN_'+str(meshN)
    ylabel = r'$\langle\sigma\rangle^A_{xy}$ (a.u.)'
    plot_hyst(hlist, hyst_data, filename, ylabel)
    hyst_data = sigavg_A[:,:,1,0]
    filename = 'sigavg_A_yx_meshN_'+str(meshN)
    ylabel = r'$\langle\sigma\rangle^A_{yx}$ (a.u.)'
    plot_hyst(hlist, hyst_data, filename, ylabel)
    hyst_data = sigavg_A[:,:,1,1]
    filename = 'sigavg_A_yy_meshN_'+str(meshN)
    ylabel = r'$\langle\sigma\rangle^A_{yy}$ (a.u.)'
    plot_hyst(hlist, hyst_data, filename, ylabel)   
    return hlist, R_S, R_A, G_S, G_A, sigavg_S, sigavg_A

def plot_hyst(hlist, hyst_data, filename, ylabel):
    ''' Plot the hysteresis data.
        hlist is the list of magnetic field values of length hnumtot
        hyst_data is a two-dimensional array of shape (hnumtot, 2) where the second
        dimension corresponds to up_sweep and down_sweep.
    '''
    plt.rcParams.update({'font.size': 16})
    x = hlist
    y = hyst_data.T
    plt.figure(1000,figsize=(8,6))
    plt.axes([0.2,0.2,0.6,0.6])
    xrange = x.max() - x.min()
    xmarginfac = 0.08  # Adjust as needed
    plt.xlim([x.min() - xmarginfac*xrange, x.max() + xmarginfac*xrange])
    yrange = y.max() - y.min()
    ymarginfac = 0.1  # Adjust as needed
    plt.ylim([y.min() - ymarginfac*yrange, y.max() + ymarginfac*yrange])
    plt.plot(x,y[0],'-',color='blue',linewidth=2)
    plt.plot(x,y[1],'--',color='red',linewidth=2)
    plt.ylabel(ylabel)
    plt.xlabel(r'$H$ (a.u.)')
    plt.tick_params(axis='x',which='both',top=False)
    plt.tick_params(axis='y',which='both',right=False)
    outfile = filename + '.pdf'
    plt.savefig(outfile,bbox_inches='tight')       
    plt.close(1000)     
    
def calc_hyst_h(_grf,
              _sig0, _sigh, _sigd1, _sigd2,
              _hc, _r,
              _eh, _grfscale, _ew,
              _h):
    ''' Calculate the same data as calc_hyst but at a single h.
        Also return the conductance tensor from directly inverting the resistance,
        and the spatially averaged conductivity tensor.
        See Docstring of sigma_gen() for the definitions of the following arguments:
        _grf, _sig0, _sigh, _sigd1, _sigd2, _hc, _r, _eh, _grfscale, _ew
        _h         :    value of the magnetic field
    '''
    meshN = _grf.shape[0]
    Rtensor = np.zeros((2, 2, 2))   # (sweep up or down, Rtensor row,  Rtensor column)
    Gtensor = np.zeros((2, 2, 2))
    sigmaavg = np.zeros((2, 2, 2))
    # Generate boundary lists only once
    meshN_fen = meshN - 1 # -1 is because fenics mesh includes boundary points
    mesh = fen.UnitSquareMesh(meshN_fen, meshN_fen)
    xtol = 0.1/meshN_fen   # tolerance for coordinates
    left_bound_list, right_bound_list, top_bound_list, bot_bound_list = gen_bound_list(mesh, xtol)
    #####################
    ###### up_sweep #####
    #####################
    # Generate sigma_grf for up_sweep
    sigma_grf_u, f_u, grf_sigh_u = sigma_gen(_grf,
                                 _sig0, _sigh, _sigd1, _sigd2,
                                 _h, _hc, _r,
                                 _eh, _grfscale, _ew) 
    # Solve phi_Ix, phi_Iy for up_sweep
    phi_Ix, mesh, sig00, sig01, sig10, sig11, xtol = solve_phi(sigma_grf_u)
    phi_Iy, mesh, sig00, sig01, sig10, sig11, xtol = solve_phi(rot_sigma_grf(sigma_grf_u))
    # Calculate R tensor for up_sweep
    R_u = calc_R_all(phi_Ix, phi_Iy, 
                    left_bound_list,
                    right_bound_list,
                    top_bound_list,
                    bot_bound_list,
                    Ixy = 1.0)
    Rtensor[0] = R_u
    # Calculate G tensor for up_sweep
    G_u = np.linalg.inv(R_u)
    Gtensor[0] = G_u
    # Calculate average conductivity for up_sweep
    sigmaavg_u = np.mean(sigma_grf_u, axis = (2, 3))
    sigmaavg[0] = sigmaavg_u
    #####################
    #### down_sweep #####
    #####################        
    # Generate sigma_grf for down_sweep
    # _grf, _r, _hc, _h should change sign
    sigma_grf_d, f_d, grf_sigh_d = sigma_gen(-1.0*_grf,
                                 _sig0, _sigh, _sigd1, _sigd2,
                                 -1.0 *_h, -1.0*_hc, -1.0*_r,
                                 _eh, _grfscale, _ew)          
    # Solve phi_Ix, phi_Iy for down_sweep
    phi_Ix, mesh, sig00, sig01, sig10, sig11, xtol = solve_phi(sigma_grf_d)
    phi_Iy, mesh, sig00, sig01, sig10, sig11, xtol = solve_phi(rot_sigma_grf(sigma_grf_d))
    # Calculate R tensor for down_sweep
    R_d = calc_R_all(phi_Ix, phi_Iy, 
                    left_bound_list,
                    right_bound_list,
                    top_bound_list,
                    bot_bound_list,
                    Ixy = 1.0)
    Rtensor[1] = R_d
    # Calculate G tensor for down_sweep
    G_d = np.linalg.inv(R_d)
    Gtensor[1] = G_d
    # Calculate average conductivity for down_sweep
    sigmaavg_d = np.mean(sigma_grf_d, axis = (2, 3))
    sigmaavg[1] = sigmaavg_d
    # Print results for the current h
    print("h = ", _h)
    print("R_u = %12.6f, %12.6f, %12.6f, %12.6f" % tuple(R_u.flatten()))
    print("R_d = %12.6f, %12.6f, %12.6f, %12.6f" % tuple(R_d.flatten()))
    print("G_u = %12.6f, %12.6f, %12.6f, %12.6f" % tuple(G_u.flatten()))
    print("G_d = %12.6f, %12.6f, %12.6f, %12.6f" % tuple(G_d.flatten()))
    print("sig_u = %12.6f, %12.6f, %12.6f, %12.6f" % tuple(sigmaavg_u.flatten()))
    print("sig_d = %12.6f, %12.6f, %12.6f, %12.6f" % tuple(sigmaavg_d.flatten()))

    # Symmetric part
    R_S = 0.5 * (Rtensor[:,:,:] + Rtensor[::-1,:,:])
    G_S = 0.5 * (Gtensor[:,:,:] + Gtensor[::-1,:,:])
    sigavg_S = 0.5 * (sigmaavg[:,:,:] + sigmaavg[::-1,:,:])
    # Antisymmetric part    
    R_A = 0.5 * (Rtensor[:,:,:] - Rtensor[::-1,:,:])
    G_A = 0.5 * (Gtensor[:,:,:] - Gtensor[::-1,:,:])
    sigavg_A = 0.5 * (sigmaavg[:,:,:] - sigmaavg[::-1,:,:])
    return _h, R_S, R_A, G_S, G_A, sigavg_S, sigavg_A    
    
def grf_plot(grf_M, fname='domain_profile'):
    ''' Plot the domain profile in a unit square.
    '''
    plt.rcParams.update({'font.size': 18})    
    Z = grf_M
    plt.figure(1000,figsize=(6,6))
    ax=plt.gca()
    ax.set_aspect('equal')
    extent = [0,1,0,1]
    plt.imshow(Z[:,::-1], extent = extent, cmap = 'seismic')
    ax.set_facecolor('black')
    plt.colorbar(fraction=0.046, pad=0.04)
    xcap = r'$x$'
    ycap = r'$y$'
    plt.ylabel(ycap)
    plt.xlabel(xcap)                    
    outfile = fname + '.pdf'
    plt.savefig(outfile,bbox_inches='tight') 
    plt.close(1000)
    
def plot_bounds(xlist, data, bounds, filename, ylabel, plotbounds = True):
    ''' Generate scatter plots of random instances of data versus x values in xlist.
    '''
    plt.rcParams.update({'font.size': 14})
    x = xlist
    y = data
    plt.figure(1000,figsize=(8,6))
    plt.axes([0.2,0.2,0.6,0.6])
    xrange = x.max() - x.min()
    xmarginfac = 0.12  # Adjust as needed
    plt.xlim([x.min() - xmarginfac*xrange, x.max() + xmarginfac*xrange])
    yrange = y.max() - y.min()
    ymarginfac = 0.15  # Adjust as needed
    plt.ylim([y.min() - ymarginfac*yrange, y.max() + ymarginfac*yrange])
    plt.scatter(x, y, s = 100, alpha = 0.8)
    if plotbounds:
        boundsnum = len(bounds)
        for i in range(boundsnum):
            boundlinex = [x.min() - xmarginfac*xrange, x.max() + xmarginfac*xrange]
            boundliney = [bounds[i], bounds[i]]
            plt.plot(boundlinex, boundliney, '--', lw = 0.75, color = 'red')
    plt.ylabel(ylabel)
    plt.xlabel(r'$H$ (a.u.)')
    plt.tick_params(axis='x',which='both',top=False)
    plt.tick_params(axis='y',which='both',right=False)
    outfile = filename + '.pdf'
    plt.savefig(outfile,bbox_inches='tight')       
    plt.close(1000)    
    
    
##############################################
#### Average sigma_h using Wick's theorem ####
##############################################    
    
def calc_sighavg_Wick(_sigh, _h, _hc, _r, _eh, _VCM = 0.3, _VM = 0.09, _ncutoff = 15, _hcut = 2):
    ''' Calculate the spatially averaged sigma_h using the analytic formula derived
        from Wick's theorem.
        _VCM      :     Covariance of the two random fields C and M at the same location.
        _VM       :     Variance of the random field M at the same location. 
                        Must be small (~0.1) for the series to be convergent. 
        _ncutoff  :     cutoff of the summation over n
        _hcut     :     If (_h - _hc)/_eh > _hcut a constant will be returned.
    '''
    hh = (_h - _hc)/_eh
    rV = _r*_VCM
    if abs(hh) > _hcut:
        sighavg = _sigh * np.sign(hh)
    else:
        def fac(p):
            ''' wrapper for factorial
            '''
            return math.factorial(p)
        def summandN(n):
            ''' wrapper for the summand at a given n
            '''
            summandn = 0.0
            for intk in range(n+1):
                summandn += ( (-1)**n 
                        * (2 * hh**2 )**intk 
                        * fac(2*n) / 2**n / fac(n)
                        / fac(n - intk)
                        / fac(2 * intk)
                        * (  hh / (2 * intk + 1) 
                            + rV
                          )
                        ) * _VM**(n - intk)
            return summandn
        sighavg = 0.0
        for n in range(_ncutoff+1):
            sumn = summandN(n)
            sighavg += sumn
        sighavg *= _sigh * 2 / np.pi**0.5
    return sighavg

def calc_hyst_Wick(_sigh, _hc, _r, _eh, _VCM,
              _VM = 1e-6, _ncutoff = 15, _hcut = 2,
              _hmax = 1.5, _hnum = 100,
              _loadfile = True):
    ''' Calculate the hysteresis loops of the spatially averaged sigma_h calculated using Wick's theorem.
        See Docstring of calc_sighavg_Wick() for the definitions of the following arguments:
        _VCM, _VM, _ncutoff, _hcut
        _hmax      :    maximum value of the magnetic field
        _hnum      :    number of magnetic field values
        _loadfile  :    whether to load data from files
    '''
    if not _loadfile:
        # Generate list of h values symmetric about 0
        hplus = np.linspace(0.0, _hmax, np.int32(_hnum/2)+1, endpoint = True)
        hminus = -1.0*hplus[-1:0:-1]
        hlist = np.concatenate((hminus,hplus))
        hnumtot = len(hlist)
        sighavglist = np.zeros((hnumtot, 2))
        for i in range(hnumtot):
            _h = hlist[i]
            #####################
            ###### up_sweep #####
            #####################
            # Calculate sigmahavg for up_sweep
            sighavg_u = calc_sighavg_Wick(_sigh = _sigh,
                                          _h = _h,
                                          _hc = _hc, 
                                          _r = _r, 
                                          _eh = _eh, 
                                          _VCM = _VCM, 
                                          _VM = _VM, 
                                          _ncutoff = _ncutoff, 
                                          _hcut = _hcut)
            sighavglist[i,0] = sighavg_u
            #####################
            #### down_sweep #####
            #####################        
            # _VCM, _hc should change sign
            sighavg_d = calc_sighavg_Wick(_sigh = _sigh,
                                          _h = _h,
                                          _hc = -1.0*_hc, 
                                          _r = _r, 
                                          _eh = _eh, 
                                          _VCM = -1.0*_VCM, 
                                          _VM = _VM, 
                                          _ncutoff = _ncutoff, 
                                          _hcut = _hcut)       
            sighavglist[i,1] = sighavg_d
            # Print results for the current h
            print("h = ", _h)
            print("sigh_u, sigh_d = %12.6f, %12.6f" % (sighavg_u, sighavg_d))
        # Symmetric part
        sighavg_S = 0.5 * (sighavglist[:,:] + sighavglist[::-1,::-1])
        # Antisymmetric part    
        sighavg_A = 0.5 * (sighavglist[:,:] - sighavglist[::-1,::-1])
        # Save results
        np.savetxt('hlist_Wick_r' + str(_r)
                   + '.dat', 
                   hlist, fmt='%16.10f')
        np.savetxt('Sighavg_S_Wick_r' + str(_r)
                   + '.dat', 
                   sighavg_S.flatten(), fmt='%16.10f')
        np.savetxt('Sighavg_A_Wick_r' + str(_r)
                   + '.dat', 
                   sighavg_A.flatten(), fmt='%16.10f')   
    else:
        # load files
        hlist = np.loadtxt('hlist_Wick_r' + str(_r)
                   + '.dat')
        hnumtot = len(hlist)
        tensorshape = (hnumtot, 2)
        sighavg_S = np.loadtxt('Sighavg_S_Wick_r' + str(_r)
                   + '.dat').reshape(tensorshape)
        sighavg_A = np.loadtxt('Sighavg_A_Wick_r' + str(_r)
                   + '.dat').reshape(tensorshape)
    # Plot the hysteresis curves
    hyst_data = sighavg_S
    filename = 'Sighavg_S_Wick_r' + str(_r)
    ylabel = r'$\langle\sigma\rangle^S_{\rm h}$ (a.u.)'
    plot_hyst(hlist, hyst_data, filename, ylabel)
    hyst_data = sighavg_A
    filename = 'Sighavg_A_Wick_r' + str(_r)
    ylabel = r'$\langle\sigma\rangle_{\rm h}$ (a.u.)'
    plot_hyst(hlist, hyst_data, filename, ylabel) 
    return hlist, sighavg_S, sighavg_A
    
    
##############################################
####### Generate plots in the paper ##########
##############################################
    
def plot_paper(loadgrf = True, loadfile = True, task = 0):
    ''' Generate plots used in the paper.
        task == 0 :  all plots
                1:   Fig. 1a, 1b
                2:   Fig. 2a, 2b
                3:   Fig. 2c, 2d
                4:   Fig. S1
                5:   Fig. R1a, R1b
                6:   Fig. R1c, R1d
                7:   Fig. R2a, R2b
                8:   Fig. R2c, R2d
    '''
    if task == 0 or task == 1:
        # Fig. 1a parameters
        hc = 0.8
        sigh = 0.05
        eph = 0.1
        rh = -4.0
        hmax = 1.5
        hnum = 250
        VCM = 0.3
        VM = 0.09
        ncutoff = 50
        hcut = 3
        figID = 'Fig1a'
        # plot hysteresis
        hlist, sighavg_S, sighavg_A = calc_hyst_Wick(_sigh = sigh, 
                                                     _hc = hc, 
                                                     _r = rh, 
                                                     _eh = eph, 
                                                     _VCM = VCM,
                                                    _VM = VM, 
                                                    _ncutoff = ncutoff, _hcut = hcut,
                                                    _hmax = hmax, _hnum = hnum,
                                                    _loadfile = loadfile)
        os.rename('Sighavg_A_Wick_r' + str(rh) + '.pdf', figID + '.pdf')
    
        # Fig. 1b parameters
        hc = 0.8
        sigh = 0.05
        eph = 0.1
        rh = 4.0
        hmax = 1.5
        hnum = 250
        VCM = 0.3
        VM = 0.09
        ncutoff = 50
        hcut = 3
        figID = 'Fig1b'
        # plot hysteresis
        hlist, sighavg_S, sighavg_A = calc_hyst_Wick(_sigh = sigh, 
                                                     _hc = hc, 
                                                     _r = rh, 
                                                     _eh = eph, 
                                                     _VCM = VCM,
                                                    _VM = VM, 
                                                    _ncutoff = ncutoff, _hcut = hcut,
                                                    _hmax = hmax, _hnum = hnum,
                                                    _loadfile = loadfile)
        os.rename('Sighavg_A_Wick_r' + str(rh) + '.pdf', figID + '.pdf')

    if task == 0 or task == 2:
        # Fig. 2a parameters
        lcorr = 3e-2
        hc = 5.0
        sigh = 0.07
        sigd1 = -0.8
        sigd2 = -0.4
        epw = 2.0
        sig0 = 1.0
        eph = 1.0
        meshN = 255
        hdomain = 3.9   # value of h for plotting the domain profile
        grfscale = 1.0
        rh = 0.0
        hmax = 11
        hnum = 250
        figID = 'Fig2a'
        # Generate the random field using random seed or load it from files
        if not loadgrf:
            rnd_seed=140440532
            np.random.seed(rnd_seed)
            grf = grf_gen(lcorr, meshN)
            np.savetxt('grf_'
                       + figID
                       +'_meshN_'
                       + str(meshN) 
                       + '.dat', 
                       grf.flatten(), fmt='%16.10f')          
        else:
            grf = np.loadtxt('grf_'
                       + figID
                       +'_meshN_'
                       + str(meshN) 
                       + '.dat').reshape((meshN,meshN))
        # plot domain profile at a given h
        sigma_grf, f, grf_M = sigma_gen(_grf = grf,
                      _sig0 = sig0, _sigh = sigh, _sigd1 = sigd1, _sigd2 = sigd2,
                      _h = hdomain, _hc = hc, _r = rh,
                      _eh = eph, _grfscale = grfscale, _ew = epw)
        fname = figID
        grf_plot(grf_M, fname)
        # plot hysteresis
        hlist, R_S, R_A, G_S, G_A, sigavg_S, sigavg_A =  calc_hyst(_grf = grf,
                          _sig0 = sig0, _sigh = sigh, _sigd1 = sigd1, _sigd2 = sigd2,
                          _hc = hc, _r = rh,
                          _eh = eph, _grfscale = grfscale, _ew = epw,
                          _hmax = hmax, _hnum = hnum, _loadfile = loadfile)
        figID = 'Fig2b'
        os.rename('R_A_yx_meshN_'+str(meshN) + '.pdf', figID + '.pdf')

    if task == 0 or task == 3:    
        # Fig. 2c parameters
        lcorr = 8e-3
        hc = 5.0
        sigh = 0.07
        sigd1 = -0.8
        sigd2 = -0.4
        epw = 2.0
        sig0 = 1.0
        eph = 1.0
        meshN = 257
        hdomain = 3.9
        grfscale = 1.0
        rh = 0.0
        hmax = 11
        hnum = 250
        figID = 'Fig2c'
        # Generate the random field using random seed or load it from files
        if not loadgrf:
            rnd_seed=140440532
            np.random.seed(rnd_seed)
            grf = grf_gen(lcorr, meshN)
            np.savetxt('grf_'
                       + figID
                       +'_meshN_'
                       + str(meshN) 
                       + '.dat', 
                       grf.flatten(), fmt='%16.10f')          
        else:
            grf = np.loadtxt('grf_'
                       + figID
                       +'_meshN_'
                       + str(meshN) 
                       + '.dat').reshape((meshN,meshN))
        # plot domain profile at a given h
        sigma_grf, f, grf_M = sigma_gen(_grf = grf,
                      _sig0 = sig0, _sigh = sigh, _sigd1 = sigd1, _sigd2 = sigd2,
                      _h = hdomain, _hc = hc, _r = rh,
                      _eh = eph, _grfscale = grfscale, _ew = epw)
        fname = figID
        grf_plot(grf_M, fname)
        # plot hysteresis
        hlist, R_S, R_A, G_S, G_A, sigavg_S, sigavg_A =  calc_hyst(_grf = grf,
                          _sig0 = sig0, _sigh = sigh, _sigd1 = sigd1, _sigd2 = sigd2,
                          _hc = hc, _r = rh,
                          _eh = eph, _grfscale = grfscale, _ew = epw,
                          _hmax = hmax, _hnum = hnum, _loadfile = loadfile)
        figID = 'Fig2d'
        os.rename('R_A_yx_meshN_'+str(meshN) + '.pdf', figID + '.pdf')        

    if task == 0 or task == 4:
        # Fig. S1 parameters
        lcrange = np.array([5e-2, 5e-3])
        sigd1range = np.array([-0.8, -0.1])
        sigd2range = np.array([-0.7, -0.1])
        sighrange = np.array([0.05, 0.5])
        epwrange = np.array([0.2, 2.0])
        ephrange = np.array([0.2, 2.0])
        hrange = np.array([-3, 3])
        randparanum = 7  # total number of randomized parameters
        hc = 0.0
        epw = 2.0
        sig0 = 1.0
        eph = 1.0
        meshN = 256
        grfscale = 1.0
        rh = 0.0
        num = 250  # total number of points
        figID = 'FigS1'
        G_A_data = np.zeros((num, 2, 2))   # G_A for each random instance
        param_data = np.zeros((num, randparanum))  # values of parameters for each random instance
        rnd_seed=140440532
        np.random.seed(rnd_seed)
        if not loadfile:
            for i in range(num):
                # Generate the random field using random seed or load it from files
                lcorr = np.random.rand()*(lcrange[1]-lcrange[0]) + lcrange[0]
                sigd1 = np.random.rand()*(sigd1range[1]-sigd1range[0]) + sigd1range[0]
                sigd2 = np.random.rand()*(sigd2range[1]-sigd2range[0]) + sigd2range[0]
                sigh = np.random.rand()*(sighrange[1]-sighrange[0]) + sighrange[0]
                h = np.random.rand()*(hrange[1]-hrange[0]) + hrange[0]
                epw = np.random.rand()*(epwrange[1]-epwrange[0]) + epwrange[0]
                eph = np.random.rand()*(ephrange[1]-ephrange[0]) + ephrange[0]
                param_data[i] = np.array([lcorr, sigd1, sigd2, sigh, epw, eph, h])
                grf = grf_gen(lcorr, meshN)
                # Calculate R, G, and sigavg
        
                th, R_S, R_A, G_S, G_A, sigavg_S, sigavg_A =  calc_hyst_h(_grf = grf,
                              _sig0 = sig0, _sigh = sigh, _sigd1 = sigd1, _sigd2 = sigd2,
                              _hc = hc, _r = rh,
                              _eh = eph, _grfscale = grfscale, _ew = epw,
                              _h = h)
                G_A_data[i] = G_A[0]
            # save data
            np.savetxt('param_'
                       + figID
                       +'_meshN_'
                       + str(meshN) 
                       + '.dat', 
                       param_data.flatten(), fmt='%16.10f')
            np.savetxt('G_A_'
                       + figID
                       +'_meshN_'
                       + str(meshN) 
                       + '.dat', 
                       G_A_data.flatten(), fmt='%16.10f')
        else:
            # load data
            param_data = np.loadtxt('param_'
                       + figID
                       +'_meshN_'
                       + str(meshN) 
                       + '.dat').reshape((num, randparanum))
            G_A_data = np.loadtxt('G_A_'
                       + figID
                       +'_meshN_'
                       + str(meshN) 
                       + '.dat').reshape((num, 2, 2))
        # Create scatter plots
        # rescale G_H by sigh
        G_H = G_A_data[:,1,0] / param_data[:,3]
        bounds = [-1.0, 1.0]
        ylabel = r'$G_{\rm h}/\sigma_{\rm h}$'
        plot_bounds(xlist = param_data[:,-1], 
                    data = G_H, 
                    bounds = bounds, 
                    filename = figID, 
                    ylabel = ylabel, 
                    plotbounds = True)

    if task == 0 or task == 5:
        # Fig. R1a parameters
        lcorr = 1e-2
        hc = 0.8
        sigh = 0.05
        sigd1 = 0.0
        sigd2 = 0.0
        epw = 2.0
        sig0 = 1.0
        eph = 0.1
        meshN = 253
        hdomain = 0.8   # value of h for plotting the domain profile
        grfscale = 0.3
        rh = -4.0
        hmax = 1.5
        hnum = 250
        figID = 'FigR1a'
        # Generate the random field using random seed or load it from files
        if not loadgrf:
            rnd_seed=140440532
            np.random.seed(rnd_seed)
            grf = grf_gen(lcorr, meshN)
            np.savetxt('grf_'
                       + figID
                       +'_meshN_'
                       + str(meshN) 
                       + '.dat', 
                       grf.flatten(), fmt='%16.10f')          
        else:
            grf = np.loadtxt('grf_'
                       + figID
                       +'_meshN_'
                       + str(meshN) 
                       + '.dat').reshape((meshN,meshN))
        # plot domain profile at a given h
        sigma_grf, f, grf_M = sigma_gen(_grf = grf,
                      _sig0 = sig0, _sigh = sigh, _sigd1 = sigd1, _sigd2 = sigd2,
                      _h = hdomain, _hc = hc, _r = rh,
                      _eh = eph, _grfscale = grfscale, _ew = epw)
        fname = figID
        grf_plot(grf_M, fname)
        # plot hysteresis
        hlist, R_S, R_A, G_S, G_A, sigavg_S, sigavg_A =  calc_hyst(_grf = grf,
                          _sig0 = sig0, _sigh = sigh, _sigd1 = sigd1, _sigd2 = sigd2,
                          _hc = hc, _r = rh,
                          _eh = eph, _grfscale = grfscale, _ew = epw,
                          _hmax = hmax, _hnum = hnum, _loadfile = loadfile)
        figID = 'FigR1b'
        os.rename('G_A_yx_meshN_'+str(meshN) + '.pdf', figID + '.pdf')
        figID = 'FigR1b2'
        os.rename('sigavg_A_yx_meshN_'+str(meshN) + '.pdf', figID + '.pdf')
        figID = 'FigR1b3'
        os.rename('R_A_yx_meshN_'+str(meshN) + '.pdf', figID + '.pdf')


    if task == 0 or task == 6:                        
        # Fig. R1c parameters
        lcorr = 1e-2
        hc = 0.8
        sigh = 0.05
        sigd1 = 0.0
        sigd2 = 0.0
        epw = 2.0
        sig0 = 1.0
        eph = 0.1
        meshN = 254
        hdomain = 0.8   # value of h for plotting the domain profile
        grfscale = 0.3
        rh = 4.0
        hmax = 1.5
        hnum = 250
        figID = 'FigR1c'
        # Generate the random field using random seed or load it from files
        if not loadgrf:
            rnd_seed=140440532
            np.random.seed(rnd_seed)
            grf = grf_gen(lcorr, meshN)
            np.savetxt('grf_'
                       + figID
                       +'_meshN_'
                       + str(meshN) 
                       + '.dat', 
                       grf.flatten(), fmt='%16.10f')          
        else:
            grf = np.loadtxt('grf_'
                       + figID
                       +'_meshN_'
                       + str(meshN) 
                       + '.dat').reshape((meshN,meshN))
        # plot domain profile at a given h
        sigma_grf, f, grf_M = sigma_gen(_grf = grf,
                      _sig0 = sig0, _sigh = sigh, _sigd1 = sigd1, _sigd2 = sigd2,
                      _h = hdomain, _hc = hc, _r = rh,
                      _eh = eph, _grfscale = grfscale, _ew = epw)
        fname = figID
        grf_plot(grf_M, fname)
        # plot hysteresis
        hlist, R_S, R_A, G_S, G_A, sigavg_S, sigavg_A =  calc_hyst(_grf = grf,
                          _sig0 = sig0, _sigh = sigh, _sigd1 = sigd1, _sigd2 = sigd2,
                          _hc = hc, _r = rh,
                          _eh = eph, _grfscale = grfscale, _ew = epw,
                          _hmax = hmax, _hnum = hnum, _loadfile = loadfile)
        figID = 'FigR1d'
        os.rename('G_A_yx_meshN_'+str(meshN) + '.pdf', figID + '.pdf')
        figID = 'FigR1d2'
        os.rename('sigavg_A_yx_meshN_'+str(meshN) + '.pdf', figID + '.pdf')
        figID = 'FigR1d3'
        os.rename('R_A_yx_meshN_'+str(meshN) + '.pdf', figID + '.pdf')


    if task == 0 or task == 7:
        # Fig. R2a parameters
        lcorr = 3e-2
        hc = 5.0
        sigh = 0.07
        sigd1 = -0.8
        sigd2 = -0.4
        epw = 2.0
        sig0 = 1.0
        eph = 1.0
        meshN = 251
        hdomain = 3.9   # value of h for plotting the domain profile
        grfscale = 1.0
        rh = 0.8
        hmax = 11
        hnum = 250
        figID = 'FigR2a'
        # Generate the random field using random seed or load it from files
        if not loadgrf:
            rnd_seed=140440532
            np.random.seed(rnd_seed)
            grf = grf_gen(lcorr, meshN)
            np.savetxt('grf_'
                       + figID
                       +'_meshN_'
                       + str(meshN) 
                       + '.dat', 
                       grf.flatten(), fmt='%16.10f')          
        else:
            grf = np.loadtxt('grf_'
                       + figID
                       +'_meshN_'
                       + str(meshN) 
                       + '.dat').reshape((meshN,meshN))
        # plot domain profile at a given h
        sigma_grf, f, grf_M = sigma_gen(_grf = grf,
                      _sig0 = sig0, _sigh = sigh, _sigd1 = sigd1, _sigd2 = sigd2,
                      _h = hdomain, _hc = hc, _r = rh,
                      _eh = eph, _grfscale = grfscale, _ew = epw)
        fname = figID
        grf_plot(grf_M, fname)
        # plot hysteresis
        hlist, R_S, R_A, G_S, G_A, sigavg_S, sigavg_A =  calc_hyst(_grf = grf,
                          _sig0 = sig0, _sigh = sigh, _sigd1 = sigd1, _sigd2 = sigd2,
                          _hc = hc, _r = rh,
                          _eh = eph, _grfscale = grfscale, _ew = epw,
                          _hmax = hmax, _hnum = hnum, _loadfile = loadfile)
        figID = 'FigR2b'
        os.rename('R_A_yx_meshN_'+str(meshN) + '.pdf', figID + '.pdf')

    if task == 0 or task == 8:
        # Fig. R2c parameters
        lcorr = 3e-2
        hc = 5.0
        sigh = 0.07
        sigd1 = -0.8
        sigd2 = -0.4
        epw = 2.0
        sig0 = 1.0
        eph = 1.0
        meshN = 252
        hdomain = 3.9   # value of h for plotting the domain profile
        grfscale = 1.0
        rh = -0.8
        hmax = 11
        hnum = 250
        figID = 'FigR2c'
        # Generate the random field using random seed or load it from files
        if not loadgrf:
            rnd_seed=140440532
            np.random.seed(rnd_seed)
            grf = grf_gen(lcorr, meshN)
            np.savetxt('grf_'
                       + figID
                       +'_meshN_'
                       + str(meshN) 
                       + '.dat', 
                       grf.flatten(), fmt='%16.10f')          
        else:
            grf = np.loadtxt('grf_'
                       + figID
                       +'_meshN_'
                       + str(meshN) 
                       + '.dat').reshape((meshN,meshN))
        # plot domain profile at a given h
        sigma_grf, f, grf_M = sigma_gen(_grf = grf,
                      _sig0 = sig0, _sigh = sigh, _sigd1 = sigd1, _sigd2 = sigd2,
                      _h = hdomain, _hc = hc, _r = rh,
                      _eh = eph, _grfscale = grfscale, _ew = epw)
        fname = figID
        grf_plot(grf_M, fname)
        # plot hysteresis
        hlist, R_S, R_A, G_S, G_A, sigavg_S, sigavg_A =  calc_hyst(_grf = grf,
                          _sig0 = sig0, _sigh = sigh, _sigd1 = sigd1, _sigd2 = sigd2,
                          _hc = hc, _r = rh,
                          _eh = eph, _grfscale = grfscale, _ew = epw,
                          _hmax = hmax, _hnum = hnum, _loadfile = loadfile)
        figID = 'FigR2d'
        os.rename('R_A_yx_meshN_'+str(meshN) + '.pdf', figID + '.pdf')
