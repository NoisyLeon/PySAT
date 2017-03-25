#!/usr/bin/env python
"""
A python module for seismic anisotropy analysis


:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu

"""
import numpy as np
# import transforms3d
import transformations

class elasticTensor(object):
    """
    An object to manipulate elastic tensor in 3D coordinate
    ===========================================================================
    Cijkl   - 4th order elastic tensor
    Cvoigt  - Voigt matrix
    rho     - density
    compl   - element is compliance or not
    ===========================================================================
    """
    def __init__(self, compl=False):
        self.Cijkl  = np.zeros([3,3,3,3])
        self.Cvoigt = np.zeros([6,6])
        self.rho    = None
        self.compl  = compl
        return
    
    def Cijkl2Voigt(self):
        """ Convert full tensor to Voigt notation
            Convert from the 3*3*3*3 elastic constants tensor to 
            to 6*6 matrix representation.
            Use the optional argument "compl" for the elastic compliance (not 
            stiffness) tensor to deal with the multiplication 
            of elements needed to keep the Voigt and full 
            notation consistant.
            Modified from script by Andrew Walker.
         """
        t2m = np.array([[0,1,2,1,2,0],[0,1,2,2,0,1]])
        for i in xrange(6):
            for j in xrange(6):
                # print i+1, j+1, ' <- ', t2m[0,i]+1,t2m[1,i]+1,t2m[0,j]+1,t2m[1,j]+1
                self.Cvoigt[i,j] = self.Cijkl[t2m[0,i],t2m[1,i],t2m[0,j],t2m[1,j]]
        if self.compl:
            self.Cvoigt = self.Cvoigt * np.array([  [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                                                    [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                                                    [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                                                    [2.0, 2.0, 2.0, 4.0, 4.0, 4.0],
                                                    [2.0, 2.0, 2.0, 4.0, 4.0, 4.0],
                                                    [2.0, 2.0, 2.0, 4.0, 4.0, 4.0]])
        return
    
    def Voigt2Cijkl(self):
        """ Convert from Voigt to full tensor notation 
            Convert from the 6*6 elastic constants matrix to 
            the 3*3*3*3 tensor representation. Use the optional 
            argument "compl" for the elastic compliance (not 
            stiffness) tensor to deal with the multiplication 
            of elements needed to keep the Voigt and full 
            notation consistant.
            Modified from script by Andrew Walker.
         """
        m2t = np.array([[0,5,4],[5,1,3],[4,3,2]])
        if self.compl:
            Cvoigt = self.Cvoigt / np.array([   [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                                                [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                                                [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                                                [2.0, 2.0, 2.0, 4.0, 4.0, 4.0],
                                                [2.0, 2.0, 2.0, 4.0, 4.0, 4.0],
                                                [2.0, 2.0, 2.0, 4.0, 4.0, 4.0]])
        else: Cvoigt = self.Cvoigt
        for i in xrange(3):
            for j in xrange(3):
                for k in xrange(3):
                    for l in xrange(3):
                        # print i+1, j+1, k+1, l+1,' <- ', m2t[i,j]+1,m2t[k,l]+1
                        self.Cijkl[i,j,k,l] = Cvoigt[m2t[i,j],m2t[k,l]]
        return
    
    
    def bondmat(self, axis, angle):
        """
        Compute Bond Matrix for rotation of Voigt matrix (eq. 1.54 in Carcione, 2014)
        
        Reference:
        Carcione, J.M., 2014. Wave fields in real media:
            Wave propagation in anisotropic, anelastic, porous and electromagnetic media (Vol. 38). Elsevier.
        
        """
        radian  = np.pi*angle/180.
        g       = transformations.rotation_matrix(radian, axis)[:3, :3]
        M       = np.array([[g[0,0]**2, g[0,1]**2, g[0,2]**2, 2.*g[0,1]*g[0,2], 2.*g[0,2]*g[0,0], 2.*g[0,0]*g[0,1]],
                            [g[1,0]**2, g[1,1]**2, g[1,2]**2, 2.*g[1,1]*g[1,2], 2.*g[1,2]*g[1,0], 2.*g[1,0]*g[1,1]],
                            [g[2,0]**2, g[2,1]**2, g[2,2]**2, 2.*g[2,1]*g[2,2], 2.*g[2,2]*g[2,0], 2.*g[2,0]*g[2,1]],
            [g[1,0]*g[2,0], g[1,1]*g[2,1], g[1,2]*g[2,2], g[1,1]*g[2,2]+g[1,2]*g[2,1], g[1,0]*g[2,2]+g[1,2]*g[2,0], g[1,1]*g[2,0]+g[1,0]*g[2,1]],
            [g[2,0]*g[0,0], g[2,1]*g[0,1], g[2,2]*g[0,2], g[0,1]*g[2,2]+g[0,2]*g[2,1], g[0,2]*g[2,0]+g[0,0]*g[2,2], g[0,0]*g[2,1]+g[0,1]*g[2,0]],
            [g[0,0]*g[1,0], g[0,1]*g[1,1], g[0,2]*g[1,2], g[0,1]*g[1,2]+g[0,2]*g[1,1], g[0,2]*g[1,0]+g[0,0]*g[1,2], g[0,0]*g[1,1]+g[0,1]*g[1,0]]
            ])
        return M
    
    def rotT(self, axis, angle, resetCvoigt=True):
        """
        Rotate a 4th order elastic tensor using a rotation matrix, g
       
        Tensor rotation involves a summation over all combinations
        of products of elements of the unrotated tensor and the 
        rotation matrix. Like this for a rank 3 tensor:
            T'(ijk) -> Sum g(i,p)*g(j,q)*g(k,r)*T(pqr)
        
        with the summation over p, q and r. The obvious implementation
        involves (2*rank) length 3 loops building up the summation in the
        inner set of loops. This optimized implementation >100 times faster 
        than that obvious implementaton using 8 nested loops. Returns a 
        3*3*3*3 numpy array representing the rotated tensor, Tprime.
        Modified from script by Andrew Walker.
        """
        radian  = np.pi*angle/180.
        # g       = transforms3d.axangles.axangle2mat(axis, radian)
        g       = transformations.rotation_matrix(radian, axis)[:3, :3]
        gg = np.outer(g, g) # Flatterns input and returns 9*9 array
                            # of all possible products
        gggg = np.outer(gg, gg).reshape(4 * g.shape)
                            # 81*81 array of double products reshaped
                            # to 3*3*3*3*3*3*3*3 array...
        axes = ((0, 2, 4, 6), (0, 1, 2, 3)) # We only need a subset 
                                            # of gggg in tensordot...
        self.Cijkl[:] = np.tensordot(gggg, self.Cijkl, axes)
        self.Cijkl[self.Cijkl<0.000001]=0.
        if resetCvoigt: self.Cijkl2Voigt()
        return
    
    def rotB(self,axis, angle, resetCijkl=True):
        """
        Rotate Voigt matrix using Bond matrix (eq. 1.58 in Carcione, 2014)
        
        Reference:
        Carcione, J.M., 2014. Wave fields in real media:
            Wave propagation in anisotropic, anelastic, porous and electromagnetic media (Vol. 38). Elsevier.
        """
        M           = self.bondmat(axis=axis, angle=angle)
        self.Cvoigt = np.dot(M, self.Cvoigt)
        self.Cvoigt = np.dot(self.Cvoigt, M.T)
        self.Cvoigt[self.Cvoigt<0.000001]=0.
        if resetCijkl: self.Voigt2Cijkl()
        return
        
    def set_love(self, A, C, L, N, F, resetCijkl=True):
        """
        Set Love parameters for a VTI media
        ============================================================================
        Input Parameters:
        A,C,L,N,F   - Love parameters (GPa)
        resetCijkl  - reset 4th order tensor or not
        ============================================================================
        """
        self.Cvoigt[:] =  np.array([[A, A-2.*N, F, 0., 0., 0.],
                                    [A-2.*N, A, F, 0., 0., 0.],
                                    [F, F, C, 0., 0., 0.],
                                    [0., 0., 0., L, 0., 0.],
                                    [0., 0., 0., 0., L, 0.],
                                    [0., 0., 0., 0., 0., N]])
        if resetCijkl: self.Voigt2Cijkl()
        return
    
    def set_radial(self, vp, vs, rho, xi, phi, eta, resetCijkl=True):
        """
        Output the elastic tensor given a set of radial anisotropy parameters
        as used typically in global seismology.  Average velocities are given by:
            15*rho*<Vp>^2 = 3*C + (8 + 4*eta)*A + 8*(1 - eta)*L
            15*rho*<Vs>^2 =   C + (1 - 2*eta)*A + (6 + 4*eta)*L + 5*N
        ============================================================================
        Input Parameters:
        vp:   Voigt average P wave velocity
        vs:   Voigt average shear wave velocity
        rho:  Density
        xi:   (Vsh^2/Vsv^2) of horizontal waves
        phi:  (Vpv^2/Vph^2)
        eta:  C13/(C11 - 2*C44)
        ============================================================================
        """
        vp=vp*1e3
        vs=vs*1e3
        L = 15.*rho*((3.*phi + 8. + 4.*eta)*vs**2 - (phi + 1. - 2.*eta)*vp**2) \
                /((6. + 4.*eta + 5.*xi)*(3.*phi + 8. + 4.*eta) 
                 - 8.*(phi + 1. - 2.*eta)*(1. - eta)) 

        A = (15.*rho*vp**2 - 8.*(1. - eta)*L) / (3.*phi + 8. + 4.*eta) 
     
        F = eta*(A - 2.*L) 
        C = phi*A 
        N = xi*L 
        C12 = A - 2.*N
        self.Cvoigt[:]  =  np.array([[A, C12, F, 0., 0., 0.],
                                    [C12, A, F, 0., 0., 0.],
                                    [F, F, C, 0., 0., 0.],
                                    [0., 0., 0., L, 0., 0.],
                                    [0., 0., 0., 0., L, 0.],
                                    [0., 0., 0., 0., 0., N]])
        self.Cvoigt     = self.Cvoigt/1e9
        # # self.Cvoigt[self.Cvoigt<0.000001]=0.
        if resetCijkl: self.Voigt2Cijkl()
        return

    
    # def set_TI(self, ):
        
    #     
    # def elastic_DB(self, dtype):
    #     s.lower()
    #     if uid is 'olivine' or 
    #     switch lower(uid)
    #   case {'olivine','ol'}
    #      info = 'Single crystal olivine (Abramson et al, JGR, 1997; doi:10.1029/97JB00682)' ; 
    #      C = [320.5  68.1  71.6   0.0   0.0   0.0 ; ...
    #            68.1 196.5  76.8   0.0   0.0   0.0 ; ...
    #            71.6  76.8 233.5   0.0   0.0   0.0 ; ...
    #             0.0   0.0   0.0  64.0   0.0   0.0 ; ...
    #             0.0   0.0   0.0   0.0  77.0   0.0 ; ...
    #             0.0   0.0   0.0   0.0   0.0  78.7 ];
    #      rh = 3355 ;   
    #   case {'fayalite', 'fa'}
    #       info = 'Isentropic elastic constants for single cystal (Fe0.94,Mn0.06)2SiO4 fayalite (Speziale et al, JGR, 2004; doi:10.1029/2004JB003162)';
    #       % Table 2 - "This study"
    #       C = [270   103    97.2   0.0   0.0   0.0 ; ...
    #            103   171.1  93.5   0.0   0.0   0.0 ; ...
    #             97.2  93.5 234.1   0.0   0.0   0.0 ; ...
    #              0.0   0.0   0.0  33.4   0.0   0.0 ; ...
    #              0.0   0.0   0.0   0.0  48.7   0.0 ; ...
    #              0.0   0.0   0.0   0.0   0.0  59.6 ];
    #      rh = 4339.1 ; % 0.3 GPa - table 1
    #   case {'albite','alb'}
    #   info = 'Single crystal albite (Brown et al, PCM, 2006; doi:10.1007/s00269-006-0074-1)' ;
    #      C = [ 69.9  34.0  30.8   5.1  -2.4  -0.9 ; ...
    #            34.0 183.5   5.5  -3.9  -7.7  -5.8 ; ...
    #            30.8   5.5 179.5  -8.7   7.1  -9.8 ; ...
    #             5.1  -3.9  -8.7  24.9  -2.4  -7.2 ; ...
    #            -2.4  -7.7   7.1  -2.4  26.8   0.5 ; ...
    #            -0.9  -5.8  -9.8  -7.2   0.5  33.5 ];
    #        rh = 2623 ;
    #    case {'anorthite', 'an96'}
    #        info = 'Single crystal anorthite 96 (Brown et al, submitted)';
    #        % Brown, Angel and Ross "Elasticity of Plagioclase Feldspars" 
    #        % http://earthweb.ess.washington.edu/brown/resources/PlagioclaseElasticity.pdf
    #        C = [132.2  64.0  55.3   9.5   5.1  -10.8 ; ...
    #              64.0 200.2  31.9   7.5   3.4   -7.2 ; ... 
    #              55.3  31.9 163.9   6.6   0.5    1.6 ; ...
    #               9.5   7.5   6.6  24.6   3.0   -2.2 ;...
    #               5.1   3.4   0.5   3.0  36.6    5.2 ;...
    #             -10.8  -7.2   1.6  -2.2   5.2   36.0];
    #        rh = 2757;
    #    case {'enstatite', 'ens'}
    #      info = 'Single crystal orthoenstatite (Weidner et al, PEPI 1978, 17:7-13)' ;
    #      C = [ 225.0  72.0  54.0   0.0  0.0   0.0 ; ...
    #            72.0  178.0  53.0   0.0  0.0   0.0 ; ...
    #            54.0   53.0 214.0   0.0  0.0   0.0 ; ...
    #             0.0    0.0   0.0  78.0  0.0   0.0 ; ...
    #             0.0    0.0   0.0   0.0 76.0   0.0 ; ...
    #             0.0    0.0   0.0   0.0  0.0  82.0 ];
    #      rh = 3200 ;  
    #    case {'jadeite', 'jd'}
    #      info = 'Single crystal jadeite (Kandelin and Weidner, 1988, 50:251-260)' ;
    #      C = [ 274.0  94.0  71.0   0.0  4.0   0.0 ; ...
    #            94.0  253.0  82.0   0.0 14.0   0.0 ; ...
    #            71.0   82.0 282.0   0.0 28.0   0.0 ; ...
    #             0.0    0.0   0.0  88.0  0.0  13.0 ; ...
    #             4.0   14.0  28.0   0.0 65.0   0.0 ; ...
    #             0.0    0.0   0.0  13.0  0.0  94.0 ];
    #      rh = 3300 ; 
    #    case {'diopside', 'di'}
    #      info = 'Single crystal chrome-diopeside (Isaak and Ohno, PCM, 2003, 30:430-439)' ;
    #      C = [228.1   94.0  71.0   0.0  7.9   0.0 ; ...
    #            94.0  181.1  82.0   0.0  5.9   0.0 ; ...
    #            71.0   82.0 245.4   0.0 39.7   0.0 ; ...
    #             0.0    0.0   0.0  78.9  0.0   6.4 ; ...
    #             7.9    5.9  39.7   0.0 68.2   0.0 ; ...
    #             0.0    0.0   0.0   6.4  0.0  78.1 ];
    #      rh = 3400 ;
    #    case {'halite', 'nacl'}
    #        info = 'Single crystal halite (NaCl, rock-salt).';
    #        C = [ 49.5  13.2  13.2  0.0  0.0  0.0 ; ...
    #              13.2  49.5  13.2  0.0  0.0  0.0 ; ...
    #              13.2  13.2  49.5  0.0  0.0  0.0 ; ...
    #               0.0   0.0   0.0 12.8  0.0  0.0 ; ...
    #               0.0   0.0   0.0  0.0 12.8  0.0 ; ...
    #               0.0   0.0   0.0  0.0  0.0 12.8];
    #        rh = 2170;
    #    case {'sylvite', 'kcl'}
    #        info = 'Single crystal sylvite (KCl).';
    #        C = [ 40.1   6.6   6.6  0.0  0.0  0.0 ; ...
    #               6.6  40.1   6.6  0.0  0.0  0.0 ; ...
    #               6.6   6.6  40.1  0.0  0.0  0.0 ; ...
    #               0.0   0.0   0.0  6.4  0.0  0.0 ; ...
    #               0.0   0.0   0.0  0.0  6.4  0.0 ; ...
    #               0.0   0.0   0.0  0.0  0.0  6.4];
    #        rh = 1990;
    #    case 'galena'
    #        info = 'Single crystal galena (Bhagavantam and Rao, Nature, 1951 168:42)';
    #        C = [127.0  29.8  29.8  0.0  0.0  0.0 ; ...
    #              29.8 127.0  29.8  0.0  0.0  0.0 ; ...
    #              29.8  29.8 127.0  0.0  0.0  0.0 ; ...
    #               0.0   0.0   0.0 24.8  0.0  0.0 ; ...
    #               0.0   0.0   0.0  0.0 24.8  0.0 ; ...
    #               0.0   0.0   0.0  0.0  0.0 24.8];
    #        rh = 7600;
    #    case 'stishovite'
    #        info = 'Single crystal stishovite, SiO2 (Weidner et al., JGR, 1982, 87:4740-4746)';
    #        C = [453.0 211.0 203.0   0.0   0.0   0.0 ; ...
    #             211.0 453.0 203.0   0.0   0.0   0.0 ; ...
    #             203.0 203.0 776.0   0.0   0.0   0.0 ; ...
    #               0.0   0.0   0.0 252.0   0.0   0.0 ; ...
    #               0.0   0.0   0.0   0.0 252.0   0.0 ; ...
    #               0.0   0.0   0.0   0.0   0.0 302.0];
    #        rh = 4290;
    #    case {'fluorapatite', 'apatite'}
    #        info = 'Single crystal fluorapatite, Ca5F(PO4)3 (Sha et al., J. Appl. Phys., 1994, 75:7784; doi:10.1063/1.357030)';
    #        C = [152.0 49.99 63.11   0.0   0.0   0.0 ; ...
    #             49.99 152.0 63.11   0.0   0.0   0.0 ; ...
    #             63.11 63.11 185.7   0.0   0.0   0.0 ; ...
    #               0.0   0.0   0.0 42.75   0.0   0.0 ; ...
    #               0.0   0.0   0.0   0.0 42.75   0.0 ; ...
    #               0.0   0.0   0.0   0.0   0.0 51.005];
    #        rh = 3150;
    # 
    #    case {'antigorite', 'atg'}
    #        info = 'Adiabatic single crystal antigorite, (Bezacier et al., EPSL 2010, 289:198-208; doi:10.1016/j.epsl.2009.11.009)';
    #        % Note that X1||a, X2||b and X3||c* - not IRE convection.
    #        % and that these are adiabatic, not isothermal, constants, but
    #        % that's what "should" be used for wave velocites (c.f. Karato
    #        % deformation of Earth materials, sec. 4.3). Velocities quoted
    #        % in the reference (Table 3) use the corrected isotropic 
    #        % adiabatic moduli... 
    #        C = [208.10   66.40    16.00    0.0    5.50   0.0 ; ...
    #              66.4   201.60     4.90    0.0   -3.10   0.0 ; ...
    #              16.00    4.90    96.90    0.0    1.60   0.0 ; ...
    #               0.0     0.0      0.0    16.90   0.0  -12.10 ; ...
    #               5.50   -3.10     1.60    0.0   18.40   0.0 ; ...
    #               0.0     0.0      0.0   -12.10   0.0   65.50];
    #        rh = 2620;
    #        
    #    case 'llm_mgsio3ppv'
    #        info = 'Adiabatic single crystal MgSiO3 post-perovskite under lowermost mantle conditions: from molecular dynamics, DFT & GGA and 2800 K and 127 GPa (Wookey et al. Nature, 2005, 438:1004-1007; doi:10.1038/nature04345 )';
    #        C = [1139  357  311    0    0    0 ; ...
    #              357  842  466    0    0    0 ; ...
    #              311  466 1137    0    0    0 ; ...
    #               0     0    0  268    0    0 ; ...
    #               0     0    0    0  210    0 ; ...
    #               0     0    0    0    0  346 ];
    #        rh = 5269;  
    #        
    #    case 'llm_mgsio3pv'
    #        info = 'Adiabatic single crystal MgSiO3 perovskite under lowermost mantle conditions: from molecular dynamics, DFT & GGA and 2800 K and 126 GPa (Wookey et al. Nature, 2005, 438:1004-1007; doi:10.1038/nature04345 )';
    #        C = [ 808  522  401    0    0    0 ; ...
    #              522 1055  472    0    0    0 ; ...
    #              401  472  993    0    0    0 ; ...
    #               0     0    0  328    0    0 ; ...
    #               0     0    0    0  263    0 ; ...
    #               0     0    0    0    0  262 ];
    #        rh = 5191;
    #        
    #    case 'ice'
    #        info = 'Adiabatic single crystal artifical water ice at -16C: from Gammon et al. (1983) Journal of Glaciology 29:433-460.';
    #        C = [13.961  7.153  5.765  0    0    0 ; ...
    #              7.153 13.961  5.765  0    0    0 ; ...
    #              5.765  5.765 15.013  0    0    0 ; ...
    #               0     0       0     3.21 0    0 ; ...
    #               0     0       0     0    3.21 0 ; ...
    #               0     0       0     0    0    3.404 ];
    #        rh = 919.10; % See page 442 of paper.
    #        
    #    case {'quartz', 'qz'}
    #        info = 'Premium cultured single crystal alpha-quartz at 22 C using resonance-ultrasound spectroscopy. From Heyliger et al. (2003) Journal of the Accoustic Society of America 114:644-650';
    #        C = [87.17  6.61  12.02 -18.23  0      0 ; ...
    #              6.61 87.17  12.02  18.23  0      0 ; ...
    #             12.02 12.02 105.80   0.0   0      0 ; ...
    #            -18.23 18.23   0.0   58.27  0      0 ; ...
    #              0     0      0      0    58.27 -18.23 ; ...
    #              0     0      0      0   -18.23  40.28 ];
    #        rh = 2649.7;
    # 
    
        