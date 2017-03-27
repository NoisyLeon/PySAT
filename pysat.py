#!/usr/bin/env python
"""
The Python Seismic Anisotropy Toolkit

:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
    
:References:
    Bond, W.L., 1943. The mathematics of the physical properties of crystals.
        Bell Labs Technical Journal, 22(1), pp.1-72.
    Riley, K.F., Hobson, M.P. and Bence, S.J., 2006. Mathematical methods for physics and engineering: a comprehensive guide.
        Cambridge university press.
    Carcione, J.M., 2014. Wave fields in real media:
        Wave propagation in anisotropic, anelastic, porous and electromagnetic media (Vol. 38). Elsevier.
::: Note :::
For rotation matrix ambiguities:
    https://en.wikipedia.org/wiki/Rotation_matrix#Ambiguities

Direction of rotation:
    The direction of rotation is given by the right-hand rule (orient the thumb of the right hand along the axis around which the rotation occurs,
    with the end of the thumb at the positive end of the axis; curl your fingers; the direction your fingers curl is the direction of rotation).
    Therefore, the rotations are counterclockwise if looking along the axis of rotation from positive to negative.
"""
import numpy as np
import copy
try:
    from opt_einsum import contract
    use_opt_einsum=True
except: use_opt_einsum=False

########################################################################################################
# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

# For testing whether a number is close to zero
_EPS4 = np.finfo(float).eps * 4.0
########################################################################################################


def euler2mat(ai, aj, ak, axes='sxyz'):
    """Return rotation matrix from Euler angles and axis sequence.
    Parameters
    ----------
    ai : float
        First rotation angle in degree (according to `axes`).
    aj : float
        Second rotation angle in degree (according to `axes`).
    ak : float
        Third rotation angle in degree (according to `axes`).
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).
    Returns
    -------
    mat : array-like shape (3, 3) or (4, 4)
        Rotation matrix or affine.
    Examples
    --------
    >>> R = euler2mat(1, 2, 3, 'syxz')
    >>> np.allclose(np.sum(R[0]), -1.34786452)
    True
    >>> R = euler2mat(1, 2, 3, (0, 1, 0, 1))
    >>> np.allclose(np.sum(R[0]), -0.383436184)
    True
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes
    ai=np.pi*ai/180.; aj=np.pi*aj/180.; ak=np.pi*ak/180.
    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    M = np.eye(3)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj*si
        M[i, k] = sj*ci
        M[j, i] = sj*sk
        M[j, j] = -cj*ss+cc
        M[j, k] = -cj*cs-sc
        M[k, i] = -sj*ck
        M[k, j] = cj*sc+cs
        M[k, k] = cj*cc-ss
    else:
        M[i, i] = cj*ck
        M[i, j] = sj*sc-cs
        M[i, k] = sj*cc+ss
        M[j, i] = cj*sk
        M[j, j] = sj*ss+cc
        M[j, k] = sj*cs-sc
        M[k, i] = -sj
        M[k, j] = cj*si
        M[k, k] = cj*ci
    return M


def mat2euler(mat, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.
    Note that many Euler angle triplets can describe one matrix.
    Parameters
    ----------
    mat : array-like shape (3, 3) or (4, 4)
        Rotation matrix or affine.
    axes : str, optional
        Axis specification; one of 24 axis sequences as string or encoded
        tuple - e.g. ``sxyz`` (the default).
    Returns
    -------
    ai : float
        First rotation angle (according to `axes`).
    aj : float
        Second rotation angle (according to `axes`).
    ak : float
        Third rotation angle (according to `axes`).
    Examples
    --------
    >>> R0 = euler2mat(1, 2, 3, 'syxz')
    >>> al, be, ga = mat2euler(R0, 'syxz')
    >>> R1 = euler2mat(al, be, ga, 'syxz')
    >>> np.allclose(R0, R1)
    True
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _TUPLE2AXES[axes]  # validation
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = np.array(mat, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = np.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS4:
            ax = np.arctan2( M[i, j],  M[i, k])
            ay = np.arctan2( sy,       M[i, i])
            az = np.arctan2( M[j, i], -M[k, i])
        else:
            ax = np.arctan2(-M[j, k],  M[j, j])
            ay = np.arctan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = np.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS4:
            ax = np.arctan2( M[k, j],  M[k, k])
            ay = np.arctan2(-M[k, i],  cy)
            az = np.arctan2( M[j, i],  M[i, i])
        else:
            ax = np.arctan2(-M[j, k],  M[j, j])
            ay = np.arctan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax/np.pi*180., ay/np.pi*180., az/np.pi*180.

def rot2mat(axis, angle, is_normalized=False):
    ''' Rotation matrix for rotation angle `angle` around `axis`
    ===============================================================================
    :::Important Note:::
    The rotation matrix is defined for rotation of a tensor in a fixed coordinate
    The output rotation matrix generated by this function is the inverse of the
    rotation matrix in Bond's book(p12-13).
    ===============================================================================
    Input Parameters:
    axis            - 3 element sequence, vector specifying axis for rotation.
    angle           - scalar, angle of rotation in degree.
    is_normalized   - bool, optional
       True if `axis` is already normalized (has norm of 1).  Default False
    -----
    output  -   mat : array shape (3,3), rotation matrix for specified rotation
    ===============================================================================
    Notes
    -----
    From: http://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
    '''
    x, y, z = axis
    if not is_normalized:
        n = np.sqrt(x*x + y*y + z*z)
        x = x/n
        y = y/n
        z = z/n
    angle  = np.pi*angle/180.
    c = np.cos(angle); s = np.sin(angle); C = 1-c
    xs = x*s;   ys = y*s;   zs = z*s
    xC = x*C;   yC = y*C;   zC = z*C
    xyC = x*yC; yzC = y*zC; zxC = z*xC
    return np.array([
            [ x*xC+c,   xyC-zs,   zxC+ys ],
            [ xyC+zs,   y*yC+c,   yzC-xs ],
            [ zxC-ys,   yzC+xs,   z*zC+c ]])

def mat2rot(mat, unit_thresh=1e-5):
    """Return axis, angle and point from (3, 3) matrix `mat`
    Parameters
    ----------
    mat : array-like shape (3, 3)
        Rotation matrix
    unit_thresh : float, optional
        Tolerable difference from 1 when testing for unit eigenvalues to
        confirm `mat` is a rotation matrix.
    Returns
    -------
    axis : array shape (3,)
       vector giving axis of rotation
    angle : scalar
       angle of rotation in radians.
    Examples
    --------
    >>> direc = np.random.random(3) - 0.5
    >>> angle = (np.random.random() - 0.5) * (2*math.pi)
    >>> R0 = axangle2mat(direc, angle)
    >>> direc, angle = mat2axangle(R0)
    >>> R1 = axangle2mat(direc, angle)
    >>> np.allclose(R0, R1)
    True
    Notes
    -----
    http://en.wikipedia.org/wiki/Rotation_matrix#Axis_of_a_rotation
    """
    M = np.asarray(mat, dtype=np.float)
    # direction: unit eigenvector of R33 corresponding to eigenvalue of 1
    L, W = np.linalg.eig(M.T)
    i = np.where(np.abs(L - 1.0) < unit_thresh)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    direction = np.real(W[:, i[-1]]).squeeze()
    # rotation angle depending on direction
    cosa = (np.trace(M) - 1.0) / 2.0
    if abs(direction[2]) > 1e-8:
        sina = (M[1, 0] + (cosa-1.0)*direction[0]*direction[1]) / direction[2]
    elif abs(direction[1]) > 1e-8:
        sina = (M[0, 2] + (cosa-1.0)*direction[0]*direction[2]) / direction[1]
    else:
        sina = (M[2, 1] + (cosa-1.0)*direction[1]*direction[2]) / direction[0]
    angle = np.arctan2(sina, cosa)
    return direction, angle/np.pi*180.

def bondmat(axis, angle):
    """
    Compute Bond Matrix for rotation of Voigt matrix (eq. 1.54 in Carcione, 2014)
    :::Important Note:::
    The rotation matrix used for Bond matrix was originally defined for rotation of the coordinate system
    ,which is in the opposite direction for rotation of a tensor in a fixed coordinate.
    We define the rotation as the rotation for the tensor itself in a fixed coordinate,
    Therefore,
    M   = bondmat(axis, angle) should be equal to the Bond Matrix for an angle of opposite sign
    =========================================================================================================
    Input Parameters:
    axis            - 3 element sequence, vector specifying axis for rotation.
    angle           - scalar, angle of rotation in degree.
    -----
    output          - array shape (3,3), Bond matrix for rotation of Voigt matrix
    =========================================================================================================
    """
    g       = rot2mat(axis = axis, angle = angle)
    M       = np.array([[g[0,0]**2, g[0,1]**2, g[0,2]**2, 2.*g[0,1]*g[0,2], 2.*g[0,2]*g[0,0], 2.*g[0,0]*g[0,1]],
                        [g[1,0]**2, g[1,1]**2, g[1,2]**2, 2.*g[1,1]*g[1,2], 2.*g[1,2]*g[1,0], 2.*g[1,0]*g[1,1]],
                        [g[2,0]**2, g[2,1]**2, g[2,2]**2, 2.*g[2,1]*g[2,2], 2.*g[2,2]*g[2,0], 2.*g[2,0]*g[2,1]],
        [g[1,0]*g[2,0], g[1,1]*g[2,1], g[1,2]*g[2,2], g[1,1]*g[2,2]+g[1,2]*g[2,1], g[1,0]*g[2,2]+g[1,2]*g[2,0], g[1,1]*g[2,0]+g[1,0]*g[2,1]],
        [g[2,0]*g[0,0], g[2,1]*g[0,1], g[2,2]*g[0,2], g[0,1]*g[2,2]+g[0,2]*g[2,1], g[0,2]*g[2,0]+g[0,0]*g[2,2], g[0,0]*g[2,1]+g[0,1]*g[2,0]],
        [g[0,0]*g[1,0], g[0,1]*g[1,1], g[0,2]*g[1,2], g[0,1]*g[1,2]+g[0,2]*g[1,1], g[0,2]*g[1,0]+g[0,0]*g[1,2], g[0,0]*g[1,1]+g[0,1]*g[1,0]]
        ])
    return M

def bondmat2(g):
    """
    Compute Bond Matrix for rotation of Voigt matrix (eq. 1.54 in Carcione, 2014)
    ================================================================================
    Input Parameters:
    g   - transformation matrix
    -----
    output  -   mat : array shape (3,3), Bond matrix for rotation of Voigt matrix
    ================================================================================
    Reference:
    Carcione, J.M., 2014. Wave fields in real media:
        Wave propagation in anisotropic, anelastic, porous and electromagnetic media (Vol. 38). Elsevier.
    """
    M       = np.array([[g[0,0]**2, g[0,1]**2, g[0,2]**2, 2.*g[0,1]*g[0,2], 2.*g[0,2]*g[0,0], 2.*g[0,0]*g[0,1]],
                        [g[1,0]**2, g[1,1]**2, g[1,2]**2, 2.*g[1,1]*g[1,2], 2.*g[1,2]*g[1,0], 2.*g[1,0]*g[1,1]],
                        [g[2,0]**2, g[2,1]**2, g[2,2]**2, 2.*g[2,1]*g[2,2], 2.*g[2,2]*g[2,0], 2.*g[2,0]*g[2,1]],
        [g[1,0]*g[2,0], g[1,1]*g[2,1], g[1,2]*g[2,2], g[1,1]*g[2,2]+g[1,2]*g[2,1], g[1,0]*g[2,2]+g[1,2]*g[2,0], g[1,1]*g[2,0]+g[1,0]*g[2,1]],
        [g[2,0]*g[0,0], g[2,1]*g[0,1], g[2,2]*g[0,2], g[0,1]*g[2,2]+g[0,2]*g[2,1], g[0,2]*g[2,0]+g[0,0]*g[2,2], g[0,0]*g[2,1]+g[0,1]*g[2,0]],
        [g[0,0]*g[1,0], g[0,1]*g[1,1], g[0,2]*g[1,2], g[0,1]*g[1,2]+g[0,2]*g[1,1], g[0,2]*g[1,0]+g[0,0]*g[1,2], g[0,0]*g[1,1]+g[0,1]*g[1,0]]
        ])
    return M

class elasticTensor(object):
    """
    An object to manipulate elastic tensor in 3D coordinate
    ===========================================================================
    Cijkl   - 4th order elastic tensor (3*3*3*3, GPa)
    Cvoigt  - Voigt matrix (6*6, GPa)
    rho     - density (kg/m^3)
    compl   - element is compliance or not
    ===========================================================================
    """
    def __init__(self, compl=False):
        self.Cijkl  = np.zeros([3,3,3,3])
        self.Cvoigt = np.zeros([6,6])
        self.rho    = np.nan
        self.compl  = compl
        self.info   = ''
        return
    
    def __str__(self):
        self.Cvoigt[np.abs(self.Cvoigt)<1e-6]=0.
        outstr=self.info
        outstr=outstr+'\n------\nVoigt matrix (Gpa):'
        outstr=outstr+'\n'+self.Cvoigt.__str__()
        outstr=outstr+'\n------\ndensity = %g' %self.rho + ' km/m^3'
        return outstr
    
    def __repr__(self): return self.__str__()
    
    def copy(self): return copy.deepcopy(self)
    
    def Cijkl2Voigt(self):
        """
        Convert 4th order elastic tensor to Voigt notation
        Use the optional argument "compl" for the elastic compliance (not 
        stiffness) tensor to deal with the multiplication 
        of elements needed to keep the Voigt and full 
        notation consistant.
        """
        t2m = np.array([[0,1,2,1,2,0],[0,1,2,2,0,1]])
        for i in xrange(6):
            for j in xrange(6):
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
        """
        Convert Voigt matrix to 4th order elastic tensor 
        Use the optional argument "compl" for the elastic compliance (not 
        stiffness) tensor to deal with the multiplication 
        of elements needed to keep the Voigt and full 
        notation consistant.
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
                        self.Cijkl[i,j,k,l] = Cvoigt[m2t[i,j],m2t[k,l]]
        return
    
    ##########################################################################
    # Methods for transformation of elastic tensor in fixed coordinate system
    ##########################################################################
    
    def rotT(self, axis, angle, resetCvoigt=True):
        """
        Rotate a 4th order elastic tensor with transformation matrix (rotation matrix),
        the coordinate system is fixed. 
        Note that the rotation is the inverse of rotation of a coordinate system.
        ==================================================================================
        Input Parameters:
        axis            - 3 element sequence, vector specifying axis for rotation.
        angle           - scalar, angle of rotation in degree.
        ==================================================================================
        ::: Note 1 :::
        Define L[i,j] = e'[i] * e[j] as transformation matrix of the coordinate,
        for axis = [0, 0, 1], rotation angle theta, we get:
            L = [cos(theta)  sin(theta)   0
                 -sin(theta) cos(theta)   0
                    0           0         1 ]
        And C'[i,j,k,l] = L[i,a]L[j,b]L[k,c]L[l,d]C[a,b,c,d]
        Note that we actually need to rotate the tensor, thus the angle has opposite sign,
        which is the case for the output from rot2mat.
        For more details, see Riley's book(p931, third edition).
        ::: Note 2 :::
        The definition of L[i,j] may be different in other books.
        Another definition is L[i,j] = e[i] * e'[j], in this case, we have:
        C'[i,j,k,l] = L[a,i]L[b,j]L[c,k]L[d,l]C[a,b,c,d]
        The rotation matrix from rot2mat should be changed if using this convention.
        ==================================================================================
        """
        g  = rot2mat(axis=axis, angle=angle)
        if use_opt_einsum:
            self.Cijkl[:]=contract('ia,jb,kc,ld,abcd->ijkl', g, g, g, g, self.Cijkl)
        else:
            self.Cijkl[:]=np.einsum('ia,jb,kc,ld,abcd->ijkl', g, g, g, g, self.Cijkl)
        if resetCvoigt: self.Cijkl2Voigt()
        return
    
    def _test_rotT(self, v=np.array([1.,0.,0.]), axis=[0,0,1], angle=45.):
        R=pysat.rot2mat(axis=axis, angle=angle)
        vprime=np.einsum('ia,a->i', R, v)
        print 'Testing rotation of a vector with fixed coordinate'
        print 'v =',v,' vprime = ',vprime
        return
    
    def rotB(self, axis, angle, resetCijkl=True):
        """
        Rotate Voigt matrix using Bond matrix (eq. 1.58 in Carcione, 2014)
        Note that the rotation is the inverse of rotation of a coordinate system,
        thus the rotation matrix used to construct Bond matrix is the inverse of the
        rotation matrix in Bond's book (p12-13)
        ============================================================================
        Input Parameters:
        axis            - 3 element sequence, vector specifying axis for rotation.
        angle           - scalar, angle of rotation in degree.
        resetCijkl      - reset 4th order tensor or not
        ============================================================================
        """
        M           = bondmat(axis=axis, angle=angle)
        self.Cvoigt = np.dot(M, self.Cvoigt)
        self.Cvoigt = np.dot(self.Cvoigt, M.T)
        if resetCijkl: self.Voigt2Cijkl()
        return
    
    def rotTB(self, axis, angle, verbose=True):
        """
        Rotate elastic tensor with both rotT and rotB, output error if incompatible
        ============================================================================
        Input Parameters:
        axis            - 3 element sequence, vector specifying axis for rotation.
        angle           - scalar, angle of rotation in degree.
        ============================================================================
        """
        et_temp = self.copy()
        self.rotT(axis=axis, angle=angle)
        et_temp.rotB(axis=axis, angle=angle)
        if not np.allclose(self.Cvoigt, et_temp.Cvoigt):
            raise ValueError('Inconsistent Rotation!')
        else:
            if verbose: print 'Consistent rotation!'
        return
    
    def rot_dip_strike(self, dip, strike, method='default'):
        """
        Rotate elastic tensor dip and strike angle, original tensor should be VTI
        Definition of geographical coordinate system:
        x   - North; y   - East; z  - depth
        ============================================================================
        Input Parameters:
        dip             - dip angle in degree (0<=dip<=90.)
        strike          - strike angle in degree (0<=strike<360.)
                            clockwise from North (x-axis)
        method          - 'default' : rotate with dip and strike in two steps
                          'euler'   : rotate with Euler angles
        ============================================================================
        """
        if dip >90. or dip < 0.: raise ValueError('Dip should be within [0., 90.]!')
        if method=='default':
            self.rotTB(axis=[1,0,0], angle=dip)
            self.rotTB(axis=[0,0,1], angle=strike)
        elif method == 'euler':
            g  = euler2mat(ai=dip, aj=0., ak=strike, axes='sxyz')
            if use_opt_einsum:
                self.Cijkl[:]=contract('ia,jb,kc,ld,abcd->ijkl', g, g, g, g, self.Cijkl)
            else:
                self.Cijkl[:]=np.einsum('ia,jb,kc,ld,abcd->ijkl', g, g, g, g, self.Cijkl)
            self.Cijkl2Voigt()
        return
    
    def rot_dip_strike2(self, dip, strike, verbose=True):
        self.rot_dip_strike(dip=dip, strike=strike)
        et_temp = self.copy()
        et_temp.rot_dip_strike(dip=dip, strike=strike, method='euler')
        if not np.allclose(self.Cvoigt, et_temp.Cvoigt):
            raise ValueError('Inconsistent dip/strike Rotation!')
        else:
            if verbose: print 'Consistent dip/strike Rotation!'
        return
    
    def check_stability(self, verbose=True):
        """Check that the elastic constants matrix is positive definite 
        That is,  check that the structure is stable to small strains. This
        is done by finding the eigenvalues of the Voigt elastic stiffness matrix
        by diagonalization and checking that they are all positive.
        See Born & Huang, "Dynamical Theory of Crystal Lattices" (1954) page 141.
        """
        stable = False
        (eigenvalues, eigenvectors) = np.linalg.eig(self.Cvoigt)
        if (np.amin(eigenvalues) > 0.0):
            stable = True
        else:
            print 'Eigenvalues:', eigenvalues
            raise ValueError('Elastic tensor not stable to small strains (Voigt matrix is not positive definite)')
        return stable
        
    ###############################################################
    # Methods for specifying elastic parameters
    ###############################################################
    
    def set_love(self, A, C, L, N, F, resetCijkl=True, mtype='VTI'):
        """
        Set Love parameters for a VTI media
        ============================================================================
        Input Parameters:
        A,C,L,N,F   - Love parameters (GPa)
        resetCijkl  - reset 4th order tensor or not
        ============================================================================
        """
        if mtype=='VTI':
            self.Cvoigt[:] =  np.array([[A, A-2.*N, F, 0., 0., 0.],
                                        [A-2.*N, A, F, 0., 0., 0.],
                                        [F, F, C, 0., 0., 0.],
                                        [0., 0., 0., L, 0., 0.],
                                        [0., 0., 0., 0., L, 0.],
                                        [0., 0., 0., 0., 0., N]])
            self.info='Love VTI'
        elif mtype=='HTI':
            self.Cvoigt[:] =  np.array([[C, F, F, 0., 0., 0.],
                                        [F, A, A-2*N, 0., 0., 0.],
                                        [F, A-2*N, A, 0., 0., 0.],
                                        [0., 0., 0., N, 0., 0.],
                                        [0., 0., 0., 0., L, 0.],
                                        [0., 0., 0., 0., 0., L]])
            self.info='Love HTI'
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
        vp  - Voigt average P wave velocity
        vs  - Voigt average shear wave velocity
        rho - Density
        xi  - (Vsh^2/Vsv^2) of horizontal waves
        phi - (Vpv^2/Vph^2)
        eta - C13/(C11 - 2*C44)
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
        self.info       = 'radial VTI'
        if resetCijkl: self.Voigt2Cijkl()
        return
    
    def set_thomsen(self, vp, vs, eps, gamma, delta, rho, resetCijkl=True):
        """
        Set Thomsen parameters for a VTI media
        ============================================================================
        Input Parameters:
        vp,vs                   - (km/s)
        eps,gamma,delta         - Thomsen parameters, dimensionless
        rho                     - density (kg/m^2)
        resetCijkl              - reset 4th order tensor or not
        ============================================================================
        """
        #  convert to m/s
        vp=vp*1e3
        vs=vs*1e3
        C       = np.zeros([6,6])
        C[2,2]  = vp*vp  # Eq 9a in Thomsen paper.
        C[3,3]  = vs*vs  # 9b
        C[5,5]  = C[3,3]*(2.0*gamma +1.0) # 8b
        C[0,0]  = C[2,2]*(2.0*eps +1.0) # 8a
           
        btm     = 2.0*C[3,3]
        term    = C[2,2] - C[3,3]
        ctm     = C[3,3]*C[3,3] - (2.0*delta*C[2,2]*term + term*term) 
        dsrmt   = (btm*btm - 4.0*ctm) 
        if dsrmt < 0: raise ValueError('S-velocity too high or delta too negative for Thomsen routine.')
           
        C[0,2] = -btm/2.0 + np.sqrt(dsrmt)/2.0 
        C[0,1] = C[0,0] - 2.0*C[5,5] 
        C[1,2] = C[0,2] 
        C[4,4] = C[3,3] 
        C[1,1] = C[0,0] 
        
        # make symmetrical
        for i in xrange(6):
             for j in xrange(6):
                 C[j,i] = C[i,j]
        #  convert to GPa
        C           = C*rho/1e9
        # output data
        self.Cvoigt = C
        self.rho    = rho
        self.info   = 'Thomsen VTI'
        if resetCijkl: self.Voigt2Cijkl()
        return
         
    def elastic_DB(self, mtype, resetCijkl=True):
        """
        Get elastic constant from predefined data base
        ============================================================================
        Input Parameters:
        mtype       - mineral type
        resetCijkl  - reset 4th order tensor or not
        ============================================================================
        """
        mtype   = mtype.lower()
        if mtype == 'olivine' or mtype == 'ol':
            self.info = 'Single crystal olivine (Abramson et al, JGR, 1997; doi:10.1029/97JB00682)'  
            self.Cvoigt[:] =  np.array([[320.5, 68.1, 71.6, 0., 0., 0.],
                                        [68.1, 196.5, 76.8, 0., 0., 0.],
                                        [71.6, 76.8, 233.5, 0., 0., 0.],
                                        [0., 0., 0., 64.0, 0., 0.],
                                        [0., 0., 0., 0., 77.0, 0.],
                                        [0., 0., 0., 0., 0., 78.7]])
            self.rho=3355.
        elif mtype == 'fayalite' or mtype == 'fa':
            self.info = 'Isentropic elastic constants for single cystal (Fe0.94,Mn0.06)2SiO4 fayalite (Speziale et al, JGR, 2004; doi:10.1029/2004JB003162)' 
            self.Cvoigt[:] =  np.array([[270., 103.,  97.2, 0., 0., 0.],
                                        [103., 171.1, 93.5, 0., 0., 0.],
                                        [97.2, 93.5, 234.1, 0., 0., 0.],
                                        [0., 0., 0., 33.4, 0., 0.],
                                        [0., 0., 0., 0., 48.7, 0.],
                                        [0., 0., 0., 0., 0., 59.6]])
            self.rho=4339.1
        elif mtype == 'albite' or mtype == 'alb':
            self.info = 'Single crystal albite (Brown et al, PCM, 2006; doi:10.1007/s00269-006-0074-1)' 
            self.Cvoigt[:] =  np.array([[69.9,  34.0,  30.8,   5.1, -2.4, -0.9],
                                        [34.0, 183.5,   5.5,  -3.9, -7.7, -5.8],
                                        [30.8,   5.5, 179.5,  -8.7,  7.1, -9.8],
                                        [ 5.1,  -3.9,  -8.7,  24.9, -2.4, -7.2],
                                        [-2.4,  -7.7,   7.1,  -2.4, 26.8,  0.5],
                                        [-0.9,  -5.8,  -9.8,  -7.2,  0.5,  33.5]])
            self.rho=2623.
        elif mtype == 'anorthite' or mtype == 'an96':
            # Brown, Angel and Ross "Elasticity of Plagioclase Feldspars" 
            self.info = 'Single crystal anorthite 96 (Brown et al, JGR, 2016)'
            self.Cvoigt[:] =  np.array([[132.2, 64.0,  55.3,   9.5,  5.1, -10.8],
                                        [ 64.0,200.2,  31.9,   7.5,  3.4,  -7.2],
                                        [ 55.3, 31.9, 163.9,   6.6,  0.5,   1.6],
                                        [  9.5,  7.5,   6.6,  24.6,  3.0,  -2.2],
                                        [  5.1,  3.4,   0.5,   3.0, 36.6,   5.2],
                                        [-10.8, -7.2,   1.6,  -2.2,  5.2,  36.0]])
            self.rho=2757.
        elif mtype == 'enstatite' or mtype == 'ens':
            self.info = 'Single crystal orthoenstatite (Weidner et al, PEPI 1978, 17:7-13)'
            self.Cvoigt[:] =  np.array([[225.0, 72.0, 54.0, 0., 0., 0.],
                                        [ 72.0,178.0, 53.0, 0., 0., 0.],
                                        [ 54.0, 53.0,214.0, 0., 0., 0.],
                                        [0., 0., 0., 78.0, 0., 0.],
                                        [0., 0., 0., 0., 76.0, 0.],
                                        [0., 0., 0., 0., 0., 82.0]])
            self.rho=3200.
        elif mtype == 'jadeite' or mtype == 'jd':
            self.info = 'Single crystal jadeite (Kandelin and Weidner, 1988, 50:251-260)'
            self.Cvoigt[:] =  np.array([[274.0, 94.0, 71.0, 0., 4., 0.],
                                        [ 94.0,253.0, 82.0, 0.,14., 0.],
                                        [ 71.0, 82.0,282.0, 0.,28., 0.],
                                        [   0.,   0.,   0.,88.0, 0., 13.0],
                                        [  4.0,  14.,  28.,  0.,65.,  0.],
                                        [   0.,   0.,   0., 13.0, 0., 94.0]])
            self.rho=3300.
        elif mtype == 'diopside' or mtype == 'di':
            self.info = 'Single crystal chrome-diopeside (Isaak and Ohno, PCM, 2003, 30:430-439)'
            self.Cvoigt[:] =  np.array([[228.1, 94.0, 71.0, 0., 7.9, 0.],
                                        [ 94.0,181.1, 82.0, 0., 5.9, 0.],
                                        [ 71.0, 82.0,245.4, 0.,39.7, 0.],
                                        [   0.,  0.0,  0.0,78.9, 0.,6.4],
                                        [  7.9,  5.9, 39.7,  0.,68.2,0.],
                                        [   0.,   0.,   0., 6.4, 0., 78.1]])
            self.rho=3400.
        elif mtype == 'halite' or mtype == 'nacl':
            self.info = 'Single crystal halite (NaCl, rock-salt).'
            self.Cvoigt[:] =  np.array([[ 49.5, 13.2, 13.2, 0.0, 0., 0.],
                                        [ 13.2, 49.5, 13.2, 0.0, 0., 0.],
                                        [ 13.2, 13.2, 49.5, 0.0, 0., 0.],
                                        [   0.,  0.0,  0.0,12.8, 0., 0.],
                                        [   0.,   0.,   0.,  0.,12.8,0.],
                                        [   0.,   0.,   0.,  0., 0., 12.8]])
            self.rho=2170.
        elif mtype == 'sylvite' or mtype == 'kcl':
            self.info = 'Single crystal sylvite (KCl).'
            self.Cvoigt[:] =  np.array([[ 40.1,  6.6,  6.6, 0.0, 0., 0.],
                                        [  6.6, 40.1,  6.6, 0.0, 0., 0.],
                                        [  6.6,  6.6, 40.1, 0.0, 0., 0.],
                                        [   0.,  0.0,  0.0, 6.4, 0., 0.],
                                        [   0.,   0.,   0.,  0.,6.4, 0.],
                                        [   0.,   0.,   0.,  0., 0., 6.4]])
            self.rho=1990.
        elif mtype == 'galena':
            self.info = 'Single crystal galena (Bhagavantam and Rao, Nature, 1951 168:42)'
            self.Cvoigt[:] =  np.array([[ 127., 29.8, 29.8, 0.0, 0., 0.],
                                        [ 29.8, 127., 29.8, 0.0, 0., 0.],
                                        [ 29.8, 29.8, 127., 0.0, 0., 0.],
                                        [   0.,  0.0,  0.0,24.8, 0., 0.],
                                        [   0.,   0.,   0.,  0.,24.8, 0.],
                                        [   0.,   0.,   0.,  0., 0., 24.8]])
            self.rho=7600.
        elif mtype == 'stishovite':
            self.info = 'Single crystal stishovite, SiO2 (Weidner et al., JGR, 1982, 87:4740-4746)'
            self.Cvoigt[:] =  np.array([[ 453., 211., 203., 0.0, 0., 0.],
                                        [ 211., 453., 203., 0.0, 0., 0.],
                                        [ 203., 203., 776., 0.0, 0., 0.],
                                        [   0.,  0.0,  0.0,252., 0., 0.],
                                        [   0.,   0.,   0.,  0.,252., 0.],
                                        [   0.,   0.,   0.,  0., 0., 252.]])
            self.rho=4290.
        elif mtype == 'fluorapatite'or mtype == 'apatite':
            self.info = 'Single crystal fluorapatite, Ca5F(PO4)3 (Sha et al., J. Appl. Phys., 1994, 75:7784; doi:10.1063/1.357030)'
            self.Cvoigt[:] =  np.array([[ 152., 49.99,  63.11, 0.0, 0., 0.],
                                        [49.99, 152.0,  63.11, 0.0, 0., 0.],
                                        [63.11, 63.11,  185.7, 0.0, 0., 0.],
                                        [   0.,  0.0,  0.0,  42.75, 0., 0.],
                                        [   0.,   0.,   0.,  0., 42.75, 0.],
                                        [   0.,   0.,   0.,  0.,    0., 51.005]])
            self.rho=3150.
        elif mtype == 'antigorite' or mtype == 'atg':
            # Note that X1||a, X2||b and X3||c* - not IRE convection.
            # and that these are adiabatic, not isothermal, constants, but
            # that's what "should" be used for wave velocites (c.f. Karato
            # deformation of Earth materials, sec. 4.3). Velocities quoted
            # in the reference (Table 3) use the corrected isotropic 
            # adiabatic moduli... 
            self.info = 'Adiabatic single crystal antigorite, (Bezacier et al., EPSL 2010, 289:198-208; doi:10.1016/j.epsl.2009.11.009)'
            self.Cvoigt[:] =  np.array([[ 208.1,  66.4,  16.00, 0.0,  5.5, 0.],
                                        [  66.4, 201.6,   4.90, 0.0, -3.1, 0.],
                                        [ 16.00,  4.90,  96.90, 0.0,  1.6, 0.],
                                        [    0.,    0.0,   0.0,16.9,  0.0, -12.10],
                                        [   5.5,  -3.10,   1.6,  0., 18.4, 0.],
                                        [   0.0,    0.0,   0.0,-12.10, 0., 65.50]])
            self.rho=2620.
        elif mtype == 'llm_mgsio3ppv':
            self.info = 'Adiabatic single crystal MgSiO3 post-perovskite under lowermost mantle conditions: from molecular dynamics, DFT & GGA and 2800 K and 127 GPa \
                        (Wookey et al. Nature, 2005, 438:1004-1007; doi:10.1038/nature04345 )'
            self.Cvoigt[:] =  np.array([[ 1139.,  357.,  311., 0.0,   0., 0.],
                                        [  357.,  842.,  466., 0.0,   0., 0.],
                                        [  311.,  466., 1137., 0.0,   0., 0.],
                                        [    0.,    0.,   0.0, 268.,  0., 0.],
                                        [    0.,    0.,   0.0,  0., 210., 0.],
                                        [    0.,    0.,   0.0,  0.,   0., 346.]])
            self.rho=5269.
        elif mtype == 'llm_mgsio3pv':
            self.info = 'Adiabatic single crystal MgSiO3 post-perovskite under lowermost mantle conditions: from molecular dynamics, DFT & GGA and 2800 K and 127 GPa \
                        (Wookey et al. Nature, 2005, 438:1004-1007; doi:10.1038/nature04345 )'
            self.Cvoigt[:] =  np.array([[  808.,  522.,  401., 0.0,   0., 0.],
                                        [  522., 1055.,  472., 0.0,   0., 0.],
                                        [  401.,  472.,  993., 0.0,   0., 0.],
                                        [    0.,    0.,   0.0, 328.,  0., 0.],
                                        [    0.,    0.,   0.0,  0., 263., 0.],
                                        [    0.,    0.,   0.0,  0.,   0., 262.]])
            self.rho=5191.
        elif mtype == 'ice':
            self.info = 'Adiabatic single crystal artifical water ice at -16C: from Gammon et al. (1983) Journal of Glaciology 29:433-460.'
            self.Cvoigt[:] =  np.array([[13.961,  7.153, 5.765, 0.0,   0., 0.],
                                        [ 7.153, 13.961, 5.765, 0.0,   0., 0.],
                                        [ 5.765,  5.765,15.013, 0.0,   0., 0.],
                                        [    0.,    0.,   0.0, 3.21,   0., 0.],
                                        [    0.,    0.,   0.0,   0., 3.21, 0.],
                                        [    0.,    0.,   0.0,   0.,   0., 3.404]])
            self.rho=919.10
        elif mtype == 'quartz' or mtype == 'qz':
            self.info = 'Premium cultured single crystal alpha-quartz at 22 C using resonance-ultrasound spectroscopy. \
                        From Heyliger et al. (2003) Journal of the Accoustic Society of America 114:644-650'
            self.Cvoigt[:] =  np.array([[ 87.17,   6.61, 12.02, -18.23,    0., 0.],
                                        [  6.61,  87.17, 12.02,  18.23,    0., 0.],
                                        [ 12.02,  12.02, 105.8,    0.0,    0., 0.],
                                        [-18.23,  18.23,   0.0,  58.27,    0., 0.],
                                        [    0.,     0.,   0.0,     0., 58.27, -18.23],
                                        [    0.,     0.,   0.0,     0.,-18.23,  40.28]])
            self.rho=2649.7
        else: raise NameError('Unexpected name of mineral!')
        if resetCijkl: self.Voigt2Cijkl()
        return
    
    
class Christoffel(object):
    """
    Contains all information about the material, such as
    density and stiffness tensor. Given a reciprocal vector
    (sound wave direction), it can produce phase and group
    velocities and associated enhancement factors.

    After initialization, set a wave vector direction with
    set_direction or set_direction_spherical, after which any and all
    information can be gained from the get_* functions. All calculations
    will be done on the fly on a need-to-know basis.

    Keyword arguments:
    stiffness -- 6x6 stiffness tensor in GPa
    density -- density of the material in kg/m^3
    """

    def __init__(self, etensor):
        self.bulk = get_bulk(stiffness)
        self.shear = get_shear(stiffness)
        self.iso_P, self.iso_S = isotropic_velocities(self.bulk, self.shear, density)

        stiffness = 0.5 * ( stiffness + stiffness.T)
        self.stiffness = np.array(de_voigt(stiffness))
        self.stiffness *= 1000.0/density
        self.density = density

        self.hessian_mat = hessian_christoffelmat(self.stiffness)

        self.clear_direction()
    
    