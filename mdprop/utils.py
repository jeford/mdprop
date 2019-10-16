import numpy as np
import scipy.linalg
import scipy.spatial
import scipy.optimize
import copy as cp

from . import data, units

def density_to_spherical_radius(masses, density):
    """
    Given list of masses and density, computes radius of sphere in a.u.

    Args:
        masses (array_like): in au
        density (float): in grams/cm^3

    Returns:
        (float): radius of sphere for given density with given mass
    """
    mass = np.sum(masses)
    density_au = density * units.KG_TO_AU * 1.E-3 / ((units.M_TO_AU/100.0)**3) # convert density units
    volume = mass / density_au
    r = pow(volume * 3.0 / 4.0 / np.pi, 1.0/3.0)
    return r

def compute_center_of_mass(coordinates, masses):
    """
    Given coordinates and masses, return center of mass coordinates.
    Also works to compute COM translational motion.

    Args:
        coordinates ({nparticle, ndim} ndarray): xyz (to compute COM) or velocities (COM velocity)
        masses ({nparticle,} array_like): masses

    Returns:
        ({ndim,} ndarray): center of mass coordinates
    """
    coordinates_cp = np.array(coordinates)
    mass_cp = np.reshape(masses, (-1, 1))
    com_coordinates = np.sum(mass_cp * coordinates_cp, axis=0)/np.sum(mass_cp)
    return com_coordinates

def compute_moment_of_inertia_tensor(coordinates, masses):
    """
    Given coordinates and masses, compute 3 x 3 moment of inertia tensor

    Args:
        coordinates ({nparticle, 3} ndarray): coordinates
        masses ({nparticle,} ndarray): masses

    Returns:
        ({3, 3} ndarray): moment of inertia tensor
    """
    X2 = coordinates**2
    I = np.zeros((3, 3))
    I[0, 0] = np.sum((X2[:, 1] + X2[:, 2]) * masses)
    I[1, 1] = np.sum((X2[:, 0] + X2[:, 2]) * masses)
    I[2, 2] = np.sum((X2[:, 0] + X2[:, 1]) * masses)
    I[0, 1] = I[1, 0] = - np.sum(coordinates[:, 0] * coordinates[:, 1] * masses)
    I[0, 2] = I[2, 0] = - np.sum(coordinates[:, 0] * coordinates[:, 2] * masses)
    I[1, 2] = I[2, 1] = - np.sum(coordinates[:, 1] * coordinates[:, 2] * masses)
    return I

def compute_angular_momentum(coordinates, velocities, masses):
    """
    Given coordinates, velocites, and masses, computes total angular momentum.

    Args:
        coordinates ({nparticle, 3} ndarray): coordinates
        velocities ({nparticle, 3} ndarray): velocities
        masses ({nparticle,} ndarray): masses

    Returns:
        ({3,} ndarray): vector of total angular momenta
    """
    linear_momentum = np.array(velocities) * np.reshape(masses, (-1, 1))
    coordinates_mat = cross_product_matrix(coordinates)
    total_angular_momentum = np.einsum('ijk,ik->j', coordinates_mat, linear_momentum)
    return total_angular_momentum

def compute_kinetic_energy(velocities, masses):
    r"""
    Compute the kinetic energy using the velocities.

    .. math:: K = 1/2 \sum_i m_i (v_{ix}^2 + v_{iy}^2 + v_{iz}^2)

    Args:
        velocities ({nparticle, ndim} ndarray): velocities
        masses ({nparticle,} ndarray): masses

    Returns:
        (float): total kinetic energy
    """
    kinetic_energy = 0.5 * np.sum(np.sum(velocities**2, axis=1) * np.reshape(masses, (-1, )))
    return kinetic_energy

def compute_kinetic_energy_momentum(momenta, masses):
    r"""
    Compute the kinetic energy using the momenta.

    .. math:: K = 1/2 \sum_i m_i^{-1} (p_{ix}^2 + p_{iy}^2 + p_{iz}^2)

    Args:
        momenta ({nparticle, ndim} ndarray): momenta
        masses ({nparticle,} ndarray): masses

    Returns:
        (float): total kinetic energy
    """
    kinetic_energy = 0.5 * np.sum(np.sum(momenta**2, axis=1) / np.reshape(masses, (-1, )))
    return kinetic_energy

def compute_temperature(velocities, masses, boltzmann_constant=1.0):
    r"""
    Compute the temperature using the velocities.

    .. math:: kT = 1/3N \sum_i m_i (v_{ix}^2 + v_{iy}^2 + v_{iz}^2)

    Args:
        velocities ({nparticle, ndim} ndarray): velocities
        masses ({nparticle,} ndarray): masses
        boltzmann_constant (float): scaling factor of the temperature to energy conversion, defaults to 1.0

    Returns:
        (float): temperature
    """
    temperature = np.sum(np.sum(velocities**2, axis=1) * np.reshape(masses, (-1, ))) / np.size(velocities) / boltzmann_constant
    return temperature

def compute_virial_temperature(coordinates, force, boltzmann_constant=1.0):
    r"""
    Compute virial temperature defined by Clausius virial equation 

    .. math:: 3NkT = - \mathbb{E}[\sum_i r_i^T F_i]

    Args:
        coordinates ({nparticle, ndim} ndarray): coordinates
        force ({nparticle, ndim} ndarray): force
        boltzmann_constant (float): scaling factor of the temperature to energy conversion, defaults to 1.0

    Returns:
        (float): temperature defined through virial equation

    References:
        doi:10.1038/052568a0
        doi:10.1103/PhysRevE.62.4757
    """
    temperature = - np.sum(coordinates * force) / np.size(force) / boltzmann_constant
    return temperature

def compute_configurational_temperature(G, H, G0=None, boltzmann_constant=1.0):
    r"""
    Compute the instantaneous configurational temperature defined by

    .. math:: kT(t) = \nabla V(t)^T \nabla V(t) / \nabla^2 V

    Args:
        G (ndarray): gradient of potential to use to compute configurational temp
        H (ndarray): Hessian of potential to use
        G0 (ndarray): model gradient (default None, uses G)
        boltzmann_constant (float): multiplicative factor to convert energy to desired temperature units (default K/au)

    Returns:
        float: configurational temperature
    """
    if G0 is None:
        G0 = G 
    config_temp = np.sum(G*G0) / np.trace(H) * boltzmann_constant
    return config_temp

def compute_nose_hoover_energy():
    """
    Compute the energy associated with the Nose-Hoover chains
    """
    raise NotImplementedError

def compute_spherical_volume(radius):
    """
    Given radius, return volume of corresponding sphere.

    Args:
        radius (float): radius of sphere

    Returns:
        (float): volume of sphere
    """
    volume = 4.0/3.0 * np.pi * radius**3
    return volume

def compute_spherical_radius(volume):
    """
    Given volume, return radius of corresponding sphere.

    Args:
        volume (float): volume of sphere

    Returns:
        (float): radius of sphere
    """
    radius = (volume * 0.75 / np.pi) ** (1.0/3.0)
    return radius

def compute_convex_volume(coordinates):
    """
    Compute the volume of the convex hull of the given points.

    Args:
        coordinates ({nparticle, ndim} ndarray): coordinates

    Returns:
        (float): volume of convex hull of coordinates
    """
    ch = scipy.spatial.ConvexHull(coordinates)
    return ch.volume

def compute_virial_pressure(volume, kinetic_energy, coordinates, force):
    r"""
    Compute the virial pressure using the virial equation

    .. math:: PV = 2/3 KE + 1/3\sum_i r_i^T F_i

    Assumes consistent units, and constant center of mass at origin

    Args:
        volume (float): volume of system
        kinetic_energy (float): kinetic_energy of system
        coordinates ({nparticle, ndim} ndarray): coordinates
        force ({nparticle, ndim} ndarray): force

    Returns:
        (float): pressure
    """
    pv = (2.0 * kinetic_energy + np.sum(coordinates * force)) / np.shape(coordinates)[-1]
    pressure = pv / volume
    return pressure

def compute_sequential_rmsd(Xs, M=None):
    r"""
    Compute sequential RMSD for all given frames of X.

    .. math:: \mathrm{RMSD}_i = \sqrt{1/\sum_j^{\mathrm{nparticle}} m_j} || \mathrm{diag}(M) X_{i+1} - \mathrm{diag}(M) X_{i} ||

    Args:
        Xs ({nframe, nparticle, ndim} ndarray): coordinates 
        M ({nparticle,} ndarray): optional masses to use, must be same for all frames, default is equivalent to ones

    Returns:
        ({nframe-1,} ndarray): root mean square deviations from frame to frame
    """
    if M is None:
        M = np.ones((np.shape(Xs)[1],))
    # Mass weight and reshape
    Xs_r = np.reshape(np.reshape(Xs, (len(Xs), -1, 3)) * np.sqrt(np.reshape(M, (1, -1, 1))), (len(Xs), -1))
    return np.linalg.norm(Xs_r[1:] - Xs_r[:-1], axis=1) / np.sqrt(np.sum(M))

def subsample_path(Xs, nsample, rmsds=None, interpolate=True):
    """
    Use the rmsds along the path to find images that are roughly evenly spaced 
        in rmsd including the endpoints

    Args:
        Xs ({nframe, nparticle, ndim} ndarray): Original images along path
        nsample (int): number of samples to return
        rmsds ({nframe-1,} ndarray): optional, Use precomputed rmsds, which may use whatever metric you want, should be sorted
        interpolate (bool): default True, interpolate nearest neighbor images weighted by distance to evenly spaced points in rmsd, if False, simply return the nearest neighbor image
    """
    if rmsds is None:
        rmsds = compute_rmsd(Xs)
    cs_rmsds = np.cumsum(np.insert(rmsds, 0, 0.0))
    normalized_rmsds = cs_rmsds / cs_rmsds[-1]
    ideal_rmsds = np.linspace(0.0, 1.0, num=nsample, endpoint=True)
    # Find the indices where it could be inserted to maintain sorting, same as the nearest neighbor
    nn_l = np.searchsorted(normalized_rmsds, ideal_rmsds) - 1
    Xs_sub = [Xs[0]]
    # The first and last images are always the original first and last images,
    # this also helps avoid the ideal_rmsd being exactly equal to the normalized
    # rmsd which can throw off how the searchsorted works
    for i in range(1, nsample-1):
        t1p = ideal_rmsds[i] - normalized_rmsds[nn_l[i]]
        t2p = normalized_rmsds[nn_l[i]+1] - ideal_rmsds[i]
        if t1p == 0.0:
            Xs_sub.append(Xs[nn_l[i]])
        elif t2p == 0.0:
            Xs_sub.append(Xs[nn_l[i]+1])
        elif interpolate: # Calculate weight as normalized difference
            w1 = t2p / (t1p + t2p)
            w2 = t1p / (t1p + t2p)
            Xs_sub.append(w1 * Xs[nn_l[i]] + w2 * Xs[nn_l[i]+1])
        else: # Use nearest neighbor
            if t1p < t2p:
                Xs_sub.append(Xs[nn_l[i]])
            else:
                Xs_sub.append(Xs[nn_l[i]+1])
    Xs_sub.append(Xs[-1])
    return np.array(Xs_sub)

def random_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.

    Args:
        deflection (float): the magnitude of the rotation. For 0, no rotation; for 1, competely random rotation. Small deflection => small perturbation.
        randnums ({3,} array_like): 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.

    Returns:
        ({3, 3} ndarray): R random rotation matrix
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c

    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.

    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    # Construct the rotation matrix  ( V Transpose(V) - I ) R.

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M

def pairwise_dist_squared(xyz1, xyz2=None):
    """
    Compute all squared distances between atoms in one or two molecules

    Args:
        xyz1 ({nparticle1, ndim} ndarray): matrix of coordinates
        xyz2 ({nparticle2, ndim} ndarray): matrix of coordinates, if not given xyz1 will be used

    Returns:
        ({nparticle1, nparticle2} ndarray): matrix of square distances between atoms
    """
    if xyz2 is None:
        xyz2 = xyz1
    xyz1_view = xyz1[:, :, None]
    xyz2_view = np.transpose(xyz2[:, :, None], (2, 1, 0))
    dists = np.sum((xyz1_view - xyz2_view)**2, axis=1)
    return dists

def pairwise_dist(xyz1, xyz2=None):
    """
    Compute all distances between atoms in one or two molecules

    Args:
        xyz1 ({nparticle1, ndim} ndarray): matrix of coordinates
        xyz2 ({nparticle2, ndim} ndarray): matrix of coordinates, if not given xyz1 will be used

    Returns:
        ({nparticle1, nparticle2} ndarray): matrix of distances between atoms
    """
    return np.sqrt(pairwise_dist_squared(xyz1, xyz2))

def symbol_to_covalent_radius(symbols):
    """
    Given iterable of symbols, return numpy array of covalent radii in bohr radius.

    Args:
        symbols ({nparticle,} array_like): atomic symbols

    Returns:
        ({nparticle,} ndarray): covalent_radii in bohr radius

    TODO: Use mtzutils
    """
    cov_rad = []
    for s in symbols:
        cov_rad.append(data.element_dict[s].covalent_radius * units.ANGSTROM_TO_AU)
    cov_rad = np.array(cov_rad)
    return cov_rad

def covalent_dist_matrix(S):
    """
    Compute all covalent distances between atoms

    Args:
        S ({nparticle,} ndarray): atomic symbols

    Returns:
        ({nparticle, nparticle} ndarray): matrix of covalent distances between atoms
    """
    covrad = symbol_to_covalent_radius(S)
    return covrad[:, None] + covrad[None, :]

def symbol_to_mass(symbols):
    """
    Given symbols, return array of masses in electron masses.

    Args:
        S ({nparticle,} ndarray): atomic symbols

    Returns:
        ({nparticle,} ndarray): atomic masses in electron masses
    """
    masses = []
    for s in symbols:
        masses.append(data.element_dict[s].exact_mass * units.AMU_TO_AU)
    return np.array(masses)

def atomic_number_to_mass(Zs):
    """
    Given atomic numbers, return array of masses in electron masses.

    Args:
        S ({nparticle,} ndarray): atomic numbers

    Returns:
        ({nparticle,} ndarray): atomic masses in electron masses
    """
    masses = []
    for Z in Zs:
        masses.append(data.element_list[Z].exact_mass * units.AMU_TO_AU)
    return np.array(masses)

def align_procrustes(X1, X2, M, mass_weight=False):
    """
    Align one molecule to the other using orthogonal procrustes algorithm.

    Args:
        X1 ({nparticle, ndim} ndarray): Geometry of first molecule (to be aligned)
        X2 ({nparticle, ndim} ndarray): Geometry of second molecule, indexing should be the same
        M ({nparticle,} ndarray): Masses, used to determine center of mass and mass-weighting
        mass_weight (bool): if True, will multiply coordinates by sqrt(M) prior to computing orthogonal procrustes rotation

    Returns:
        ({nparticle, ndim} ndarray): Coordinate of first molecule after rotating, reflecting, translating to the second molecule
    """
    # Move center of mass to origin
    com1 = compute_center_of_mass(X1, M)
    com2 = compute_center_of_mass(X2, M)
    X1center = X1 - com1
    X2center = X2 - com2
    # Compute optimal orthogonal transformation to overlap
    if mass_weight:
        X1m = X1center * np.sqrt(np.reshape(M, (-1, 1)))
        X2m = X2center * np.sqrt(np.reshape(M, (-1, 1)))
        R, alpha = scipy.linalg.orthogonal_procrustes(X1m, X2m)
    else:
        R, alpha = scipy.linalg.orthogonal_procrustes(X1center, X2center)
    X1rot = np.dot(X1center, R)
    # Move center of mass to second center
    X1trans = X1rot + com2
    return X1trans, R

def align_principal_axes(X, M):
    """
    Align molecule to the principal axes frame.

    Args:
        X ({nparticle, 3} ndarray): Coordinates of molecule
        M ({nparticle,} ndarray): Masses (used to determine center of mass)

    Returns:
        ({nparticle, 3} ndarray, {3, 3} ndarray): tuple of coordinates after moving COM to origin and rotating axes to align with intertial tensor and rotation matrix used
    """
    com = compute_center_of_mass(X, M)
    Xtrans = np.array(X) - com
    I = compute_moment_of_inertia_tensor(Xtrans, M)
    I /= np.sum(M) # Renormalizing shouldn't change eigenvectors
    L, U = np.linalg.eigh(I)
    X_align = np.dot(Xtrans, U)
    return X_align, U

def align(X1, X2, M1, M2=None, S1=None, S2=None, prealign=False, permutation=False, reflection=False, mass_weight=False):
    """
    Align one molecule to another. If permutations are allowed, the molecules 
    are first aligned to the Eckart frame, then reflections are checked, and 
    then the permutation is found through linear sum assignment with the 
    pair distance matrix. Then the molecules are aligned using orthogonal 
    procrustes, which can be mass weighted.

    Args:
        X1 ({nparticle, 3} ndarray): Coordinates of first configuration
        X2 ({nparticle, 3} ndarray): Coordinates of second configuration
        M1 ({nparticle,} ndarray): masses of first configuration
        M2 ({nparticle,} ndarray): optional masses of second configuration, defaults to M1
        S1 ({nparticle,} ndarray): optional atomic symbols for first configuration; if given, entries with same symbols will only be permuted within their own group; symbols can be arbitrary specific atom subclasses such as CO or CA for better accuracy
        S2 ({nparticle,} ndarray): optional atomic symbols for second configuration; both S1 and S2 must be given to allow class-based permutations
        prealign (bool): True to align molecules to principle axes prior to anything else, default False
        permutation (bool): True to allow permutation of atoms within systems using linear assignment; this is usually good, but may not be perfect, default False
        reflection (bool): True to allow reflections of geometry, default False
        mass_weight (bool): True to mass_weight by sqrt(M) for procrustes alignment, default False

    Returns:
        X1_align: X1 after alignment (and permutation/reflection)
        if permutation:
            perm: permutation indices such that X1[perm] ~= X2
        if reflection:
            ref: (1, 3) np.array of 1 if not reflected, -1 if reflected

    TODO: Include all permutations of axes after principal axis alignment for cases of symmetry.
    TODO: Rewrite as class, because this is complicated.
    """
    M1curr = np.reshape(M1, (-1, 1))
    if M2 is None:
        M2curr = np.copy(M1curr)
    else:
        M2curr = np.reshape(M2, (-1, 1))
    if S1 is not None:
        S1curr = np.array(S1)
    if S2 is not None:
        S2curr = np.array(S2)
    COM1 = compute_center_of_mass(X1, M1curr)
    COM2 = compute_center_of_mass(X2, M2curr)
    if prealign:
        X1curr, _ = align_principal_axes(X1, M1curr)
        X2curr, R_eck = align_principal_axes(X2, M2curr)
    else:
        X1curr = np.copy(X1)
        X2curr = np.copy(X2)
    # Compute permutation and reflection
    if reflection: # Reflection may be necessary for systems with improper rotations as well
        ref = np.ones((1, 3))
        for i in range(3):
            order1 = np.argsort(np.abs(X1curr[:, i]))
            order2 = np.argsort(np.abs(X2curr[:, i]))
            diff = np.sum((X1curr[order1, i] - X2curr[order2, i])**2)
            add =  np.sum((X1curr[order1, i] + X2curr[order2, i])**2)
            if add < diff:
                X1curr[:, i] *= -1
                ref[0, i] = -1

    if permutation:
        N1 = len(M1)
        D = scipy.spatial.distance.cdist(X1curr, X2curr, "euclidean")
        if S1 is not None and S2 is not None:
            col = -np.ones((N1,), dtype=np.int) # Instantiate to -1 to allow checking if nothing is assigned
            unique_symbols = set(S1)    
            # Iterate through unique symbols
            for u_s in unique_symbols:
                # Get indices of atoms with corresponding symbol
                i1 = np.array([i for i, s in enumerate(S1) if s == u_s])
                i2 = np.array([i for i, s in enumerate(S2) if s == u_s])
                # Get pairwise dist submatrix with correct elements
                d = D[i1][:, i2]
                # Compute linear assignment
                u_row, u_col = scipy.optimize.linear_sum_assignment(d)
                # Assign permuted indices to subset of indices for final permutation
                col[i1] = i2[u_col]
            
        elif S1 is not None or S2 is not None:
            raise ValueError("Symbol lists for both configurations must be given to be used.")

        else:
            # Use linear assignment on distances if symbols not given
            row, col = scipy.optimize.linear_sum_assignment(D)

        perm = -np.ones((len(M1), ), dtype=np.int)
        perm[col] = np.arange(N1)
        if -1 in perm:
            raise ValueError("Permutation assignment unsuccesful.")
        X1curr = X1curr[perm]
        M1curr = M1curr[perm]
        if S1 is not None:
            S1curr = S1curr[perm]

    # Now that permutations are done, find orthogonal procrustes rotation
    if mass_weight:
        R_pro, _ = scipy.linalg.orthogonal_procrustes(X1curr * np.sqrt(M1curr), X2curr * np.sqrt(M2curr))
    else:
        R_pro, _ = scipy.linalg.orthogonal_procrustes(X1curr, X2curr)
    # Rotate first geometry into second geometry's frame
    X1curr = np.dot(X1curr, R_pro)
    if prealign:
        # Must undo eckart rotation
        X1curr = np.dot(X1curr, R_eck.T)
        # Move center of mass to second center
        X1curr += COM2 
    else:
        X1curr += COM2 - COM1
    
    if permutation and reflection:
        return X1curr, perm, ref
    elif permutation:
        return X1curr, perm
    elif reflection:
        return X1curr, ref
    else:
        return X1curr

def cross_product_matrix(X):
    """
    Returns the skew-symmetric matrix representation of X that when multiplied 
    with a 3-vector returns the cross product.

    Args:
        X ({..., 3} ndarray): vector or batch to convert

    Returns:
        ({..., 3, 3} ndarray): (batch of) matrices representing cross product operation
    """
    sh = np.shape(X)
    if sh[-1] != 3:
        raise ValueError("Cross product matrix requires the last dimension to be 3")
    Xmat_sh = tuple(list(sh) + [3])
    Xmat = np.zeros(Xmat_sh)
    Xmat[..., 1, 0] =  X[..., 2]
    Xmat[..., 0, 1] = -X[..., 2]
    Xmat[..., 2, 0] = -X[..., 1]
    Xmat[..., 0, 2] =  X[..., 1]
    Xmat[..., 2, 1] =  X[..., 0]
    Xmat[..., 1, 2] = -X[..., 0]
    return Xmat

def row_permute(A, Pinds):
    """
    Use numpy index slicing to permute a matrix as B = P'A
    where A is the given matrix, P is a permutation matrix and B is returned.
    Because this is done in sparse manner, P is instead given as Pinds which 
    should be the permuted indices, with shape (nperm, 2).

    Args:
        A ({N, M} ndarray): array to permute
        Pinds ({nperm, 2} ndarray): pairs of indices to permute

    Returns:
        ({N, M} ndarray): permuted matrix P'A 

    Note: 
        This routine does not handle resizing if permuting into a larger or smaller matrix.
    """
    Anew = np.copy(A)
    Anew[[Pinds[:, 0], Pinds[:, 1]], :] = A[[Pinds[:, 1], Pinds[:, 0]], :]
    return Anew

def column_permute(A, Pinds):
    """
    Use numpy index slicing to permute a matrix as B = AP
    where A is the given matrix, P is a permutation matrix and B is returned.
    Because this is done in sparse manner, P is instead given as Pinds which 
    should be the permuted indices, with shape (nperm, 2).

    Args:
        A ({N, M} ndarray): array to permute
        Pinds ({nperm, 2} ndarray): pairs of indices to permute

    Returns:
        ({N, M} ndarray): permuted matrix AP

    Note: 
        This routine does not handle resizing if permuting into a larger or smaller matrix.
    """
    Anew = np.copy(A)
    Anew[:, [Pinds[:, 0], Pinds[:, 1]]] = Anew[:, [Pinds[:, 1], Pinds[:, 0]]]
    return Anew

def symmetric_permute(A, Pinds):
    """
    Use numpy index slicing to permute a matrix as B = P'AP
    where A is the given matrix, P is a permutation matrix and B is returned.
    Because this is done in sparse manner, P is instead given as Pinds which 
    should be the permuted indices, with shape (nperm, 2).

    Args:
        A ({N, N} ndarray): array to permute
        Pinds ({nperm, 2} ndarray): pairs of indices to permute

    Returns:
        ({N, N} ndarray): permuted matrix P'AP

    Note: 
        This routine does not handle resizing if permuting into a larger or smaller matrix.
        There is also not error checking for if the same index is included multiple times.
    """
    Anew = np.copy(A)
    Anew[[Pinds[:, 0], Pinds[:, 1]], :] = A[[Pinds[:, 1], Pinds[:, 0]], :]
    Anew[:, [Pinds[:, 0], Pinds[:, 1]]] = Anew[:, [Pinds[:, 1], Pinds[:, 0]]]
    return Anew

def axis_align_matrix(v, axis_ind=0):
    """
    Return matrix that yields Rv = v' where v' is aligned to axis_ind axis

    Args:
        v ({K,} ndarray): vector to align
        axis_ind (int): index of dimension to align to

    Returns:
        ({K, K} ndarray): R, rotation matrix that aligns given vector
    """
    vnew = np.zeros_like(v)
    vnew[axis_ind] = 1.0
    u = v/np.linalg.norm(v) - vnew
    R = np.eye(np.size(u)) - 2.0 * np.outer(u, u) / np.sum(u * u)
    R[-1] *= -1.0
    return R

def time_reversible_propagation(D, Ps):
    """
    Use the Niklasson time-reversible propagation with dissipation for extrapolating a density matrix (or other quantity)
    Order of propagation to use is determined by the length of Ps

    References:
        https://doi.org/10.1063/1.3148075

    Args:
        D ({N, N} ndarray): last converged density matrix
        Ps ({nframe, N, N} ndarray}): last k initial guess density matrices in order of [n-k, n-k+1, ..., n], nframe must be in [4, 10].

    Returns:
        ({N, N} ndarray): Reversibly propagated density matrix
    """
    order = len(Ps)
    if not (4 <= order <= 10):
        raise ValueError("len(Ps) must be in [4, 10] for propagation.")
    
    # Order 3 to 9
    kappas = [1.69, 1.75, 1.82, 1.84, 1.86, 1.88, 1.89]
    alphas = [0.150, 0.057, 0.018, 0.0055, 0.0016, 0.00044, 0.00012]
    coeffs = [ # Ordered as in the paper, c_i goes with P_n-i, i.e. c_0 with P_n, c_1 P_n-1
            [-2.0, 3.0, 0.0, -1.0],
            [-3.0, 6.0, -2.0, -2.0, 1.0],
            [-6.0, 14.0, -8.0, -3.0, 4.0, -1.0],
            [-14.0, 36.0, -27.0, -2.0, 12.0, -6.0, 1.0],
            [-36.0, 99.0, -88.0, 11.0, 32.0, -25.0, 8.0, -1.0],
            [-99.0, 286.0, -286.0, 78.0, 78.0, -90.0, 42.0, -10.0, 1.0],
            [-286.0, 858.0, -936.0, 364.0, 168.0, -300.0, 184.0, -63.0, 12.0, -1.0],
        ]
    order_ind = order - 4
    kappa = kappas[order_ind]
    alpha = alphas[order_ind]
    coeff = [alpha * c for c in coeffs[order_ind]]
    coeff[0] += 2.0 - kappa 
    coeff[1] -= 1.0
    
    P_new = kappa * np.copy(D)
    for c, P in zip(coeff, Ps[::-1]):
        P_new += c * P

    return P_new

def langevin_energy(X1, V1, F1, X2, V2, F2, M, dt):
    """
    The change in 'effective energy' arising from integrating an NVT ensemble 
    using a langevin thermostat

    Args:
        X1 ({nparticle, 3} ndarray): Coordinates of first configuration
        V1 ({nparticle, 3} ndarray): Velocities of first configuration
        F1 ({nparticle, 3} ndarray): Forces acting on first configuration
        X2 ({nparticle, 3} ndarray): Coordinates of second configuration
        V2 ({nparticle, 3} ndarray): Velocities of second configuration
        F2 ({nparticle, 3} ndarray): Forces acting on second configuration
        M ({nparticle,} ndarray): masses
        dt (float): time step used to integrate between X1 and X2

    Returns:
        (float): Effective energy change between two configurations that arises from Langevin integration 
    """
    KE1 = compute_kinetic_energy(V1, M)
    KE2 = compute_kinetic_energy(V2, M)
    dX = X2 - X1
    XF = np.sum(dX * (F1 + F2) * 0.5)
    KEF1 = compute_kinetic_energy_momentum(F1, M)
    KEF2 = compute_kinetic_energy_momentum(F2, M)
    Fsq = 0.25 * dt * dt * (KEF2 - KEF1)
    return KE1 - KE2 + XF + Fsq
    

def numerical_gradient(inp, func, eps=1.E-6, output_ind=None, *args, **kwargs):
    """
    Moves inp by eps in each dimension and calls the function to compute the
    gradient numerically with a symmetric stencil.

    Args:
        inp (ndarray): input to function
        func (function): function to differentiate at inp
        eps (float): small number by which to move each element of inp (default 1.E-6)
        output_ind (int): if given, takes the numerical gradient wrt the i'th output of the function given
        *args: additional function arguments
        **kwargs: function keyword arguments

    Returns:
        (ndarray): numerical gradient of function at input with shape (inp.shape x function_output.shape)
    """
    g = []
    inpr = inp.ravel()
    for i in range(len(inpr)):
        plus = np.copy(inpr)
        plus[i] += eps
        plus = plus.reshape(np.shape(inp))
        minus = np.copy(inpr)
        minus[i] -= eps
        minus = minus.reshape(np.shape(inp))
        if output_ind is not None:
            de = func(plus, *args, **kwargs)[output_ind] - func(minus, *args, **kwargs)[output_ind]
        else:
            de = func(plus, *args, **kwargs) - func(minus, *args, **kwargs)
        dedx = de/2.0/eps
        g.append(dedx)
    dims = []
    if np.shape(inp):
        dims += [d for d in np.shape(inp)]
    if np.shape(g[0]):
        dims += [d for d in np.shape(g[0])]
    grad = np.array(g).reshape(dims)
    return grad

def np_cache(cache_size=4, arg_ind=0):
    """
    Naive LRU cache implementation for functions whose first parameter is a numpy array.
    Makes array_equal comparison, rather than using a hash of a tuple form of the array.

    Kwargs:
        cache_size (int): size of cache to save inputs and outputs (default 4)
        arg_ind (int): index of input argument to check to determine if the output should be identical

    Note:
        For class methods, the arg_ind should be 1, as argument 0 implies the class instance.
        For inherited class methods, this will not work if there multiple base class instances, unless used very carefully.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            inp_curr = args[arg_ind]
            is_cached = False
            for i in range(len(wrapper.inp_cache)):
                inp = wrapper.inp_cache[-i-1]
                if np.array_equal(inp_curr, inp):
                    is_cached = True
                    out = wrapper.out_cache[-i-1]
                    if i > 0:
                        # Move recalled value to end of cache
                        wrapper.inp_cache.append(wrapper.inp_cache.pop(-i-1))
                        wrapper.out_cache.append(wrapper.out_cache.pop(-i-1))
                    break 

            if not is_cached:
                out = func(*args, **kwargs)
                wrapper.inp_cache.append(np.copy(inp_curr))
                wrapper.inp_cache = wrapper.inp_cache[-wrapper.cache_size:]
                wrapper.out_cache.append(out)
                wrapper.out_cache = wrapper.out_cache[-wrapper.cache_size:]
                    
            return cp.deepcopy(out)

        # Bind a reset method to the new function
        def reset_cache(cache_size=cache_size):
            wrapper.inp_cache = []
            wrapper.out_cache = []
            wrapper.cache_size = cache_size

        wrapper.reset_cache = reset_cache
        wrapper.reset_cache()

        return wrapper
    return decorator

def sin_sigmoid(x):
    r"""
    sin sigmoid transfer function

    .. math:: 

        f(x) &= x_c - 1/2\pi \sin(2\pi x_c) \\
        x_c &= \mathrm{min}(\mathrm{max}(0, x), 1)

    Args:
        x (float): input
    Returns:
        (float): output in [0, 1]
    """
    xc = np.clip(x, 0.0, 1.0)
    return xc - 0.5 / np.pi * np.sin(2.0 * np.pi * xc)

def clamp(x):
    r"""
    Linear transfer function

    .. math:: 

        f(x) &= x_c \\
        x_c &= \mathrm{min}(\mathrm{max}(0, x), 1)

    Args:
        x (float): input
    Returns:
        (float): output in [0, 1]
    """
    return np.clip(x, 0.0, 1.0)

def smoothstep(x):
    r"""
    Sigmoidal transfer function

    .. math:: 

        f(x) &= 3x_c^2 - 2x_c^3 \\
        x_c &= \mathrm{min}(\mathrm{max}(0, x), 1)

    Args:
        x (float): input
    Returns:
        (float): output in [0, 1]
    """
    xc = np.clip(x, 0.0, 1.0)
    return 3.0 * xc**2 - 2.0 * xc**3

def smootherstep(x):
    r"""
    Sigmoidal transfer function

    .. math:: 

        f(x) &= 6x_c^5 - 15x_c^4 + 10x_c^3 \\
        x_c &= \mathrm{min}(\mathrm{max}(0, x), 1)

    Args:
        x (float): input
    Returns:
        (float): output in [0, 1]
    """
    xc = np.clip(x, 0.0, 1.0)
    return 6.0 * xc**5 - 15 * xc**4 + 10 * xc**3
