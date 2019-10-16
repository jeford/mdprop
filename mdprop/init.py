import numpy as np
import scipy
import scipy.optimize

from . import units, utils

def center(coordinates, masses):
    """
    Given coordinates and masses, return coordinates with center of mass at 0.
    Also works to remove COM translational motion.

    Args:
        coordinates ({nparticle, ndim} ndarray): xyz (to compute COM) or velocities (COM velocity)
        masses ({nparticle,} array_like): masses

    Returns:
        centered_coordinates ({nparticle, ndim} ndarray): xyz with COM at origin or velocities with COM velocity removed 
    """
    com_coordinates = utils.compute_center_of_mass(coordinates, masses)
    new_coordinates = np.array(coordinates) - com_coordinates
    return new_coordinates

def angular_momentum_components(coordinates, velocities, masses):
    """
    Computes the minimal change in 3D momentum vectors to remove the angular 
    momentum by solving 

    min. || \Delta p ||^2_2
    s.t. x \cross (p + \Delta p) = 0

    Args:
        coordinates ({nparticle, 3} ndarray): xyz coordinates 
        velocities ({nparticle, 3} ndarray): particle velocities
        masses ({nparticle,} array_like): masses

    Returns:
        dv ({nparticle, 3} ndarray): minimal perturbation to velocities that have no angular momentum, such that v + dv has no angular momentum
    """
    mass_r = np.reshape(masses, (-1, 1))
    linear_momentum = velocities * mass_r
    angular_constraint_mat = np.reshape(np.transpose(utils.cross_product_matrix(coordinates), (1, 0, 2)), (3, -1)) # 3 x (N x 3)
    constraint_vec = - np.dot(angular_constraint_mat, np.reshape(linear_momentum, (-1, 1))) # 3 x 1
    dp = np.linalg.lstsq(angular_constraint_mat, constraint_vec, rcond=None)[0]
    dv = np.reshape(dp, (-1, 3)) / mass_r
    return dv
    
def initialize_centered(coordinates, velocities, masses):
    """
    Initialize coordinates to have COM at the origin and velocities to have COM
    translation and angular momentum removed by minimizing

    min. || \Delta p ||^2_2
    s.t. x \cross (p + \Delta p) = 0
         \sum_i (p_i + \Delta p_i) = 0
         \sum_i x_i m_i = 0

    Args:
        coordinates ({nparticle, 3} ndarray): particle coordinates 
        velocities ({nparticle, 3} ndarray): particle velocities
        masses ({nparticle,} array_like): masses

    Returns:
        centered_coordinates ({nparticle, 3} ndarray): particle coordinates 
        centered_velocities ({nparticle, 3} ndarray): particle velocities
    """
    center_coordinates = center(coordinates, masses)
    center_velocities = center(velocities, masses)
    mass_r = np.reshape(masses, (-1, 1))
    linear_momentum = center_velocities * mass_r
    linear_constraint_mat = np.hstack([np.eye(3) for i in range(np.shape(center_coordinates)[0])]) # 3 x (N x 3)
    angular_constraint_mat = np.reshape(np.transpose(utils.cross_product_matrix(center_coordinates), (1, 0, 2)), (3, -1)) # 3 x (N x 3)
    total_constraint_mat = np.vstack([linear_constraint_mat, angular_constraint_mat]) # 6 x (N x 3)
    constraint_vec = - np.dot(total_constraint_mat, np.reshape(linear_momentum, (-1, 1))) # 6 x 1
    dp = np.linalg.lstsq(total_constraint_mat, constraint_vec, rcond=None)[0]
    dv = np.reshape(dp, (-1, 3)) / mass_r
    center_velocities = center_velocities + dv
    return center_coordinates, center_velocities

def rescale_velocity(velocities, masses, kT):
    """
    Rescale velocities to correspond to a specific temperature.

    Args:
        velocities ({nparticle, ndim} ndarray): particle velocities
        masses ({nparticle,} ndarray): particle masses
        kT (float): Temperature in energy units

    Returns:
        rescaled_velocities ({nparticle, ndim} ndarray): particle velocities after scaling to target kT
    """
    alpha = np.sqrt(kT / utils.compute_temperature(velocities, masses))
    rescaled_velocities = velocities * alpha
    return rescaled_velocities

def boltzmann(kT, masses, dim=3, recenter=True):
    """
    Take sample in dim dimensions for each masses at each temperature.

    Args:
        kT (float): Temperature in energy units
        masses ({nparticle,} ndarray): particle masses
        dim (int): number of dimensions to sample
        recenter (bool): True to remove center of mass motion

    Returns:
        velocities ({nparticle, ndim} ndarray): particle velocities sampled from boltzmann distribution at energy kT
    """
    mass_r = np.reshape(masses, (-1, 1))
    factor = np.sqrt(kT/mass_r)
    velocities = np.random.normal(scale=factor, size=(mass_r.shape[0], dim))
    if recenter:
        velocities = center(velocities, mass_r)
    return velocities

def sin_velocity(masses, Qs, kT):
    """
    Samples velocities and auxiliary velocities from Boltzmann distribution and then rescales the joint velocity and auxiliary velocity to fit the isokinetic Nose Hoover constraint.

    Args:
        masses ({nparticle,} ndarray): masses of nonauxiliary dofs
        Qs ({2, L, nparticle, ndim} ndarray): auxiliary masses
        kT (float): temperature in energy units

    Returns:
        ({nparticle, ndim} ndarray, {2, L, nparticle, ndim} ndarray): tuple of velocity and auxiliary velocity arrays
    """
    _, Lint, natom, ndim = np.shape(Qs)
    factor = np.sqrt(kT/Qs)
    velocities = boltzmann(kT, masses, dim=ndim)
    aux_v = np.random.normal(scale=factor)
    L = float(Lint)
    for i in range(natom):
        for a in range(ndim):
            v = np.concatenate((velocities[i, a], aux_v[0, :, i, a]), axis=None)
            m = np.concatenate((masses[i] / L, 1.0/(L+1.0)*Qs[0, :, i, a]), axis=None)
            curr_kT = np.sum(v**2 * m)
            alpha = np.sqrt(kT / curr_kT)
            velocities[i, a] *= alpha
            aux_v[0, :, i, a] *= alpha

    Lhalf = Lint//2
    # The first aux velocity should not change sign during dynamics, so we
    # initialize half of them to each sign (if there are an even number of them)
    if Lhalf*2 == Lint:
        aux_v[0, :Lhalf, :, :] =  np.abs(aux_v[0, :Lhalf, :, :])
        aux_v[0, Lhalf:, :, :] = -np.abs(aux_v[0, Lhalf:, :, :])
        
    return velocities, aux_v


def boltzmann_range(kT, masses, ppf_range):
    """
    Take single sample of the Boltzmann distribution within a probability range.
 
    Args:
        kT (float): Temperature in energy units
        masses ({nparticle,} ndarray): particle masses
        ppf_range ({2, nparticle} array_like): percent range to sample in probability distribution            

    Returns:
        speeds ({nparticle,} ndarray): speeds sampled from boltzmann distribution in given probability range
    """
    ppf = np.random.uniform(ppf_range[0], ppf_range[1])
    factor = np.sqrt(kT/masses)
    sample = scipy.stats.maxwell.ppf(ppf, scale=factor)
    return sample

def bimol(
        coordinates1, 
        coordinates2, 
        masses1, 
        masses2, 
        impact_kT=10000.0 * units.K_TO_AU, 
        vibration_kT=600.0 * units.K_TO_AU,
        nosc=100, 
        freq=3000.0 * units.INV_CM_TO_AU, 
        impact_ppf_range=(0.4,0.6), 
        tolerance=4.0,
    ):
    """
    Function to go from two molecular coordinates to a total configuration for bimolecular collision

    Args:
        coordinates1 ({nparticle1, 3} ndarray): coordinates of molecule 1
        coordinates2 ({nparticle2, 3} ndarray): coordinates of molecule 2
        masses1 ({nparticle1,} ndarray): masses of atoms in molecule 1
        masses2 ({nparticle2,} ndarray): masses of atoms in molecule 2
        impact_kT (float): temperature at which to sample the impact in energy units
        vibration_kT (float): temperature at which to sample molecular vibrations in energy units
        nosc (float): number of oscillations at given frequency, used to compute distance
        freq (float): vibrational frequency to use in energy units
        impact_ppf_range ({2,} array_like): percents to sample of Boltzmann cumulative distribution function for impact velocity
        tolerance (float): minimum usable distance between two molecules

    Returns:
        coordinates_tot ({nparticle1+nparticle2, 3} ndarray): coordinates of two molecules randomly rotated, separated on x axis
        velocities_tot ({nparticle1+nparticle2, 3} ndarray): velocities drawn from boltzmann distributions pointed at one another on x axis
    """
    # Convert to atomic units
    mass_tot = np.concatenate([masses1, masses2])
    natom1 = len(masses1)
    natom2 = len(masses2)

    # Get randomly rotated molecules
    coordinates1_rot = np.dot(center(coordinates1, masses1), utils.random_rotation_matrix())
    coordinates2_rot = np.dot(center(coordinates2, masses2), utils.random_rotation_matrix())

    # View each molecule as rigid body of their own mass at the given temperature,
    # sample boltzmann distribution of velocities within given percentage bounds
    sample_velocities1 = boltzmann_range(impact_kT, np.sum(masses1), impact_ppf_range)
    sample_velocities2 = boltzmann_range(impact_kT, np.sum(masses2), impact_ppf_range)
    speed_tot = sample_velocities1 + sample_velocities2

    # Compute time to impact based on number of oscillations at the given frequency
    impact_time = nosc / freq

    # Use velocities and time to impact to compute minimal distance
    impact_dist = speed_tot * impact_time

    # Place first molecule with center of mass zero, second molecule on 1d line with minimal distance
    min_dist = impact_dist + np.max(coordinates1_rot[:, 0]) - np.min(coordinates2_rot[:, 0])
    coordinates2_rot[:, 0] = coordinates2_rot[:, 0] + min_dist

    # Ensure molecules are beyond tolerance
    while np.min(utils.pairwise_dist(coordinates1_rot, coordinates2_rot)) < tolerance:
        coordinates2_rot[:, 0] += 0.2 

    # Construct total coordinate matrix, center total mass at zero
    coordinates_tot = np.vstack([coordinates1_rot, coordinates2_rot])
    coordinates_tot = center(coordinates_tot, mass_tot)

    # Assign velocities
    velocities_tot = np.zeros_like(coordinates_tot)
    velocities_tot[:natom1, 0] = sample_velocities1
    velocities_tot[natom1:, 0] = -sample_velocities2

    # Add random boltzmann energy to each atom (may be different temperature)
    velocities_tot[:natom1, :] += boltzmann(vibration_kT, masses1)
    velocities_tot[natom1:, :] += boltzmann(vibration_kT, masses2)

    # Remove COM motion and total angular momentum
    coordinates_tot, velocities_tot = initialize_centered(coordinates_tot, velocities_tot, mass_tot)

    return coordinates_tot, velocities_tot

#def gaussian_x(x, a=1.0, x0=0.0, k0=1.0):
#    """
#    Gaussian wave packet of width a, centered at x0, with momentum k0
#    """
#    return ((a * np.sqrt(np.pi)) ** (-0.5)
#            * np.exp(-0.5 * ((x - x0) * 1. / a) ** 2 + 1j * x * k0))
#
#def gaussian_k(k, a=1.0, x0=0.0, k0=1.0):
#    """
#    Fourier transform of gaussian_x(x)
#    """
#    return ((a / np.sqrt(np.pi))**0.5
#            * np.exp(-0.5 * (a * (k - k0)) ** 2 - 1j * (k - k0) * x0))
