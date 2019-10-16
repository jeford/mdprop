import numpy as np

from . import utils

"""
potential.py contains several harmonic boundary potentials, such as soft
spherical boundaries (SoftSphere) or soft hyperplane boundaries (SoftHalfSpace).
It also contains simple interparticle potentials such as InteratomicSpring and
Morse. Lastly it contains model potentials for optimization or dynamics such as
the Rosenbrock function and the HenonHeiles model.

All potentials have a compute_energy and compute_gradient function, which takes in a geometry X. 
Many model potentials also have a compute_energy_per_particle method which returns an (n, ) array containing each particle's energy.
Model potentials taking in a magnitude parameter can generally use a vector of magnitudes for cases when perhaps mass-dependent magnitudes are desired.

All methods expect an X argument which has the shape (n, k) where n is the 
number of particles, and k is the dimension in space each particle lives, 
generally 1 to 3.
"""

class Potential(object):
    """
    Base class for simple potentials and the forces they exert on dynamic
    simulations.  Each potential object has defined compute_energy(X) and
    compute_gradient(X); compute_force(X) is automatically defined in this base
    class. Certain per-particle potentials also have
    compute_energy_per_particle(X) to return componentwise potential energies.

    Cached versions of compute_quantity() are simply named 'quantity()'
    """
    def __init__(self):
        if self.compute_energy is not Potential.compute_energy:
            self.energy = utils.np_cache(cache_size=4, arg_ind=0)(self.compute_energy)
        if self.compute_energy_per_particle is not Potential.compute_energy_per_particle:
            self.energy_per_particle = utils.np_cache(cache_size=4, arg_ind=0)(self.compute_energy_per_particle)
        if self.compute_gradient is not Potential.compute_gradient:
            self.gradient = utils.np_cache(cache_size=4, arg_ind=0)(self.compute_gradient)
        self.hessian = utils.np_cache(cache_size=4, arg_ind=0)(self.compute_hessian)

    def compute_energy(self, *args, **kwargs):
        """
        Compute the energy corresponding to the potential
        Args:
            X ({nparticle, ndim} ndarray): coordinates
            *args
            **kwargs

        Returns:
            (float): energy corresponding to configuration
        """
        raise NotImplementedError

    def compute_energy_per_particle(self, *args, **kwargs):
        """
        Compute the energy per particle corresponding to the potential

        Args:
            X ({nparticle, ndim} ndarray): coordinates
            *args
            **kwargs

        Returns:
            ({nparticle,} ndarray): energies
        """
        raise NotImplementedError

    def compute_gradient(self, *args, **kwargs):
        """
        Compute the gradient of the potential
        Args:
            X ({nparticle, ndim} ndarray): coordinates
            *args
            **kwargs

        Returns:
            (float, {nparticle, ndim} ndarray): energy, gradient tuple
        """
        raise NotImplementedError

    def compute_force(self, *args, **kwargs):
        """
        Compute the force of the potential
        Args:
            X ({nparticle, ndim} ndarray): coordinates
            *args
            **kwargs

        Returns:
            (float, {nparticle, ndim} ndarray): energy, force tuple
        """
        pe, g = self.compute_gradient(*args, **kwargs)
        return pe, -g

    def compute_hessian(self, *args, **kwargs):
        """
        Compute the hessian of the potential, defaults to numerical hessian but can be overwritten
        Args:
            X ({nparticle, ndim} ndarray): coordinates
            *args
            **kwargs

        Returns:
            {nparticle*ndim, nparticle*ndim} ndarray: hessian matrix
        """
        X = args[0]
        if len(args) > 1:
            H = utils.numerical_gradient(X, self.gradient, eps=1.E-5, output_ind=1, *(args[1:]), **kwargs)
        else:
            H = utils.numerical_gradient(X, self.gradient, eps=1.E-5, output_ind=1, **kwargs)
        H_r = np.reshape(H, (np.size(X), np.size(X)))
        H_sym = 0.5 * (H_r + np.transpose(H_r))
        return H_sym

    def compute_hessian_vector_product(self, *args, **kwargs):
        """
        Compute the hessian vector product, equivalent to a symmetrized directional derivative of the gradient
        Args:
            X ({nparticle, ndim} ndarray): coordinates
            V ({nparticle, ndim} ndarray): direction in which to evaluate change of gradient
            *args
            **kwargs

        Returns:
            {nparticle, ndim} ndarray: hessian vector product
        """
        X = args[0]
        V = args[1]
        eps = 1E-5
        if len(args) > 2:
            Hv = 0.5 / eps * (self.gradient(X + eps * V, *(args[2:]), **kwargs) - self.gradient(X - eps * V, *(args[2:]), **kwargs))
        else:
            Hv = 0.5 / eps * (self.gradient(X + eps * V, **kwargs) - self.gradient(X - eps * V, **kwargs))
        return Hv

    def force(self, *args, **kwargs):
        """
        Compute the force of the potential, includes caching
        Args:
            X ({nparticle, ndim} ndarray): coordinates
            *args
            **kwargs

        Returns:
            (float, {nparticle, ndim} ndarray): energy, force tuple
        """
        pe, g = self.gradient(*args, **kwargs)
        return pe, -g

    def add(self, other):
        """
        Construct a PotentialList object from two potentials

        Args:
            pot1 (Potential): first potential
            pot2 (Potential): second potential

        Returns:
            (PotentialList): combination of two potentials with unit coefficients
        """
        if isinstance(self, PotentialList):
            potentials = [p for p in self.potentials]
        else:
            potentials = [self]
        if isinstance(other, PotentialList):
            potentials += [p for p in other.potentials]
        else:
            potentials += [other]
        return PotentialList(potentials)

    @property
    def name(self):
        return self.__class__.__name__.lower()

class PotentialWrapper(Potential):
    """
    Simple wrapper for energy/gradient/force functions so that they play nice with Update class
    """
    def __init__(self, energy=None, energy_per_particle=None, gradient=None, force=None):
        """
        Args:
            energy (function): returns energy of given configuration
            energy_per_particle (function): returns ({nparticle,} ndarray) of particle energies for given configuration
            gradient (function): returns (energy, gradient) tuple corresponding to given configuration
            force (function): returns (energy, force) tupe corresponding to given configuration
        """
        if energy is not None:
            self.compute_energy = energy
        if energy_per_particle is not None:
            self.compute_energy_per_particle = energy_per_particle
        if gradient is not None:
            self.compute_gradient = gradient
        if force is not None:
            self.compute_force = force
        super(PotentialWrapper, self).__init__()    

class PotentialList(Potential):
    """
    Simple class to combine potentials into a single potential, for ease of use with a single Update

    Args:
        potentials (list of Potential objects): 
        energy_coeffs ({npotentials,} ndarray): coefficients with which to add potential energies together (default ones)
        grad_coeffs ({npotentials,} ndarray): coefficients with which to add potential gradients together (default ones)

    Note:
        The difference in coefficients is useful when you only want to hold onto the energy corresponding to a certain potential but want to use the gradient for multiple.
    """
    def __init__(self, potentials, energy_coeffs=None, grad_coeffs=None):
        self.potentials = potentials
        self.energy_coeffs = energy_coeffs or np.ones((len(potentials),))
        self.grad_coeffs = grad_coeffs or np.ones((len(potentials),))
        super(PotentialList, self).__init__()    

    def __getitem__(self, index):
        return self.potentials[index]

    def compute_energy(self, X, **state):
        E = 0.0
        for coeff, potential in zip(self.energy_coeffs, self.potentials):
            E += coeff * potential.compute_energy(X, **state)
        return E

    def compute_gradient(self, X, **state):
        E = 0.0
        G = np.zeros_like(X)
        for e_coeff, g_coeff, potential in zip(self.energy_coeffs, self.grad_coeffs, self.potentials):
            Ecurr, Gcurr = potential.compute_gradient(X, **state)
            E += e_coeff * Ecurr
            G += g_coeff * Gcurr
        return E, G


class SoftHalfSpace(Potential):
    """
    Exerts linearly growing forces along the negative of the normal if the
        particle position minus the offset has a positive projection along the
        normal.

    Args:
        normal ({ndim,} ndarray): vector normal to the plane defining the halfspace
        magnitude (float or {nparticle,} ndarray): Magnitude of applied force
        offset ({ndim,} ndarray): vector of offset from origin to any point in hyperplane defining the halfspace
    """
    def __init__(self,
            normal,
            magnitude=1.0,
            offset=0.0,
            ):
        self.normal = np.reshape(normal / np.linalg.norm(normal), (1, -1))
        self.offset = np.reshape(offset, (1, -1))
        self.magnitude = magnitude
        super(SoftHalfSpace, self).__init__()    

    def compute_energy_per_particle(self, X, **state):
        X_offset = X - self.offset
        X_dot_normal = np.reshape(np.inner(X_offset, self.normal), (-1, 1))
        violation = np.maximum(0.0, X_dot_normal)
        pe = 0.5 * self.magnitude * violation**2
        return pe

    def compute_energy(self, X, **state):
        pe = self.compute_energy_per_particle(X, **state)
        return np.sum(pe)

    def compute_gradient(self, X, **state):
        X_offset = X - self.offset
        X_dot_normal = np.reshape(np.inner(X_offset, self.normal), (-1, 1))
        violation = np.maximum(0.0, X_dot_normal)
        pe = 0.5 * self.magnitude * violation**2
        grad = self.normal * self.magnitude * violation
        pe = np.sum(pe)
        return pe, grad


class SoftCube(Potential):
    """
    Exerts linearly growing forces along each coordinate that is beyond the side length of the cube centered at the origin

    Args:
        bound (float): Side length of cube to use divided by 2; if abs(x[0, 0]) is greater than bound, then the force is exerted
        magnitude (float or {nparticle,} ndarray): Magnitude of applied force
        offset ({ndim,} ndarray): coordinates of center of cube
    """
    def __init__(
            self,
            bound,
            magnitude=1.0,
            offset=0.0,
            ):
        self.bound = bound
        self.offset = np.reshape(offset, (1, -1))
        self.magnitude = magnitude
        super(SoftCube, self).__init__()    

    def compute_energy_per_particle(self, X, **state):
        diff = np.abs(X - self.offset) - self.bound
        pe = np.where(diff > 0.0, 0.5 * self.magnitude * diff**2, 0.0)
        pe = np.sum(pe, axis=1)
        return pe

    def compute_energy(self, X, **state):
        pe = self.compute_energy_per_particle(X, **state)
        return np.sum(pe)

    def compute_gradient(self, X, **state):
        X_offset = X - self.offset
        diff = np.abs(X_offset) - self.bound
        pe = np.where(diff > 0.0, 0.5 * self.magnitude * diff**2, 0.0)
        grad = np.where(np.abs(X_offset) > self.bound, np.sign(X_offset) * self.magnitude * diff, 0.0)
        pe = np.sum(pe)
        return pe, grad


class SoftSphere(Potential):
    r"""
    Exerts linearly growing force along vector of particles displacement from sphere's center

    .. math:: V(x_i) = 0.5 * k * \mathrm{max}(0, ||x_i - x_0|| - r) ** 2

    Args:
        radius (float): radius of sphere to use; if norm(x[0, :]) is greater than radius, then the force is exerted
        magnitude (float or {nparticle,} ndarray): Magnitude of applied force
        offset ({ndim,} ndarray): origin of spherical potential
    """

    def __init__(
            self,
            radius,
            magnitude=1.0,
            offset=0.0,
            ):
        self.radius = radius
        self.offset = np.reshape(offset, (1, -1))
        self.magnitude = magnitude
        super(SoftSphere, self).__init__()    

    def compute_energy_per_particle(self, X, **state):
        X_offset = X - self.offset
        dists = np.linalg.norm(X_offset, axis=1).reshape(-1, 1)
        pe = 0.5 * self.magnitude * np.maximum(0.0, dists - self.radius)**2
        return pe

    def compute_energy(self, X, **state):
        pe = self.compute_energy_per_particle(X, **state)
        return np.sum(pe)

    def compute_gradient(self, X, **state):
        X_offset = X - self.offset
        dists = np.linalg.norm(X_offset, axis=1).reshape(-1, 1)
        violation = np.maximum(0.0, dists - self.radius)
        pe = 0.5 * self.magnitude * violation**2
        grad = X_offset / dists * self.magnitude * violation
        pe = np.sum(pe)
        return pe, grad


class InteratomicSpring(Potential):
    """
    Soft equivalent of the PositionDriver Update, places harmonic well about
    around the selected atoms bond length so that they move toward or away from
    another, but still allows them to vibrate.

    Args:
        atom_ind1 (int): atom index of first atom to pull toward one another
        atom_ind2 (int): atom index of second atom to pull toward one another
        dist_stop (float): distance at which to stop pulling the atoms together
        magnitude (float): Magnitude of applied force
        interpolate (bool): Linearly move the wells based on time_frac if Update is TimeDependent
    """
    def __init__(self,
            atom_ind1,
            atom_ind2,
            dist_stop,
            magnitude,
            interpolate=False,
            ):
        self.atom_ind1 = atom_ind1
        self.atom_ind2 = atom_ind2
        self.dist_start = None
        self.dist_stop = dist_stop
        self.magnitude = magnitude
        self.interpolate = interpolate
        self.time_frac = 1.0
        super(InteratomicSpring, self).__init__()    

    def compute_energy(self, X, **state):
        if self.interpolate:
            # Restart movement cycle
            if state['time_frac'] <= self.time_frac:
                # Compute initial distance
                self.dist_start = np.linalg.norm(X[self.atom_ind1, :] - X[self.atom_ind2, :])
            self.time_frac = state['time_frac']
            des_dist = (1.0 - self.time_frac) * self.dist_start + self.time_frac * self.dist_stop
        else:
            des_dist = self.dist_stop

        vec = X[self.atom_ind2, :] - X[self.atom_ind1, :]
        curr_dist = np.linalg.norm(vec)
        diff = curr_dist - des_dist
        pe = 0.5 * self.magnitude * diff**2
        return pe

    def compute_gradient(self, X, **state):
        if self.interpolate:
            # Restart movement cycle
            if state['time_frac'] <= self.time_frac:
                # Compute initial distance
                self.dist_start = np.linalg.norm(X[self.atom_ind1, :] - X[self.atom_ind2, :])
            self.time_frac = state['time_frac']
            des_dist = (1.0 - self.time_frac) * self.dist_start + self.time_frac * self.dist_stop
        else:
            des_dist = self.dist_stop

        vec = X[self.atom_ind2, :] - X[self.atom_ind1, :]
        curr_dist = np.linalg.norm(vec)
        vec /= curr_dist
        diff = curr_dist - des_dist
        pe = 0.5 * self.magnitude * diff**2
        grad = np.zeros_like(X)
        grad[self.atom_ind1, :] = -self.magnitude * diff * vec
        grad[self.atom_ind2, :] =  self.magnitude * diff * vec
        return pe, grad

class InteratomicLinear(Potential):
    """
    Linear potential V(r) = kr between two atoms. Useful for force modified PES given a negative magnitude.

    Args:
        atom_ind1 (int): atom index of first atom to pull toward one another
        atom_ind2 (int): atom index of second atom to pull toward one another
        magnitude (float): Magnitude of applied force
    """
    def __init__(self,
            atom_ind1,
            atom_ind2,
            magnitude,
            ):
        self.atom_ind1 = atom_ind1
        self.atom_ind2 = atom_ind2
        self.magnitude = magnitude
        super(InteratomicLinear, self).__init__()    

    def compute_energy(self, X, **state):
        vec = X[self.atom_ind2, :] - X[self.atom_ind1, :]
        curr_dist = np.linalg.norm(vec)
        pe = self.magnitude * curr_dist
        return pe

    def compute_gradient(self, X, **state):
        vec = X[self.atom_ind2, :] - X[self.atom_ind1, :]
        curr_dist = np.linalg.norm(vec)
        pe = self.magnitude * curr_dist
        grad = np.zeros_like(X)
        grad[self.atom_ind1, :] = -self.magnitude * vec / curr_dist
        grad[self.atom_ind2, :] =  self.magnitude * vec / curr_dist
        return pe, grad


class ClassicalCoulomb(Potential):
    """
    Pairwise additive potential of q1q2/r, charges will not be updated from initial values

    Args:
        q ({nparticle,} ndarray): charges of each particle
        magnitude (float or {nparticle, nparticle} ndarray): magnitudes to scale the energy/gradient contributions
    """
    def __init__(self, q, magnitude):
        self.q = np.reshape(q, (-1, 1))
        self.q2 = self.q * np.transpose(self.q)
        self.magnitude = magnitude
        super(ClassicalCoulomb, self).__init__()    

    def compute_energy_per_particle(self, X, **state):
        dists = utils.pairwise_dist(X)
        dists = np.where(dists <= 1E-12, np.inf, dists)
        pairwise_energy = self.magnitude * self.q2 / dists
        return 0.5 * np.sum(pairwise_energy, axis=1)

    def compute_energy(self, X, **state):
        return np.sum(self.compute_energy_per_particle(X, **state))

    def compute_gradient(self, X, **state):
        #return (self.compute_energy(X), utils.numerical_gradient(X, self.compute_energy))
        nparticle = X.shape[0]
        xyz_view = X[:, :, None]
        xyz_tran = np.transpose(xyz_view, (2, 1, 0))
        dist_vec = xyz_view - xyz_tran
        dists = np.linalg.norm(dist_vec, axis=1)
        dists = np.where(dists <= 1E-12, np.inf, dists)
        inv_dists = 1.0/dists
        pairwise_energy = self.magnitude * self.q2 * inv_dists
        pe_particle = 0.5 * np.sum(pairwise_energy, axis=1)
        pe_tot = np.sum(pe_particle)

        grad_mag = np.reshape(pairwise_energy * inv_dists * inv_dists, (nparticle, 1, nparticle))
        grad = -np.sum(dist_vec * grad_mag, axis=2)
        return pe_tot, grad

class Harmonic(Potential):
    r"""
    Harmonic force given as Taylor expansion of energy.

    .. math:: E(X) = E(X0) + (X-X0)^T \nabla E(X0) + 1/2 (X-X0)^T \nabla^2 E(X0) (X-X0)

    Args:
        X0 ({nparticle, ndim} ndarray): coordinates at which expansion is made
        E0 (float): energy at X0
        grad ({nparticle, ndim} ndarray): gradient at X0
        hessian ({nparticle*ndim, nparticle*ndim} ndarray): hessian at X0
    """
    def __init__(self, X0, E0=0.0, grad=None, hessian=None):
        self.X0 = X0
        self.E0 = E0
        if grad is None:
            self.grad = np.zeros_like(X0)
        else:
            self.grad = grad
        if hessian is None:
            self.hessian = np.zeros_like(np.outer(X0, X0))
        else:
            self.hessian = hessian
        super(Harmonic, self).__init__()    

    def compute_energy(self, X, **state):
        dX = np.reshape(X-self.X0, (-1, 1))
        E = self.E0 + np.inner(self.grad.reshape((-1, 1)), dX) + 0.5 * np.dot(np.transpose(dX), np.dot(self.hessian, dX))
        return E

    def compute_gradient(self, X, **state):
        dX = np.reshape(X-self.X0, (-1, 1))
        hp = np.dot(self.hessian, dX)
        E = self.E0 + np.sum(self.grad.reshape((-1, 1)) * dX) + 0.5 * np.sum(dX * hp)
        G = self.grad + np.reshape(hp, np.shape(X))
        return E, G
        

class Morse(Potential):
    r"""
    Morse inter particle potential function.

    Can be given a single parameters to treat all pairs, or matrix of parameters
    for unique pairwise interactions.

    .. math:: E_{ij} = D_{ij} * ((1 - \exp(-a_{ij} * (r_{ij} - r_{ij}^{eq})))^2 - 1)

    Args:
        D (float or {nparticle, nparticle} ndarray): dissociation energy parameter
        a (float or {nparticle, nparticle} ndarray): exponential constant / width of well
        r_equil (float or {nparticle, nparticle} ndarray): equilibrium distance
    """
    def __init__(self, D, a, r_equil):
        self.D = D
        self.a = a
        self.r_equil = r_equil
        super(Morse, self).__init__()    

    def compute_energy_per_particle(self, X, **state):
        dists = utils.pairwise_dist(X)
        pe_contrib = self.D * ((1.0 - np.exp(-self.a * (dists - self.r_equil)))**2 - 1.0)
        np.fill_diagonal(pe_contrib, 0.0)
        pe_particle = 0.5 * np.sum(pe_contrib, axis=1)
        return pe_particle

    def compute_energy(self, X, **state):
        pe_particle = self.compute_energy_per_particle(X, **state)
        pe_tot = np.sum(pe_particle)
        return pe_tot

    def compute_gradient(self, X, **state):
        nparticle = X.shape[0]
        xyz_view = X[:, :, None]
        xyz_tran = np.transpose(xyz_view, (2, 1, 0))
        dist_vec = xyz_view - xyz_tran
        dists = np.linalg.norm(dist_vec, axis=1)
        with np.errstate(divide='ignore'):
            inv_dist = np.reshape(np.where(dists > 0.0, 1.0/dists, 0.0), (nparticle, 1, nparticle))
        exp_term = np.exp( - self.a * (dists - self.r_equil))
        pe_contrib = self.D * ((1.0 - exp_term)**2 - 1.0)
        np.fill_diagonal(pe_contrib, 0.0)
        pe_particle = 0.5 * np.sum(pe_contrib, axis=1)
        pe_tot = np.sum(pe_particle)
        grad_mag = np.reshape(self.D * self.a * ( exp_term - exp_term**2 ), (nparticle, 1, nparticle))
        grad = 2.0 * np.sum(dist_vec * inv_dist * grad_mag, axis=2)
        return pe_tot, grad


class Kepler(Potential):
    r"""
    Planetary model with inverse distance potential from the origin.

    .. math:: E = k / ||r||

    Args:
        magnitude (float): magnitude of applied force
    """
    def __init__(self, magnitude):
        self.magnitude = magnitude
        super(Kepler, self).__init__()    

    def compute_energy_per_particle(self, X, **state):
        return -self.magnitude / np.linalg.norm(X, axis=-1)
        
    def compute_energy(self, X, **state):
        return np.sum(self.compute_energy_per_particle(X, **state))

    def compute_gradient(self, X, **state):
        V = -self.magnitude / np.linalg.norm(X, axis=-1)
        G = self.magnitude * X / np.linalg.norm(X, axis=-1, keepdims=True)**3
        return V, G

    @staticmethod
    def init_cond(eccentricity):
        q = np.array([[1.0 - eccentricity, 0.0]])
        p = np.array([[0.0, np.sqrt((1.0 + eccentricity) / (1.0 - eccentricity))]])
        return q, p


class Rosenbrock(Potential):
    r"""
    Classic model for non-convex optimization functions.
    
    .. math:: f(x,y) = (a-x)^2 + b(y - x^2)^2

    References:
        https://en.wikipedia.org/wiki/Rosenbrock_function

    Args:
        a (float): a in equation
        b (float): b in equation
    """
    def __init__(
            self,
            a=1.0,
            b=100.0,
            ):
        self.a = a
        self.b = b
        super(Rosenbrock, self).__init__()    

    def compute_energy_per_particle(self, X, **state):
        if X.shape[1] != 2:
            raise ValueError("Shape for Rosenbrock function must be N x 2")
        pe = (self.a - X[:, 0])**2 + self.b*(X[:, 1] - X[:, 0]**2)**2
        return pe

    def compute_energy(self, X, **state):
        return np.sum(self.compute_energy_per_particle(X, **state))

    def compute_gradient(self, X, **state):
        if X.shape[1] != 2:
            raise ValueError("Shape for Rosenbrock function must be N x 2")
        grad = np.empty_like(X)
        ymx2 = X[:, 1] - X[:, 0]**2
        pe = (self.a - X[:, 0])**2 + self.b*ymx2**2
        grad[:, 0] = 2.0*(X[:, 0] - self.a) - 4.0*self.b*X[:, 0]*ymx2
        grad[:, 1] = 2.0*self.b*ymx2
        return pe, grad


class HeinonHeiles(Potential):
    r"""
    Chaotic system model

    .. math:: V(x, y) = 1/2 (x^2 + y^2) + \alpha (x^2 y - y^3 / 3)

    Args:
        alpha (float): as in equation
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        super(HeinonHeiles, self).__init__()    

    def compute_energy_per_particle(self, X, **state):
        if X.shape[1] != 2:
            raise ValueError("Shape for Rosenbrock function must be N x 2")
        X02 = X[:, 0]**2 
        pe = 0.5 * (X02 + X[:, 1]**2) + self.alpha * (X02 * X[:, 1] - X[:, 1]**3)

    def compute_energy(self, X, **state):
        return np.sum(self.compute_energy_per_particle(self, X, **state))

    def compute_gradient(self, X, **state):
        if X.shape[1] != 2:
            raise ValueError("Shape for Rosenbrock function must be N x 2")
        X02 = X[:, 0]**2
        pe = np.sum(0.5 * (X02 + X[:, 1]**2) + self.alpha * (X02 * X[:, 1] - X[:, 1]**3))
        G = np.zeros_like(X)
        G[:, 0] = X[:, 0] + 2.0 * self.alpha * X[:, 0] * X[:, 1]
        G[:, 1] = X[:, 1] + self.alpha * (X02 - X[:, 1]**2)
        

class MullerBrown(Potential):
    r"""
    Model potential often used for transition state search testing.

    .. math:: E = \sum_i A_i \exp(a_i * (x - x_i^0)^2 + b_i(x - x_i^0)(y - y_0) + c_i(y - y_i^0)^2)

    Args:
        A ({1, nterms} ndarray):
        a ({1, nterms} ndarray):
        b ({1, nterms} ndarray):
        c ({1, nterms} ndarray):
        x0 ({1, nterms} ndarray):
        y0 ({1, nterms} ndarray):
    """
    def __init__(self,
        A=None,
        a=None,
        b=None,
        c=None,
        x0=None,
        y0=None,
        ):

        self.A =  A  or np.array([[-200.0, -100.0, -170.0,  15.0]])
        self.a =  a  or np.array([[  -1.0,   -1.0,   -6.5,   0.7]])
        self.b =  b  or np.array([[   0.0,    0.0,   11.0,   0.6]])
        self.c =  c  or np.array([[ -10.0,  -10.0,   -6.5,   0.7]])
        self.x0 = x0 or np.array([[   1.0,    0.0,   -0.5,  -1.0]])
        self.y0 = y0 or np.array([[   0.0,    0.5,    1.5,   1.0]])
        super(MullerBrown, self).__init__()    

    def compute_energy_per_particle(self, X, **state):
        x = X[:, 0:1] # Maintain (N, 1) shape
        y = X[:, 1:2]
        V = np.sum(self.A * np.exp(
                         self.a * (x - self.x0)**2
                       + self.b * (x - self.x0) * (y - self.y0)
                       + self.c * (y - self.y0)**2
                   ), axis=1)
        return V

    def compute_energy(self, X, **state):
        return np.sum(self.compute_energy_per_particle(X, **state))

    def compute_gradient(self, X, **state):
        x = X[:, 0:1] # Maintain (N, 1) shape
        y = X[:, 1:2]
        Vcomp = self.A * np.exp(
                         self.a * (x - self.x0)**2
                       + self.b * (x - self.x0) * (y - self.y0)
                       + self.c * (y - self.y0)**2
                   )
        V =  np.sum(Vcomp)
        G = np.zeros_like(X)
        G[:, 0] = np.sum(Vcomp * (2.0 * self.a * (x - self.x0) + self.b * (y - self.y0)), axis=1)
        G[:, 1] = np.sum(Vcomp * (self.b * (self.x - x0) + 2.0 * self.c * (y - self.y0)), axis=1)
        return V, G

class Prinz(Potential):
    r"""
    1D model potential for Markov state modeling, should only be used in [-1, 1] for numerical stability

    .. math:: V(x) = A * (x^b + \sum_i c_i \exp(d_i * (x + e_i)^2)

    Args:
        A (float):
        b (float):
        c ({1, nterms} ndarray):
        d ({1, nterms} ndarray):
        e ({1, nterms} ndarray):
    """
    def __init__(self, A=None, b=None, c=None, d=None, e=None):
        self.A = A or 4.0
        self.b = b or 8.0
        self.c = c or np.array([[0.8, 0.2, 0.5]])
        self.d = d or np.array([[-80, -80, -40]])
        self.e = e or np.array([[0.0, -0.5, 0.5]])
        super(Prinz, self).__init__()    

    def compute_energy_per_particle(self, X, **state):
        V = self.A * (X[:, 0] ** self.b + np.sum(self.c * np.exp(self.d * (X - self.e)**2), axis=-1))
        return V

    def compute_energy(self, X, **state):
        return np.sum(self.compute_energy_per_particle(X, **state))

    def compute_gradient(self, X, **state):
        V = self.compute_energy(X, **state)
        G = self.A * (self.b * X[:, 0] ** (self.b - 1.0) + np.sum(self.c * np.exp(self.d * (X - self.e) ** 2) * self.d * 2.0 * (X - self.e), axis=-1))
        G = np.reshape(G, np.shape(X))
        return V, G
