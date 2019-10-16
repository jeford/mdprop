import numpy as np
from scipy.linalg import expm, cholesky
import warnings

from . import init, units, utils

class Update(object):
    """
    Abstract base class describing single updates to position or velocity (or 
    other members of the state dict), a list of these is used to construct 
    an integrator; each update is similar to a single term in a Liouvillian

    The __init__ method of each Update object should construct self.params dict
    and self.requirements set that specifies the object.

    Each update object contains a params dict which governs how to conduct updates,
    and should not change. Most __init__ functions are written to take in natural
    units, so that they can easily be applied to any system desired.
    """
    h5_keys = []
    h5_shapes = []
    h5_types = []

    def __init__(self):
        raise NotImplementedError

    def __str__(self):
        return self.params.get('name', self.__class__.__name__)

    def __repr__(self):
        st = self.__class__.__name__ \
                + "\n\nParams:\n" \
                + str(self.params) \
                + "\n\nRequirements:\n" \
                + str(self.requirements)
        return st

    def update(self, step_length, state):
        """
        update functions are called in sequence by the ExplicitIntegrator

        Args:
            step_length: length of step taken, generally given by integrator coeff*dt
            state: input dict containing keys/vals needed by update

        Returns:
            state_update: dict of updates to the current state
        """
        raise NotImplementedError

    def get_h5_data(self, **kwargs):
        h5_shapes_trans = []
        for shape in self.h5_shapes:
            curr_shape = []
            for element in shape:
                if isinstance(element, str):
                    curr_shape.append(kwargs[element])
                else:
                    curr_shape.append(element)
            h5_shapes_trans.append(curr_shape)
        return self.h5_keys, h5_shapes_trans, self.h5_types

    @staticmethod
    def get_list_h5_data(hooks, **kwargs):
        """
        Given a list of updates, returns H5 tuple containing all uniquely named data
        Dynamic shapes such as 'natom' can be specified with **kwargs
        """
        h5_keys = []
        h5_shapes = []
        h5_types = []
        h5_key_set = set([])
        for h in hooks:
            keys, shapes, types = h.get_h5_data(**kwargs)
            for k, s, t in zip(keys, shapes, types):
                if k not in h5_key_set:
                    h5_keys.append(k)
                    h5_shapes.append(s)
                    h5_types.append(t)
                    h5_key_set = h5_key_set.union([k])
        return h5_keys, h5_shapes, h5_types

class PositionUpdate(Update):
    """
    Update position X based on velocity V

    Params:
        recenter (bool): True to remove COM / COM translation and rotation prior to position update
        masses ({nparticle,} ndarray): masses, only required if recenter is True
        coord_key (str): key to positions in state
        vel_key (str): key to velocities in state
        name (str): name of update
    """
    h5_keys = ['X']
    h5_shapes = [('natom', 3)]
    h5_types = ['f']

    def __init__(self, recenter=False, masses=None, coord_key='X', vel_key='V', name="position"):
        self.params = {
            'recenter' : recenter,
            'coord_key' : coord_key,
            'vel_key' : vel_key,
            'name' : name,
        }
        self.requirements = set([coord_key, vel_key])
        if recenter:
            if masses is None:
                raise ValueError("Must give masses to enforce recentering in PositionUpdate")
            else:
                self.params['masses'] = np.reshape(masses, (-1, 1))
        self.dX = None
        self.X = None
        self.V = None

    def update(self, step_length, state):
        if self.params['recenter']:
            self.X, self.V = init.initialize_centered(state[self.params['coord_key']], state[self.params['vel_key']], self.params['masses'])
        else:
            self.X = state[self.params['coord_key']]
            self.V = state[self.params['vel_key']]
        self.dX = step_length * self.V
        self.X = self.X + self.dX
        self.state_update = {
            self.params['coord_key']: self.X,
            }
        if self.params['recenter']:
            self.state_update[self.params['vel_key']] = self.V
        return self.state_update

class VelocityUpdate(Update):
    """
    Update velocities V based on potential.force given

    Params:
        potential (Potential object): potential.force must take in state['X'], outputing (potential_energy, force_vector)
        masses ({nparticle,} ndarray): masses for each particle
        coord_key (str): key to positions in state
        vel_key (str): key to velocities in state
        name (str): name of update, used for naming the energy contribution (default 'potential')
    """
    h5_keys = ['V']
    h5_shapes = [('natom', 3)]
    h5_types = ['f']

    def __init__(self, potential, masses, coord_key='X', vel_key='V', name="potential"):
        self.params = {
            'potential': potential,
            'masses': np.reshape(masses, (-1, 1)),
            'coord_key' : coord_key,
            'vel_key' : vel_key,
            'name': name,
            }
        self.requirements = set([coord_key, vel_key])
        self.E = None
        self.F = None
        self.dV = None
        self.V = None
        self.state_update = {}
        
    def update(self, step_length, state):
        self.E, self.F = self.params['potential'].force(state[self.params['coord_key']])
        self.dV = step_length * self.F / self.params['masses']
        self.V = state[self.params['vel_key']] + self.dV
        KE = utils.compute_kinetic_energy(self.V, self.params['masses'])
        self.state_update = {
            self.params['vel_key'] : self.V,  
            self.params['name'] + '_energy': self.E,
            self.params['name'] + '_gradient': -self.F,
            'kinetic_energy': KE,
            }
        return self.state_update

class GeneralVelocityUpdate(Update):
    """
    Update velocities V based on force function given.  
    This object is subtly different from VelocityUpdate in that the force 
    function can use any object in the state dict, but the forces still 
    propagate the velocities the same way.

    Params:
        potential (Potential object): potential.force must take in state['X'], outputing (potential_energy, force_vector)
        masses ({nparticle,} ndarray): masses for each particle
        recalculate (bool): True to always recalculate force
        coord_key (str): key to positions in state
        vel_key (str): key to velocities in state
        name (str): name of update, used for naming the energy contribution (default 'potential')
    """
    h5_keys = ['V']
    h5_shapes = [('natom', 3)]
    h5_types = ['f']

    def __init__(self, 
            potential, 
            masses,
            recalculate=False,
            vel_key='V',
            name="potential", 
            ):
        self.params = {
            'potential': potential,
            'masses': np.reshape(masses, (-1, 1)),
            'recalculate': recalculate,
            'vel_key' : vel_key,
            'name': name,
            }
        self.requirements = set([vel_key])
        self.E = None
        self.F = None
        self.dV = None
        self.V = None
        self.state_update = {}
        
    def update(self, step_length, state):
        if self.params['recalculate']:
            self.E, self.F = self.params['potential'].compute_force(**state)
        else:
            self.E, self.F = self.params['potential'].force(**state)
        self.dV = step_length * self.F / self.params['masses']
        self.V = state[self.params['vel_key']] + self.dV
        KE = utils.compute_kinetic_energy(self.V, self.params['masses'])
        self.state_update = {
            self.params['vel_key']: self.V,  
            self.params['name'] + '_energy': self.E,
            self.params['name'] + '_gradient': -self.F,
            'kinetic_energy': KE,
            }
        return self.state_update

class IsokineticVelocityUpdate(Update):
    """
    Update velocities while enforcing an isokinetic distribution.

    Params:
        potential (Potential object): potential.force must take in state['X'], outputing (potential_energy, force_vector)
        masses ({nparticle,} ndarray): masses for each particle
        kT (float): kinetic energy to constrain to
        nhc (bool): True to apply joint isokinetic constraint to velocities and first NHC dofs
        name (str): name of update, used for naming the energy contribution (default 'potential')

    References:
        The Journal of Chemical Physics 118, 2510 (2003); doi: 10.1063/1.1534582
        https://www.tandfonline.com/doi/abs/10.1080/00268976.2013.844369
    """
    h5_keys = ['V']
    h5_shapes = [('natom', 3)]
    h5_types = ['f']

    def __init__(self, potential, masses, kT, nhc=False, name="potential"):
        self.params = {
            'potential': potential,
            'masses': np.reshape(masses, (-1, 1)),
            'kT': kT,
            'name': name,
            }
        self.requirements = set(['X', 'V'])
        self.nhc = nhc
        if nhc:
            self.requirements.add('aux_velocity_NH')
        self.E = None
        self.F = None
        self.V = None
        self.K = None
        self.lmbd = None
        self.state_update = {}

    def update(self, step_length, state):
        self.E, self.F = self.params['potential'].force(state['X'])

        if self.nhc:
            self.L = np.shape(state['aux_velocity_NH'])[1]
            self.lmbd = self.lmbd or self.L*self.params['kT']
            self.a = self.F * state['V'] / self.lmbd
            self.b = self.F**2 / self.params['masses'] / self.lmbd
        else:
            self.K = self.K or (np.size(state['V']) - 1) * 0.5 * self.params['kT']
            self.a = 0.5 / self.K * np.sum(state['V'] * self.F)
            self.b = 0.5 / self.K * np.sum(self.F**2 / self.params['masses'])
        sqb = np.sqrt(self.b)
        arg = step_length * sqb
        with np.errstate(divide='ignore', invalid='ignore'): # Hide all the divide by zero warnings
            self.s = np.where(
                arg > 0.00001, 
                self.a / self.b * (np.cosh(arg) - 1.0) + 1.0 / sqb * np.sinh(arg),
                ((((self.b*self.a/24.0)*step_length + self.b/6.0)*step_length + 0.5*self.a)*step_length + 1.0)*step_length
                )
            self.sdot = np.where(
                arg > 0.00001, 
                self.a / sqb * np.sinh(arg) + np.cosh(arg),
                (((self.b*self.a/6.0)*step_length + 0.5*self.b)*step_length + self.a)*step_length + 1.0
                )
        self.V = (state['V'] + self.s * self.F / self.params['masses']) / self.sdot
        KE = utils.compute_kinetic_energy(self.V, self.params['masses'])
        self.state_update = {
            'V': self.V,  
            self.params['name'] + '_energy': self.E,
            self.params['name'] + '_gradient': -self.F,
            'kinetic_energy': KE,
            }
        if self.nhc:
            self.aux_v = np.copy(state['aux_velocity_NH'])
            self.aux_v[0] = self.aux_v[0] / self.sdot
            self.state_update['aux_velocity_NH'] = self.aux_v
        return self.state_update

class TimeDependent(Update):
    """
    Update state based on update given, if the condition
    simulation_time % time_modulus >= time_start && simulation_time % time_modulus < time_stop
    
    Unlike other updates, this update wraps an existing update and makes it time dependent

    Params:
        update (Update): Update to make time dependent 
        time_start (float): scalar to add to remainder (see above) (default 0.0) 
        time_stop (float): scalar to add to remainder (see above) (default np.inf)
        time_modulus (float): modulus parameter of simulation time to determine if to apply Update (see above) (default None)
        scale_step (bool): True to scale the step length by 
            (1.0 - s(time_frac)) * scale_start + s(time_frac) * scale_stop 
            allows adiabatically turning updates on or off, i.e. for adiabatic switching
        scale_start (float): see scale_step (default 0.0)
        scale_stop (float): see scale_step (default 1.0)
        switching_func (function): switching function with range and domain [0, 1], see scale_step
        null_return (dict): returned in the case that the update is not currently turned on (default {})
        name_prefix (str): Renames update with prefix (default 'time_dependent_')
    """
    def __init__(self, 
            update, 
            time_start=0.0, 
            time_stop=np.inf, 
            time_modulus=None,
            scale_step=False,
            scale_start=0.0,
            scale_stop=1.0,
            switching_func=utils.smootherstep,
            null_return={}, 
            name_prefix="timedependent_",
            ):
        self.params = update.params.copy()
        self.params.update({
            'update': update,
            'time_start': time_start,
            'time_stop': time_stop,
            'time_modulus': time_modulus,
            'scale_step': scale_step,
            'scale_start': scale_start,
            'scale_stop': scale_stop,
            'switching_func': switching_func,
            'null_return': null_return,
            'name': name_prefix + update.params['name'],
            })
        self.requirements = set(list(update.requirements) + ['simulation_time'])
        self.h5_keys = update.h5_keys
        self.h5_shapes = update.h5_shapes
        self.h5_types = update.h5_types
        self.curr_mod = None
        self.curr_frac = None
        self.curr_scale = None
        self.state_update = {}

    def update(self, step_length, state):
        if self.params['time_modulus'] is not None:
            self.curr_mod = state['simulation_time'] % self.params['time_modulus']
        else:
            self.curr_mod = state['simulation_time']

        self.curr_frac = (self.curr_mod - self.params['time_start']) / (self.params['time_stop'] - self.params['time_start'])
        self.curr_frac = np.clip(self.curr_frac, 0.0, 1.0)
        state['time_frac'] = self.curr_frac

        if self.params['scale_step']:
            self.curr_scale = (1.0 - self.params['switching_func'](self.self.curr_frac)) * self.params['scale_start'] + self.params['switching_func'](self.curr_frac) * self.params['scale_stop']
        else:
            self.curr_scale = 1.0

        cond1 = self.curr_mod >= self.params['time_start']
        cond2 = self.curr_mod < self.params['time_stop']
        if cond1 and cond2:
            self.state_update = self.params['update'].update(self.curr_scale * step_length, state)
        elif self.params['scale_step'] and np.abs(self.curr_scale) > 1E-8 and self.params['time_modulus'] is None: 
            self.state_update = self.params['update'].update(self.curr_scale * step_length, state)
        else:
            self.state_update = self.params['null_return']
        return self.state_update

class Langevin(Update):
    """
    Update velocities using Bussi-Parrinello Langevin integrator

    Params:
        masses ({nparticle,} ndarray): masses for each particle
        kT (float): temperature in energy units
        damptime (float): damping time
        rescale (bool): True to project the new momentum vector along the old
        name (str): name of update (default 'langevin')

    References:
        doi:10.1103/PhysRevE.75.056707
        doi:10.1063/1.5029833
    """
    h5_keys = ['V']
    h5_shapes = [('natom', 3)]
    h5_types = ['f']

    def __init__(self, 
            masses,
            kT, 
            damptime,
            rescale=False,
            vel_key='V',
            name='langevin',
            ):
        self.params = {
            'masses': np.reshape(masses, (-1, 1)),
            'kT': kT,
            'damptime': damptime,
            'gamma': 1.0 / damptime,
            'rescale': rescale,
            'vel_key' : vel_key,
            'name' : name,
            }
        self.requirements = set(['V'])
        self.step_length = None
        self.c1 = None
        self.c2 = None
        self.dV = None
        self.V = None
        self.state_update = {}

    def update(self, step_length, state):
        if self.step_length != step_length:
            self.c1 = np.exp(-self.params['gamma'] * abs(step_length))
            self.c2 = np.sqrt((1.0 - self.c1**2) * self.params['kT'] / self.params['masses'])
            self.step_length = step_length

        self.dV = (self.c1 - 1.0) * state[self.params['vel_key']] + self.c2 * np.random.standard_normal(state[self.params['vel_key']].shape)
        self.V = state[self.params['vel_key']] + self.dV
        if self.params['rescale']:
            self.V = np.linalg.norm(self.V, axis=1, keepdims=True) / np.linalg.norm(state[self.params['vel_key']], axis=1, keepdims=True) * state[self.params['vel_key']]
        self.state_update = {
            self.params['vel_key'] : self.V,
            }
        return self.state_update

class AdaptiveLangevin(Update):
    """
    Update velocities using adaptive Langevin integrator

    Params:
        masses ({nparticle,} ndarray): masses for each particle
        kT (float): temperature in energy units
        aux_mass (float): mass to use in for auxiliary degree of freedom corresponding to thermostat frequency
        sigma (float): variance of additional noise (default is sqrt(2kT gamma_0))
        name (str): name of update (default 'langevin')

    References:
        https://epubs.siam.org/doi/pdf/10.1137/15M102318X
        https://aip.scitation.org/doi/10.1063/1.3626941
    """
    h5_keys = ['V']
    h5_shapes = [('natom', 3)]
    h5_types = ['f']

    def __init__(self, 
            masses,
            kT, 
            aux_mass,
            sigma=None,
            name='adaptive_langevin',
            ):
        self.params = {
            'masses': np.reshape(masses, (-1, 1)),
            'kT': kT,
            'aux_mass': aux_mass,
            'name' : name,
            }
        self.requirements = set(['V', 'gamma'])
        self.step_length = None
        self.sigma = sigma
        self.gamma = None
        self.c1 = None
        self.c2 = None
        self.V = None
        self.state_update = {}

    @staticmethod
    def initialize(kT, ndof, tau):
        r"""
        Compute 'optimal damping' parameters given characteristic timescale

        .. math:: 

            \gamma &= 2 / \tau \\
            Q &= N_d k_B T \tau^2 / 2

        Args:
            kT (float): temperature in energy units
            ndof (int): total number of degrees of freedom
            tau (float): characteristic time scale

        Returns:
            (float, float): tuple of initial gamma and auxiliary mass to use
        """
        gamma = 2.0 / tau
        Q = 0.5 * ndof * kT * tau**2
        return gamma, Q

    def update(self, step_length, state):
        if self.sigma is None:
            self.sigma = np.sqrt(2.0 * self.params['kT'] * state['gamma'])
        KE = state.get('kinetic_energy', utils.compute_kinetic_energy(state['V'], self.params['masses']))
        self.gamma = state['gamma'] + 0.5 * step_length / self.params['aux_mass'] * (2.0 * KE - np.size(state['V']) * self.params['kT'])

        self.c1 = np.exp(-self.gamma * abs(step_length))
        self.c2 = np.sqrt((1.0 - self.c1**2) * 0.5 / self.gamma / self.params['masses'])
        self.V = self.c1 * state['V'] + self.sigma * self.c2 * np.random.standard_normal(state['V'].shape)

        KE = utils.compute_kinetic_energy(self.V, self.params['masses'])

        self.gamma += 0.5 * step_length / self.params['aux_mass'] * (2.0 * KE - np.size(state['V']) * self.params['kT'])
        self.state_update = {
            'V': self.V,
            'gamma': self.gamma,
            'kinetic_energy': KE,
            }
        return self.state_update

class ColoredNoise(Update):
    """
    Update velocities using colored noise 

    Params:
        masses ({nparticle,} ndarray): masses for each particle
        kT (float): temperature in energy units
        gamma ({naux+1, naux+1} ndarray): friction coefficient matrix in inverse units of time
        regularization (float): Small number to add to diagonal of gamma to ensure successful cholesky decomposition
        gamma_inf (float): noise at high frequency, used to build gamma if it's not given
        gamma_0 (float): noise at omega_til, used to build gamma if it's not given
        omega_til (float): displacement of exponential distributions from 0, used to build gamma if it's not given
        name (str): name of update (default 'colored_noise')

    References:
        doi:10.1063/1.3518369
    """
    h5_keys = ['V', 'aux_momentum_CN']
    h5_shapes = [('natom', 3), ('naux', 3)]
    h5_types = ['f', 'f']

    def __init__(self, 
            masses,
            kT=300.0 * units.K_TO_AU, 
            gamma=None, 
            gamma_inf=83.33/units.PS_TO_AU, # Using GLE 12fs parameters from ref
            gamma_0=0.01/units.PS_TO_AU, 
            omega_til=300.0/units.PS_TO_AU, 
            regularization=1E-8,
            dim=3,
            name='colored_noise',
            ):

        # Build gamma as in reference
        if gamma is None:
            var = np.sqrt(omega_til * (gamma_inf - gamma_0))
            tmp = 3.0**(0.25)
            gamma = np.array([
                        [gamma_inf, tmp*var, 1.0/tmp * var],
                        [tmp*var, tmp**2 * omega_til, omega_til],
                        [-1.0/tmp * var, -omega_til, 0.0]
                        ])

        gamma = gamma + np.eye(gamma.shape[0]) * regularization

        self.params = {
            # Broadcast masses to match dimension of velocities
            'masses': (np.reshape(masses, (-1, 1)) * np.ones((dim,))).reshape((1, -1)), # (N x 1) -> (1 x 3N)
            'kT': kT * units.K_TO_AU,
            'gamma': gamma,
            'name' : name,
            }
        self.requirements = set(['V', 'aux_momentum_CN'])
        self.step_length = None
        self.C1 = None
        self.C2 = None
        self.dV = None
        self.V = None
        self.state_update = {}

    def update(self, step_length, state):
        if self.step_length != step_length:
            self.C1 = expm(-self.params['gamma'] * abs(step_length))
            self.C1_update = self.C1 - np.eye(self.C1.shape[0]) # Subtract identity to compute \Delta p
            self.C2 = cholesky(np.eye(self.C1.shape[0]) - np.dot(np.transpose(self.C1), self.C1))
            self.step_length = step_length

        # Unroll everything to compute the update as a matrix multiplication
        V_unroll = state['V'].reshape(1, -1) # (N x 3) -> (1 x 3N)
        P_unroll = V_unroll * self.params['masses'] # Elementwise multiplication

        # construct matrix that is (#aux mom per DOF + 1) x (DOF)
        P_tot = np.vstack([P_unroll, state['aux_momentum_CN']]) # (M+1 x 3N)
        friction_contrib = np.dot(self.C1_update, P_tot) # (M+1 x 3N)

        noise = np.dot(self.C2, np.random.standard_normal(P_tot.shape))
        noise_contrib = noise * np.sqrt(self.params['masses'] * self.params['kT']) # The masses are broadcasted here

        update = friction_contrib + noise_contrib
        self.dV = (update[0,:] / self.params['masses']).reshape(-1, state['V'].shape[1])
        self.V = state['V'] + self.dV
        self.dAux = update[1:,:]
        self.Aux = state['aux_momentum_CN'] + self.dAux

        KE = utils.compute_kinetic_energy(self.V, self.params['masses'])
        self.state_update = {
            'V': self.V,
            'aux_momentum_CN': self.Aux,
            'kinetic_energy': KE,
            }
        
        return self.state_update

class NoseHooverNVT(Update):
    """
    Update velocities using massive Nose-Hoover chains
       DOI: 10.1080/00268979600100761

    Params:
        masses ({nparticle,} ndarray): masses for each particle
        kT (float): temperature in energy
        Qs ({chain_length, natom} ndarray): auxiliary masses
        nc (int): number of integration substeps
        name (str): name of update (default 'nosehooverchain')
    """
    h5_keys = ['V', 'aux_position_NH', 'aux_velocity_NH']
    h5_shapes = [('natom', 3), ('naux', 'natom'), ('naux', 'natom')]
    h5_types = ['f', 'f', 'f']
    requirements = set(['V', 'masses', 'aux_position_NH', 'aux_velocity_NH'])

    def __init__(self,
            masses,
            kT,
            Qs,
            nc=5,
            name='nosehooverchain',
            ):
        self.params = {
            'masses': np.reshape(masses, (-1, 1)),
            'kT': kT,
            'Qs': Qs,
            'nc': nc,
            'name': name,
            }
        self.aux_q = None
        self.aux_v = None
        self.aux_a = None
        self.V = None
        w1 = 1.0 / (2.0 - 2.0**(1.0/3.0))
        w3 = w1
        w2 = 1.0 - w1 - w3
        self.ws = np.array([w1, w2, w3])
        self.M = len(Qs)
        self.state_update = {}

    @classmethod
    def build(cls, masses, kT, tau=0.5 * units.PS_TO_AU, chain_length=5, nc=5, dim=3, mass_weight=True):
        """
        Construct the update for molecular system with auxiliary masses based on rules given in reference
        References:
            doi: 10.1021/j100151a052

        Args:
            masses ({nparticle,} ndarray): masses of original dofs
            kT (float): temperature in energy units
            tau (float): 'natural timescale' to set the masses of the NHCs using Q = kT tau^2 (default 500 fs in au)
            chain_length (int): length of NHC per atom (default 5)
            nc (int): number of integration substeps for NHCs (default 5)
            dim (int): dimension of original system (default 3)
            mass_weight (bool): True to multiply Qs by ratio of particle mass / hydrogen mass (default True)

        Returns:
            NHC_update: a constructed NHC thermostat update
        """
        if mass_weight:
            mass_r = np.reshape(masses, (1, -1)) / utils.symbol_to_mass(['H'])[0]
        else:
            mass_r = np.ones((1, np.size(masses)))
        Qs = np.ones((chain_length, np.size(mass_r))) * kT * mass_r * tau **2
        Qs[0] *= dim
        return cls(masses, kT, Qs, nc)

    @staticmethod
    def initialize(kT, Qs):
        """
        Create initial positions and velocities of auxiliary degrees of freedom,
        positions are set to zero and velocities are boltzmann distributed

        Args:
            kT (float): Temperature in energy units
            Qs ({chain_length, natom} ndarray): masses of NHCs
        
        Returns:
            ({chain_length, natom} ndarray, {chain_length, natom} ndarray): tuple of initial auxiliary positions and auxiliary velocities sampled from Boltzmann distribution
        """
        aux_q = np.zeros_like(Qs)
        factor = np.sqrt(kT/Qs)
        aux_v = np.random.normal(scale=factor)
        return aux_q, aux_v

    def compute_nose_kinetic_energy(self, velocities, masses):
        """
        Calculate kinetic energy corresponding to NHC velocities

        Args:
            velocities ({chain_length, natom} ndarray): NHC velocities
            masses ({chain_length, natom} ndarray): NHC masses

        Returns:
            (float): NHC kinetic energy
        """
        return 0.5 * np.sum(masses * velocities**2) 

    def compute_nose_potential_energy(self, coordinates, gkt, gnkt):
        """
        Calculate potential energy corresponding to NHC coordinates

        Args:
            coordinates ({chain_length, natom} ndarray): NHC coordinates
            gkt (float): temperature in energy units
            gnkt (float): dofs per chain * temperature in energy

        Returns:
            (float): NHC potential energy
        """
        return np.sum(coordinates[0] * gnkt) + np.sum(coordinates[1:] * gkt)

    def update(self, step_length, state):
        self.aux_q = np.copy(state['aux_position_NH'])
        self.aux_v = np.copy(state['aux_velocity_NH'])
        # Atomwise KE*2 
        akin = np.sum(state['V']**2 * self.params['masses'], axis=1)
        scale = np.ones_like(akin)
        self.aux_a = np.zeros_like(self.aux_q) 
        self.gnkt = np.shape(state['V'])[-1] * self.params['kT']
        self.gkt = self.params['kT']
        self.aux_a[0] = (akin - self.gnkt) / self.params['Qs'][0]
        self.aux_a[1:] = (self.params['Qs'][:-1] * self.aux_v[:-1]**2 - self.gkt) / self.params['Qs'][1:]

        for k in range(self.params['nc']): # loop of integrations substeps
            for w in self.ws: # loop of steps in Yoshida Suzuki integrator
                # This is sort of hacky due to translation from TeraChem, which
                # was itself translated from DOI: 10.1080/00268979600100761
                # appendix A
                wdts2 = w * step_length / self.params['nc'] 
                wdts4 = wdts2 * 0.5 
                wdts8 = wdts4 * 0.5

                self.aux_v[self.M-1] += self.aux_a[self.M-1] * wdts4
                # Intra chain coupling M to 0
                for Mi in range(self.M-1):
                    aa = np.exp(-wdts8 * self.aux_v[self.M-(Mi+1)])
                    self.aux_v[self.M-1-(Mi+1)] = self.aux_v[self.M-1-(Mi+1)] * aa**2 + wdts4 * aa * self.aux_a[self.M-1-(Mi+1)]

                # Update kinetic energy
                aa = np.exp(-wdts2 * self.aux_v[0])
                scale *= aa
                self.aux_a[0] = (akin * scale**2 - self.gnkt) / self.params['Qs'][0]

                # Update positions
                self.aux_q += wdts2 * self.aux_v

                # Intra chain coupling 0 to M
                for Mi in range(self.M-1):
                    aa = np.exp(-wdts8 * self.aux_v[Mi+1])
                    self.aux_v[Mi] = self.aux_v[Mi] * aa**2 + wdts4 * aa * self.aux_a[Mi]
                    self.aux_a[Mi+1] = (self.params['Qs'][Mi] * self.aux_v[Mi]**2 - self.gkt) / self.params['Qs'][Mi+1]

                self.aux_v[self.M-1] += self.aux_a[self.M-1] * wdts4

        # All this work to rescale velocities
        self.V = state['V'] * np.reshape(scale, (-1, 1))
        self.energy = self.compute_nose_kinetic_energy(self.aux_v, self.params['Qs'])
        self.energy += self.compute_nose_potential_energy(self.aux_q, self.gkt, self.gnkt)
        KE = utils.compute_kinetic_energy(self.V, self.params['masses'])
        self.state_update = {
                'V' : self.V,
                'aux_position_NH': self.aux_q,
                'aux_velocity_NH': self.aux_v,
                'NHC_energy': self.energy,
                'kinetic_energy': KE,
            }
        return self.state_update

class IsokineticNoseHoover(NoseHooverNVT):
    """
    Update velocities using massive Nose-Hoover chains that contain joint isokinetic constraint

    Params:
        masses ({nparticle,} ndarray): masses for each original particles
        kT (float): temperature in energy
        Qs ({2, L, nparticle, dim} ndarray): auxiliary masses
        nc (int): number of integration substeps
        name (str): name of update (default 'nosehooverchain')

    References:
        https://www.tandfonline.com/doi/pdf/10.1080/00268976.2013.844369
        https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.93.150201
        https://aip.scitation.org/doi/pdf/10.1063/1.1534582?class=pdf
    """
    @classmethod
    def build(cls, masses, kT, L=4, tau=0.5 * units.PS_TO_AU, nc=5, dim=3, mass_weight=False):
        """
        Construct the update for molecular system with auxiliary masses based on rules given in
            doi: 10.1021/j100151a052

        Args:
            masses ({nparticle,} ndarray): masses of original dofs
            kT (float): temperature in energy units
            L (int): number of auxiliary dofs per original dof (default 4)
            tau (float): 'natural timescale' to set the masses of the NHCs using Q = kT tau^2 (default 10 fs in au)
            damptime (float): rate of damping for Ornstein-Uhlenbeck/Langevin process applied to last NHC dofs (default 10 fs in au)
            nc (int): number of integration substeps for NHCs (default 5)
            dim (int): dimension of original system (default 3)
            mass_weight (bool): True to multiply Qs by ratio of particle mass / hydrogen mass (default False)

        Returns:
            NHC_update: a constructed Isokinetic NHC thermostat update
        """
        if mass_weight:
            mass_r = np.reshape(masses, (1, 1, -1, 1)) / utils.symbol_to_mass(['H'])[0]
        else:
            mass_r = np.ones((1, 1, np.size(masses), 1))
        Qs = np.ones((2, L, np.size(mass_r), dim)) * kT * mass_r * tau **2
        return cls(masses, kT, Qs, nc)

    def update(self, step_length, state):
        self.aux_v = np.copy(state['aux_velocity_NH'])
        self.V = np.copy(state['V'])
        self.L = float(np.shape(self.params['Qs'])[1])
        self.lmbd = self.L * self.params['kT']

        for k in range(self.params['nc']): # loop of integrations substeps
            for w in self.ws: # loop of steps in Yoshida Suzuki integrator
                # step_length generally already the total \Delta t / 2, making
                # sub_step = w_i * \Delta t / 2 / nc
                sub_step = w * step_length / self.params['nc'] 
                half_sub_step = 0.5 * sub_step

                # Take half substep for vk2
                G = (self.params['Qs'][0] * self.aux_v[0]**2 - self.params['kT']) / self.params['Qs'][1]
                self.aux_v[1] += half_sub_step * G

                # Take substep for v, vk1
                aa = np.exp(-sub_step * self.aux_v[1])
                tt = self.V**2 * self.params['masses'] + self.L / (self.L + 1.0) * np.sum(self.params['Qs'][0]*(self.aux_v[0]**2)*(aa**2), axis=0)
                srtop = np.sqrt(self.lmbd/tt)
                self.V = self.V * srtop
                self.aux_v[0] = self.aux_v[0] * srtop * aa

                # Take half substep for vk2
                G = (self.params['Qs'][0] * self.aux_v[0]**2 - self.params['kT']) / self.params['Qs'][1]
                self.aux_v[1] += half_sub_step * G

        KE = utils.compute_kinetic_energy(self.V, self.params['masses'])
        self.state_update = {
                'V' : self.V,
                'aux_position_NH': self.aux_q,
                'aux_velocity_NH': self.aux_v,
                'kinetic_energy': KE,
            }
        return self.state_update

class NoseHooverLangevin(Update):
    """
    Update the last auxiliary velocity in each NHC with Langevin thermostatting

    Params:
        kT (float): temperature in energy
        Qs ({2, L, nparticle, dim} ndarray): auxiliary masses
        damptime (float): rate of damping for Ornstein-Uhlenbeck/Langevin process applied to last NHC dofs (default 10 fs in au)
        name (str): name of update (default 'nosehooverlangevin')

    References:
        https://www.tandfonline.com/doi/pdf/10.1080/00268976.2013.844369
        https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.93.150201
        https://aip.scitation.org/doi/pdf/10.1063/1.1534582?class=pdf
    """
    def __init__(self,
            kT,
            Qs,
            damptime,
            name='nosehooverlangevin',
            ):
        self.params = {
            'kT': kT,
            'Qs': Qs,
            'damptime': damptime,
            'gamma': 1.0 / damptime,
            'name': name,
            }
        self.requirements = set(['aux_velocity_NH'])
        self.step_length = None
        self.c1 = None
        self.c2 = None
        self.sigma = np.sqrt(2.0 * self.params['gamma'] * self.params['kT'] / self.params['Qs'][-1])
        self.state_update = {}

    def update(self, step_length, state):
        if self.step_length != step_length:
            self.c1 = np.exp(-self.params['gamma'] * abs(step_length))
            self.c2 = np.sqrt((1.0 - np.exp(-2.0 * self.params['gamma'] * abs(step_length))) * 0.5 / self.params['gamma'])
            self.step_length = step_length

        self.aux_v = np.copy(state['aux_velocity_NH'])
        self.aux_v[-1] = self.c1 * self.aux_v[-1] + self.sigma * self.c2 * np.random.standard_normal(np.shape(self.aux_v[-1]))
        self.state_update = {
            'aux_velocity_NH': self.aux_v,
            }
        return self.state_update

#class NoseHooverSphericalNPT(Update):
#    """
#    Update velocities using massive Nose-Hoover chains
#       DOI: 10.1080/00268979600100761
#
#    state required:
#        V: velocities
#        masses: masses of each degree of freedom 
#            (a vector should also work with broadcasting for atoms)
#        aux_position_NH: numpy array of auxiliary positions
#        aux_velocity_NH: numpy array of auxiliary velocities
#
#    Params:
#        kT: temperature in energy
#        Qs: numpy array of masses with shape (chain_length, natom)
#        nc: number of integration substeps
#    """
#    h5_keys = ['V', 'aux_position_NH', 'aux_velocity_NH']
#    h5_shapes = [('natom', 3), ('naux', 'natom'), ('naux', 'natom')]
#    h5_types = ['f', 'f', 'f']
#    requirements = set(['V', 'masses', 'aux_position_NH', 'aux_velocity_NH'])
#
#    def __init__(self,
#            kT,
#            Pext,
#            Qs,
#            nc=5,
#            name='nosehooverchain',
#            ):
#        self.params = {
#            'kT': kT,
#            'Qs': Qs,
#            'nc': nc,
#            'name': name,
#            }
#        self.aux_q = None
#        self.aux_v = None
#        self.aux_a = None
#        self.V = None
#        w1 = 0.41449077179437571194
#        w3 = -0.65796308717750284778
#        self.ws = np.array([w1, w1, w3, w1, w1])
#        self.M = len(Qs)
#        self.state_update = {}
#
#    @classmethod
#    def build(cls, kT, masses, tau=0.5 * units.PS_TO_AU, chain_length=5, nc=5, dim=3, mass_weight=True):
#        """
#        Construct the update for molecular system with auxiliary masses based on rules given in
#            doi: 10.1021/j100151a052
#
#        Args:
#            kT: Temperature in energy units
#            masses: np.array of atomic masses
#            tau: relaxation time scale
#            chain_length: length of Nose-Hoover chain
#            nc: number of Yoshida-Suzuki integration substeps used to integrate NHC degrees of freedom
#            dim: number of degrees of freedom per particle
#            mass_weight: if True, will scale masses of NHCs by mass_i / mass_H 
#                where mass_i is the mass of atom i and mass_H is a proton mass
#
#        Returns:
#            NHC_update: a constructed NHC thermostat update
#        """
#        if mass_weight:
#            mass_r = np.reshape(masses, (1, -1)) / utils.symbol_to_mass(['H'])[0]
#        else:
#            mass_r = np.ones((1, np.size(masses)+1))
#        Qs = np.ones((chain_length, np.size(mass_r))) * kT * mass_r * tau **2
#        Qs[0, :] *= dim
#        Qs[0, -1] *= dim
#        return cls(kT, Qs, nc)
#
#    def initialize(self, kT=None, Qs=None):
#        """
#        Create initial positions and velocities of auxiliary degrees of freedom,
#        positions are set to zero and velocities are boltzmann distributed
#
#        Args:
#            kT: Temperature in energy units
#            Qs: np.array (chain_length, natom) of masses of NHCs
#        
#        Returns:
#            aux_q, aux_v: auxiliary variables for position and velocity
#        """
#        if kT is None:
#            kT = self.params['kT']
#        if Qs is None:
#            Qs = self.params['Qs']
#
#        aux_q = np.zeros_like(Qs)
#        factor = np.sqrt(kT/Qs)
#        aux_v = np.random.normal(scale=factor)
#        return aux_q, aux_v
#
#    def compute_nose_kinetic_energy(self, velocities, masses):
#        return 0.5 * np.sum(velocities ** 2 * masses) 
#
#    def compute_nose_potential_energy(self, coordinates, gkt, gnkt):
#        return np.sum(coordinates[0] * gnkt) + np.sum(coordinates[1:] * gkt)
#
#    def update(self, step_length, state):
#        self.aux_q = np.copy(state['aux_position_NH'])
#        self.aux_v = np.copy(state['aux_velocity_NH'])
#        # Atomwise KE (note the factor of two)
#        akin = np.sum(state['V']**2, axis=1) * np.reshape(state['masses'], (-1, ))
#        vkin = vmass * vlogv**2
#        kin = np.concatenate([akin, vkin])
#        scale = np.ones_like(kin)
#        self.aux_a = np.zeros_like(self.aux_q) 
#        self.gnkt = np.shape(state['V'])[-1] * self.params['kT']
#        self.gkt = self.params['kT']
#        self.aux_a[0] = (kin - self.gnkt) / self.params['Qs'][0]
#        self.aux_a[1:] = (self.params['Qs'][:-1] * self.aux_v[:-1]**2 - self.gkt) / self.params['Qs'][1:]
#        self.aux_a_V = 3.0 * (self.Pint - self.params['pressure']) / vmass #TODO
#
#        for k in range(self.params['nc']): # loop of integrations substeps
#            for w in self.ws: # loop of steps in Yoshida Suzuki integrator
#                # This is sort of hacky due to translation from TeraChem, which
#                # was itself translated from DOI: 10.1080/00268979600100761
#                # appendix A
#                wdts2 = w * step_length / self.params['nc'] 
#                wdts4 = wdts2 * 0.5 
#                wdts8 = wdts4 * 0.5
#
#                self.aux_v[self.M-1] += self.aux_a[self.M-1] * wdts4
#                # Intra chain coupling M to 0
#                for Mi in range(self.M-1):
#                    aa = np.exp(-wdts8 * self.aux_v[self.M-(Mi+1)])
#                    self.aux_v[self.M-1-(Mi+1)] = self.aux_v[self.M-1-(Mi+1)] * aa**2 + wdts4 * aa * self.aux_a[self.M-1-(Mi+1)]
#
#                # Update kinetic energy
#                aa = np.exp(-wdts2 * self.aux_v[0])
#                scale *= aa
#                self.aux_a[0] = (akin * scale**2 - self.gnkt) / self.params['Qs'][0]
#
#                # Update positions
#                self.aux_q += wdts2 * self.aux_v
#
#                # Intra chain coupling 0 to M
#                for Mi in range(self.M-1):
#                    aa = np.exp(-wdts8 * self.aux_v[Mi+1])
#                    self.aux_v[Mi] = self.aux_v[Mi] * aa**2 + wdts4 * aa * self.aux_a[Mi]
#                    self.aux_a[Mi+1] = (self.params['Qs'][Mi] * self.aux_v[Mi]**2 - self.gkt) / self.params['Qs'][Mi+1]
#
#                self.aux_v[self.M-1] += self.aux_a[self.M-1] * wdts4
#
#        # All this work to rescale velocities
#        self.V = state['V'] * np.reshape(scale, (-1, 1))
#        self.energy = self.compute_nose_kinetic_energy(self.aux_v, self.params['Qs'])
#        self.energy += self.compute_nose_potential_energy(self.aux_q, self.gkt, self.gnkt)
#        self.state_update = {
#                'V' : self.V,
#                'aux_position_NH': self.aux_q,
#                'aux_velocity_NH': self.aux_v,
#                'NHC_energy': self.energy,
#            }
#        return self.state_update

#class NoseHooverNPTPositionUpdate(PositionUpdate):
#    coeffs = np.array([1.0/6.0, 1.0/120.0, 1.0/5040.0, 1.0/362880.0])
#
#    def update(self, step_length, state):
#        vlogv = 
#        aa = np.exp(0.5 * step_length * vlogv)
#        aa2 = aa * aa
#        arg2 = (0.5 * vlogv * step_length) ** 2
#        poly = (((self.coeffs[3] * arg2 + self.coeffs[2]) * arg2 + self.coeffs[1]) * arg2 + coeffs[0]) * arg2 + 1.0
#        bb = aa * poly * step_length
#        self.X = state['X'] * aa2 + state['V'] * bb
#        self.aux_q = state['aux_position_NH'] + vlogv * step_length
#        self.state_update = {
#                'X' : self.X,
#            }
#        return self.state_update

class DistanceAnchor(Update):
    """
    Move atoms by mass weighted coordinates to given distance. Without being 
    wrapped by TimeDependent Update, the positions are held constant at 
    dist_stop. With it, they can be interpolated from their initial distance to 
    the final distance. The rate is determined by linearly going from the 
    interatomic distance at the time_start to the dist_stop at the time_stop.
    Velocities of the selected atoms are also set to zero.
    This update should be placed immediately before or after the position update.

    Params:
        mass1 (float): mass of first atom
        mass2 (float): mass of second atom
        atom_ind1 (int): first atom index to pull toward one another
        atom_ind2 (int): second atom index to pull toward one another
        dist_stop (float): distance at which to stop pulling the atoms together
        interpolate (bool): True to linearly move the wells based on time_frac if Update is TimeDependent
    """
    h5_keys = ['X']
    h5_shapes = [('natom', 3)]
    h5_types = ['f']

    def __init__(self, 
            mass1,
            mass2,
            atom_ind1, 
            atom_ind2, 
            dist_stop,
            interpolate=False,
            name='distance_anchor',
            ):
        self.params = {
            'mass1': mass1,
            'mass2': mass2,
            'atom_ind1': atom_ind1,
            'atom_ind2': atom_ind2,
            'dist_stop': dist_stop,
            'interpolate': interpolate,
            'name' : name,
            }
        self.requirements = set(['X', 'V'])
        self.time_frac = 1.0 # Use the time_frac to determine when to restart
        self.X1_start = None
        self.X2_start = None
        self.X1_move = None
        self.X2_move = None
        self.X = None
        self.V = None
        self.state_update = {}

    def reset(self, state):
        # Compute vector between atoms and initial distance
        self.X1_start = state['X'][self.params['atom_ind1'], :]
        self.X2_start = state['X'][self.params['atom_ind2'], :]
        vec_start = self.X2_start - self.X1_start 
        dist_start = np.linalg.norm(vec_start)
        # Compute mass weighted distances that each atom should move
        dist1 = (dist_start - self.params['dist_stop']) * self.params['mass2'] / (self.params['mass1'] + self.params['mass2']) / dist_start
        dist2 = (dist_start - self.params['dist_stop']) * self.params['mass1'] / (self.params['mass1'] + self.params['mass2']) / dist_start

        # Compute vector that atoms will travel along 
        self.X1_move =  vec_start * dist1
        self.X2_move = -vec_start * dist2

    def update(self, step_length, state):
        self.X = np.copy(state['X'])
        self.V = np.copy(state['V'])
        if self.params['interpolate']:
            # Restart movement cycle
            if state['time_frac'] <= self.time_frac:
                self.reset(state)
            self.time_frac = state['time_frac']

        else:
            self.reset(state)
            self.time_frac = 1.0
            
        # Linearly interpolate along vector as time goes by
        self.X[self.params['atom_ind1'], :] = self.X1_start + self.time_frac * self.X1_move
        self.X[self.params['atom_ind2'], :] = self.X2_start + self.time_frac * self.X2_move
        # Remove velocities
        self.V[self.params['atom_ind1'], :] = 0.0 
        self.V[self.params['atom_ind2'], :] = 0.0 
        self.state_update = { 
            'X' : self.X,
            'V' : self.V,
            }
        return self.state_update

class Recenter(Update):
    """
    Move center of mass to origin, remove center of mass 
    translational/rotational velocity.

    Useful in combination with forces that do not preserve such quantities, such
    as stochastic thermostats.

    Should probably be placed prior to a position update.

    Params:
        masses ({nparticle,} ndarray): masses of particles
    """
    h5_keys = ['X']
    h5_shapes = [('natom', 3)]
    h5_types = ['f']

    def __init__(self, 
            masses,
            name='recenter',
            ):
        self.params = {
            'masses': masses,
            'name' : name,
            }
        self.requirements = set(['X', 'V'])
        self.X = None
        self.V = None
        self.state_update = {}

    def update(self, step_length, state):
        self.X, self.V = init.initialize_centered(state['X'], state['V'], self.params['masses'])
        self.state_update = {
            'X': self.X,
            'V': self.V,
            }
        return self.state_update

class MetropolisHastings(Update):
    """
    This update proceeds in two steps, the first step simply holds the position 
    and momentum of the state, the second checks whether the new state is 
    probable, and if not the state is reset to the previous with flipped 
    momentum.

    The order of integration should generally be (according to Free Energy Computations):
        [Thermostat, MetropolisHastings, Velocity, Position, Velocity, MetropolisHastings, Thermostat]

    Params:
        masses ({nparticle,} ndarray): masses for particles
        kT (float): temperature in energy
        potential_key (str): state key that corresponds to desired potential energy to check
    """
    h5_keys = []
    h5_shapes = []
    h5_types = []

    def __init__(self,
            masses,
            kT,
            potential_key='potential_energy',
            name='hmc',
            ):
        self.requirements = set(['X', 'V', potential_key])
        self.params = {
            'masses': masses,
            'kT': kT,
            'potential_key': potential_key,
            }
        self.potential_key = potential_key
        self.counter = 0
        self.X_init = None
        self.V_init = None
        self.PE_init = None
        self.KE_init = None
        self.PE_final = None
        self.KE_final = None
        self.state_update = {}

    def update(self, step_length, state):
        self.counter += 1
        if self.counter % 2 == 1: # First call in integration loop, just tabulate current state
            self.X_init = state['X']
            self.V_init = state['V']
            self.PE_init = state[self.potential_key]
            self.KE_init = utils.compute_kinetic_energy(state['V'], self.params['masses'])
            self.state_update = {}

        else: # Second call in integration loop
            self.PE_final = state[self.potential_key]
            self.KE_final = utils.compute_kinetic_energy(state['V'], self.params['masses'])
            diff = self.PE_final + self.KE_final - (self.PE_init + self.KE_init)
            if np.random.uniform() < np.min(1.0, np.exp(-diff / self.params['kT'])):
                self.state_update = {} # Keep current trajectory
            else: 
                self.state_update = { # Revert to before, flip momentum
                        'X':  self.X_init,
                        'V': -self.V_init,
                    }
            
        return self.state_update
        
class BXDE(Update):
    """
    This update proceeds in two steps, the first step simply holds the position 
    /momentum/gradient of the state, the second checks whether the new state has
    crossed an energy barrier, if so the velocities are reflected away from the
    barrier.
    
    Different from the paper, the user may give a delta_PE which defines a 
    maximum energy to reflect from. This makes it easy to window the energy 
    within the adaptive scheme.

    The order of integration should generally be:
        [Thermostat, BXDE, Velocity, Position, Velocity, BXDE, Thermostat]

    Params:
        masses ({nparticle,} ndarray): masses for particles
        PE_min (float): minimum potential energy allowed by barrier (default -np.inf)
        dPE (float): Max energy allowed given by PE_min + dPE (default np.inf)
        potential_name (str): used to get potential_energy and potential_gradient state values
        adaptive (bool): True to dynamically change PE_min according to reference (default True)
        nstep_sample (int): number of steps to sample for adaptive barriers (default 100)
        name (str): update name (default 'bxde')

    References:
        doi: 10.1021/acs.jctc.8b00515
    """
    h5_keys = []
    h5_shapes = []
    h5_types = []

    def __init__(self,
            masses,
            PE_min=-np.inf,
            dPE=np.inf,
            potential_name='potential',
            adaptive=True,
            nstep_sample=100,
            name='bxde',
            ):
        self.requirements = set(['X', 'V', potential_name + '_energy', potential_name + '_gradient'])
        self.params = {
            'masses': np.reshape(masses, (-1, 1)),
            'potential_key': potential_name + '_energy',
            'gradient_key': potential_name + '_gradient',
            'adaptive': adaptive,
            'nstep_sample': nstep_sample,
            }
        self.PE_min = PE_min
        self.dPE = dPE
        self.potential_name = potential_name
        self.adaptive = adaptive
        self.nstep_sample = nstep_sample
        self.counter = 0
        self.curr_PE_max = None
        self.X_init = None
        self.V_init = None
        self.V = None
        self.PE_final = None
        self.lmbda = None
        self.state_update = {}

    def update(self, step_length, state):
        self.counter += 1
        self.state_update = {}
        if self.counter % 2 == 1: # First call in integration loop, just tabulate current state
            self.X_init = state['X']
            self.V_init = state['V']
            self.PE_init = state[self.params['potential_key']]
            self.G_init = state[self.params['gradient_key']]

        else: # Second call in integration loop
            self.PE_final = state[self.params['potential_key']]
            if self.adaptive:
                if self.PE_final > self.curr_PE_max:
                    if self.counter//2 > self.nstep_sample:
                        self.PE_min = self.curr_PE_max
                        self.counter = 0
                    else:
                        # Don't let PE_max go over PE_min + dPE
                        PE_cutoff = self.PE_min + self.dPE
                        if self.PE_min > -np.inf:
                            self.curr_PE_max = min(PE_cutoff, self.PE_final) 
                        else:
                            self.curr_PE_max = self.PE_final
                self.state_update['BXDE_PE_curr_max'] = self.curr_PE_max
                self.state_update['BXDE_PE_min'] = self.PE_min

            if (self.PE_final < self.PE_min):
                gke = utils.compute_kinetic_energy_momentum(self.G_init, self.params['masses'])
                self.lmbda = np.sum(self.G_init * self.V_init) / gke
                self.V = self.V_init + self.lmbda * self.G_init / self.params['masses']
                # Revert to before, reflect velocities about PE boundary
                self.state_update[self.params['potential_key']] = self.PE_init
                self.state_update['X'] = self.X_init
                self.state_update['V'] = self.V
            elif (self.PE_final > (self.PE_min + self.dPE) and self.PE_min > -np.inf):
                gke = utils.compute_kinetic_energy_momentum(self.G_init, self.params['masses'])
                self.lmbda = - np.sum(self.G_init * self.V_init) / gke
                self.V = self.V_init + self.lmbda * self.G_init / self.params['masses']
                # Revert to before, reflect velocities about PE boundary
                self.state_update[self.params['potential_key']] = self.PE_init
                self.state_update['X'] = self.X_init
                self.state_update['V'] = self.V

        return self.state_update

class FIRE(Update):
    """
    Fast inertial relaxation engine step

    Can be used to add a minimization step to the dynamics,
    recommended use is to append to an existing MD ExplicitIntegrator

    Params:
        deltat_max (float): maximum time step
        N_min (int): see ref
        f_inc (float): see ref
        f_dec (float): see ref
        alpha_start (float): see ref
        f_alpha (float): see ref
        grad_key (str): key to pull gradient from state (default 'potential_gradient')

    References:
        doi 10.1103/PhysRevLett.97.170201
    """
    h5_keys = []
    h5_shapes = []
    h5_types = []

    def __init__(self,
            deltat_max,
            N_min=5,
            f_inc=1.1,
            f_dec=0.5,
            alpha_start=0.1,
            f_alpha=0.99,
            grad_key='potential_gradient',
            ):
        self.deltat_max = deltat_max
        self.N_min = N_min
        self.f_inc = f_inc
        self.f_dec = f_dec
        self.alpha_start = alpha_start
        self.f_alpha = f_alpha
        self.grad_key = grad_key
        self.P = None
        self.Ppos_nstep = 0
        self.alpha = alpha_start
        self.state_update = {}

    def update(self, step_length, state):
        self.P = - np.sum(state[self.grad_key] * state['V'])
        if self.P > 0.0:
            self.state_update = {
                    'V': (1.0 - self.alpha) * state['V'] + self.alpha * state[self.grad_key] / np.linalg.norm(state[self.grad_key]) * np.abs(state['V'])
                }
            self.Ppos_nstep += 1
            if self.Ppos_nstep > self.N_min:
                self.state_update['dt'] = np.min(state['dt'] * self.f_inc, self.deltat_max)
                self.alpha = self.alpha * f_alpha
        else:
            self.state_update = {
                'V' : np.zeros_like(state['V']),
                'dt' : state['dt'] * self.f_dec,
                }
            self.alpha = self.alpha_start
            self.Ppos_nstep = 0

        return self.state_update
