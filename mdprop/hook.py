import numpy as np
import os
import pickle as pkl
import time
import sys
import atexit

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False

from . import io, units, update, utils

class Hook(update.Update):
    """
    Hooks are specialized updates that don't require step lengths.
    They simply take in the state and compute a property of that state such as
    kinetic energy, temperature, etc.

    Define the compute function which returns a state dict just as an update would.
    """
    def update(self, step_length, state):
        return self.compute(state)

    def initialize(self, state):
        """ Initialization to be done prior to trajectory run """
        return self.compute(state)

    def finalize(self, state):
        """ Cleanup to be done after trajectory run """
        self.state_update = {}
        return self.state_update

    def compute(self, state):
        raise NotImplementedError

class KineticEnergy(Hook):
    """
    Compute kinetic energy and temperature

    state required:
        V: velocities

    Params:
        masses: masses of each degree of freedom
        boltzmann_constant: conversion factor from KE to temperature (default au/K)
    """
    h5_keys = ['kinetic_energy', 'temperature']
    h5_shapes = [(1, ), (1, )]
    h5_types = ['f', 'f']

    def __init__(self, masses, boltzmann_constant=units.K_TO_AU):
        self.params = {
            'masses': masses,
            'boltzmann_constant': boltzmann_constant,
            }
        self.requirements = set(['V', 'masses'])

    def compute(self, state):
        self.kinetic_energy = utils.compute_kinetic_energy(state['V'], self.params['masses'])
        self.state_update = {
           'kinetic_energy': self.kinetic_energy,
            }
        return self.state_update

class Temperature(Hook):
    """
    Compute temperature from kinetic energy

    state required:
        V: velocities (for ndof)

    Params:
        boltzmann_constant: conversion factor from KE to temperature (default au/K)
    """
    h5_keys = ['temperature']
    h5_shapes = [(1, )]
    h5_types = ['f']

    def __init__(self, boltzmann_constant=units.K_TO_AU):
        self.params = {
            'boltzmann_constant': boltzmann_constant,
            }
        self.requirements = set(['V', 'kinetic_energy'])

    def initialize(self, state):
        return {}

    def compute(self, state):
        self.temperature = 2.0 * state['kinetic_energy'] / np.size(state['V']) / self.params['boltzmann_constant']
        self.state_update = {
           'temperature': self.temperature,
            }
        return self.state_update

class VirialTemperature(Hook):
    """
    Compute virial definition of temperature from position and gradient

    state required:
        X: coordinates
        grad_key: gradient of desired potential

    Params:
        grad_key: state[grad_key] == desired gradient to use
        boltzmann_constant: conversion factor from KE to temperature (default au/K)
    """
    h5_keys = ['virial_temperature']
    h5_shapes = [(1, )]
    h5_types = ['f']

    def __init__(self, grad_key='potential_gradient', boltzmann_constant=units.K_TO_AU):
        self.params = {
            'grad_key': grad_key,
            'boltzmann_constant': boltzmann_constant,
            }
        self.requirements = set(['X', grad_key])

    def initialize(self, state):
        if state.get(self.params['grad_key'], None) is None:
            return {}

    def compute(self, state):
        self.virial = utils.compute_virial_temperature(state['X'], -state[self.params['grad_key']], boltzmann_constant=self.params['boltzmann_constant'])
        self.state_update = {
           'virial_temperature': self.virial,
            }
        return self.state_update

class ConfigurationalTemperature(Hook):
    r"""
    Compute the instantaneous configurational temperature defined by

    .. math:: kT(t) = \nabla V(t)^T \nabla V_0(t) / Tr(\nabla^2 V_0)

    Params:
        pot (Potential): potential to use to compute gradient in numerator
        pot0 (Potential): model potential to compute gradient and Hessian term (default pot)
        boltzmann_constant (float): multiplicative factor to convert energy to desired temperature units (default K/au)

    References:
        Equation 26 of https://www.tandfonline.com/doi/abs/10.1080/00268970500054664
    """
    def __init__(self, pot, pot0=None, boltzmann_constant=1.0/units.K_TO_AU):
        self.pot = pot
        if pot0 is None:
            self.pot0 = pot
        else:
            self.pot0 = pot0
        self.boltzmann_constant = boltzmann_constant
        self.state_update = {}

    def compute(self, state):
        _, G = self.pot.gradient(state['X'])
        _, G0 = self.pot0.gradient(state['X'])
        H0 = self.pot0.hessian(state['X'])
        config_temp1 = np.mean(G*G0) / np.mean(np.diag(H0)) * self.boltzmann_constant
        config_temp2 = np.mean(np.ravel(G*G0) / np.diag(H0)) * self.boltzmann_constant
        self.state_update = {
            #'config_temperature': utils.compute_configurational_temperature(G, H0, G0=G0, boltzmann_constant=self.boltzmann_constant),
            'config1_temperature': config_temp1,
            'config2_temperature': config_temp2,
            }
        return self.state_update

class VolumePressure(Hook):
    """
    Compute the instantaneous convex volume, use it with kinetic_energy,
    gradient, position to compute the pressure and add it to the state.
    Requires the simulation to always have a 3d volume, which needs 4 or more
    particles.

    state required:
        X, kinetic_energy, gradient

    Params:
        N/A
    """
    h5_keys = ['volume', 'pressure']
    h5_shapes = [(1, ), (1, )]
    h5_types = ['f', 'f']

    def __init__(self):
        self.params = {}
        self.requirements = set(['X', 'kinetic_energy', 'gradient'])

    def compute(self, state):
        volume = utils.compute_convex_volume(state['X'])
        pressure = utils.compute_virial_pressure(volume, state['kinetic_energy'], state['X'], -state['gradient'])
        virial_temp = utils.compute_virial_temperature(state['X'], -state['gradient'])
        self.state_update = {
            'volume': volume,
            'pressure': pressure,
            'virial_temperature': virial_temp,
            }
        return self.state_update

class TotalEnergy(Hook):
    """
    Compute total energy from state variables that end in 'energy'.

    state required:
        *energy: potential_energy, kinetic_energy, boundary_energy, etc.

    Params:
        N/A

    Note: The energy of this Update will not be right if multiple potentials are 
        used with the same name
    """
    h5_keys = ['total_energy']
    h5_shapes = [(1,)]
    h5_types = ['f']

    def __init__(self):
        self.params = {}
        self.requirements = set([])

    def compute(self, state):
        TE = 0.0
        for key in state:
            if key[-6:] == 'energy' and key != 'total_energy':
                TE += state[key]

        self.state_update = {'total_energy': TE}
        return self.state_update

class Gradient(Hook):
    """
    Copy the desired gradient into the state for easy access, or for saving
    state required:
        None

    Params:
        update: copy the -force from this VelocityUpdate object
    """
    h5_keys = ['gradient']
    h5_shapes = [('natom', 3)]
    h5_types = ['f']

    def __init__(self, update):
        self.params = {
            'update': update,
            }
        self.requirements = set([])

    def compute(self, state):
        # Ensure the update contains gradient information
        if self.params['update'].F is None:
            self.params['update'].update(0.0, state)
        self.state_update = {'gradient' : -self.params['update'].F}
        return self.state_update

class GradientCombination(Hook):
    """
    Copy a linear combination of gradients into the state for easy access, or for saving
    state required:
        None

    Params:
        updates: copy the -force from this list of VelocityUpdate objects
        coeffs: coefficients to scale gradients by
    """
    h5_keys = ['gradient']
    h5_shapes = [('natom', 3)]
    h5_types = ['f']

    def __init__(self, updates, coeffs):
        self.params = {
            'updates': updates,
            'coeffs': coeffs,
            }
        self.requirements = set([])

    def compute(self, state):
        # Ensure the updates contain gradient information
        for u in self.params['updates']:
            if u.F is None:
                u.update(0.0, state)

        # Copy it over
        total_grad = - self.params['coeffs'][0] * self.params['updates'][0].F
        for u, c in zip(self.params['updates'][1:], self.params['coeffs'][1:]):
            total_grad = total_grad - c * u.F
        self.state_update = {'gradient' : total_grad}
        return self.state_update

class BondOrder(Hook):
    """
    Copy the desired bond order into the state for easy access, or for saving
    state required:
        None

    Params:
        potential: copy the bond_order from this object.result dict

    TODO: This is currently a bit a of hack to work with ReaxFF, and needs improving
    """
    h5_keys = ['bond_order']
    h5_shapes = [('natom', 'natom')]
    h5_types = ['f']

    def __init__(self, potential):
        self.params = {
            'potential': potential,
            }
        self.requirements = set([])

    def compute(self, state):
        # Ensure the update contains bond order information
        if self.params['potential'].result.get('bond_order', None) is None:
            return {}
        self.state_update = {'bond_order' : self.params['potential'].result['bond_order']}
        return self.state_update

class Dipole(Hook):
    """
    Copy the desired dipole into the state for easy access, or for saving
    state required:
        None

    Params:
        potential: copy the dipole from this object.result dict

    TODO: This is currently a bit a of hack to work with ReaxFF, and needs improving
    """
    h5_keys = ['dipole']
    h5_shapes = [(3, )]
    h5_types = ['f']

    def __init__(self, potential):
        self.params = {
            'potential': potential,
            }
        self.requirements = set([])

    def compute(self, state):
        # Ensure the update contains bond order information
        if self.params['potential'].result.get('dipole', None) is None:
            return {}
        self.state_update = {'dipole' : self.params['potential'].result['dipole']}
        return self.state_update

class Extract(Hook):
    """
    Pulls the given key out of an dict and copies it into the state.
    Can be used with vars(Object) or similar constructs to easily get object 
    attributes that are changing over the course of the trajectory.
    """
    def __init__(self, dct, key, save_key):
        self.params = {
            'dct': dct,
            'key': key,
            'save_key': save_key,
            }
        self.requirements = set([])

    def compute(self, state):
        self.state_update = {
            self.params['save_key']: self.params['dct'][self.params['key']]
        }
        return self.state_update

class Callback(Hook):
    """
    Calls a given function with the set of given state variables, places returned value (if not None) in state with given key

    Args:
        callback: function to call
        arg_keys: state variable keys used to call function
        state_key: key of returned value (or None)
    """
    def __init__(self, callback, arg_keys=[], state_key=None, initialize=True):
        self.params = {
            'callback': callback,
            'arg_keys': arg_keys,
            'state_key': state_key,
            'initialize': initialize,
            }
        self.requirements = set([])

    def initialize(self, state):
        if self.params['initialize']:
            return self.compute(state)
        else:
            self.state_update = {}
            return self.state_update

    def compute(self, state):
        args = [state[key] for key in self.params['arg_keys']]
        val = self.params['callback'](*args)
        if self.params['state_key'] is not None:
            self.state_update = {
                self.params['state_key'] : val,
                }
        else:
            self.state_update = {}
        return self.state_update

class WriteArrayXYZ(Hook):
    """
    Append the array of a given key of current state to file.

    state required:
        key: string key to state to append
        symbols: atomic symbols to write in XYZ format
        frame: current frame (for subtext)
        simulation_time: current simulation time (for subtext)
    """
    def __init__(self, key, filename, unit_conversion=1.0, format_str="% .11E"):
        self.params = {
            'key': key,
            'filename': filename,
            'unit_conversion': unit_conversion,
            'format_str': format_str,
            }
        self.requirements = set([key, 'symbols', 'frame', 'simulation_time'])
        self.filehandle = None

    def initialize(self, state):
        self.filehandle = open(self.params['filename'], 'w') # Create file or error out if it's not writeable
        # If the attribute doesn't exist initially, don't write an initial frame
        # for it
        if self.params['key'] in state:
            return self.compute(state)
        else:
            self.state_update = {}
            return self.state_update

    def compute(self, state):
        Xc = state[self.params['key']] * self.params['unit_conversion']
        text = "Frame %d, time %f" % (state['frame'], state['simulation_time'])
        io.save_xyz_filehandle(Xc, state['symbols'], self.filehandle, text, format_str=self.params['format_str'])
        self.filehandle.flush()

        self.state_update = {}
        return self.state_update

    def finalize(self, state):
        self.filehandle.close()
        self.state_update = {}
        return self.state_update

class WriteCheckpoint(Hook):
    """
    Save the state as a pickled dict.
    """
    def __init__(self, filename="mdprop.chkpt", frequency=100, num_backups=2):
        self.params = {
            'filename': filename,
            'frequency': frequency,
            'num_backups': num_backups,
            }
        self.requirements = set([])

    def compute(self, state):
        if state['frame'] % self.params['frequency'] == 0:
            for i in range(0, self.params['num_backups']-1, -1):
                curr_filename = self.params['filename'] + ('_%1d' % i)
                next_filename = self.params['filename'] + ('_%1d' % (i+1))
                if os.path.isfile(curr_filename):
                    os.rename(curr_filename, next_filename)

            if self.params['num_backups'] > 0 and os.path.isfile(self.params['filename']):
                os.rename(self.params['filename'], self.params['filename'] + '_0')
            with open(self.params['filename'], 'wb') as fout:
                pkl.dump(state, fout)
        self.state_update = {}
        return self.state_update

class Timing(Hook):
    """
    Save information about the calculation time
    """
    h5_keys = ['calc_time', 'total_calc_time', 'simulation_time', 'frame']
    h5_shapes = [(1,), (1,), (1,), (1,)]
    h5_types = ['f', 'f', 'f', 'i']

    def __init__(self):
        self.params = {}
        self.requirements = set([])
        self.init_time = None # Initialized at first compute call with Trajectory.run()
        self.last_time = None
        self.curr_time = None
        self.state_update = {}

    def compute(self, state):
        if self.init_time is None:
            self.init_time = time.time()
            self.curr_time = self.init_time
        self.last_time = self.curr_time
        self.curr_time = time.time()
        self.state_update = {
                'calc_time': self.curr_time - self.last_time,
                'total_calc_time': self.curr_time - self.init_time,
                'frame': state.get('frame', -1) + 1,
            }
        return self.state_update

class RandomState(Hook):
    """ 
    Save information necessary to reproduce a specific random state for numpy, 
    initializes the random state if it is given in the state dict
    """
    h5_keys = ['random_state_str', 'random_state_keys', 'random_state_pos', 'random_state_has_gauss', 'random_state_cached_gaussian']
    h5_shapes = [(1,), (624,), (1,), (1,), (1,)]
    h5_types = ['S7', 'uint32', 'int64', 'int64', 'float64']

    def __init__(self):
        self.params = {}
        self.requirements = set([])
        self.random_state = None

    def initialize(self, state):
        self.random_state = state.get('random_state', None)
        if self.random_state is not None:
            np.random.set_state(*self.random_state)
        self.state_update = {}
        return self.state_update

    def compute(self, state):
        self.random_state = np.random.get_state()
        self.random_state_str = self.random_state[0].encode('ascii')
        self.random_state_keys = self.random_state[1]
        self.random_state_pos = self.random_state[2]
        self.random_state_has_gauss = self.random_state[3]
        self.random_state_cached_gaussian = self.random_state[4]
        self.state_update = {
            'random_state': self.random_state,
            'random_state_str': self.random_state_str,
            'random_state_keys': self.random_state_keys,
            'random_state_pos': self.random_state_pos,
            'random_state_has_gauss': self.random_state_has_gauss,
            'random_state_cached_gaussian': self.random_state_cached_gaussian,
            }
        return self.state_update

class Print(Hook):
    """
    Prints a list of keys to given output.
    """
    def __init__(self, printkeys, output=sys.stdout):
        self.keys = printkeys
        self.params = {
            'printkeys': self.keys,
            'output': output,
            }
        self.requirements = set(printkeys)

    def print_header(self):
        self.params['output'].write("|" + "|".join([" %-18s "% pk[:18]  for pk in self.params['printkeys']]) + "|\n")
        self.params['output'].flush()

    def print_line(self):
        self.params['output'].write ("|" + "|".join(["-%-18s-"% ("-"*18) for pk in self.params['printkeys']]) + "|\n")
        self.params['output'].flush()

    def print_key_values(self, state):
        self.params['output'].write("|" + "|".join([" %- 1.11E " % state.get(pk, 0.0) for pk in self.params['printkeys']]) + "|\n")
        self.params['output'].flush()

    def initialize(self, state):
        self.print_header()
        self.print_line()
        self.state_update = {}
        return self.state_update

    def compute(self, state):
        self.print_key_values(state)
        self.state_update = {}
        return self.state_update

    def finalize(self, state):
        self.print_line()
        self.state_update = {}
        return self.state_update

class WriteH5(Hook):
    """
    Given a list of (state key, array size) tuples, write the state to a given h5 file.
    The arrays are written per frame, and so are written with size (nframe, *array_size).
    The size of nframes is adjusted dynamically each time the

    Args:
        h5filename: h5 filename to write values to
        keys: list of keys to take from state and write
        shapes: shapes of corresponding state[key]
        key_map: if given, maps original keys to different keys in the h5file
        writemode: 'r+', 'w', 'w-', 'a'; defaults to 'w' which overwrites any existing file
        kwargs: list of arguments for create_dataset on the h5file;
            keys, shapes are given by state variables, chunks=True for resizing

    Note: This strictly appends to the H5 file, so if data exists already, or if
        different datasets are written more often, then they are not guaranteed to
        line up.
    """
    def __init__(self, h5filename, keys, shapes, types, key_map=None, cache_size=10, writemode='w', **kwargs):
        if not _HAS_H5PY:
            raise ImportError("Package h5py required for writing h5 file.")
        self.h5file = h5py.File(h5filename, writemode, libver='latest')
        if key_map is None:
            key_map = keys
        self.params = {
            'h5file': self.h5file,
            'keys': keys,
            'shapes': shapes,
            'key_map': key_map,
            'cache_size': cache_size,
            }
        self.requirements = set([k for k in keys])
        self.tmp_arrays = []
        self.counter = 0
        for k, s, t in zip(key_map, shapes, types):
            shape = tuple([0] + list(s))
            maxshape = tuple([None] + list(s))
            cache_shape = tuple([cache_size] + list(s))
            self.tmp_arrays.append(np.zeros(cache_shape, dtype=t))
            self.h5file.require_dataset(k, shape=shape, maxshape=maxshape, dtype=t, chunks=True, **kwargs)
        self.h5file.swmr_mode = True # Allow reading the file while the trajectory runs if the libver is new enough

    def initialize(self, state):
        """
        Initialize the H5 writer by attempting to write from state, but not 
        returning a failure if the key does not exist.
        """
        counter_mod = self.counter % self.params['cache_size']
        # Write data to a cache
        for i, k, in enumerate(self.params['keys']):
            if state.get(k, None) is not None:
                self.tmp_arrays[i][counter_mod, ...] = state[k]
        # Write to file if cache is full
        if counter_mod + 1 == self.params['cache_size']:
            for i, (m, s) in enumerate(zip(self.params['key_map'], self.params['shapes'])):
                self.params['h5file'][m].resize(self.params['h5file'][m].shape[0]+self.params['cache_size'], axis=0) # Resize
                self.params['h5file'][m][-self.params['cache_size']:, ...] = self.tmp_arrays[i] # Add data

        self.counter += 1

        # Register the finalize method to run atexit so that h5file is closed
        # properly, state isn't actually needed so we just pass an empty dict
        atexit.register(self.finalize, {})

        self.state_update = {}
        return self.state_update
        
    def compute(self, state):
        counter_mod = self.counter % self.params['cache_size']
        # Write data to a cache
        for i, k, in enumerate(self.params['keys']):
            self.tmp_arrays[i][counter_mod, ...] = state[k]
        # Write to file if cache is full
        if counter_mod + 1 == self.params['cache_size']:
            for i, (m, s) in enumerate(zip(self.params['key_map'], self.params['shapes'])):
                self.params['h5file'][m].resize(self.params['h5file'][m].shape[0]+self.params['cache_size'], axis=0) # Resize
                self.params['h5file'][m][-self.params['cache_size']:, ...] = self.tmp_arrays[i] # Add data
            self.params['h5file'].flush()
        self.counter += 1
        self.state_update = {}
        return self.state_update

    def finalize(self, state):
        counter_mod = self.counter % self.params['cache_size']
        # Write to file with partial cache (cache should already be populated)
        if counter_mod > 0:
            for i, (m, s) in enumerate(zip(self.params['key_map'], self.params['shapes'])):
                self.params['h5file'][m].resize(self.params['h5file'][m].shape[0]+counter_mod, axis=0) # Resize
                self.params['h5file'][m][-counter_mod:, ...] = self.tmp_arrays[i][:counter_mod, ...] # Add data
            self.params['h5file'].flush()
        self.params['h5file'].close()
        self.state_update = {}
        return self.state_update

class TimestepController(Hook):
    """
    Uses the state and a given control function to update the timestep.
    Expects the control function to take the state as first argument, and
    parameters after.
    """
    def __init__(self, controller, *controller_args, **controller_kwargs):
        self.params = {
            'controller': controller,
            'controller_args': controller_args,
            'controller_kwargs': controller_kwargs,
            }
        self.requirements = set(['gradient', 'V'])
        self.rho = 1.0
        self.epsilon = None
        self.control = None
        self.dt = None

    def initialize(self, state):
        self.epsilon = state['dt']
        self.control = self.params['controller'](state, *self.params['controller_args'], **self.params['controller_kwargs'])
        self.rho += 0.5 * self.epsilon * self.control
        self.dt = self.epsilon / self.rho
        self.state_update = {
            'dt': self.dt,
            'control': self.control,
            }
        return self.state_update

    def compute(self, state):
        self.control = self.params['controller'](state, *self.params['controller_args'], **self.params['controller_kwargs'])
        self.rho += self.epsilon * self.control
        self.dt = self.epsilon / self.rho
        self.state_update = {
            'dt': self.dt,
            'control': self.control,
            }
        return self.state_update

    def finalize(self, state):
        self.control = self.params['controller'](state, *self.params['controller_args'], **self.params['controller_kwargs'])
        self.rho -= 0.5 * self.epsilon * self.control
        self.dt = self.epsilon / self.rho
        self.state_update = {
            'dt': self.dt,
            'control': self.control,
            }
        return self.state_update

class SINConstraintViolation(Hook):
    r"""
    Calculates SIN thermostat constraint violation defined by

    .. math:: \sqrt(\sum_a (m_a v_a^2 + L/(L+1)\sum_i Q_1 {v_{1,a}^(i)}^2 - L k_B T)^2)
    """
    def __init__(self, masses, Qs, kT):
        self.params = {
                'masses': np.reshape(masses, (-1, 1)),
                'Qs': Qs,
                'kT': kT,
                'L': np.shape(Qs)[1],
            }
        self.state_update = {}

    def compute(self, state):
        self.state_update = {
            "constraint_violation" : np.linalg.norm(state['V']**2 * self.params['masses'] + self.params['L']/(self.params['L']+1.0) * np.sum(self.params['Qs'][0] * state['aux_velocity_NH'][0]**2, axis=0) - self.params['L'] * self.params['kT'])
            }
        return self.state_update


# Instantiate simple cases of certain Hooks
temperature = Temperature()
virial_temperature = VirialTemperature()
total_energy = TotalEnergy()
volume_pressure = VolumePressure()
timing = Timing()
random_state = RandomState()
write_xyz = WriteArrayXYZ('X', 'coors.xyz', 1.0/units.ANGSTROM_TO_AU)
write_vel = WriteArrayXYZ('V', 'vel.log', 1.0/units.AMBERVELOCITY_TO_AU)
write_grad = WriteArrayXYZ('gradient', 'gradient.log', 1.0)
write_bond_order = WriteArrayXYZ('bond_order', 'bond_order.log', 1.0)
write_checkpoint = WriteCheckpoint()
screen_printer = Print(printkeys=['potential_energy', 'kinetic_energy', 'temperature', 'total_energy', 'calc_time', 'simulation_time'], output=sys.stdout)
default_hooks = [total_energy, temperature, timing, write_xyz, write_vel, write_checkpoint, screen_printer]
