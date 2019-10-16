import time
import numpy as np

from . import hook

class Trajectory(object):
    """
    Trajectory object takes state and time step, and propagates it forward
    nsteps using the integrator given. Calculation times are recorded
    automatically, everything else is computed as specified using Hook objects.

    Args:
        integ: Integrator class used to step forward the trajectory
        hooks: list of Hook objects; compute(state) is run after each step.
    """

    def __init__(
            self,
            integ,
            hooks=hook.default_hooks,
            ):
        self.integ = integ
        self.hooks = hooks

    def append_hook(self, h):
        self.hooks.append(h)

    def append_hooks(self, hooks):
        """ Append a list of new hooks to Trajectory.hooks """
        self.hooks.extend(hooks)
    
    def append_printkey(self, key):
        """ Find the (first) Print hook in self.hooks, append given key"""
        for h in self.hooks:
            if isinstance(h, hook.Print):
                h.keys.append(key)
                return
        raise ValueError("Print Hook does not appear to be in the Trajectory's hook list.")

    def append_printkeys(self, keys):
        """ Find the (first) Print hook in self.hooks, append given keys"""
        for h in self.hooks:
            if isinstance(h, hook.Print):
                h.keys.extend(keys)
                return
        raise ValueError("Print Hook does not appear to be in the Trajectory's hook list.")

    def del_printkey(self, key):
        """ Find the (first) Print hook in self.hooks, delete entry corresponding to given key """
        for h in self.hooks:
            if isinstance(h, hook.Print):
                h.keys.remove(key)
                return

    @property
    def printkeys(self):
        """ Find the (first) screen printer hook in self.hooks, append given key"""
        for h in self.hooks:
            if isinstance(h, hook.Print):
                return h.keys
        return []

    def initialize_hooks(self):
        """ Run hook.initialize(self.state) for hook in self.hooks, updating the state_dict"""
        for h in self.hooks:
            self.curr_state.update(h.initialize(self.curr_state))

    def update_hooks(self):
        """ Call all of the hooks in the order given, and update the state_dict """
        for h in self.hooks:
            self.curr_state.update(h.compute(self.curr_state))

    def finalize_hooks(self):
        """ Run hook.finalize(self.state) for hook in self.hooks, updating the state_dict"""
        for h in self.hooks:
            self.curr_state.update(h.finalize(self.curr_state))

    def run(self, dt, sim_time, state):
        """
        Run the trajectory forward in time by sim_time time.
        The hooks are all initialized, and then steps are taken until sim_time
        is reached. Each step consists of an integrator step, updating the time
        and then running the hook computations. Last the hooks are finalized.

        Args:
            dt: time step (a.u.t.)
            sim_time: time to integrate the trajectory
            state: initial state dict to propagate

        Returns:
            self.curr_state: the final state of the trajectory after 
                integration
        """
        self.sim_time = sim_time
        self.curr_state = state
        self.curr_state['dt'] = dt
        if 'simulation_time' not in self.curr_state:
            self.curr_state['simulation_time'] = 0.0
        self.initialize_hooks()

        while self.curr_state['simulation_time'] < sim_time:
            # Copy of dict with updates returned
            self.curr_state = self.integ.step(self.curr_state['dt'], self.curr_state)
            self.curr_state['simulation_time'] += self.curr_state['dt']
            self.update_hooks()

        self.finalize_hooks()

        return self.curr_state
