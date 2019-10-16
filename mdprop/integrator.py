import copy as cp
import numpy as np

from . import potential, update, units

class Integrator(object):
    """
    Integrator abstract base class
    """
    def __init__(self):
        raise NotImplementedError("Integrator is an abstract base class, try using the VelocityVerlet integrator.")

    def step(self):
        raise NotImplementedError("Integrator is an abstract base class, try using the VelocityVerlet integrator.")


class ExplicitIntegrator(Integrator):
    """
    Class of integrators that can be defined by single velocity/position 
    steps with given coefficients.
    """

    def __init__(self, coeff, updates):
        """Construct explicit integrator given a set of coefficients and updates

        Args:
            coeff (list of floats): amount to integrate each update
            updates (list of Update objects): describing how to integrate system

        Returns:
            (ExplicitIntegrator):
        """
        if len(coeff) != len(updates):
            raise ValueError("Number of updates and coefficients do not match")
        self.coeff = coeff
        self.updates = updates
        self.iter_ind = 0
        self.dt = 0.0

    def __len__(self):
        return len(self.coeff)

    def __iter__(self):
        self.iter_ind = 0
        return self

    def __next__(self):
        if self.iter_ind < len(self):
            i = self.iter_ind
            self.iter_ind += 1
            return (self.coeff[i], self.updates[i])
        else:
            raise StopIteration

    next = __next__

    def step(self, dt, state):
        """
        Take single step according to integrator.

        Args:
            dt (float): time step to integrate
            state (dict of positions('X'), velocities('V'), etc.); all things that the update functions need to update the state.

        Returns:
            (dict): state dictionary at time (t+dt)
        """
        new_state = cp.deepcopy(state)
        for coeff, update in self:
            state_update = update.update(dt*coeff, new_state)
            new_state.update(state_update)

        return new_state
            
    def append(self, integ, coeffA=1.0, coeffB=1.0):
        """A.append(B) returns
            C := A B
            
        Args:
            integ (ExplicitIntegrator): integrator to append to A

        Returns:
            (ExplicitIntegrator):

        Note: 
            Unlike append for python lists, this function returns a new object
        """
        coeff = self.coeff + integ.coeff # List additions
        updates = self.updates + integ.updates 
        C = ExplicitIntegrator(coeff, updates)
        return C

    def insert(self, integ, pos=None):
        """A.insert(B, pos) returns 
            C := A[:pos] B A[pos:]
            C := A1 B A2
        
        Args:
            integrator (ExplicitIntegrator): to insert into A
            pos (int): position to insert update into; defaults to the middle

        Returns:
            (ExplicitIntegrator):
        """
        if pos is None:
            pos = len(self.coeff)//2

        A1 = self.slice(pos1=0, pos2=pos)
        A2 = self.slice(pos1=pos, pos2=None)

        C = A1.append(integ)
        C = C.append(A2)
        return C

    def repeat(self, N):
        """
        A.repeat(N) returns
            C := A^N

        Args:
            N (int): number of repetitions of A

        Returns:
            (ExplicitIntegrator):
        """
        C = ExplicitIntegrator(self.coeff, self.updates)
        for i in range(N-1):
            C = C.append(self)
        return C

    def scale(self, alpha):
        """
        A.scale(c) returns C = A*c where A*c means the coefficients of A are multiplied by c

        Args:
            alpha (float): multiplying coefficient

        Returns:
            (ExplicitIntegrator):
        """
        coeff_scaled = [c*alpha for c in self.coeff]
        C = ExplicitIntegrator(coeff_scaled, self.updates)
        return C

    def slice(self, pos1=None, pos2=None):
        """A.slice(pos1, pos2) returns 
            C := A[pos1:pos2]
        
        Args:
            pos1 (int): starting position of slice (default 0)
            pos2 (int): starting position of slice (default len(integrator))

        Returns:
            (ExplicitIntegrator):
        """
        if not pos1:
            pos1 = 0
        if not pos2:
            pos2 = len(self.coeff)
        C = ExplicitIntegrator(self.coeff[pos1:pos2], self.updates[pos1:pos2])
        return C

    def reverse(self):
        """A.reverse() returns
            C := A[::-1]

        Returns:
            (ExplicitIntegrator):
        """
        C = ExplicitIntegrator(self.coeff[::-1], self.updates[::-1])
        return C

    def compose_with(self, integrator, coeffA=1.0, coeffB=1.0, N=1):
        """A.compose_with(B, cA, cB, N) returns 
            C := A*cA (B*cB/N)^N A*cA

        Args:
            integrator (ExplicitIntegrator): to compose into/between
            coeffA (float): coefficient to scale A
            coeffB (float): coefficient to scale B
            N (int): Number of times to repeat B

        Returns:
            (ExplicitIntegrator):
        """
        A = self.scale(coeffA)
        B = integrator.scale(coeffB/float(N))
        B = B.repeat(N)
        C = A.append(B)
        C = C.append(A)
        return C

    def compose_into(self, integrator, coeffA=1.0, coeffB=1.0, N=1):
        """A.compose_into(B, cA, cB, N) returns 
            C := B*cB (A*cA/N)^N B*cB

        Args:
            integrator (ExplicitIntegrator): to compose into/between
            coeffA (float): coefficient to scale A
            coeffB (float): coefficient to scale B
            N (int): Number of times to repeat B

        Returns:
            (ExplicitIntegrator):
        """
        C = ExplicitIntegrator.compose_with(integrator, self, coeffA=coeffB, coeffB=coeffA, N=N)
        return C

    def remove_null(self, eps=1.0E-10):
        """A.remove_null() returns
            C[i] := A[i] if abs(A.coeff[i]) < eps
                else A[i] is not included

        Args:
            eps (float): tolerance to remove small/negligible updates

        Returns:
            (ExplicitIntegrator): 
        """
        coeff = []
        updates = []
        for i in range(len(self.coeff)):
            if abs(self.coeff[i]) > eps:
                coeff.append(self.coeff[i])
                updates.append(self.updates[i])
        C = ExplicitIntegrator(coeff, updates)
        return C

    def squash(self, eps=1.0E-10):
        """A.squash() returns
            C[i] := A.coeff[i] + A.coeff[j] if A.update[i] == A.update[j],
            A.update[j] is not added
            Basically, all repeated updates are reduced to a single update
                and null updates are removed

        Args:
            eps (float): tolerance to remove small/negligible updates

        Returns:
            (ExplicitIntegrator): 
        """
        coeff = list(self.coeff) # copy over lists
        updates = list(self.updates)
        newlistlen = len(self.coeff)
        i = 0
        while i < newlistlen-1:
            if updates[i] is updates[i+1]:
                coeff[i] += coeff[i+1]
                del coeff[i+1]
                del updates[i+1]
                newlistlen -= 1
            else:
                i += 1

        C = ExplicitIntegrator(coeff, updates)
        C = C.remove_null(eps)
        return C

    @property
    def is_symmetric(self):
        """
        Returns:
            (bool): True if the updates and coefficients are symmetrically split
        """
        nup = len(self.updates)
        nup_check = nup // 2
        if nup % 2 == 1:
            nup_check += 1
        return all([((self.updates[i] is self.updates[-i]) and (self.coeff[i] == self.coeff[-i])) for i in range(nup_check)])

    def symmetrize(self):
        """
        If ExplicitIntegrator is not symmetric, return a symmetrized version, else return original ExplicitIntegrator

        Returns:
            ExplicitIntegrator: 
        """
        if self.is_symmetric:
            return self
        else:
            return self.append(self.reverse()).squash().scale(0.5)

    @property
    def requirements(self):
        """
        Determines what state the updates require for application of the integrator

        Returns:
            (set): union of requirements of all updates
        """
        required = set()
        for u in self.updates:
            required = set.union(required, u.requirements)
        return required

    def coeff_totals(self):
        """
        Computes the sum of coefficients of each update

        Returns:
            (list of Updates, list of coefficient sums)
        """
        update_set = []
        coeff_tots = []
        for i,u in enumerate(self.updates):
            if not u in update_set:
                update_set.append(u)
                coeff_tots.append(0.0)
            for j,u2 in enumerate(update_set):
                if u is u2:
                    coeff_tots[j] += self.coeff[i]
        return update_set, coeff_tots

    def renormalize(self):
        """
        Find sums of coefficients for all unique updates and scale sums to 1.0

        Returns:
            ExplicitIntegrator
        """
        update_set, coeff_tots = self.coeff_totals()
        coeffs = [c for c in self.coeff]
        for i,u in enumerate(self.updates):
            for j,u2 in enumerate(update_set):
                if u is u2:
                    coeffs[i] /= coeff_tots[j]
        return ExplicitIntegrator(coeffs, self.updates)

    def __repr__(self):
        update_set, coeff_tots = self.coeff_totals()

        return ("Integrator properties:"
                +   "\nUpdates:\n"
                +   str([str(u) for u in self.updates])
                +   "\n"
                +   "\nCoefficients:\n"
                +   str(self.coeff)
                +   "\n"
                +   "\nUnique Updates:\n"
                +   str([str(u) for u in update_set])
                +   "\n"
                +   "\nCoefficient Sums:\n"
                +   str(coeff_tots)
                +   "\n"
                +   "\nRequired state:\n"
                +   str(self.requirements)
                +   "\n"
                #+ "\nParamaters:\n"
                #+     str([str(u.params) for u in self.updates])
                #+     "\n"
                    )

    @staticmethod
    def _one_step_integrator(update, coeff=1.0):
        """Single update integrator, used with composition to construct standard integrators
        
        Args:
            update (Update): update to integrate
            coeff (float): magnitude of update (default 1.0)

        Returns:
            (ExplicitIntegrator): single update with given coefficient
        """
        return ExplicitIntegrator([coeff], [update])

    @staticmethod
    def _symmetric(updates):
        """
        Generic symmetric integrator, runs through updates given in order with 
        half time steps, and then backwards with half time steps
        0.5 U_0, 0.5 U_1, ... 0.5 U_{N-1}, 1.0 U_{N}, 0.5_{N-1}, ..., 0.5 U_0

        Args:
            updates (list of Update objects): updates to integrate

        Returns:
            (ExplicitIntegrator): Symmetric integrator
        """
        sym_updates = updates[:-1] + [updates[-1]] + updates[:-1][::-1]
        coeff = [0.5]*(len(updates)-1) + [1.0] + [0.5]*(len(updates) - 1)
        return ExplicitIntegrator(coeff, sym_updates)

    @staticmethod
    def _velocity_verlet(vel_update, pos_update=update.PositionUpdate()):
        """
        Standard velocity verlet integrator with steps
            0.5 V, 1.0 P, 0.5 V
        
        Args:
            vel_update (Update): used to update velocity
            pos_update (Update): used to update position (default update.PositionUpdate)

        Returns:
            (ExplicitIntegrator): Velocity Verlet integrator
        """
        return Symmetric([vel_update, pos_update])

    @staticmethod
    def _velocity_verlet_multiple(vel_updates, pos_update=update.PositionUpdate()):
        """ 
        Velocity verlet-like integrator with multiple velocity updates that are 
        all symmetrically split about the pos_update; for example the 
        Bussi-Parrinello integrator can be formulated like this.
            0.5 V_0, 0.5 V_1, ... 0.5 V_N, 1.0 P, 0.5 V_N, 0.5 V_{N-1}, ..., 0.5 V_0
        
        Args:
            vel_updates (list of Update objects): used to update velocity
            pos_update (Update): used to update position (default update.PositionUpdate)

        Returns:
            (ExplicitIntegrator): Symmetric integrator with position in the middle of all given velocity updates
        """
        return Symmetric(vel_updates + [pos_update])

    @staticmethod
    def _ruth_4th(vel_update, pos_update=update.PositionUpdate()):
        """
        Symmetric 4th order integrator by Ruth
        
        Args:
            vel_update (Update): used to update velocity
            pos_update (Update): used to update position (default update.PositionUpdate)

        Returns:
            (ExplicitIntegrator): Ruth symmetric 4th order integrator with  given velocity update

        References:
            http://www.slac.stanford.edu/cgi-wrap/getdoc/slac-pub-5071.pdf
        """
        updates = [vel_update, pos_update]*3 + [vel_update]
        ctmp = 2.0**(1.0/3.0)
        coeff = [
                1.0/(2.0*(2.0-ctmp)),
                1.0/(2.0-ctmp),
                (1.0-ctmp)/(2.0*(2.0-ctmp)),
                -ctmp/(2.0-ctmp),
                (1.0-ctmp)/(2.0*(2.0-ctmp)),
                1.0/(2.0-ctmp),
                1.0/(2.0*(2.0-ctmp)),
            ]
        return ExplicitIntegrator(coeff, updates)

    @staticmethod
    def _mclachlan_atela_4th(vel_update, pos_update=update.PositionUpdate()):
        """
        'Optimal' 4th order integrator by McLachlan and Atela
        
        Args:
            vel_update (Update): used to update velocity
            pos_update (Update): used to update position (default update.PositionUpdate)

        Returns:
            (ExplicitIntegrator): Asymmetric 4th order integrator

        References:
            doi: 10.1088/0951-7715/5/2/011
        """
        As = [
             0.5153528374311229364,
            -0.085782019412973646,
             0.4415830236164665242,
             0.1288461583643841854,
            ]
        Bs = [
             0.1344961992774310892,
             -0.2248198030794208058,
             0.7563200005156682911,
             0.3340036032863214255,
            ]
        updates = [vel_update, pos_update]*4
        coeff = []
        for a, b in zip(As, Bs):
            coeff.extend([a, b])
        return ExplicitIntegrator(coeff, updates)

    @staticmethod
    def _yoshida_suzuki_6th(vel_update, pos_update=update.PositionUpdate()):
        """
        Symmetric 6th order integrator by Yoshida and Suzuki (Solution A)
                
        Args:
            vel_update (Update): used to update velocity
            pos_update (Update): used to update position (default update.PositionUpdate)

        Returns:
            (ExplicitIntegrator): Symmetric 6th order integrator

        References:
            https://www.sciencedirect.com/science/article/pii/0375960190900923
        """
        updates = [vel_update, pos_update]*7 + [vel_update]
        As = [
             0.78451361047756,
             0.23557321335936,
            -1.1776799841789,
             1.3151863206839, 
        ]
        Bs = [
             0.39225680523878,
             0.51004341191846,
            -0.47105338540976, 
             0.068753168252520,
        ]
        coeff = []
        for i in range(4):
            coeff.append(Bs[i])
            coeff.append(As[i])
        coeff.append(Bs[3])
        for i in range(2,-1,-1):
            coeff.append(As[i])
            coeff.append(Bs[i])
        return ExplicitIntegrator(coeff, updates)

    @staticmethod
    def _adiabatic_switching(vel_update_start, vel_update_stop, time_stop, time_start=0.0):
        """
        Construct an integrator from two forces, each is time dependent.

        Args:
            vel_update_start (VelocityUpdate): initial VelocityUpdate to use
            vel_update_stop (VelocityUpdate): final VelocityUpdate to switch to
            time_stop (float): time over which to make the switch
            time_start (float): time to begin the switchover (default 0.0)
            
        Note: This integrator should be swapped out for a standard integrator after the adiabatic switching has finished.
        """
        td_update_start = update.TimeDependent(
                            vel_update_start, 
                            time_start=time_start,
                            time_stop=time_stop,
                            time_modulus=None,
                            scale_step=True,
                            scale_start=1.0,
                            scale_stop=0.0,
                            null_return={},
                            )
        td_update_stop = update.TimeDependent(
                            vel_update_stop, 
                            time_start=time_start,
                            time_stop=time_stop,
                            time_modulus=None,
                            scale_step=True,
                            scale_start=0.0,
                            scale_stop=1.0,
                            null_return={},
                            )
        integ = VelocityVerlet(td_update_stop)
        integ = integ.compose_into(OneStepIntegrator(td_update_start, coeff=0.5))
        return integ

    @staticmethod
    def _sin(pot, masses, kT, L=4, tau=10.0 * units.FS_TO_AU, damptime=10.0 * units.FS_TO_AU, nc=5, dim=3, mass_weight=False):
        """
        Construct stochastic isokinetic Nose Hoover integrator 

        Args:
            pot (Potential): Potential to integrate/sample
            masses ({nparticle,} ndarray): masses of original dofs
            kT (float): temperature in energy units
            L (int): number of auxiliary dofs per original dof (default 4)
            tau (float): 'natural timescale' to set the masses of the NHCs using Q = kT tau^2 (default 10 fs in au)
            damptime (float): rate of damping for Ornstein-Uhlenbeck/Langevin process applied to last NHC dofs (default 10 fs in au)
            nc (int): number of integration substeps for NHCs (default 5)
            dim (int): dimension of original system (default 3)
            mass_weight (bool): True to multiply Qs by ratio of particle mass / hydrogen mass (default False)

        Returns:
            (ExplicitIntegrator):

        References:
            https://www.tandfonline.com/doi/abs/10.1080/00268976.2013.844369
            https://pubs.acs.org/doi/10.1021/acs.jctc.6b00188

        Notes:
            SIN samples the canonical distribution in configuration space ONLY and NOT in momentum/velocity space, so the standard definition of temperature may be very misleading.
        """
        return SIN_RESPA([pot], [], masses, kT, L=L, tau=tau, damptime=damptime, nc=nc, dim=dim, mass_weight=mass_weight)

    @staticmethod
    def _respa(vel_updates, Ns, inner_integ=None, thermostat_update=None, XI=True):
        """
        Construct a RESPA integrator from multiple VelocityUpdate objects and with given thermostat Update object

        Args:
            vel_updates ({nup,} list of VelocityUpdate objects): ordered from lowest frequency to highest frequency
            Ns ({nup-1,} list of ints): number of inner steps of each inner potential
            inner_integ (ExplicitIntegrator): Inner-most integration without VelocityUpdate (default PositionUpdate)
            XI (bool): True to place thermostat update in intermediate RESPA time scales, see references (default True)

        Returns:
            (ExplicitIntegrator):

        References:
            https://pubs.acs.org/doi/10.1021/acs.jctc.6b00188
        """
        if len(Ns)+1 != len(vel_updates):
            raise ValueError("The length of Ns should always be one shorter than that of vel_updates.")

        # Start with inner most update integration
        if inner_integ is not None:
            integ = inner_integ
        else:
            integ = OneStepIntegrator(update.PositionUpdate())
        Ns_tot = [1] + Ns
        Nprod = np.prod(Ns_tot)
        coeff_scale_tot = 0.5 / Nprod
        coeff_scale = coeff_scale_tot
        integ = integ.scale(1.0/Nprod)
        do_XI = ((thermostat_update is not None) and XI)
        do_XO = ((thermostat_update is not None) and (not XI))
        
        # Loop through the updates in reverse order
        for n, vu in zip(Ns_tot[::-1], vel_updates[::-1]):
            
            # Integrate forward steps
            integ = integ.compose_into(OneStepIntegrator(vu, coeff=coeff_scale))
            if do_XI:
                integ = integ.compose_into(OneStepIntegrator(thermostat_update, coeff=coeff_scale_tot))
            # Repeat integration n times, scale to preserve step size
            integ = integ.repeat(n)
            if do_XI:
                integ = integ.compose_into(OneStepIntegrator(thermostat_update, coeff=-coeff_scale_tot))
            coeff_scale *= n
            
        # Lastly make sure the outer NHC update is applied properly
        if do_XI:
            integ = integ.compose_into(OneStepIntegrator(thermostat_update, coeff=coeff_scale_tot))
        elif do_XO:
            integ = integ.compose_into(OneStepIntegrator(thermostat_update, coeff=0.5))
        integ = integ.squash()
        return integ

    @staticmethod
    def _nhc_respa(potentials, Ns, masses, kT, tau=10.0 * units.FS_TO_AU, chain_length=5, nc=5, dim=3, mass_weight=True, XI=True):
        """
        Construct a RESPA integrator from multiple potentials with NHC thermostat
        Args:
            potentials ({npot,} list of Potential objects): ordered from lowest frequency to highest frequency
            Ns ({npot-1,} list of ints): number of inner steps of each inner potential
            masses ({nparticle,} ndarray): masses of original dofs
            kT (float): temperature in energy units
            tau (float): 'natural timescale' to set the masses of the NHCs using Q = kT tau^2 (default 10 fs in au)
            chain_length (int): length of NHC per atom (default 5)
            nc (int): number of integration substeps for NHCs (default 5)
            dim (int): dimension of original system (default 3)
            mass_weight (bool): True to multiply Qs by ratio of particle mass / hydrogen mass (default True)
            XI (bool): True to update NHC dofs in intermediate RESPA time scales, see references (default True)

        References:
            https://pubs.acs.org/doi/10.1021/acs.jctc.6b00188
        """
        if len(Ns)+1 != len(potentials):
            raise ValueError("The length of Ns should always be one shorter than that of potentials.")

        # Start with innermost update integration
        vel_updates = [update.VelocityUpdate(p, masses, name=p.name) for p in potentials]
        pos_up = update.PositionUpdate()
        nhc_up = update.NoseHooverNVT.build(masses, kT, tau=tau, chain_length=chain_length, nc=nc, dim=dim, mass_weight=mass_weight)
        integ = OneStepIntegrator(pos_up)
        return RESPA(vel_updates, Ns, inner_integ=integ, thermostat_update=nhc_up, XI=XI)

    @staticmethod
    def _sin_respa(potentials, Ns, masses, kT, L=4, tau=10.0 * units.FS_TO_AU, damptime=10.0 * units.FS_TO_AU, nc=5, dim=3, mass_weight=False, XI=True):
        """
        Construct a RESPA integrator from multiple potentials with a stochastic isokinetic Nose-Hoover (SIN) thermostat

        Args:
            potentials ({npot,} list of Potential objects): ordered from lowest frequency to highest frequency
            Ns ({npot-1,} list of ints): number of inner steps of each inner potential
            masses ({nparticle,} ndarray): masses of original dofs
            kT (float): temperature in energy units
            L (int): number of auxiliary dofs per original dof (default 4)
            tau (float): 'natural timescale' to set the masses of the NHCs using Q = kT tau^2 (default 10 fs in au)
            damptime (float): rate of damping for Ornstein-Uhlenbeck/Langevin process applied to last NHC dofs (default 10 fs in au)
            nc (int): number of integration substeps for NHCs (default 5)
            dim (int): dimension of original system (default 3)
            mass_weight (bool): True to multiply Qs by ratio of particle mass / hydrogen mass (default False)
            XI (bool): True to update NHC dofs in intermediate RESPA time scales, see references (default True)

        References:
            https://www.tandfonline.com/doi/abs/10.1080/00268976.2013.844369
            https://pubs.acs.org/doi/10.1021/acs.jctc.6b00188

        Notes:
            SIN samples the canonical distribution in configuration space ONLY and NOT in momentum/velocity space, so the standard definition of temperature may be very misleading.
        """
        if len(Ns)+1 != len(potentials):
            raise ValueError("The length of Ns should always be one shorter than that of potentials.")

        # Construct Update objects
        vel_updates = [update.IsokineticVelocityUpdate(p, masses, kT, nhc=True, name=p.name) for p in potentials]
        pos_up = update.PositionUpdate()
        isok_nhc_up = update.IsokineticNoseHoover.build(masses, kT, L=L, tau=tau, nc=nc, dim=dim, mass_weight=mass_weight)
        Qs = isok_nhc_up.params['Qs']
        ou_nhc_up = update.NoseHooverLangevin(kT, Qs, damptime=damptime)
        integ = Symmetric([pos_up, ou_nhc_up])
        return RESPA(vel_updates, Ns, inner_integ=integ, thermostat_update=isok_nhc_up, XI=XI)

OneStepIntegrator = ExplicitIntegrator._one_step_integrator
Symmetric = ExplicitIntegrator._symmetric
VelocityVerlet = ExplicitIntegrator._velocity_verlet
VelocityVerletMultiple = ExplicitIntegrator._velocity_verlet_multiple
Ruth4th = ExplicitIntegrator._ruth_4th
McLachlanAtela4th = ExplicitIntegrator._mclachlan_atela_4th
YoshidaSuzuki6th = ExplicitIntegrator._yoshida_suzuki_6th
AdiabaticSwitching = ExplicitIntegrator._adiabatic_switching
SIN = ExplicitIntegrator._sin
RESPA = ExplicitIntegrator._respa
NHC_RESPA = ExplicitIntegrator._nhc_respa
SIN_RESPA = ExplicitIntegrator._sin_respa
