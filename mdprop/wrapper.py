import numpy as np
import os
import datetime
import contextlib
import subprocess
import glob
import shutil

import ase
from ase.calculators.dftb import Dftb
    
from . import data, utils, units, potential

class Wrapper(potential.Potential):
    """
    Wrappers for other codes that provide potential energies.
    """
    def __init__(self):
        super(Wrapper, self).__init__()    

    def connect(self):
        """
        Create folder for running calculation.

        This function can be overwritten in inheriting classes as necessary.
        """
        self.curr_dir = os.getcwd()
        dir_name = self.__class__.__name__.lower() + "_" + str(datetime.datetime.now()).replace(" ", "_")
        os.mkdir(dir_name)
        self.working_dir = os.path.abspath(dir_name)
        self.job_count = 0

    @contextlib.contextmanager
    def run_in_job_dir(self):
        """A simple context manager which switches into the specified job
        directory, run enclosed commands then switch back.  Context managers
        ensure that whatever happens (exit, return, exceptions, etc), the
        working directory will be switched back

        This is to alleviate the pain caused by ASE writing input and output
        files directly in the current directory.  Wrap any code that generate
        I/O with the following with block:

            with self.run_in_job_dir():
                ...
        """
        try:
            self.job_count += 1
            self.job_dir = os.path.join(self.working_dir, "job_%d" % self.job_count)
            os.mkdir(self.job_dir)
            os.chdir(self.job_dir)
            yield
        finally:
            if self.job_count > 4:
                del_job_dir = os.path.join(self.working_dir, "job_%d" % (self.job_count - 4))
                shutil.rmtree(del_job_dir)
            os.chdir(self.curr_dir)

    def disconnect(self):
        """
        Function called to cleanup for use with 'with' statement, can be 
        overwritten if certain cleanup functionality is desired.
        """
        return         

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.disconnect()

class DFTBPlus(Wrapper):
    """
    This is a wrapper of ASE DFTB+ (a wrapper of a wrapper...) to compute
    single point energies/gradients/forces.
    """
    default_HD = {
        'Br' : -0.0573,  
        'C'  : -0.1492,  
        'Ca' : -0.0340,  
        'Cl' : -0.0697,  
        'F'  : -0.1623,  
        'H'  : -0.1857,  
        'I'  : -0.0433,  
        'K'  : -0.0339,      
        'Mg' : -0.0200,
        'N'  : -0.1535,
        'Na' : -0.0454,
        'O'  : -0.1575,
        'P'  : -0.1400,
        'S'  : -0.1100,
        'Zn' : -0.0300, 
    }
    default_AM = {
        'Br' : '"d"',   
        'C'  : '"p"',   
        'Ca' : '"p"',   
        'Cl' : '"d"',   
        'F'  : '"p"',   
        'H'  : '"s"',   
        'I'  : '"d"',   
        'K'  : '"p"',       
        'Mg' : '"p"',
        'N'  : '"p"',
        'Na' : '"p"',
        'O'  : '"p"',
        'P'  : '"p"',
        'S'  : '"d"',
        'Zn' : '"d"',
    }
    def __init__(self, symbols, **dftb_opts):
        self.reset(symbols, **dftb_opts)
        super(DFTBPlus, self).__init__()    

    def reset(self, symbols, **dftb_opts):
        """
        Give the wrapper a new molecule and options.
        """
        self.ase_mol = ase.Atoms(symbols=symbols, pbc=False)
        mol_atoms = set(symbols)
        thirdorder_opt = dftb_opts.get('Hamiltonian_ThirdOrderFull', None) or dftb_opts.get('Hamiltonian_ThirdOrder', None)
        for atom_sym in mol_atoms:
            dftb_opts['Hamiltonian_MaxAngularMomentum_' + atom_sym] = DFTBPlus.default_AM[atom_sym]
            if thirdorder_opt is "Yes" or thirdorder_opt is "YES":
                dftb_opts['Hamiltonian_HubbardDerivs_'] = ''
                if not ('Hamiltonian_HubbardDerivs_' + atom_sym) in dftb_opts:
                    dftb_opts['Hamiltonian_HubbardDerivs_' + atom_sym] = DFTBPlus.default_HD[atom_sym]

        self.engine = Dftb(
                          atoms=self.ase_mol,
                          run_manyDftb_steps=False,
                          Hamiltonian_MaxAngularMomentum_='',
                          **dftb_opts
                          )

        self.ase_mol.set_calculator(self.engine)
        self.dftb_opts = dftb_opts

    def update_charges(self):
        scc_opt = self.dftb_opts.get("Hamiltonian_SCC", None)
        if scc_opt == "YES" or scc_opt == "Yes":
            charges = self.engine.results['charges']
            charge_str = "\n\t\t".join([str(c) for c in charges])
            charge_dict = {'Hamiltonian_InitialCharges_AllAtomCharges_empty' : charge_str}
            charge_dict['Hamiltonian_InitialCharges_'] = ''
            charge_dict['Hamiltonian_InitialCharges_AllAtomCharges_'] = ''
            self.engine.set(**charge_dict)

    def compute_energy(self, X):
        with self.run_in_job_dir():
            # Positions must be set directly to Angstroms, ASE writes them in Bohr under the hood
            self.ase_mol.set_positions(X / units.ANGSTROM_TO_AU) 
            # ASE converts DFTB+ Hartree to eV so we convert back to Hartree
            pe = self.ase_mol.get_potential_energy() / ase.units.Hartree
            self.update_charges()
        return pe

    def compute_force(self, X):
        with self.run_in_job_dir():
            # Positions must be set directly to Angstroms, ASE writes them in Bohr under the hood
            self.ase_mol.set_positions(X / units.ANGSTROM_TO_AU) 
            # ASE converts DFTB+ Hartree to eV so we convert back to Hartree
            pe = self.ase_mol.get_potential_energy() / ase.units.Hartree
            # ASE converts DFTB+ Hartree/Bohr to eV/Angstrom so we convert back to Hartree/Bohr
            force = self.ase_mol.get_forces() / ase.units.Hartree * ase.units.Bohr
            self.update_charges()
        return pe, force

    def compute_gradient(self, X):
        pe, f = self.compute_force(X)
        return pe, -f

class ReaxFF(Wrapper):
    def __init__(self, 
            symbols, 
            lammps_exec, 
            ffieldfile, 
            inputfile="lammps.in", 
            datafile="lammps.data", 
            dumpfile="lammps.dump", 
            outputfile="lammps.out", 
            bondfile="lammps.bonds",
            neb_coord_file="coord.initial",
            etol=0.0,
            ftol=0.36,
            neb_iter=100000,
            cineb_iter=50000,
            dump_freq=50,
            ):
        """
        Note: masses correspond to unique atoms and are mostly included for 
        completeness when writing the data file. None will result in looking up 
        the most frequent isotopic mass in AMU.

        Args:
            symbols: atomic symbols to be used
            lammps_exec: command to be subprocessed to compute the lammps forces
            ffieldfile: path to reaxFF forcefield parameter file
            masses: masses to use for given computation, included for completeness
            inputfile: name of lammps input
            datafile: name of lammps data file
            dumpfile: name of lammps dump file
            outputfile: name of lammps output file
        """
        self.symbols = symbols
        self.unique_atoms = sorted(list(set(symbols)))
        self.masses = utils.symbol_to_mass(self.unique_atoms) / units.AMU_TO_AU
        self.inputfile = inputfile
        self.datafile = datafile
        self.ffieldfile = ffieldfile
        self.dumpfile = dumpfile
        self.outputfile = outputfile
        self.bondfile = bondfile
        self.neb_coord_file = neb_coord_file
        self.etol = etol
        self.ftol = ftol
        self.neb_iter = neb_iter
        self.cineb_iter = cineb_iter
        self.dump_freq = dump_freq
        self.lammps_exec = lammps_exec.split()
        self.result = {}
        super(ReaxFF, self).__init__()    

    def get_bound_box(self, Xs):
        Xarray = np.array(Xs)
        return [
                np.min(Xarray[..., 0]), np.max(Xarray[..., 0]), 
                np.min(Xarray[..., 1]), np.max(Xarray[..., 1]), 
                np.min(Xarray[..., 2]), np.max(Xarray[..., 2]), 
                ]

    def write_data(self, X, box=None, datafile=None):
        if datafile is None:
            datafile = self.datafile
        Xconv = X / units.ANGSTROM_TO_AU
        if box is None:
            box = self.get_bound_box(Xconv)
        
        with open(datafile, "w") as fout:
            fout.write("\n#Units in Angstroms and grams/mole for masses")
            fout.write("\n%d atoms\n%d atom types\n" % (X.shape[0], len(self.unique_atoms)))
            fout.write("\n%.3f %.3f xlo xhi" % (round(box[0]) - 1.0, round(box[1]) + 1.0))
            fout.write("\n%.3f %.3f ylo yhi" % (round(box[2]) - 1.0, round(box[3]) + 1.0))
            fout.write("\n%.3f %.3f zlo zhi\n" % (round(box[4]) - 1.0, round(box[5]) + 1.0))
            fout.write("\nMasses\n\n")
            for i, ua in enumerate(self.unique_atoms):
                # "real" units use grams/mole, equivalent to AMU
                fout.write("%d\t%.9f\n" % (i + 1, self.masses[i]))
            fout.write("\nAtoms\n\n")
            for a in range(X.shape[0]):
                fout.write("%3d %3d   0.0\t % .11E\t % .11E\t % .11E\n" %
                  (a+1, self.unique_atoms.index(self.symbols[a])+1, Xconv[a, 0], Xconv[a, 1], Xconv[a, 2]))

    def write_input(self):
        with open(self.inputfile, 'w') as fout:
            fout.write(
                      "# File auto-generated via ReaxFF wrapper code in mdprop python package\n"
                    + "units real\n" 
                    + "atom_style charge\n"
                    + "boundary s s s\n"
                    + "read_data %s\n" % self.datafile
                    + "pair_style reax/c NULL safezone 10.0 mincap 1000\n" 
                    + "pair_coeff * * %s %s\n" % (self.ffieldfile, " ".join([a for a in self.unique_atoms]))
                    + "neighbor 2.0 nsq\n"
                    + "neigh_modify every 1 delay 0 check no\n"
                    + "compute reax all pair reax/c\n"
                    + "fix             10 all qeq/reax 1 0.0 10.0 1.0e-6 reax/c\n"
                    + "fix 97 all reax/c/bonds 1 %s\n" % self.bondfile
                    + "compute 2 all property/atom fx fy fz\n"
                    + "dump 15   all custom  100 %s fx fy fz\n" % self.dumpfile
                    + 'dump_modify   15   format line "%.11E %.11E %.11E" sort id\n'
                    + "run	0\n"
                    )

    def write_minimization_input(self):
        with open(self.inputfile, 'w') as fout:
            fout.write(
                      "# File auto-generated via ReaxFF wrapper code in mdprop python package\n"
                    + "units real\n" 
                    + "atom_style charge\n"
                    + "boundary s s s\n"
                    + "read_data %s\n" % self.datafile
                    + "pair_style reax/c NULL safezone 10.0 mincap 1000\n" 
                    + "pair_coeff * * %s %s\n" % (self.ffieldfile, " ".join([a for a in self.unique_atoms]))
                    + "neighbor 2.0 nsq\n"
                    + "neigh_modify every 1 delay 0 check no\n"
                    + "compute reax all pair reax/c\n"
                    + "fix             10 all qeq/reax 1 0.0 10.0 1.0e-6 reax/c\n"
                    + "thermo 1\n"
                    + "dump 15   all custom  100 %s x y z fx fy fz\n" % self.dumpfile
                    + "dump_modify   15   format line \"%.11E %.11E %.11E %.11E %.11E %.11E\" sort id\n"
                    + "minimize 1.0e-6 1.0e-6 5000 10000\n"
                    + "run	0\n"
                    )

    def write_neb_input(self):
        with open(self.inputfile, 'w') as fout:
            fout.write(
                      "# File auto-generated via ReaxFF wrapper code in mdprop python package\n"
                    + "units real\n" 
                    + "atom_style charge\n"
                    + "atom_modify map yes\n"
                    + "boundary s s s\n"
                    + "read_data %s\n" % self.datafile
                    + "pair_style reax/c NULL safezone 10.0 mincap 1000\n" 
                    + "pair_coeff * * %s %s\n" % (self.ffieldfile, " ".join([a for a in self.unique_atoms]))
                    + "neighbor 2.0 nsq\n"
                    + "neigh_modify every 1 delay 0 check no\n"
                    + "compute reax all pair reax/c\n"
                    + "variable i equal part\n"
                    + "fix 10 all qeq/reax 1 0.0 10.0 1.0e-6 reax/c\n"
                    + "thermo 1\n"
                    + "dump 15 all custom  100 %s.$i x y z fx fy fz\n" % self.dumpfile
                    + "dump_modify 15 format line \"%.11E %.11E %.11E %.11E %.11E %.11E\" sort id\n"
                    + "min_style quickmin\n"
                    + "fix 1 all neb 0.1\n"
                    + "neb %f %f %d %d %d each %s.$i\n" % (self.etol, self.ftol, self.neb_iter, self.cineb_iter, self.dump_freq, self.neb_coord_file)
                    + "run	0\n"
                    )

    def write_neb_coord(self, X, coordfile):
        Xconv = X / units.ANGSTROM_TO_AU
        with open(coordfile, "w") as fout:
            N = X.shape[0]
            fout.write("%d\n" % N)
            for a in range(N):
                fout.write("%3d\t % .11E\t % .11E\t % .11E\n" %
                  (a+1, Xconv[a, 0], Xconv[a, 1], Xconv[a, 2]))

    def write_neb_data(self, Xs):
        traj_box = self.get_bound_box(np.array(Xs) / units.ANGSTROM_TO_AU)
        self.write_data(Xs[0], box=traj_box, datafile=self.datafile)
        for i, X in enumerate(Xs):
            coordfile = self.neb_coord_file + ".%d" % i
            self.write_neb_coord(X, coordfile)

    def parse_dump(self):
        with open(self.outputfile, 'r') as fin:
            lines = fin.readlines()
        pe = None
        for i, line in enumerate(lines):
            if line[:9] == "Step Temp":
                pe = float(lines[i+1].split()[2]) * units.KCAL_MOL_TO_AU
                break
        with open(self.dumpfile, 'r') as fin:
            lines = fin.readlines()
        output = np.array([l.split() for l in lines[9:]], dtype=np.float64)
        force = output * units.KCAL_MOL_TO_AU / units.ANGSTROM_TO_AU # Convert to a.u. from real
        return pe, force

    def parse_minimization_dump(self):
        with open(self.outputfile, 'r') as fin:
            lines = fin.readlines()
        pe = None
        begin_line = None
        end_line = None
        for i, line in enumerate(lines):
            if line[:9] == "Step Temp" and begin_line is None:
                begin_line = i+1
            elif line[:9] == "Loop time" and begin_line is not None:
                end_line = i
                break
        pe = np.array([lines[i].split()[2] for i in range(begin_line, end_line)], dtype=np.float64) * units.KCAL_MOL_TO_AU

        with open(self.dumpfile, 'r') as fin:
            lines = fin.readlines()
        nframes = len(pe)
        frame_len = 9 + len(self.symbols)
        output = []
        for i in range(nframes):
            output.append([l.split() for l in lines[frame_len*i + 9:frame_len*(i+1)]])
        output_np = np.array(output)
        Xs = output_np[:, :, :3] * units.ANGSTROM_TO_AU # Convert to a.u. from real
        force = output_np[:, :, 3:] * units.KCAL_MOL_TO_AU / units.ANGSTROM_TO_AU # Convert to a.u. from real
        return pe, Xs, forces

    def parse_neb_dump(self, dumpfiles=None):
        """
        Strip dump files to collect the geometry and gradients of the final frames
        """
        Natom = None
        Nframe = 0
        Xs = []
        Gs = []
        if dumpfiles is None:
            dumpfiles = glob.glob(self.dumpfile + ".*")
        for curr_dump in dumpfiles:
            Nframe += 1
            with open(curr_dump, 'r') as fin:
                lines = fin.readlines()
            if Natom is None:
                Natom = int(lines[3])
            Xcurr = np.array([l.split() for l in lines[-Natom:]], dtype=np.float64)
            Xs.append(Xcurr[:, :3])
            Gs.append(Xcurr[:, 3:])
        Xs =  np.array(Xs) * units.ANGSTROM_TO_AU 
        Gs = -np.array(Gs) * units.KCAL_MOL_TO_AU / units.ANGSTROM_TO_AU
        return Xs, Gs

    def parse_bond_order(self):
        """Parse the ReaxFF/C bond order information, saved in self.bondfile"""
        natom = len(self.symbols)
        bond_order_matrix = np.zeros((natom, natom))
        n_lone_pair = np.zeros(natom)
        with open(self.bondfile) as fin:
            for line in fin:
                if line.startswith('#'):
                    continue
                data = line.split()
                iatom = int(data[0]) - 1
                nbond = int(data[2])
                n_lone_pair[iatom] = float(data[-2])
                for ibond in range(nbond):
                    iatom2 = int(data[3 + ibond]) - 1
                    bond_order = float(data[4 + nbond + ibond])
                    bond_order_matrix[iatom, iatom2] = bond_order
        return bond_order_matrix, n_lone_pair

    def run_job(self, X):
        with self.run_in_job_dir():
            self.write_data(X)
            self.write_input()
            with open(self.inputfile, 'r') as fin, open(self.outputfile, 'w') as fout:
                subprocess.call(self.lammps_exec, stdin=fin, stdout=fout)
            self.result['potential_energy'], self.result['force'] = self.parse_dump()
            self.result['gradient'] = -self.result['force']
            self.result['bond_order'], self.result['lone_pairs'] = self.parse_bond_order()
        return self.result

    def run_neb(self, Xs):
        with self.run_in_job_dir():
            self.write_neb_data(Xs)
            self.write_neb_input()
            sub_command = ("mpirun -np %d " % len(Xs)).split() + self.lammps_exec + ("-partition %dx1" % len(Xs)).split()
            with open(self.inputfile, 'r') as fin, open(self.outputfile, 'w') as fout:
                subprocess.call(sub_command, stdin=fin, stdout=fout)

    def compute_energy(self, X):
        res = self.run_job(X)
        return res['potential_energy']

    def compute_force(self, X):
        res = self.run_job(X)
        return res['potential_energy'], res['force']

    def compute_gradient(self, X):
        pe, force = self.compute_force(X)
        return pe, -force

    def compute_bond_order(self, X):
        res = self.run_job(X)
        return res['bond_order']

    def compute_lone_pairs(self, X):
        res = self.run_job(X)
        return res['lone_pairs']

    def compute_all(self, X):
        results = self.run_job(X)
        return results
