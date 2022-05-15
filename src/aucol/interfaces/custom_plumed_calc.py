from ase.calculators.calculator import Calculator, all_changes
from ase.io.trajectory import Trajectory
from ase.parallel import broadcast
from ase.parallel import world
import numpy as np
from ase.units import kJ, mol, fs, kB, GPa, nm
from os.path import exists
import torch

from ase import *


#This is a modified calculator from the ASE project. Clone has beed made to suport
#the version of PLUMED with engine CVs.

def restart_from_trajectory(prev_traj, *args, prev_steps=None, atoms=None, **kwargs):
    atoms.calc = Plumed(*args, atoms=atoms, restart=True, **kwargs)

    """This function helps the user to restart a plumed simulation
    from a trajectory file.

    Parameters
        ----------
        calc: Calculator object
            It  computes the unbiased forces

    .. note:: As alternative for restarting a plumed simulation, the user
            has to fix the positions, momenta and Plumed.istep
    """
    with Trajectory(prev_traj) as traj:
        if prev_steps is None:
            atoms.calc.istep = len(traj) - 1
        else:
            atoms.calc.istep = prev_steps
        atoms.set_positions(traj[-1].get_positions())
        atoms.set_momenta(traj[-1].get_momenta())
    return atoms.calc


class Plumed(Calculator):
    """Plumed calculator is used for simulations of enhanced sampling methods
    with the open-source code PLUMED (plumed.org).


    [1] The PLUMED consortium, Nat. Methods 16, 670 (2019)
    [2] Tribello, Bonomi, Branduardi, Camilloni, and Bussi,
    Comput. Phys. Commun. 185, 604 (2014).
    """

    implemented_properties = ['energy', 'forces']

    def __init__(self, calc, input, timestep, cv_model=None, atoms=None, kT=1., log='', restart=False):
        """
        Plumed calculator is used for simulations of enhanced sampling methods
        with the open-source code PLUMED (plumed.org).

        [1] The PLUMED consortium, Nat. Methods 16, 670 (2019)
        [2] Tribello, Bonomi, Branduardi, Camilloni, and Bussi,
        Comput. Phys. Commun. 185, 604 (2014)

        Parameters
        ----------
        calc: Calculator object
            It  computes the unbiased forces and CVs

        input: List of strings
            It contains the setup of plumed actions

        timestep: float
            Timestep of the simulated dynamics

        atoms: Atoms
            Atoms object to be attached


        .. note:: For this case, the calculator is defined strictly with the
            object atoms inside. This is necessary for initializing the
            Plumed object. For conserving ASE convention, it can be initialized as
            atoms.calc = (..., atoms=atoms, ...)

        kT: float. Default 1.
            Value of the thermal energy in eV units. It is important for
            some of the methods of plumed like Well-Tempered Metadynamics.

        log: string
            Log file of the plumed calculations

        restart: boolean. Default False
            True if the simulation is restarted.


        .. note:: In order to guarantee a well restart, the user has to fix momenta,
            positions and Plumed.istep, where the positions and momenta corresponds
            to the last coniguration in the previous simulation, while Plumed.istep
            is the number of timesteps performed previously. This can be done
            using ase.calculators.plumed.restart_from_trajectory.

        """
        from plumed import Plumed as pl

        if atoms is None:
            raise TypeError('plumed calculator has to be defined with the object atoms inside.')

        self.istep = 0
        Calculator.__init__(self, atoms=atoms)

        self.input = input
        self.calc = calc
        self.cv_model = cv_model

        self.name = '{}+Plumed'.format(self.calc.name)

        if world.rank == 0:
            natoms = len(atoms.get_positions())

            self.plumed = pl()

            #Properly set units!!!
            ps = 1000*fs
            self.plumed.cmd("setMDEnergyUnits", mol/kJ) # EV to kjoule/mol
            self.plumed.cmd("setMDLengthUnits", 1/nm) # Angstrom to nm
            self.plumed.cmd("setMDTimeUnits", 1/ps) #ase time units to ps

            self.plumed.cmd("setNatoms", natoms)
            self.plumed.cmd("setMDEngine", "ASE")
            self.plumed.cmd("setLogFile", log)
            self.plumed.cmd("setTimestep", float(timestep))
            self.plumed.cmd("setRestart", restart)
            self.plumed.cmd("setKbT", float(kT))
            self.plumed.cmd("init")
            if input is None:
                self.plumed.cmd("read")
            else:
                for line in input:
                    self.plumed.cmd("readInputLine", line)

        self.atoms = atoms


    def calculate(self, atoms=None, properties=['energy', 'forces', 'cv', 'cv_grad'], system_changes=all_changes):
        #Calculator.calculate(self, atoms, properties, system_changes)
        Calculator.calculate(self, atoms, ['energy', 'forces', 'cv', 'cv_grad'], system_changes)
        #We use internally calculated CVs
        energy, forces = self.compute_energy_and_forces(atoms, self.atoms.get_positions(), self.istep)
        self.istep += 1
        self.results['energy'], self. results['forces'] = energy, forces

    def compute_energy_and_forces(self, atoms, pos, istep):
        if "cv" in self.calc.implemented_properties:
            cvs = self.calc.get_property('cv', atoms=self.atoms)
            grads = self.calc.get_property('cv_grad', atoms=self.atoms)
        elif self.cv_model is not None:
            cvs, grads = self.cv_model.cvs_with_grad(atoms)
        else:
            raise NotImplementedError("Calculator do not implement CV and no CV model provided")

        if world.rank == 0:
            ener_forc = self.compute_bias(pos, istep, cvs, grads)
        else:
            ener_forc = None
        energy_bias, forces_bias = broadcast(ener_forc)
        energy = self.calc.get_potential_energy(self.atoms) + energy_bias
        forces = self.calc.get_forces(self.atoms) + forces_bias
        return energy, forces

    def compute_bias(self, pos, istep, cvs, grads):
        self.plumed.cmd("setStep", istep)
        self.plumed.cmd("setPositions", pos)

        self.plumed.cmd("setMasses", self.atoms.get_masses())
        forces_bias = np.zeros((self.atoms.get_positions()).shape, dtype=np.float64)
        self.plumed.cmd("setForces", forces_bias)
        virial = np.zeros((3, 3))
        self.plumed.cmd("setVirial", virial)

        #Now pass the calculated value to Plumed
        self.cvs = []
        self.grads = []
        for i, (cv, grad) in enumerate(zip(cvs, grads)):
            self.cvs.append(np.array(cv, dtype=np.float64, copy=True))
            self.grads.append(nm*np.array(grad, dtype=np.float64, copy=True))
            self.plumed.cmd("setExtraCV cv"+str(i+1), self.cvs[i])
            self.plumed.cmd("setExtraCVDerivative cv"+str(i+1), self.grads[i])

        #print(forces_bias)
        self.plumed.cmd("prepareCalc")
        self.plumed.cmd("performCalc")
        energy_bias = np.zeros((1,))
        self.plumed.cmd("getBias", energy_bias)
        #print(forces_bias)
        return [energy_bias, forces_bias]

    def write_plumed_files(self, images):
        """ This function computes what is required in
        plumed input for some trajectory.

        The outputs are saved in the typical files of
        plumed such as COLVAR, HILLS """
        for i, image in enumerate(images):
            pos = image.get_positions()
            self.compute_energy_and_forces(pos, i)
        return self.read_plumed_files()

    def read_plumed_files(self, file_name=None):
        read_files = {}
        if file_name is not None:
            read_files[file_name] = np.loadtxt(file_name, unpack=True)
        else:
            for line in self.input:
                if line.find('FILE') != -1:
                    ini = line.find('FILE')
                    end = line.find(' ', ini)
                    if end == -1:
                        file_name = line[ini+5:]
                    else:
                        file_name = line[ini+5:end]
                    read_files[file_name] = np.loadtxt(file_name, unpack=True)

            if len(read_files) == 0:
                if exists('COLVAR'):
                    read_files['COLVAR'] = np.loadtxt('COLVAR', unpack=True)
                if exists('HILLS'):
                    read_files['HILLS'] = np.loadtxt('HILLS', unpack=True)

        assert not len(read_files) == 0, "There are not files for reading"
        return read_files

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.plumed.finalize()
