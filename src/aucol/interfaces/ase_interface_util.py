import ase.io
import ase.build

from schnetpack import Properties
from schnetpack.environment import SimpleEnvironmentProvider
from schnetpack.data.atoms import AtomsConverter
from schnetpack.md.utils import MDUnits

from ase.calculators.calculator import Calculator, all_changes

import numpy as np


class CVNNPCalculator(Exception):
    pass

class CVNNPCalculator(Calculator):
    """
    ASE calculator for schnetpack with a collective variable propery.

    Args:
        ml_model (schnetpack.AtomisticModel): Trained model for
            calculations
        device (str): select to run calculations on 'cuda' or 'cpu'
        collect_triples (bool): Set to True if angular features are needed,
            for example, while using 'wascf' models
        environment_provider (callable): Provides neighbor lists
        pair_provider (callable): Provides list of neighbor pairs. Only
            required if angular descriptors are used. Default is none.
        **kwargs: Additional arguments for basic ase calculator class
    """

    energy = Properties.energy
    forces = Properties.forces
    stress = Properties.stress
    cv = "cv"
    cv_grad = "cv_grad"
    implemented_properties = [energy, forces, stress, cv, cv_grad]

    def __init__(
        self,
        model,
        device="cpu",
        collect_triples=False,
        environment_provider=SimpleEnvironmentProvider(),
        energy=None,
        forces=None,
        stress=None,
        cv=None,
        cv_grad=None,
        energy_units="eV",
        forces_units="eV/Angstrom",
        stress_units="eV/Angstrom/Angstrom/Angstrom",
        cv_units="",
        cv_grad_units="A",
        **kwargs
    ):
        Calculator.__init__(self, **kwargs)

        self.model = model
        self.model.to(device)

        self.atoms_converter = AtomsConverter(
            environment_provider=environment_provider,
            collect_triples=collect_triples,
            device=device,
        )

        self.model_energy = energy
        self.model_forces = forces
        self.model_stress = stress
        self.model_cv = cv
        self.model_cv_grad = cv_grad

        # Convert to ASE internal units (energy=eV, length=A)
        self.energy_units = MDUnits.unit2unit(energy_units, "eV")
        self.forces_units = MDUnits.unit2unit(forces_units, "eV/Angstrom")
        self.stress_units = MDUnits.unit2unit(stress_units, "eV/A/A/A")
        self.cv_units = MDUnits.unit2unit(stress_units, "")
        self.cv_grad_units = MDUnits.unit2unit(stress_units, "A")

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
        """
        Args:
            atoms (ase.Atoms): ASE atoms object.
            properties (list of str): do not use this, no functionality
            system_changes (list of str): List of changes for ASE.
        """
        # First call original calculator to set atoms attribute
        # (see https://wiki.fysik.dtu.dk/ase/_modules/ase/calculators/calculator.html#Calculator)

        if self.calculation_required(atoms, properties):
            Calculator.calculate(self, atoms)
            # Convert to schnetpack input format
            model_inputs = self.atoms_converter(atoms)
            # Call model
            model_results = self.model(model_inputs)

            results = {}
            # Convert outputs to calculator format
            if self.model_energy is not None:
                if self.model_energy not in model_results.keys():
                    raise SpkCalculatorError(
                        "'{}' is not a property of your model. Please "
                        "check the model "
                        "properties!".format(self.model_energy)
                    )
                energy = model_results[self.model_energy].cpu().data.numpy()
                results[self.energy] = (
                    energy.item() * self.energy_units
                )  # ase calculator should return scalar energy

            if self.model_forces is not None:
                if self.model_forces not in model_results.keys():
                    raise SpkCalculatorError(
                        "'{}' is not a property of your model. Please "
                        "check the model"
                        "properties!".format(self.model_forces)
                    )
                forces = model_results[self.model_forces].cpu().data.numpy()
                results[self.forces] = (
                    forces.reshape((len(atoms), 3)) * self.forces_units
                )

            if self.model_stress is not None:
                if atoms.cell.volume <= 0.0:
                    raise SpkCalculatorError(
                        "Cell with 0 volume encountered for stress computation"
                    )

                if self.model_stress not in model_results.keys():
                    raise SpkCalculatorError(
                        "'{}' is not a property of your model. Please "
                        "check the model"
                        "properties! If desired, stress tensor computation can be "
                        "activated via schnetpack.utils.activate_stress_computation "
                        "at ones own risk.".format(self.model_stress)
                    )
                stress = model_results[self.model_stress].cpu().data.numpy()
                results[self.stress] = stress.reshape((3, 3)) * self.stress_units

            if self.model_cv is not None:
                if self.model_cv not in model_results.keys():
                    raise SpkCalculatorError(
                        "'{}' is not a property of your model. Please "
                        "check the model "
                        "properties!".format(self.model_cv)
                    )
                cv = model_results[self.model_cv].cpu().data.numpy()
                results[self.cv] = (
                    cv #* self.cv_units
                )  # ase calculator should return scalar energy

            if self.model_cv_grad is not None:
                if self.model_cv_grad not in model_results.keys():
                    raise SpkCalculatorError(
                        "'{}' is not a property of your model. Please "
                        "check the model"
                        "properties!".format(self.model_cv_grad)
                    )
                cv_grad = model_results[self.model_cv_grad]
                cv_grad = [g.cpu().data.numpy() for g in cv_grad]
                results[self.cv_grad] = cv_grad #* self.cv_grad_units


            self.results = results

"""
Args:
    atoms (ase.Atoms): ASE atoms object to correct. ASE needs upper diagonal matrix
        for non-orthogonal cell.
Return:
    sstr (ase.Atoms): ASE corrected atoms
"""
def repair_nonortogonal_ase(sstr):

    if not (sstr.get_cell() == np.triu(sstr.get_cell())).all()==True:
        ase.build.niggli_reduce(sstr)

        # swap cell vectors
        cn = np.copy(sstr.get_cell())
        cn[[0,2],:] = cn[[2,0],:]
        cn[:,[0,2]] = cn[:,[2,0]]
        sstr.set_cell(cn)

        # swap coordinates
        spos = sstr.get_positions()
        spos[:,[0,2]] = spos[:,[2,0]]
        sstr.set_positions(spos)
        print('''### WARNING: ASEs NPT module does not work if the UC is not a upper triangular matrix
               a- and c-axes are swapped - Niggli reduced and new input structure have been
               written to niggli.vasp and new_input.vasp, respectively''')

    return sstr
