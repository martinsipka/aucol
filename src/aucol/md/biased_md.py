#Running actual MD
#--------------------------------------------------------------
from ase.md.npt import NPT
from ase.optimize import BFGS
from ase.units import kJ, mol, fs, kB, GPa
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

import schnetpack as spk
from schnetpack.environment import AseEnvironmentProvider

import icv.interfaces.custom_plumed_calc as ase_plumed
from icv.utils.build_plumed_dat import build_moving_restrain
from icv.interfaces.ase_interface_util import CVNNPCalculator
from ase.calculators.cp2k import CP2K

"""
Preoptimize geometry using BFGS
Args:
    sstr (ase.Atoms): Atom structure
    nnp_model (callable): Neural network potential to use as calculator
    device (str): Device to run on
    args (args): Runtime arguments
    out_path (str): Where to output results
"""
def pre_opt(sstr, nnp_model, device, args, out_path='results/'):
    # set energy/force calculator
    ffcalc = spk.interfaces.SpkCalculator(
         model=nnp_model,
         device=device,
         energy='energy',
         forces='forces',
         environment_provider=AseEnvironmentProvider(6.0    )
         )

    # set up MD run
    sstr.calc = ffcalc
    # RUN pre-optimization if requested
    relax = BFGS(sstr, trajectory=out_path+'opt.traj', logfile=out_path+'opt.log')
    relax.run(fmax = 1e-2, steps=2000)

"""
Run actual simulation using custom calculator that outputs also CVs
Args:
    sstr (ase.Atoms): Atom structure
    cv_nnp ()
    nnp_model (callable): Neural network potential to use as calculator
    device (str): Device to run on
    args (args): Runtime arguments
    out_path (str): Where to output results
"""
def run_md(structure, cv_nnp, plumed_dat, args, device, lpre_opt=True,tstep=0.5,nstep=1000000,nh_tcoupling=20.,
            temperature=300,ext_pressure=0.0,out_interval=100,out_path='results/', cutoff=6.0):

    # set energy/force calculator
    cv_ffcalc = CVNNPCalculator(
         model=cv_nnp,
         device=device,
         energy='energy',
         forces='forces',
         cv="cv",
         cv_grad="cv_grad",
         environment_provider=AseEnvironmentProvider(6.0)
         )
    # Set plumed calculator
    calc = ase_plumed.Plumed(cv_ffcalc, plumed_dat, tstep*fs, atoms=structure, kT=temperature*kB)
    structure.calc = calc

    # set initial velocities according to temperature
    MaxwellBoltzmannDistribution(structure, temperature*kB)

    dyn = NPT(structure,
            timestep=tstep*fs,
            ttime=nh_tcoupling*fs,
            temperature=temperature*kB,
            externalstress=ext_pressure*GPa,
            loginterval=out_interval, trajectory=out_path+'md.traj', logfile=out_path+'md.log')
    dyn.run(nstep)

"""
Run actual simulation using custom calculator that outputs also CVs
Args:
    sstr (ase.Atoms): Atom structure
    cv_nnp ()
    nnp_model (callable): Neural network potential to use as calculator
    device (str): Device to run on
    args (args): Runtime arguments
    out_path (str): Where to output results
"""
def run_two_model_md(structure, nnp, cv_vae, plumed_dat, args, device, lpre_opt=True,tstep=0.5,nstep=1000000,nh_tcoupling=20.,
            temperature=300,ext_pressure=0.0,out_interval=100,out_path='results/', cutoff=6.0):

    # set energy/force calculator
    ffcalc = spk.interfaces.SpkCalculator(
         model=nnp,
         device=device,
         energy='energy',
         forces='forces',
         #stress='stress',                   # uncomment if stress calc. is required
         environment_provider=spk.environment.AseEnvironmentProvider(cutoff)
         )

    # Set plumed calculator
    calc = ase_plumed.Plumed(ffcalc, plumed_dat, tstep*fs, cv_model=cv_vae, atoms=structure, kT=temperature*kB)
    structure.calc = calc

    # set initial velocities according to temperature
    MaxwellBoltzmannDistribution(structure, temperature*kB)

    dyn = NPT(structure,
            timestep=tstep*fs,
            ttime=nh_tcoupling*fs,
            temperature=temperature*kB,
            externalstress=ext_pressure*GPa,
            loginterval=out_interval, trajectory=out_path+'md.traj', logfile=out_path+'md.log')
    dyn.run(nstep)

"""
Run actual simulation using CP2K and loaded CV model separately
Args:
    sstr (ase.Atoms): Atom structure
    cv_vae (callable): CV model
    nnp_model (callable): Neural network potential to use as calculator
    device (str): Device to run on
    args (args): Runtime arguments
    out_path (str): Where to output results
"""
def run_cp2k_md(structure, cv_vae, cp2k_inp, plumed_dat, args, device, lpre_opt=True,tstep=0.5,nstep=1000000,nh_tcoupling=20.,
            temperature=300,ext_pressure=0.0,out_interval=100,out_path='results/', cutoff=6.0, restart=False):

    ffcalc = CP2K(inp=cp2k_inp, command="mpirun -np "+ str(args.cores) +" cp2k_shell.popt", print_level="SILENT")

    calc = ase_plumed.Plumed(ffcalc, plumed_dat, tstep*fs, cv_model=cv_vae, atoms=structure, kT=temperature*kB, restart=restart)
    structure.calc = calc

    MaxwellBoltzmannDistribution(structure, temperature*kB)

    dyn = NPT(structure,
            timestep=tstep*fs,
            ttime=nh_tcoupling*fs,
            temperature=temperature*kB,
            externalstress=ext_pressure*GPa,
            loginterval=out_interval, trajectory=out_path+'md.traj', logfile=out_path+'md.log')
    dyn.run(nstep)
