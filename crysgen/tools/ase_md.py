import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union, Sequence, List
from dataclasses import dataclass, asdict, field

import numpy as np
from ase import Atoms, units
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.analysis import DiffusionCoefficient
from ase.optimize import LBFGS
from ase.filters import UnitCellFilter

from crysgen.constants import ase_log_name, ase_traj_name

@dataclass
class MDParameters:
    """MD simulation parameters that can be serialized."""
    # For bot optimization and MD
    ensemble: str = "nvt"  # "nvt", "npt", or "both"
    # for MD
    temp: float = 300.0  # K
    press: Optional[float] = None  # Bar (None for NVT)
    dt: float = 2.0  # fs
    nsteps: int = 30000  # NVT production steps
    traj_freq: int = 100  # frames
    log_freq: int = 100  # steps
    tau_t: float = 100.0  # damping factor * timestep
    tau_p: float = 1000  # damping factor for pressure (NPT)
    compressibility: float = 4.5e-5  # 1/bar (NPT)
    custom_config: Dict[str, Any] = field(default_factory=dict)   # Custom configuration
    output_prefix: str = "md"
    ## for optimization
    max_step: int =1000 # maximum steps in optimization
    scalar_pressure: float = 0.0  # target scalar pressure for optimization
    fmax: float = 0.01  # force convergence criterion for optimization
    constant_volume: bool = False  # whether to keep volume constant during optimization
    filter_config: Dict[str, Any] = field(default_factory=dict)  # Custom filter configuration
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=4)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'MDParameters':
        return cls(**json.loads(json_str))
    
    @classmethod
    def from_file(cls, filename: Union[str, Path]) -> 'MDParameters':
        with open(filename, 'r') as f:
            return cls.from_json(f.read())
        

class MDRunner:
    """
    MD simulation runner that wraps an ASE Atoms object.
    
    Examples
    --------
    >>> # Create from structure file
    >>> md_runner = MDRunner.from_file("structure.cif")
    >>> md_runner.atoms.set_calculator(calculator)
    >>> 
    >>> # Run MD with parameters from JSON
    >>> md_runner.run_md_from_json("md_params.json")
    >>> 
    >>> # Or run with direct parameters
    >>> params = MDParameters(temperature=500.0, nsteps_nvt=50000)
    >>> md_runner.run_md(params)
    """

    def __init__(self, atoms: Atoms):
        """Initialize MDRunner with an Atoms object."""
        self.atoms = atoms
        self.md_history = []
        self.logger = self._setup_logger()
    
    def __len__(self) -> int:
        """Return number of atoms."""
        return len(self.atoms)
    
    def set_calculator(self, calc) -> None:
        """Set calculator for the atoms."""
        self.atoms.set_calculator(calc)
    
    @property
    def calc(self):
        """Get the calculator."""
        return self.atoms.calc
    
    @calc.setter
    def calc(self, calc):
        """Set the calculator."""
        self.atoms.calc = calc

    def _setup_logger(self) -> logging.Logger:
        """Setup logging for MD simulation."""
        logger = logging.getLogger(f"{self.__class__.__name__}")
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    @classmethod
    def from_file(cls, filename: Union[str, Path]) -> 'MDRunner':
        """Create MDRunner from structure file."""
        atoms_data = read(filename, index=0)
        # Ensure we have a single Atoms object, not a list
        if isinstance(atoms_data, list):
            atoms = atoms_data[0]
        else:
            atoms = atoms_data
        return cls(atoms)
    
    @classmethod
    def from_atoms(cls, atoms: Atoms) -> 'MDRunner':
        """Create MDRunner from existing Atoms object."""
        return cls(atoms)
    
    def initialize_velocities(self, temperature: float, seed: Optional[int] = None) -> None:
        """Initialize Maxwell-Boltzmann velocity distribution."""
        if seed is not None:
            np.random.seed(seed)
        MaxwellBoltzmannDistribution(self.atoms, temperature_K=temperature)
        self.logger.info(f"Initialized velocities at {temperature} K")
    
    def run_npt(self, 
                params: MDParameters,
                log_file: Optional[str] = ase_log_name,
                traj_file: Optional[str] = ase_traj_name
                ) -> None:
        """Run NPT simulation."""

        if params.press is None:
            raise ValueError("Pressure must be specified for NPT simulation")
        timestep = params.dt * units.fs
        
        # Initialize velocities
        if not hasattr(self, '_velocities_initialized'):
            self.initialize_velocities(params.temp)
            self._velocities_initialized = True
        
        # Setup NPT dynamics
        dyn = NPTBerendsen(
            self.atoms,
            timestep,
            temperature_K=params.temp,
            pressure_au=params.press * units.bar,
            taut=params.tau_t * units.fs,
            taup=params.tau_p * units.fs,
            compressibility_au=params.compressibility/units.bar,
            logfile=log_file,
            loginterval=params.log_freq,
            **params.custom_config
        )
        traj = Trajectory(traj_file, 'w', atoms=self.atoms)
        dyn.attach(traj.write, interval=params.traj_freq)
            # Run NPT
        self.logger.info("#### Starting MD...")
        start_time = time.time()
        dyn.run(params.nsteps)
        elapsed = time.time() - start_time
        self.logger.info(f"#### MD simulation completed in {elapsed:.2f} s")
        
        # Store history
        self.md_history.append({
            'type': 'NPT',
            'steps': params.nsteps,
            'temperature': params.temp,
            'pressure': params.press,
            'duration': elapsed
        })
    
    def run_nvt(
        self, 
        params: MDParameters,
        log_file: Optional[str] = ase_log_name,
        traj_file: Optional[str] = ase_traj_name
        ) -> None:
        """Run NVT simulation."""
        timestep = params.dt * units.fs
        tdamp = params.tau_t * timestep
        
        # Initialize velocities if not already done
        if not hasattr(self, '_velocities_initialized'):
            self.initialize_velocities(params.temp)
            self._velocities_initialized = True
        
        # Setup NVT dynamics
        dyn = NoseHooverChainNVT(
            self.atoms,
            timestep,
            temperature_K=params.temp,
            tdamp=tdamp,
            logfile=log_file,
            loginterval=params.log_freq,
            **params.custom_config
        )
        traj = Trajectory(traj_file, 'w', atoms=self.atoms)
        dyn.attach(traj.write, interval=params.traj_freq)

        # Run NVT
        self.logger.info("#### Starting NVT simulation...")
        start_time = time.time()
        dyn.run(params.nsteps)
        elapsed = time.time() - start_time
        self.logger.info(f"#### NVT simulation completed in {elapsed:.2f} s")
        # Store history
        self.md_history.append({
            'type': 'NVT',
            'steps': params.nsteps,
            'temperature': params.temp,
            'duration': elapsed
        })
    
    def run_md(self, params: MDParameters,**kwargs) -> None:
        """Run MD simulation based on ensemble parameter."""
        if not hasattr(self.atoms, 'calc') or self.atoms.calc is None:
            raise ValueError("Calculator must be set before running MD")
            
        self.logger.info(f"Starting MD simulation with {len(self.atoms)} atoms")
        self.logger.info(f"Ensemble: {params.ensemble.upper()}")
        self.logger.info(f"Temperature: {params.temp} K")
        
        total_start = time.time()
        
        if params.ensemble.lower() == "npt":
            self.run_npt(params)
        elif params.ensemble.lower() == "nvt":
            self.run_nvt(params)

        elif params.ensemble.lower() == "lbfgs":
            self.run_opt_LBFGS(params,**kwargs)
        else:
            raise ValueError(f"Unknown ensemble: {params.ensemble}")
        
        total_elapsed = time.time() - total_start
        self.logger.info(f"#### Total MD simulation completed in {total_elapsed:.2f} s")

    def run_md_stages(
        self,
        stages: Union[Sequence[Union[Dict[str, Any], MDParameters]], Dict[str, Any], MDParameters],
        *,
        log_dir: Optional[Union[str, Path]] = None,
        traj_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> None:
        """Run multiple MD stages sequentially using the single-stage helpers.

        Each entry in ``stages`` may be an ``MDParameters`` instance or a
        dictionary of keyword arguments accepted by ``MDParameters``. The
        ``ensemble`` field selects which underlying method to call (``run_nvt``,
        ``run_npt``, or ``run_opt_LBFGS``).
        """

        if not hasattr(self.atoms, 'calc') or self.atoms.calc is None:
            raise ValueError("Calculator must be set before running MD")

        if isinstance(stages, (MDParameters, dict)):
            stages = [stages]

        self._run_md_stages_collect(stages, log_dir=log_dir, traj_dir=traj_dir)

    def _run_md_stages_collect(
        self,
        stages: Sequence[Union[Dict[str, Any], MDParameters]],
        *,
        log_dir: Optional[Union[str, Path]] = None,
        traj_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Internal helper that runs stages and returns paths and last params."""

        log_base = Path(log_dir) if log_dir is not None else None
        traj_base = Path(traj_dir) if traj_dir is not None else None
        if log_base is not None:
            log_base.mkdir(parents=True, exist_ok=True)
        if traj_base is not None:
            traj_base.mkdir(parents=True, exist_ok=True)

        last_traj: Optional[Path] = None
        last_params: Optional[MDParameters] = None
        stage_paths: List[Dict[str, Path]] = []

        for idx, stage in enumerate(stages, start=1):
            params = stage if isinstance(stage, MDParameters) else MDParameters(**stage)
            stage_name = f"{idx:04d}"

            log_file = (log_base / f"{stage_name}.log") if log_base is not None else Path(f"{stage_name}.log")
            traj_file = (traj_base / f"{stage_name}.traj") if traj_base is not None else Path(f"{stage_name}.traj")

            ensemble = params.ensemble.lower()
            if ensemble == "npt":
                self.run_npt(params, log_file=str(log_file), traj_file=str(traj_file))
            elif ensemble == "nvt":
                self.run_nvt(params, log_file=str(log_file), traj_file=str(traj_file))
            elif ensemble == "lbfgs":
                self.run_opt_LBFGS(params, log_file=str(log_file), traj_file=str(traj_file))
            else:
                raise ValueError(f"Unknown ensemble in stage {idx}: {params.ensemble}")

            stage_paths.append({"log": log_file, "traj": traj_file})
            last_traj = traj_file
            last_params = params

        return {"stage_paths": stage_paths, "last_traj": last_traj.resolve() if last_traj else None, "last_params": last_params}

    def run_md_ion_stages(
        self,
        stages: Union[Sequence[Union[Dict[str, Any], MDParameters]], Dict[str, Any], MDParameters],
        *,
        log_dir: Optional[Union[str, Path]] = None,
        traj_dir: Optional[Union[str, Path]] = None,
        charges: Optional[Dict[str, float]] = None,
        start_frame: int = 0,
    ) -> Dict[str, Any]:
        """Run multi-stage MD and auto-compute diffusion/conductivity on last stage.

        Uses the last stage's parameters (``temp``, ``dt``, ``traj_freq``) to feed
        ``calc_diff`` so analysis matches the written trajectory without manual input.
        """

        if isinstance(stages, (MDParameters, dict)):
            stages = [stages]

        res = self._run_md_stages_collect(stages, log_dir=log_dir, traj_dir=traj_dir)

        last_traj: Optional[Path] = res["last_traj"]
        last_params: Optional[MDParameters] = res["last_params"]

        if last_traj is None or last_params is None:
            raise ValueError("No stages were executed; cannot compute diffusion")

        diff_res = MDRunner.calc_diff(
            traj_path=last_traj,
            temperature_K=last_params.temp,
            timestep_fs=last_params.dt,
            dump_interval=last_params.traj_freq,
            charges=charges,
            start_frame=start_frame,
            plot_file="msd_plot.png"
        )

        return {"last_traj": last_traj, "analysis": diff_res}

    def run_opt_LBFGS(self, 
        params: MDParameters,
        log_file: str = ase_log_name,
        traj_file: str = ase_traj_name) -> None:
        """Run MD simulation using parameters from dictionary."""
        
        # add unitcell filters
        ucf = UnitCellFilter(
            self.atoms,
            scalar_pressure=params.scalar_pressure,
            constant_volume=params.constant_volume,
            **params.filter_config
            )
        dyn = LBFGS(
            ucf, 
            trajectory=traj_file, 
            logfile=log_file,
            #maxstep= params.max_step,
            **params.custom_config
            )
        traj = Trajectory(traj_file, 'w', atoms=self.atoms)
        dyn.attach(traj.write, interval=params.traj_freq)
        
        self.logger.info("#### Starting LBFGS optimization...")
        start_time = time.time()
        dyn.run(fmax=params.fmax,
                steps=params.max_step)
        elapsed = time.time() - start_time
        self.logger.info(f"#### LBFGS optimization completed in {elapsed:.2f} s")
        # Store history
        self.md_history.append({
            'type': 'opt-LBFGS',
            #'steps': params.nsteps,
            'duration': elapsed
        })
        #params = MDParameters(**config)
        #self.run_md(params)
    
    def run_md_from_json(
        self, 
        json_file: Union[str, Path],
        **kwargs
        ) -> None:
        """Run MD simulation using parameters from JSON file."""
        params = MDParameters.from_file(json_file)
        self.logger.info(f"Loaded MD parameters from {json_file}")
        self.run_md(params, **kwargs)

    def run_md_from_dict(self, params_dict: Dict[str, Any], **kwargs) -> None:
        """Run MD simulation using parameters from dictionary."""
        params = MDParameters(**params_dict)
        self.run_md(params)
    
    def get_md_summary(self) -> Dict[str, Any]:
        """Get summary of completed MD runs."""
        return {
            'total_runs': len(self.md_history),
            'runs': self.md_history,
            'total_steps': sum(run.get('steps', 0) for run in self.md_history),
            'total_duration': sum(run['duration'] for run in self.md_history)
        }

    @staticmethod
    def calc_diff(
        traj_path: Union[str, Path],
        *,
        temperature_K: float,
        timestep_fs: float,
        dump_interval: int = 1,
        charges: Optional[Dict[str, float]] = None,
        start_frame: int = 0,
        plot_file: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Compute diffusion coefficients and ionic conductivities from an ASE trajectory.

        Parameters
        ----------
        traj_path : str | Path
            Path to the ASE trajectory file (``*.traj``).
        temperature_K : float
            Temperature in Kelvin used for the conductivity calculation.
        timestep_fs : float
            MD time step in femtoseconds.
        dump_interval : int
            Number of MD steps between stored frames in the trajectory.
        charges : dict, optional
            Mapping from element symbol to ionic charge (in units of e). Defaults to +1
            for any element not provided.
        start_frame : int
            Frame index to use as the reference for displacements.
        plot_file : str | Path, optional
            If provided, save the diffusion plot to this file path.

        Returns
        -------
        dict
            ``diff`` (diffusion coefficients in cm²/s per element with errors),
            ``sigma`` (per-element conductivity in S/m), and ``total_sigma``.
        """

        traj = read(traj_path, index=":")
        if len(traj) < 2:
            raise ValueError("Trajectory must contain at least two frames")

        if start_frame < 0 or start_frame >= len(traj) - 1:
            raise ValueError("start_frame must be within trajectory range and before the last frame")

        # Use ASE's DiffusionCoefficient analyzer
        traj_slice = traj[start_frame:]
        timestep_internal = timestep_fs * units.fs * dump_interval
        diff_coeff_analyzer = DiffusionCoefficient(traj_slice, timestep=timestep_internal)
        diff_coeffs_raw = diff_coeff_analyzer.get_diffusion_coefficients()
        
        # Generate and optionally save plot
        diff_coeff_analyzer.plot(show=False)
        if plot_file is not None:
            import matplotlib.pyplot as plt
            fig = plt.gcf()
            fig.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        #diff_coeff_analyzer.plot()
        base_frame = traj[start_frame]
        volume_m3 = base_frame.get_volume() * 1e-30
        if volume_m3 <= 0:
            raise ValueError("Volume must be positive to compute conductivity")

        symbols = base_frame.get_chemical_symbols()
        diffusion: Dict[str, List[float]] = {}
        conductivity: Dict[str, float] = {}

        for idx, element in enumerate(diff_coeff_analyzer.types_of_atoms):
            # Convert from ASE internal units to cm²/s
            D_cm2_s = diff_coeffs_raw[0][idx] * units.fs * 0.1
            D_err_cm2_s = diff_coeffs_raw[1][idx] * units.fs * 0.1
            diffusion[element] = [D_cm2_s.item(), D_err_cm2_s.item()]

            # Compute conductivity in S/cm
            D_m2_s = D_cm2_s * 1e-4  # cm²/s to m²/s
            charge_e = 1.0 if charges is None else charges.get(element, 1.0)
            q_coul = charge_e * 1.602176634e-19
            n_indices = sum(1 for sym in symbols if sym == element)
            n_number_density = n_indices / volume_m3
            sigma = n_number_density * (q_coul ** 2) * D_m2_s / (units.kB * temperature_K)
            conductivity[element] = sigma.item()*1e-2 # S/cm

        total_sigma = sum(conductivity.values())

        return {
            "temp": temperature_K,
            "diff": diffusion,  # cm²/s with [value, error]
            "sigma": conductivity,  # S/cm
            "total_sigma": total_sigma,  # S/cm
        }



