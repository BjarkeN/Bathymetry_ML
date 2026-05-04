"""HPC job generation and submission utilities."""

import subprocess
from pathlib import Path
from typing import Dict, Union, Optional
import yaml

from .hpc_utils import generate_lsf_header, validate_lsf_config


class HPCConfig:
    """Load and manage HPC configuration."""

    def __init__(self, config_path: Union[str, Path]):
        """Load HPC configuration from YAML file.
        
        Args:
            config_path: Path to hpc.yaml configuration file
        """
        self.path = Path(config_path)
        with open(self.path) as f:
            self.config = yaml.safe_load(f)

        validate_lsf_config(self.config)

    def get_lsf_directives(self) -> Dict:
        """Get LSF configuration dictionary.
        
        Returns:
            Dictionary containing LSF settings
        """
        return self.config.get("lsf", {})

    def get_environment_setup(self) -> Dict:
        """Get environment setup configuration.
        
        Returns:
            Dictionary containing conda_env and cuda_module
        """
        return self.config.get("environment", {})

    def get_job_output_dirs(self) -> Dict:
        """Get job output directory configuration.
        
        Returns:
            Dictionary with logs_dir, model_save_path, results_dir
        """
        return self.config.get("job_output", {})


class HPCJobTemplate:
    """Generate LSF job scripts from configuration."""

    def __init__(self, hpc_config: HPCConfig, training_config: Optional[Dict] = None):
        """Initialize job template generator.
        
        Args:
            hpc_config: HPCConfig instance
            training_config: Optional training configuration for merged settings
        """
        self.hpc_config = hpc_config
        self.training_config = training_config or {}

    def generate(self, command: str) -> str:
        """Generate complete LSF job script.
        
        Args:
            command: Command to execute in HPC job
            
        Returns:
            Complete job script as string
        """
        hpc_cfg = self.hpc_config.config
        env_cfg = hpc_cfg.get("environment", {})
        job_out_cfg = hpc_cfg.get("job_output", {})

        # Build job script
        script_lines = ["#!/bin/sh", ""]

        # Create log directory
        logs_dir = job_out_cfg.get("logs_dir", "logs/")
        script_lines.append(f'mkdir -p "{logs_dir}"')
        script_lines.append("")

        # Add LSF directives
        lsf_header = generate_lsf_header(hpc_cfg)
        script_lines.append(lsf_header)

        # Add output file specifications
        job_name = hpc_cfg.get("lsf", {}).get("job_name", "bathy_job")
        script_lines.append(f"#BSUB -o {logs_dir}/gpu_{job_name}%J.out")
        script_lines.append(f"#BSUB -e {logs_dir}/gpu_{job_name}%J.err")
        script_lines.append("# -- end of LSF options --")
        script_lines.append("")

        # Environment setup
        script_lines.append("# activate env")
        conda_env = env_cfg.get("conda_env", "bathymetry_ml")
        script_lines.append("source ~/miniconda3/bin/activate")
        script_lines.append("")

        # Check CUDA
        script_lines.append("nvidia-smi")

        # Load CUDA module
        cuda_module = env_cfg.get("cuda_module", "cuda/11.6")
        script_lines.append(f"# Load the cuda module")
        script_lines.append(f"module load {cuda_module}")
        script_lines.append("")

        # Run command
        script_lines.append("# Run script")
        script_lines.append(command)
        script_lines.append("")

        return "\n".join(script_lines)

    def save(self, output_path: Union[str, Path]):
        """Save generated job script to file.
        
        Args:
            output_path: Path to save script
            command: Command to execute
        """
        raise NotImplementedError("Use generate() and save manually or use HPCJobSubmitter")


class HPCJobSubmitter:
    """Submit jobs to HPC cluster via bsub."""

    def __init__(self, job_script_path: Union[str, Path]):
        """Initialize job submitter with job script.
        
        Args:
            job_script_path: Path to LSF job script
        """
        self.script_path = Path(job_script_path)
        if not self.script_path.exists():
            raise FileNotFoundError(f"Job script not found: {self.script_path}")

    def submit(self) -> str:
        """Submit job to HPC cluster via bsub.
        
        Returns:
            Job ID string
            
        Raises:
            RuntimeError: If job submission fails
        """
        try:
            result = subprocess.run(
                ["bsub"],
                stdin=open(self.script_path),
                capture_output=True,
                text=True,
                check=True,
            )
            output = result.stdout.strip()
            # Extract job ID from bsub output: "Job <12345> is submitted"
            job_id = output.split("<")[1].split(">")[0]
            return job_id
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Job submission failed: {e.stderr}")
        except (IndexError, ValueError) as e:
            raise RuntimeError(f"Could not parse job ID from bsub output: {output}")

    def get_status(self, job_id: str) -> str:
        """Check job status on HPC.
        
        Args:
            job_id: LSF job ID
            
        Returns:
            Job status string
        """
        try:
            result = subprocess.run(
                ["bjobs", job_id],
                capture_output=True,
                text=True,
                check=True,
            )
            # Return relevant status lines
            lines = result.stdout.strip().split("\n")
            return "\n".join(lines[-2:]) if len(lines) > 1 else result.stdout.strip()
        except subprocess.CalledProcessError:
            return f"Job {job_id} not found or completed"

    def cancel(self, job_id: str) -> bool:
        """Cancel running job.
        
        Args:
            job_id: LSF job ID
            
        Returns:
            True if cancellation successful
        """
        try:
            subprocess.run(
                ["bkill", job_id],
                capture_output=True,
                check=True,
            )
            return True
        except subprocess.CalledProcessError:
            return False


def generate_and_save_job_script(
    hpc_config_path: Union[str, Path],
    command: str,
    output_path: Union[str, Path],
) -> Path:
    """Generate HPC job script and save to file.
    
    Args:
        hpc_config_path: Path to hpc.yaml
        command: Command to execute in job
        output_path: Where to save generated script
        
    Returns:
        Path to generated script
    """
    output_path = Path(output_path)
    hpc_cfg = HPCConfig(hpc_config_path)
    template = HPCJobTemplate(hpc_cfg)
    script_content = template.generate(command)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(script_content)

    # Make executable
    output_path.chmod(0o755)

    return output_path
