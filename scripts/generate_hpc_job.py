"""CLI for HPC job generation and submission."""

from pathlib import Path
from typing import Optional

import typer

from bathymetry_ml import resolve_path
from bathymetry_ml.hpc import HPCConfig, HPCJobTemplate, HPCJobSubmitter, generate_and_save_job_script

app = typer.Typer(help="HPC job management for Bathymetry ML")


@app.command()
def generate(
    config: Path = typer.Option(
        "configs/training.yaml",
        help="Path to training configuration YAML",
    ),
    output: Path = typer.Option("job_script.sh", help="Output script path"),
    command: str = typer.Option(
        "python -m bathymetry_ml.train --config configs/training.yaml",
        help="Command to execute in HPC job",
    ),
):
    """Generate HPC job script for review (does not submit).
    
    Example:
        python scripts/generate_hpc_job.py generate --config configs/training.yaml --output job_train.sh
    """
    print("=" * 80)
    print("HPC JOB SCRIPT GENERATION")
    print("=" * 80)

    try:
        from bathymetry_ml.train import load_yaml

        config = resolve_path(str(config))
        training_config = load_yaml(config)
        hpc_config_path = resolve_path(
            training_config.get("execution", {}).get("hpc_config", "configs/hpc.yaml")
        )

        script_path = generate_and_save_job_script(str(hpc_config_path), command, str(output))

        print(f"\nJob script generated successfully!")
        print(f"  Output: {script_path}")
        print(f"  Command: {command}")
        print(f"\nTo submit the job, run:")
        print(f"  bsub < {script_path}")

    except Exception as e:
        print(f"Error generating job script: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(code=1)


@app.command()
def submit(
    config: Path = typer.Option(
        "configs/training.yaml",
        help="Path to training configuration YAML",
    ),
    auto_submit: bool = typer.Option(
        False,
        help="Automatically submit without asking for confirmation",
    ),
    command: str = typer.Option(
        "python -m bathymetry_ml.train --config configs/training.yaml",
        help="Command to execute",
    ),
):
    """Generate and submit HPC job to cluster.
    
    Example:
        python scripts/generate_hpc_job.py submit --config configs/training.yaml --auto-submit
    """
    print("=" * 80)
    print("HPC JOB SUBMISSION")
    print("=" * 80)

    try:
        from bathymetry_ml.train import load_yaml

        config = resolve_path(str(config))
        training_config = load_yaml(config)
        hpc_config_path = resolve_path(
            training_config.get("execution", {}).get("hpc_config", "configs/hpc.yaml")
        )

        # Generate script
        print("\n[1] Generating job script...")
        script_path = Path("_temp_job.sh")
        generate_and_save_job_script(str(hpc_config_path), command, str(script_path))
        print(f"    Generated: {script_path}")

        # Review if not auto-submit
        if not auto_submit:
            print("\n[2] Job script preview:")
            print("    " + "=" * 70)
            with open(script_path) as f:
                for line in f:
                    print(f"    {line.rstrip()}")
            print("    " + "=" * 70)

            confirm = typer.confirm("\nSubmit this job to HPC?")
            if not confirm:
                print("Job submission cancelled.")
                script_path.unlink()
                raise typer.Exit(code=0)

        # Submit
        print("\n[3] Submitting job...")
        submitter = HPCJobSubmitter(script_path)
        job_id = submitter.submit()

        print(f"\n✓ Job submitted successfully!")
        print(f"  Job ID: {job_id}")
        print(f"\nTo check job status, run:")
        print(f"  python scripts/generate_hpc_job.py status --job-id {job_id}")

        # Cleanup temp file
        script_path.unlink()

    except Exception as e:
        print(f"Error submitting job: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(code=1)


@app.command()
def status(
    job_id: str = typer.Option(..., help="LSF job ID (from job submission)"),
):
    """Check status of HPC job.
    
    Example:
        python scripts/generate_hpc_job.py status --job-id 12345
    """
    print("=" * 80)
    print(f"HPC JOB STATUS - Job {job_id}")
    print("=" * 80)

    try:
        # We need to be on HPC to check status via bjobs
        # For now, provide informational output
        print(f"\nTo check job status on HPC, run:")
        print(f"  bjobs {job_id}")
        print(f"\nOr to see more details:")
        print(f"  bjobs -a {job_id}")
        print(f"\nTo see job output:")
        print(f"  bpeek {job_id}  (while running)")
        print(f"  cat logs/gpu_bathy*.out (after completion)")

    except Exception as e:
        print(f"Error checking job status: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(code=1)


@app.command()
def create_eval_job(
    config: Path = typer.Option(
        "configs/training.yaml",
        help="Path to training configuration",
    ),
    model_path: Path = typer.Option(..., help="Path to trained model checkpoint"),
    output: Path = typer.Option("job_eval.sh", help="Output script path"),
):
    """Generate evaluation job script.
    
    Example:
        python scripts/generate_hpc_job.py create-eval-job --model-path models/svdkl.pt
    """
    print("=" * 80)
    print("HPC EVALUATION JOB GENERATION")
    print("=" * 80)

    try:
        from bathymetry_ml.train import load_yaml

        config = resolve_path(str(config))
        model_path = resolve_path(str(model_path))
        training_config = load_yaml(config)
        hpc_config_path = resolve_path(
            training_config.get("execution", {}).get("hpc_config", "configs/hpc.yaml")
        )

        command = f"python -m bathymetry_ml.evaluate --config {config} --model-path {model_path}"
        script_path = generate_and_save_job_script(str(hpc_config_path), command, str(output))

        print(f"\nEvaluation job script generated successfully!")
        print(f"  Output: {script_path}")
        print(f"  Model: {model_path}")
        print(f"\nTo submit the job, run:")
        print(f"  bsub < {script_path}")

    except Exception as e:
        print(f"Error generating evaluation job: {e}")
        import traceback

        traceback.print_exc()
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
