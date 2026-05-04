"""LSF-specific utilities for HPC job submission."""

from typing import Dict


def generate_lsf_header(config: Dict) -> str:
    """Generate LSF directives based on configuration.
    
    Handles adaptive GPU type selection (v100 vs v32gb).
    
    Args:
        config: LSF configuration dictionary
        
    Returns:
        String containing LSF directives
    """
    lsf = config.get("lsf", {})
    queue = lsf.get("queue", "gpuv100")
    job_name = lsf.get("job_name", "bathy_job")
    num_cores = lsf.get("num_cores", 4)
    num_gpus = lsf.get("num_gpus", 1)
    gpu_type = lsf.get("gpu_type", "v100")
    walltime = lsf.get("walltime", "18:00")
    memory = lsf.get("memory", "16GB")
    email_notifications = lsf.get("email_notifications", False)
    email_address = lsf.get("email_address", "")

    directives = []
    directives.append("### General options")
    directives.append(f"#BSUB -q {queue}")
    directives.append(f"#BSUB -J {job_name}")
    directives.append(f"#BSUB -n {num_cores}")
    directives.append('#BSUB -R "span[hosts=1]"')
    directives.append(f'#BSUB -gpu "num={num_gpus}:mode=exclusive_process"')
    directives.append(f"#BSUB -W {walltime}")
    directives.append(f'#BSUB -R "rusage[mem={memory}]"')

    # Add GPU type selection if v32gb
    if gpu_type.lower() == "v32gb":
        directives.append('#BSUB -R "select[gpu32gb]"')

    # Add email notifications if enabled
    if email_notifications and email_address:
        directives.append(f"#BSUB -u {email_address}")
        directives.append("#BSUB -B")
        directives.append("#BSUB -N")

    return "\n".join(directives)


def get_gpu_select_directive(gpu_type: str) -> str:
    """Return GPU selection directive for LSF.
    
    Args:
        gpu_type: GPU type ("v100" or "v32gb")
        
    Returns:
        LSF directive string (empty for v100, v32gb selector for v32gb)
    """
    if gpu_type.lower() == "v32gb":
        return '#BSUB -R "select[gpu32gb]"'
    return ""


def format_memory(memory_str: str) -> str:
    """Convert memory string to LSF format.
    
    Args:
        memory_str: Memory specification (e.g., "16GB", "32000")
        
    Returns:
        Formatted memory string for LSF
    """
    if isinstance(memory_str, str):
        # Already formatted or contains unit
        return memory_str
    return str(memory_str)


def validate_lsf_config(config: Dict) -> bool:
    """Validate LSF configuration has required fields.
    
    Args:
        config: LSF configuration dictionary
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_fields = ["queue", "num_cores", "num_gpus", "walltime", "memory"]
    lsf_config = config.get("lsf", {})

    for field in required_fields:
        if field not in lsf_config:
            raise ValueError(f"Missing required LSF config field: {field}")

    # Validate gpu_type
    gpu_type = lsf_config.get("gpu_type", "v100").lower()
    if gpu_type not in ["v100", "v32gb"]:
        raise ValueError(f"Invalid gpu_type: {gpu_type}. Must be 'v100' or 'v32gb'")

    return True
