

def is_env_enabled(env_var: str, default: str = "0") -> bool:
    r"""Check if the environment variable is enabled."""
    return os.getenv(env_var, default).lower() in ["true", "y", "1"]

def use_ray() -> bool:
    return is_env_enabled("USE_RAY")

