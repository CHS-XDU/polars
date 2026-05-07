try:
    import polars._plr as plr

    _POLARS_VERSION = f"{plr.__version__}+chs.autocache.0"
except ImportError:
    # This is only useful for documentation
    import warnings

    warnings.warn("Polars binary is missing!", stacklevel=2)
    _POLARS_VERSION = ""


def get_polars_version() -> str:
    """
    Return the version of the Python Polars package as a string.

    If the Polars binary is missing, returns an empty string.
    """
    return _POLARS_VERSION
