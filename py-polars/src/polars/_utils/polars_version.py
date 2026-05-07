try:
    import polars._plr as plr

    _CHS_VERSION_SUFFIX = "+chs.autocache.0"
    _POLARS_VERSION = (
        plr.__version__
        if plr.__version__.endswith(_CHS_VERSION_SUFFIX)
        else f"{plr.__version__}{_CHS_VERSION_SUFFIX}"
    )
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
