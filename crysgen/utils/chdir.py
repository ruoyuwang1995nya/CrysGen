import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
import uuid

@contextmanager
def set_directory(path: Path):
    """Sets the current working path within the context.

    Parameters
    ----------
    path : Path
        The path to the cwd

    Yields
    ------
    None

    Examples
    --------
    >>> with set_directory("some_path"):
    ...    do_something()
    """
    cwd = Path().absolute()
    path.mkdir(exist_ok=True, parents=True)
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(cwd)

def make_path(prefix: Path) -> Path:
    """Create a timestamped path under ``prefix``.

    The path format is ``prefix.YYMMDD.RANDOM`` where ``RANDOM``
    is a zero-padded integer with ``random_digits`` digits.

    Parameters
    ----------
    prefix : Path
        Base directory under which the dated path is constructed.

    Returns
    -------
    Path
        The constructed path. Directories are not created.

    Examples
    --------
    >>> base = Path("runs")
    >>> p = dated_path(base)
    >>> p.parts[:1] == ("runs",)
    True
    """
    current_time = time.strftime("%Y%m%d%H%M%S")
    random_string = str(uuid.uuid4())[:8]
    work_path=f"{prefix}.{current_time}.{random_string}"
    return Path(work_path)