import os
import logging
import time
import functools

def setup_logging(log_file: str = "mlff_qd.log", level: int = logging.INFO):
    """
    Initialize a generic logger for MLFF_QD modules.
    Writes logs both to console and to a file (default: mlff_qd.log in cwd).
    Safe to call multiple times — won't duplicate handlers.
    """
    logger = logging.getLogger()
    if logger.handlers:
        return  # already configured once

    log_path = os.path.join(os.getcwd(), log_file)
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(fmt))

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(fmt))

    logger.setLevel(level)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"[LOGGER] Initialized logging → {log_path}")

def timer(func):
    """A decorator that records the execution time of the function it decorates and logs the time."""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        logging.info(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer