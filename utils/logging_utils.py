import os
import logging
import time
import functools

def setup_logging():
    script_directory = os.getcwd()
    log_file_path = os.path.join(script_directory, 'Output_Training_times.log')
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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