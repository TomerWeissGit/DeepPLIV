import time
import sys
import threading
import itertools
import numpy as np

# Spinner decorator to add loading animation to specific functions
def spinner_decorator(task_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            stop_event = threading.Event()
            spinner_thread = threading.Thread(target=spinner, args=(task_name, stop_event))
            spinner_thread.start()
            try:
                result = func(*args, **kwargs)  # Call the actual function
            finally:
                stop_event.set()  # Stop the spinner once the function is done
                spinner_thread.join()
            return result
        return wrapper
    return decorator

# Spinner function to display a rotating spinner
def spinner(task_name, stop_event):
    spinner_symbols = itertools.cycle(['|', '/', '-', '\\'])
    print(f"{task_name} ", end="", flush=True)
    while not stop_event.is_set():
        sys.stdout.write(next(spinner_symbols))
        sys.stdout.flush()
        time.sleep(0.1)
        sys.stdout.write('\b')
    print("Done")