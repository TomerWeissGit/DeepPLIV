import time
import sys
import threading
import itertools
from typing import Iterable

import seaborn as sns
import matplotlib.pyplot as plt


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


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        """
        Early stopping to stop the training when the loss does not improve after certain epochs.
        :param patience: int, how many epochs to wait before stopping when loss is not improving.
        :param min_delta: float, minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def save_to_pickle(lst: Iterable, name: str = 'results.pkl'):
    import pickle
    with open(name, 'wb') as f:
        pickle.dump(lst, f)

def plot_boxplot(df):
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='coefficient', y='value', hue='method', data=df)
    plt.title('Coefficient Distribution by Method')
    plt.show()

