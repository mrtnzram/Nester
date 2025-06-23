from pathlib2 import Path
import pathlib2
import os
from datetime import datetime
import numpy as np

PROJECT_DIR = Path.cwd()
DATA_DIR = PROJECT_DIR / "data"
FIGURE_DIR = PROJECT_DIR / "figures"


def ensure_dir(file_path):
    """ create a safely nested folder
    """
    if type(file_path) == str:
        if "." in os.path.basename(os.path.normpath(file_path)):
            directory = os.path.dirname(file_path)
        else:
            directory = os.path.normpath(file_path)
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except FileExistsError as e:
                # multiprocessing can cause directory creation problems
                print(e)
    elif type(file_path) == pathlib2.PosixPath:
        # if this is a file
        if len(file_path.suffix) > 0:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            file_path.mkdir(parents=True, exist_ok=True)


def most_recent_subdirectory(dataset_loc):
    """
    Return the most recently created subdirectory matching the AVGN timestamp format.
    Skips non-directories and hidden files like `.DS_Store`.
    """
    valid_dirs = []
    directory_dates = []

    for i in dataset_loc.iterdir():
        if not i.is_dir():
            continue
        try:
            dt = datetime.strptime(i.name, "%Y-%m-%d_%H-%M-%S")
            valid_dirs.append(i)
            directory_dates.append(dt)
        except ValueError:
            continue  # skip if folder name doesn't match the timestamp format

    if not valid_dirs:
        raise FileNotFoundError(f"No valid timestamped directories found in {dataset_loc}")

    # Return the most recent directory
    return valid_dirs[np.argsort(directory_dates)[-1]]