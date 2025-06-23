import os
from datetime import datetime
from avgn.utils.paths import DATA_DIR

def initialize_paths(dataset_id="birdsong",dt_id = None):
    if dt_id is None:
        dt_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    json_path = DATA_DIR / "processed" / dataset_id / dt_id / "JSON"
    output_dir = DATA_DIR / "output_figures"
    save_loc = DATA_DIR / f'{dataset_id}.pickle'
    csv_path = DATA_DIR / f'{dataset_id}.csv'
    csv_path_corrected = DATA_DIR / f'{dataset_id}_corrected.csv'
    pickle_corrected = DATA_DIR / f'{dataset_id}_corrected.pickle'

    os.makedirs(output_dir, exist_ok=True)

    return {
        "json_path": json_path,
        "output_dir": output_dir,
        "save_loc": save_loc,
        "csv_path": csv_path,
        "csv_path_corrected": csv_path_corrected,
        "pickle_corrected": pickle_corrected
    }

def find_gzip_directories(root_directory):
    """
    Recursively finds all directories containing `.gzip` files.

    Args:
        root_directory (str): The root directory to search from.

    Returns:    
        List[str]: A list of directories containing `.gzip` files.
    """
    gzip_directories = set()

    # Walk through the directory tree
    for root, _, files in os.walk(root_directory):
        for file in files:
            if file.endswith(".gzip"):
                gzip_directories.add(root)  # Add the directory to the set
                break  # No need to check further files in this directory

    return list(gzip_directories)  