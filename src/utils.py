import yaml
import os
import re
import itertools
import scipy.io as sio


def load_config(config_path):
    """Load configuration settings from a YAML file."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_files_dict(file_path):
    """
    Recursively navigates through a directory tree and generates a nested dictionary
    representing the file structure, along with extracted information from the file paths.

    Args:
        file_path (str): The path to the root directory.

    Returns:
        dict: A nested dictionary representing the file structure and extracted information.
            The keys are directory names, and the values are nested dictionaries or file names
            and extracted information parsed with RegEx from file paths.

    Example:
        file_dict = get_files_dict('/path/to/directory')
        :param file_path:
    """
    file_dict = {}

    if isinstance(file_path, tuple):  # If it's a tuple of paths
        all_iterators = (os.walk(path, topdown=True) for path in file_path)
        walker = itertools.chain(*all_iterators)
    else:  # If it's a single path
        walker = os.walk(file_path, topdown=True)

    for root, dirs, files in walker:
        rat_conditions = re.findall('Rat\d|HC|OR|OD|presleep|SD\d|posttrial\d', root)
        if len(rat_conditions) < 4:
            continue

        file_dir_parts = root.split(os.sep)[7:]
        for file in files[1:]:
            state = bool(re.search('states.*.mat', file))
            hpc = bool(re.search(r"\bHPC.*.mat", file))
            if state or hpc:
                file_dir_parts.append(file)

        sub_dict = file_dict
        for part in file_dir_parts[:-2]:
            sub_dict = sub_dict.setdefault(part, {})

        sub_dict['rat'] = int(re.findall('\d', rat_conditions[0])[0])
        sub_dict['study_day'] = int(re.findall('\d', rat_conditions[1])[0])
        sub_dict['condition'] = rat_conditions[2]
        sub_dict['trial'] = rat_conditions[3]
        sub_dict['HPC'] = file_dir_parts[-1]
        sub_dict['states'] = file_dir_parts[-2]

    return file_dict


def process_function(func, **kwargs):
    func(**kwargs)


# def dict_walk(file_dict, folder_root, func, **kwargs):
def dict_walk(file_dict, folder_root):
    for rat, day_trial_dict in zip(file_dict.keys(), file_dict.values()):
        # Get the last sub-dictionary in the current sub-dictionary
        for day_trial_type, sleep_type_dict in zip(day_trial_dict.keys(), day_trial_dict.values()):
            for sleep_type, last_dict in zip(sleep_type_dict.keys(), sleep_type_dict.values()):
                state = last_dict['states']
                hpc = last_dict['HPC']
                states_file_path = f'{folder_root}/{rat}/{day_trial_type}/{sleep_type}/{state}'
                states = sio.loadmat(f'{states_file_path}')
                HPC_file_path = f'{folder_root}/{rat}/{day_trial_type}/{sleep_type}/{hpc}'
                HPC = sio.loadmat(f'{HPC_file_path}')
    return states, HPC
                # process_function(func, **kwargs)
