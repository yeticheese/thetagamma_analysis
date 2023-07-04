import yaml
import os
import re
import itertools
import scipy.io as sio
import inspect
import h5py
import numpy as np


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
        rat_conditions = re.findall('Rat\d|HC|OR|OD|presleep|SD\d\d|SD\d|posttrial\d', root)
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
        sub_dict['study_day'] = int(re.findall('\d\d|\d', rat_conditions[1])[0])
        sub_dict['condition'] = rat_conditions[2]
        sub_dict['trial'] = rat_conditions[3]
        sub_dict['HPC'] = file_dir_parts[-1]
        sub_dict['states'] = file_dir_parts[-2]

    return file_dict


def process_functions(*functions, **kwargs):
    kwargs['output_data'] = None
    for func in functions:
        # Get the function signature
        signature = inspect.signature(func)
        # Get the function parameters
        params = signature.parameters
        if kwargs['output_data'] is None:
            kwargs = kwargs
        else:
            kwargs[str(next(iter(params)))] = kwargs['output_data']
        # Get the remaining keyword arguments
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in params}
        print(filtered_kwargs)
        # # Call the function with the positional arguments and filtered kwargs
        kwargs['output_data'] = func(**filtered_kwargs)


def write_dict_to_hdf5(file, data, parent_key=''):
    for key, value in data.items():
        current_key = f'{parent_key}/{key}' if parent_key else key

        if isinstance(value, dict):
            # If the value is a dictionary, create a new group in the HDF5 file
            group = file.create_group(current_key)
            print(group.name)
            write_dict_to_hdf5(file, value, parent_key=current_key)
        else:
            # If the value is not a dictionary, write it as a dataset in the HDF5 file
            if isinstance(value, np.ndarray):
                # If the value is a NumPy array, convert it to a dataset
                file.create_dataset(current_key, data=value)
            else:
                # If the value is a scalar or list, convert it to a dataset
                file[current_key] = value


def read_hdf5_to_dict(write_filename):
    def traverse_hdf5_group(group):
        result = {}
        for key, item in group.items():
            if isinstance(item, h5py.Group):
                result[key] = traverse_hdf5_group(item)
            elif isinstance(item, h5py.Dataset):
                if isinstance(item[()], np.ndarray):
                    result[key] = np.array(item[()])
                else:
                    result[key] = item[()].decode()
        return result

    with h5py.File(write_filename, 'r') as file:
        return traverse_hdf5_group(file)


def dict_walk(file_dict, folder_root, *functions, **kwargs):
    for rat, day_trial_dict in zip(file_dict.keys(), file_dict.values()):
        # Get the last sub-dictionary in the current sub-dictionary
        for day_trial_type, sleep_type_dict in zip(day_trial_dict.keys(), day_trial_dict.values()):
            for sleep_type, last_dict in zip(sleep_type_dict.keys(), sleep_type_dict.values()):
                file_path = r'{folder_root}\{rat}\{day_trial_type}\{sleep_type}\{state}'
                filename_struct = r'{folder_root}\{rat}\{day_trial_type}\{sleep_type}\Rat{rat}_SD{study_day}_{' \
                                  r'condition}_{trial}'
                states_file_path = file_path.format(folder_root=folder_root,
                                                    rat=rat,
                                                    day_trial_type=day_trial_type,
                                                    sleep_type=sleep_type,
                                                    state=last_dict['states'])
                HPC_file_path = file_path.format(folder_root=folder_root,
                                                 rat=rat,
                                                 day_trial_type=day_trial_type,
                                                 sleep_type=sleep_type,
                                                 state=last_dict['HPC'])
                states = sio.loadmat(f'{states_file_path}')
                HPC = sio.loadmat(f'{HPC_file_path}')
                write_filename = filename_struct.format(folder_root=folder_root,
                                                        rat=rat,
                                                        day_trial_type=day_trial_type,
                                                        sleep_type=sleep_type,
                                                        study_day=last_dict['study_day'],
                                                        condition=last_dict['condition'],
                                                        trial=last_dict['trial'])
                if os.path.exists(write_filename):
                    print("Filename already exists")
                    continue
                else:
                    header_dict = {'header_info': {'rat': str(last_dict['rat']),
                                                   'study_day': str(last_dict['study_day']),
                                                   'condition': str(last_dict['condition']),
                                                   'trial': str(last_dict['trial'])}}

                kwargs.update({'x': HPC['HPC'],
                               'rem_states': states['states'],
                               'header_dict': header_dict,
                               'write_filename': write_filename})
                process_functions(*functions, **kwargs)
