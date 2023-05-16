import yaml
import os
import re


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
        path (str): The path to the root directory.

    Returns:
        dict: A nested dictionary representing the file structure and extracted information.
            The keys are directory names, and the values are nested dictionaries or file names
             and extracted information parsed with RegEx from file paths.

    Example:
        file_dict = get_files_dict('/path/to/directory')
        :param file_path:
    """
    file_dict = {}

    for root, dirs, files in os.walk(file_path, topdown=True):
        rat_conditions = re.findall('Rat\d|HC|OR|OD|presleep|SD\d|posttrial\d', root)
        if len(rat_conditions) < 4:
            continue

        file_dir_parts = root.split('\\')[7:]
        for file in files[1:]:
            state = bool(re.search('states.*.mat', file))
            hpc = bool(re.search(r"\bhpc.*.mat", file))
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
