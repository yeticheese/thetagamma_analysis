from .utils import write_dict_to_hdf5
import h5py
import numpy as np
import pandas as pd


def dict_write(rem_dict, header_dict, write_filename):
    rem_record = header_dict
    rem_record.update(rem_dict)
    file = h5py.File(write_filename, 'w')
    write_dict_to_hdf5(file, rem_record)
    file.close()


def cog_dataset_loader(rem_dict):
    cog = np.empty((0, 2))
    for k, sub_dict in rem_dict.items():
        if k == 'header_info':
            continue
        else:
            for key, value in sub_dict.items():
                if key == 'CoG':
                    cog = np.vstack((cog, value))
                else:
                    continue
    return cog


def dataset_to_csv(cog, data_filename):
    cog_df = pd.DataFrame(cog)
    cog_df.to_csv(data_filename, mode='a', header=False, index=False)
