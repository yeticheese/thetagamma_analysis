from .utils import write_dict_to_hdf5
import h5py


def dict_write(rem_dict, header_dict, write_filename):
    rem_record = header_dict
    rem_record.update(rem_dict)
    file = h5py.File(write_filename, 'w')
    write_dict_to_hdf5(file, rem_record)
    file.close()
