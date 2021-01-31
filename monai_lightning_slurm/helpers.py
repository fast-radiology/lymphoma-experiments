import logging
from monai.transforms import DataStatsd

PIXDIM = [0.615234, 0.615234, 1.000000]

VAL_PATIENTS = ['P16B16', 'P08B08', 'P02B02']


def train_val_split(data_dicts):
    train_files, val_files = [], []
    for patient_dict in data_dicts:
        if patient_dict['patient'] in VAL_PATIENTS:
            val_files.append(patient_dict)
        else:
            train_files.append(patient_dict)
    assert len(val_files) == len(VAL_PATIENTS)
    return train_files, val_files


class DataStatsdWithPatient(DataStatsd):
    def __call__(self, data):
        d = dict(data)
        logger = logging.getLogger("DataStats")
        logger.debug('\n')
        logger.debug(d['patient'])
        super().__call__(data)
        return d
