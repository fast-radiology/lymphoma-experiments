import os
import pickle
import numpy as np
from collections import OrderedDict

VAL_PATIENTS = ['P16B16', 'P08B08', 'P02B02']

DATA_PATH = os.environ.get(
    'DATA_PATH',
    '/home/users/nozdi/lymphoma_data/nnunet/nnUNet_raw_data/Task200_LymphomaSeg/imagesTr',
)
PROCESSED_PATH = os.environ.get(
    'PROCESSED_PATH',
    '/home/users/nozdi/lymphoma_data/nnunet/processed/Task200_LymphomaSeg'
)

filenames = os.listdir(DATA_PATH)

fold = OrderedDict()
fold['train'] = np.array(
    sorted([
        filename.rstrip('.nii.gz')
        for filename in filenames
        if filename.split('_')[1] not in VAL_PATIENTS
    ]),
    dtype='<U16',
)
fold['val'] = np.array(
    sorted([
        filename.rstrip('.nii.gz')
        for filename in filenames
        if filename.split('_')[1] in VAL_PATIENTS
    ]),
    dtype='<U16',
)

with open(os.path.join(PROCESSED_PATH, 'splits_final.pkl'), 'wb') as fp:
    pickle.dump([fold], fp)

print([fold])
