from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.dataset_conversion.utils import generate_dataset_json
import shutil


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


data_path = os.environ.get(
    "DATA_PATH", "/home/users/nozdi/lymphoma_data/nifti_package_lymphoma"
)
target_data_path = os.environ.get(
    "TARGET_DATA_PATH", "/home/users/nozdi/lymphoma_data/nnunet/nnUNet_raw_data"
)


task_name = "Task200_LymphomaSeg"
target_base = join(target_data_path, "Task200_LymphomaSeg")

maybe_mkdir_p(target_base)


data_images = sorted(
    [join(data_path, x) for x in os.listdir(data_path) if x.startswith("data")]
)
data_labels = sorted(
    [join(data_path, x) for x in os.listdir(data_path) if x.startswith("label")]
)
data_dicts = [
    {
        "patient": image_name.split("/")[-1].replace("data", "").replace(".nii.gz", ""),
        "img_filename": image_name,
        "img_target_filename": 'lymph_'
        + image_name.split("/")[-1].replace('data', '').replace(".nii.gz", "")
        + f"_{_id:03d}_0000.nii.gz",
        "label_filename": label_name,
        "label_target_filename": 'lymph_'
        + label_name.split("/")[-1].replace('label', '').replace(".nii.gz", "")
        + f"_{_id:03d}.nii.gz",
    }
    for _id, (image_name, label_name) in enumerate(zip(data_images, data_labels))
]
# train_files, val_files = train_val_split(data_dicts)


target_imagesTr = join(target_base, "imagesTr")
target_imagesTs = join(target_base, "imagesTs")
target_labelsTs = join(target_base, "labelsTs")
target_labelsTr = join(target_base, "labelsTr")
maybe_mkdir_p(target_imagesTr)
maybe_mkdir_p(target_labelsTs)
maybe_mkdir_p(target_imagesTs)
maybe_mkdir_p(target_labelsTr)

for config in data_dicts:
    shutil.copy(
        join(data_path, config['img_filename']),
        join(target_imagesTr, config['img_target_filename']),
    )
    shutil.copy(
        join(data_path, config['label_filename']),
        join(target_labelsTr, config['label_target_filename']),
    )

# for config in val_files:
#     shutil.copy(
#         join(data_path, config['img_filename']),
#         join(target_imagesTs, config['img_target_filename']),
#     )
#     shutil.copy(
#         join(data_path, config['label_filename']),
#         join(target_labelsTs, config['label_target_filename']),
#     )


generate_dataset_json(
    join(target_base, 'dataset.json'),
    target_imagesTr,
    target_imagesTs,
    ('contrast',),
    labels={0: 'background', 1: 'lymphoma'},
    dataset_name=task_name,
)
