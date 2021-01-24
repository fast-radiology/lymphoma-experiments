import os
import json
from pathlib import Path
import nibabel as nib

import torch
import monai
from monai.data import CacheDataset, NiftiSaver, DataLoader, Dataset
from monai.transforms import *
from monai.inferers import sliding_window_inference
from monai.networks.layers import Norm
from monai.metrics import compute_meandice
from monai.utils import set_determinism

monai.config.print_config()

device = torch.device('cuda:0')

PIXDIM = json.loads(os.environ.get('PIXDIM', '[0.7, 0.7, 0.7]'))
PATCH_SIZE = json.loads(os.environ.get('PATCH_SIZE', '[256, 256, 16]'))

data_path = os.environ.get('DATA_PATH')
results_path = os.environ.get('RESULTS_PATH')
job_id = os.environ.get('SLURM_JOBID')

output_path = Path(f"{results_path}/{job_id}")
output_path.mkdir(exist_ok=True)

train_images = sorted(
    [os.path.join(data_path, x) for x in os.listdir(data_path) if x.startswith('data')]
)
train_labels = sorted(
    [os.path.join(data_path, x) for x in os.listdir(data_path) if x.startswith('label')]
)

data_dicts = [
    {'image': image_name, 'label': label_name, 'patient': image_name.split('/')[1].replace('data', '').replace('.nii.gz', '')}
    for image_name, label_name in zip(train_images, train_labels)
]
train_files, val_files = data_dicts[:6] + data_dicts[8:], data_dicts[6:8]
print(f'Training patients: {len(train_files)}, Validation patients: {len(val_files)}')

set_determinism(seed=0)

train_transforms = Compose(
    [
        LoadNiftid(keys=['image', 'label']),
        AddChanneld(keys=['image', 'label']),
        Spacingd(keys=["image", "label"], pixdim=PIXDIM, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=["image"], a_min=-100, a_max=300, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=PATCH_SIZE,
            pos=1,
            neg=1,
            num_samples=16,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(["image", "label"], spatial_axis=[0, 1, 2], prob=0.5),
        ToTensord(keys=['image', 'label']),
    ]
)
val_transforms = Compose(
    [
        LoadNiftid(keys=['image', 'label']),
        AddChanneld(keys=['image', 'label']),
        Spacingd(keys=["image", "label"], pixdim=PIXDIM, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=["image"], a_min=-100, a_max=300, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=['image', 'label']),
    ]
)

train_ds = CacheDataset(
    data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=0
)
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)

val_ds = CacheDataset(
    data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=0
)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)

model = monai.networks.nets.UNet(
    dimensions=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
    dropout=0.2,
).to(device)

loss_function = monai.losses.GeneralizedDiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-3)

epoch_num = 5000
val_interval = 50
best_metric = -1
best_metric_epoch = -1

epoch_loss_values = list()
metric_values = list()

for epoch in range(epoch_num):
    if epoch % 100 == 99:
        print(f"epoch {epoch + 1}/{epoch_num}")

    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data['image'].to(device), batch_data['label'].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            metric_sum = 0.0
            metric_count = 0

            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data['image'].to(device),
                    val_data['label'].to(device),
                )
                roi_size = PATCH_SIZE
                sw_batch_size = 1
                val_outputs = sliding_window_inference(
                    val_inputs, roi_size, sw_batch_size, model
                )
                value = compute_meandice(
                    y_pred=val_outputs,
                    y=val_labels,
                    include_background=False,
                    to_onehot_y=True,
                    mutually_exclusive=True,
                )
                metric_count += len(value)
                metric_sum += value.sum().item()
            metric = metric_sum / metric_count
            metric_values.append(metric)

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), output_path / 'best_metric_model.pth')
                print('saved new best metric model')
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
            )


model.eval()
with torch.no_grad():
    for i, val_data in enumerate(val_loader):
        patient = val_data["patient"][0]

        roi_size = PATCH_SIZE
        sw_batch_size = 1
        val_outputs = sliding_window_inference(
            val_data['image'].to(device), roi_size, sw_batch_size, model
        )

        data_path = output_path / f'transformed_data_{patient}.nii.gz'
        label_path = output_path / f'transformed_label_{patient}.nii.gz'
        pred_path = output_path / f'predicted_segmentation_{patient}.nii.gz'
        nib.save(nib.Nifti1Image(val_data['image'][0][0].numpy(), np.eye(4)), data_path)
        nib.save(
            nib.Nifti1Image(val_data['label'][0][0].numpy(), np.eye(4)), label_path
        )
        nib.save(
            nib.Nifti1Image(
                torch.argmax(val_outputs, dim=1)[0].cpu().numpy().astype(np.float32),
                np.eye(4),
            ),
            pred_path,
        )
