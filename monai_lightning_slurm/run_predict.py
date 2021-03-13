import os
import json
from pathlib import Path
import nibabel as nib

import torch
import pytorch_lightning
import mlflow
import monai
from monai.data import NiftiSaver, DataLoader, Dataset, PersistentDataset
from monai.transforms import *
from monai.inferers import sliding_window_inference
from monai.networks.layers import Norm
from monai.metrics import compute_meandice
from monai.utils import set_determinism

from helpers import PIXDIM, train_val_split, DataStatsdWithPatient, StoreShaped

monai.config.print_config()

device = torch.device('cuda:0')

PATCH_SIZE = json.loads(os.environ.get('PATCH_SIZE', '[256, 256, 16]'))
CHANNELS = json.loads(os.environ.get('CHANNELS', '[16, 32, 64, 128]'))
STRIDES = json.loads(os.environ.get('STRIDES', '[2, 2, 2, 2]'))
BATCH_SIZE = 2
NUM_EPOCHS = 5000


model_path = os.environ.get('MODEL_PATH')
prediction_path = os.environ.get('PREDICTION_PATH')
data_path = os.environ.get('DATA_PATH')
cache_path = os.environ.get('CACHE_PATH')
results_path = os.environ.get('RESULTS_PATH')
job_id = os.environ.get('SLURM_JOBID')


class LymphomaNet(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self._model = monai.networks.nets.UNet(
            dimensions=3,
            in_channels=1,
            out_channels=2,
            channels=CHANNELS,
            strides=STRIDES,
            num_res_units=2,
            norm=Norm.BATCH,
            dropout=0.2,
        )
        self.loss_function = monai.losses.GeneralizedDiceLoss(
            to_onehot_y=True, softmax=True
        )
        # is it necessary?
        self.post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=2)
        self.post_label = AsDiscrete(to_onehot=True, n_classes=2)

        # is it necessary?
        self.best_val_dice = 0
        self.best_val_epoch = 0

    def forward(self, x):
        return self._model(x)

    def log_params(self):
        mlflow.log_params(
            {
                "NUM_EPOCHS": NUM_EPOCHS,
                "BATCH_SIZE": BATCH_SIZE,
                "PATCH_SIZE": PATCH_SIZE,
                "PIXDIM": PIXDIM,
                'MODEL': {
                    'net': 'UNet',
                    'channels': CHANNELS,
                    'strides': STRIDES,
                    'dropout': 0.2,
                },
            }
        )

    def prepare_data(self):
        data_images = sorted(
            [
                os.path.join(data_path, x)
                for x in os.listdir(data_path)
                if x.startswith('data')
            ]
        )
        data_labels = sorted(
            [
                os.path.join(data_path, x)
                for x in os.listdir(data_path)
                if x.startswith('label')
            ]
        )
        data_dicts = [
            {
                "image": image_name,
                "label": label_name,
                "patient": image_name.split("/")[-1]
                .replace("data", "")
                .replace(".nii.gz", ""),
            }
            for image_name, label_name in zip(data_images, data_labels)
        ]
        train_files, val_files = train_val_split(data_dicts)

        print(
            f'Training patients: {len(train_files)}, Validation patients: {len(val_files)}'
        )

        set_determinism(seed=0)

        train_transforms = Compose(
            [
                LoadNiftid(keys=['image', 'label']),
                AddChanneld(keys=['image', 'label']),
                Spacingd(
                    keys=["image", "label"], pixdim=PIXDIM, mode=("bilinear", "nearest")
                ),
                DataStatsdWithPatient(keys=["image", "label"]),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-100,
                    a_max=300,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
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
                Spacingd(
                    keys=["image", "label"], pixdim=PIXDIM, mode=("bilinear", "nearest")
                ),
                DataStatsdWithPatient(keys=["image", "label"]),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=-100,
                    a_max=300,
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                StoreShaped(keys=['image']),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ToTensord(keys=['image', 'label']),
            ]
        )

        self.train_ds = PersistentDataset(
            data=train_files, transform=train_transforms, cache_dir=cache_path
        )
        self.val_ds = PersistentDataset(
            data=val_files, transform=val_transforms, cache_dir=cache_path
        )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_ds, batch_size=1, num_workers=0)
        return val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), 2e-5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, total_steps=NUM_EPOCHS * BATCH_SIZE, verbose=True
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        # tensorboard_logs = {"train_loss": loss.item()}
        self.log("train_loss", loss, on_epoch=True, on_step=False)  # mlflow
        return {"loss": loss}  # "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        roi_size = PATCH_SIZE
        sw_batch_size = 1
        outputs = sliding_window_inference(
            images, roi_size, sw_batch_size, self.forward
        )
        loss = self.loss_function(outputs, labels)
        outputs = self.post_pred(outputs)
        labels = self.post_label(labels)
        value = compute_meandice(y_pred=outputs, y=labels, include_background=False)
        return {"val_loss": loss, "val_dice": value}

    def validation_epoch_end(self, outputs):
        val_dice, val_loss, num_items = 0, 0, 0
        for output in outputs:
            val_dice += output["val_dice"].sum().item()
            val_loss += output["val_loss"].sum().item()
            num_items += len(output["val_dice"])
        mean_val_dice = torch.tensor(val_dice / num_items)
        mean_val_loss = torch.tensor(val_loss / num_items)
        tensorboard_logs = {"val_dice": mean_val_dice, "val_loss": mean_val_loss}
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        print(
            f"current epoch: {self.current_epoch} current mean dice: {mean_val_dice:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} at epoch: {self.best_val_epoch}"
        )

        # MLFlow
        self.log('best_val_dice', self.best_val_dice, on_epoch=True, on_step=False)
        self.log("val_dice", mean_val_dice, on_epoch=True, on_step=False)
        self.log("val_loss", mean_val_loss, on_epoch=True, on_step=False)

        return {"log": tensorboard_logs}


output_path = Path(f"{results_path}/{job_id}")
output_path.mkdir(exist_ok=True)


device = torch.device("cuda:0")
net = LymphomaNet.load_from_checkpoint(checkpoint_path=model_path)
net.eval()
net.to(device)
net.prepare_data()
with torch.no_grad():
    for i, val_data in enumerate(net.val_dataloader()):
        patient = val_data["patient"][0]

        roi_size = PATCH_SIZE
        sw_batch_size = 4
        val_outputs = sliding_window_inference(
            val_data["image"].to(device), roi_size, sw_batch_size, net
        )

        data_path = output_path / f"transformed_data_{patient}.nii.gz"
        label_path = output_path / f"transformed_label_{patient}.nii.gz"
        pred_path = output_path / f"predicted_segmentation_{patient}.nii.gz"
        original_size_pred_path = (
            output_path / f"predicted_segmentation_original_size_{patient}.nii.gz"
        )

        nib.save(
            nib.Nifti1Image(
                val_data["image"][0][0].numpy(),
                val_data["image_meta_dict"]["affine"][0].numpy(),
            ),
            data_path,
        )
        nib.save(
            nib.Nifti1Image(
                val_data["label"][0][0].numpy(),
                val_data["label_meta_dict"]["affine"][0].numpy(),
            ),
            label_path,
        )
        preds = torch.argmax(val_outputs, dim=1).cpu().numpy().astype(np.float32)
        nib.save(
            nib.Nifti1Image(
                preds[0],
                val_data["label_meta_dict"]["affine"][0].numpy(),
            ),
            pred_path,
        )

        # Reverse transform order
        original_size_pred = np.zeros(
            val_data['image_last_seen_shape'][0].numpy(), dtype=np.float32
        )
        original_size_pred[
            :,
            val_data['foreground_start_coord'][0] : val_data['foreground_end_coord'][0],
            val_data['foreground_start_coord'][1] : val_data['foreground_end_coord'][1],
            val_data['foreground_start_coord'][2] : val_data['foreground_end_coord'][2],
        ] = preds

        original_spacing_transform = Spacing(
            val_data['image_meta_dict']['pixdim'][0, 1:4], mode="nearest"
        )
        original_spacing_preds = original_spacing_transform(
            original_size_pred, affine=val_data['image_meta_dict']['affine'].numpy()[0]
        )[0]

        assert (
            original_spacing_preds[0].shape
            == val_data['image_meta_dict']['spatial_shape'].numpy()
        ).all()

        nib.save(
            nib.Nifti1Image(
                original_spacing_preds[0],
                val_data['image_meta_dict']['original_affine'][0].numpy(),
            ),
            original_size_pred_path,
        )
