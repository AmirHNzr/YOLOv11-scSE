import os
import yaml
from glob import glob
import logging
import cv2
import numpy as np
import torch
from math import ceil
from typing import Tuple
import albumentations as A
from YOLOv11_scSE.Utility import pad_to,pad_xywh

log = logging.getLogger("dataset")
logging.basicConfig(level=logging.DEBUG)


class Dataset(torch.utils.data.Dataset):
    """
    Dataset class for loading images and annotations.

    Args:
        config (str): path to dataset config file
        batch_size (optional, int): batch size for dataloader
        mode (optional, str): dataset mode (train, val, test)
        img_size (optional, Tuple[int,int]): image size to pad images to
    """
    def __init__(self, config:str, batch_size:int=8, mode:str='train', img_size:Tuple[int,int]=(640, 640)):
        super().__init__()
        self.nonExistent = []
        self.config = yaml.safe_load(open(config, 'r'))
        self.dataset_path = os.path.join(os.path.dirname(config), self.config['path'])
        self.batch_size = batch_size
        self.img_size = img_size

        assert mode in ('train', 'val', 'test'), f'Invalid mode: {mode}'
        self.mode = mode

        if self.mode == 'train':
            self.augment = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.ColorJitter(p=0.3),
                A.MotionBlur(p=0.2)
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['cls'], min_visibility=0.1))
        else:
            self.augment = None

        self.im_files = self.get_image_paths()
        log.debug(f'Found {len(self.im_files)} images in {os.path.join(self.dataset_path, self.config[self.mode])}')

        self.label_files = self.get_label_paths()
        if self.label_files is not None:
            log.debug(f'Found {len(self.label_files)} labels in {os.path.join(self.dataset_path, self.config[self.mode+"_labels"])}')
        else:
            log.debug(f'No labels found in {os.path.join(self.dataset_path, self.config[self.mode+"_labels"])}')
        self.labels = self.get_labels()
        self.remove_nonexistent()
        self.seen_idxs = set()

    def get_image_paths(self):
        """
        Get image paths from dataset directory

        Searches recursively for .jpg, .png, and .jpeg files.
        """
        im_dir = os.path.join(self.dataset_path, self.config[self.mode])
        image_paths = glob(os.path.join(im_dir, '*.jpg')) + \
                      glob(os.path.join(im_dir, '*.png')) + \
                      glob(os.path.join(im_dir, '*.jpeg'))

        return image_paths

    def get_label_paths(self):
        """
        Get label paths from dataset directory

        Uses ids from image paths to find corresponding label files.

        If no label directory is found, returns None.
        """
        label_dir = os.path.join(self.dataset_path, self.config[self.mode+'_labels'])
        if os.path.isdir(label_dir):
            return [os.path.join(label_dir, os.path.splitext(os.path.basename(p))[0]+".txt") for p in self.im_files]
        return None

    def get_labels(self):
        """
        Gets labels from label files (assumes COCO formatting)

        Returns a list of dictionaries for each file
            {
                'cls': torch.Tensor of shape (num_boxes,)
                'bboxes': torch.Tensor of shape (num_boxes, 4) in (xywh) format
            }

        If no label files were found, returns a list of empty dictionaries.
        """
        if self.label_files is None:
            return [{} for _ in range(len(self.im_files))]
        labels = []
        for label_file in self.label_files:
            if not os.path.exists(label_file):
                self.nonExistent.append(os.path.splitext(os.path.basename(label_file))[0])
                continue
            annotations = open(label_file, 'r').readlines()
            cls, boxes = [], []
            for ann in annotations:
                ann = ann.strip('\n').split(' ')
                cls.append(int(ann[0]))

                # box provided in xywh format
                boxes.append(torch.from_numpy(np.array(ann[1:5], dtype=float)))

            labels.append({
                'cls': torch.tensor(cls),
                'bboxes': torch.vstack(boxes)
            })
        return labels

    def remove_nonexistent(self):
      self.im_files = list(filter(lambda x: os.path.splitext(os.path.basename(x))[0] not in self.nonExistent, self.im_files))


    def load_image(self, idx):
        """
        Loads image at specified index and prepares for model input.

        Changes image shape to be specified img_size, but preserves aspect ratio.
        """
        im_file = self.im_files[idx]
        im_id = os.path.splitext(os.path.basename(im_file))[0]
        image = cv2.cvtColor(cv2.imread(im_file), cv2.COLOR_BGR2RGB)

        h0, w0 = image.shape[:2]

        if h0 > self.img_size[0] or w0 > self.img_size[1]:
            # Resize to have max dimension of img_size, but preserve aspect ratio
            ratio = min(self.img_size[0]/h0, self.img_size[1]/w0)
            h, w = min(ceil(h0*ratio), self.img_size[0]), min(ceil(w0*ratio), self.img_size[1])
            image = cv2.resize(image, (h, w), interpolation=cv2.INTER_LINEAR)

        image = image.transpose((2, 0, 1))  # (h, w, 3) -> (3, h, w)
        image = torch.from_numpy(image).float() / 255.0

        # Pad image with black bars to desired img_size
        image, pads = pad_to(image, shape=self.img_size)

        h, w = image.shape[-2:]

        return image, pads, (h0,w0), im_id

    def get_image_and_label(self, idx):
        """
        Gets image and annotations at specified index
        """
        label = self.labels[idx]
        if idx in self.seen_idxs:
            return label
        if self.augment is not None and len(label.get('bboxes', [])) > 0:
            im_file = self.im_files[idx]
            image = cv2.cvtColor(cv2.imread(im_file), cv2.COLOR_BGR2RGB)

            # Convert bboxes to list
            bboxes = label['bboxes'].numpy()
            cls = label['cls'].numpy().tolist()

            # Convert YOLO (x_center, y_center, w, h) to (x_min, y_min, w, h)
            x_min = bboxes[:, 0] - bboxes[:, 2] / 2
            y_min = bboxes[:, 1] - bboxes[:, 3] / 2

            # Clip coordinates to [0, 1] with small epsilon
            eps = 1e-6
            x_min = np.clip(x_min, eps, 1.0 - eps)
            y_min = np.clip(y_min, eps, 1.0 - eps)
            bboxes[:, 2] = np.clip(bboxes[:, 2], eps, 1.0)  # width
            bboxes[:, 3] = np.clip(bboxes[:, 3], eps, 1.0)  # height

            # Replace original x_center, y_center so Albumentations sees clipped coords
            bboxes[:, 0] = x_min + bboxes[:, 2] / 2
            bboxes[:, 1] = y_min + bboxes[:, 3] / 2

            bboxes = bboxes.tolist()

            try:
                augmented = self.augment(image=image, bboxes=bboxes, cls=cls)
            except ValueError as e:
                print(f"Augmentation failed for sample {idx}: {e}")
                augmented = {'image': image, 'bboxes': bboxes, 'cls': cls}
            image = augmented['image']

            # Update labels with augmented bboxes
            if len(augmented['bboxes']) > 0:
                label['bboxes'] = torch.tensor(augmented['bboxes'])
                label['cls'] = torch.tensor(augmented['cls'])
            else:
                # If all boxes were filtered out, keep original boxes
                pass

            # Process augmented image
            h0, w0 = image.shape[:2]
            if h0 > self.img_size[0] or w0 > self.img_size[1]:
                ratio = min(self.img_size[0]/h0, self.img_size[1]/w0)
                h, w = min(ceil(h0*ratio), self.img_size[0]), min(ceil(w0*ratio), self.img_size[1])
                image = cv2.resize(image, (h, w), interpolation=cv2.INTER_LINEAR)

            image = image.transpose((2, 0, 1))
            image = torch.from_numpy(image).float() / 255.0
            image, pads = pad_to(image, shape=self.img_size)

            label['images'] = image
            label['padding'] = pads
            label['orig_shapes'] = (h0, w0)
            label['ids'] = os.path.splitext(os.path.basename(self.im_files[idx]))[0]
            label['bboxes'] = pad_xywh(label['bboxes'], label['padding'], label['orig_shapes'], return_norm=True)
        else:
            label['images'], label['padding'], label['orig_shapes'], label['ids'] = self.load_image(idx)
            label['bboxes'] = pad_xywh(label['bboxes'], label['padding'], label['orig_shapes'], return_norm=True)
        self.seen_idxs.add(idx)

        return label

    def __len__(self) -> int:
        return len(self.im_files)

    def __getitem__(self, index):
        return self.get_image_and_label(index)

    @staticmethod
    def collate_fn(batch):
        """
        Collate function to specify how to combine a list of samples into a batch
        """
        collated_batch = {}
        for k in batch[0].keys():
            if k == "images":
                collated_batch[k] = torch.stack([b[k] for b in batch], dim=0)
            elif k in ('cls', 'bboxes'):
                collated_batch[k] = torch.cat([b[k] for b in batch], dim=0)
            elif k in ('padding', 'orig_shapes', 'ids'):
                collated_batch[k] = [b[k] for b in batch]

        collated_batch['batch_idx'] = [torch.full((batch[i]['cls'].shape[0],), i) for i in range(len(batch))]
        collated_batch['batch_idx'] = torch.cat(collated_batch['batch_idx'], dim=0)

        return collated_batch