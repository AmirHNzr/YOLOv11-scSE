from sched import scheduler
import torch
import torch.optim as optim
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import gc
import random
from YOLOv11_scSE.Loss import DetectionLoss
from YOLOv11_scSE.Data import *
from YOLOv11_scSE.model import YOLOv11


class trainer:
    def __init__(self, model, optimizer, criterion, EPOCH=50, BATCH=8, CHUNK=64, device = torch.device('cpu')):
        self.epoch = EPOCH
        self.batch = BATCH
        self.chunk = CHUNK
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.scheduler = ReduceLROnPlateau(
                                            optimizer,
                                            mode='min',             # 'min' for loss, 'max' for accuracy/metric
                                            factor=0.1,              # LR reduction factor
                                            patience=5,              # epochs to wait before reducing
                                            threshold=1e-1
                                            )


    def batch_data(self, batch):
        images = batch['images'].to(self.device, non_blocking=True)
        targets = {
            'batch_idx': batch['batch_idx'].to(self.device, non_blocking=True),
            'cls': batch['cls'].to(self.device, non_blocking=True),
            'bboxes': batch['bboxes'].to(self.device, non_blocking=True),
        }

        preds = self.model(images)
        loss = self.criterion.compute_loss(targets, preds)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.running_loss += loss.item()

        del images, targets, preds, loss
        torch.cuda.empty_cache()

    def chunk_data(self, Trainset):
        for chunk_start in range(0, len(Trainset), self.chunk):
            chunk_end = min(chunk_start + self.chunk, len(Trainset))
            chunk_indices = self.shuffled_indices[chunk_start:chunk_end]

            subset = Subset(Trainset, chunk_indices)
            loader = DataLoader(
                subset,
                batch_size=self.batch,
                shuffle=True,
                collate_fn=Trainset.collate_fn,
                pin_memory=True,
            )

            print(f"  Started chunk: {chunk_start // self.chunk + 1}/{self.total_chunks}")
            for batch in loader:
                self.batch_data(batch)

            del loader, subset
            gc.collect()
            torch.cuda.empty_cache()

    def saveRun(self, epoch):
        ckpt = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            # include scaler if using AMP:
            # "scaler": scaler.state_dict(),
            # optional but useful:
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        }
        torch.save(ckpt, f"/content/AttnOnly_scSE_{epoch}.pth")

    def run(self, trainset):
        for epoch in range(1, self.epoch + 1):
            self.model.train()

            self.total_chunks = (len(trainset) + self.chunk - 1) // self.chunk
            self.steps_per_epoch = (len(trainset) // self.chunk + 1) * (self.chunk // self.batch)
            self.shuffled_indices = np.random.permutation(len(trainset))

            self.running_loss = 0.0
            self.chunk_data(trainset)

            print(f"Epoch {epoch}/{self.epoch}, Loss: {self.running_loss/len(trainset):.4f}")
            self.saveRun(epoch)
            print(f"LR:{self.scheduler.get_last_lr()}")
            self.scheduler.step(self.running_loss / len(trainset))

        return self.running_loss / len(trainset)



