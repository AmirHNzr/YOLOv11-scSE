import torch
import torch.optim as optim
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import gc
from YOLOv11_scSE.Loss import DetectionLoss
from YOLOv11_scSE.Data import *
from YOLOv11_scSE.model import YOLOv11

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLOv11().build_model(version='sm', num_classes=2).to(device)
BATCH_SIZE = 8
chunk_size = 64
EPOCHS = 50
total_chunks = (len(Trainset) + chunk_size - 1) // chunk_size
steps_per_epoch = (len(Trainset) // chunk_size + 1)*(chunk_size // BATCH_SIZE)
shuffled_indices = np.random.permutation(len(Trainset))
criterion = DetectionLoss(model, device=device)

# AdamW optimizer
optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-2,                # base learning rate
    betas=(0.9, 0.999),     # defaults
    eps=1e-8,               # numerical stability
    weight_decay=1e-4       # decoupled weight decay
)

# Plateau scheduler - reduces LR when a metric has stopped improving
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',             # 'min' for loss, 'max' for accuracy/metric
    factor=0.5,              # LR reduction factor
    patience=10,              # epochs to wait before reducing
    threshold=1e-1
    )


running_loss = 0.0
for epoch in range(EPOCHS):
    running_loss = 0.0
    # Shuffle data every epoch for randomness
    shuffled_indices = np.random.permutation(len(Trainset))

    print(f"Epoch {epoch + 1}/{EPOCHS}")
    # Preventing RAM or VRAM overload
    for chunk_start in range(0, len(Trainset), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(Trainset))
        chunk_indices = shuffled_indices[chunk_start:chunk_end]

        subset = Subset(Trainset, chunk_indices)
        loader = DataLoader(
            subset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=Trainset.collate_fn,
            pin_memory=True,
        )

        model.train()
        print(f"  Started chunk: {chunk_start // chunk_size + 1}/{total_chunks}")
        for batch in loader:
            images = batch['images'].to(device, non_blocking=True)
            targets = {
                'batch_idx': batch['batch_idx'].to(device, non_blocking=True),
                'cls': batch['cls'].to(device, non_blocking=True),
                'bboxes': batch['bboxes'].to(device, non_blocking=True),
            }

            preds = model(images)
            loss = criterion.compute_loss(targets, preds)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            del images, targets, preds, loss
            torch.cuda.empty_cache()

        del loader, subset
        gc.collect()
        torch.cuda.empty_cache()
    # End of epoch: save checkpoint
    torch.save(model.state_dict(), f'/AttnOnly_scSE_{epoch+1}.pt')
    scheduler.step(running_loss/len(Trainset))
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(Trainset):.4f}")
    print(f"LR:{scheduler.get_last_lr()}")

ckpt = {
    "epoch": epoch,
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
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
torch.save(ckpt, f"/content/AttnOnly_scSE.pth")