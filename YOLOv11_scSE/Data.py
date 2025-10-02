from YOLOv11_scSE.Dataset import Dataset

Trainset = Dataset("/content/data.yaml", mode="train", img_size=(640, 640))
Validset = Dataset("/content/data.yaml", mode="val", img_size=(640, 640))
Testset = Dataset("/content/data.yaml", mode="test", img_size=(640, 640))

# Check dataset lengths
print(len(Trainset))
print(len(Validset))
print(len(Testset))
