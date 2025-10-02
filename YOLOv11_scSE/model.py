from YOLOv11_scSE.Stages import Backbone, Neck, Head
from YOLOv11_scSE.Blocks import Conv,fuse_conv
import torch
import torch.nn as nn

class YOLO(torch.nn.Module):
    def __init__(self, width, depth, c3k, num_classes):
        super().__init__()
        self.net = Backbone(width, depth, c3k).to(device)
        self.fpn = Neck(width, depth, c3k).to(device)
        self.loss_gains = {'cls': 0.5, 'iou': 7.5, 'dfl': 1.5}

        img_dummy = torch.zeros(1, width[0], 256, 256).to(device)
        self.head = Head(num_classes, (width[3], width[4], width[5])).to(device)
        input_size = img_dummy.shape[-1]
        self.head.stride = torch.tensor([input_size / x.shape[-2] for x in self.forward(img_dummy)])
        self.stride = self.head.stride
        self.head.initialize_biases()



    def forward(self, x):
        x = self.net(x)
        x = self.fpn(x)
        return self.head(list(x))

    def fuse(self):
        for m in self.modules():
            if type(m) is Conv and hasattr(m, 'norm'):
                m.conv = fuse_conv(m.conv, m.norm)
                m.forward = m.fuse_forward
                delattr(m, 'norm')
        return self
    
class YOLOv11:
  def __init__(self):

    self.dynamic_weighting = {
      'n':{'c3k': [False, True], 'depth' : [1, 1, 1, 1, 1, 1], 'width' : [3, 16, 32, 64, 128, 256]},
      's':{'c3k': [False, True], 'depth' : [1, 1, 1, 1, 1, 1], 'width' : [3, 32, 64, 128, 256, 512]},
      's_mod':{'c3k': [True, True], 'depth' : [2, 2, 2, 2, 2, 2], 'width' : [3, 32, 64, 128, 256, 512]},
      'm':{'c3k': [True, True], 'depth' : [1, 1, 1, 1, 1, 1], 'width' : [3, 64, 128, 256, 512, 512]},
      'l':{'c3k': [True, True], 'depth' : [2, 2, 2, 2, 2, 2], 'width' : [3, 64, 128, 256, 512, 512]},
      'x':{'c3k': [True, True], 'depth' : [2, 2, 2, 2, 2, 2], 'width' : [3, 96, 192, 384, 768, 768]},
    }
  def build_model(self, version, num_classes):
    c3k = self.dynamic_weighting[version]['c3k']
    depth = self.dynamic_weighting[version]['depth']
    width = self.dynamic_weighting[version]['width']
    return YOLO(width, depth, c3k, num_classes)