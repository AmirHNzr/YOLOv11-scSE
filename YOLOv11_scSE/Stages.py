import torch
import torch.nn as nn
import math
from YOLOv11_scSE.Blocks import Conv, C3K2, SPPF, C2PSA, ChannelSpatialSELayer, DFL
from YOLOv11_scSE.Utility import dist2bbox, make_anchors


class Backbone(torch.nn.Module):
    def __init__(self, width, depth, csp):
        super().__init__()
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.p4 = []
        self.p5 = []

        # p1/2
        self.p1.append(Conv(width[0], width[1], torch.nn.SiLU(), k=3, s=2, p=1))
        # p2/4
        self.p2.append(Conv(width[1], width[2], torch.nn.SiLU(), k=3, s=2, p=1))
        self.p2.append(C3K2(width[2], width[3], depth[0], csp[0], r=4))
        # p3/8
        self.p3.append(Conv(width[3], width[3], torch.nn.SiLU(), k=3, s=2, p=1))
        self.p3.append(C3K2(width[3], width[4], depth[1], csp[0], r=4))
        # p4/16
        self.p4.append(Conv(width[4], width[4], torch.nn.SiLU(), k=3, s=2, p=1))
        self.p4.append(C3K2(width[4], width[4], depth[2], csp[1], r=2))
        # p5/32
        self.p5.append(Conv(width[4], width[5], torch.nn.SiLU(), k=3, s=2, p=1))
        self.p5.append(C3K2(width[5], width[5], depth[3], csp[1], r=2))
        self.p5.append(SPPF(width[5], width[5]))
        self.p5.append(C2PSA(width[5], depth[4]))

        self.p1 = torch.nn.Sequential(*self.p1)
        self.p2 = torch.nn.Sequential(*self.p2)
        self.p3 = torch.nn.Sequential(*self.p3)
        self.p4 = torch.nn.Sequential(*self.p4)
        self.p5 = torch.nn.Sequential(*self.p5)

    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(p1)
        p3 = self.p3(p2)
        p4 = self.p4(p3)
        p5 = self.p5(p4)
        return p3, p4, p5

class Neck(torch.nn.Module):
    def __init__(self, width, depth, csp):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.h1 = C3K2(width[4] + width[5], width[4], depth[5], csp[0], r=2)
        self.h2 = C3K2(width[4] + width[4], width[3], depth[5], csp[0], r=2)
        self.h3 = Conv(width[3], width[3], torch.nn.SiLU(), k=3, s=2, p=1)
        self.h4 = C3K2(width[3] + width[4], width[4], depth[5], csp[0], r=2)
        self.h5 = Conv(width[4], width[4], torch.nn.SiLU(), k=3, s=2, p=1)
        self.h6 = C3K2(width[4] + width[5], width[5], depth[5], csp[1], r=2)

        self.attn1 = ChannelSpatialSELayer(width[3],reduction_ratio = 4)
        self.attn2 = ChannelSpatialSELayer(width[4],reduction_ratio = 4)
        self.attn3 = ChannelSpatialSELayer(width[5],reduction_ratio = 4)


    def forward(self, x):
        p3, p4, p5 = x
        p4 = self.h1(torch.cat(tensors=[self.up(p5), p4], dim=1))
        p3 = self.h2(torch.cat(tensors=[self.up(p4), p3], dim=1))
        p4 = self.h4(torch.cat(tensors=[self.h3(p3), p4], dim=1))
        p5 = self.h6(torch.cat(tensors=[self.h5(p4), p5], dim=1))

        p3 = self.attn1(p3)
        p4 = self.attn2(p4)
        p5 = self.attn3(p5)

        return p3, p4, p5


class Head(torch.nn.Module):
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)

    def __init__(self, nc=80, filters=()):
        super().__init__()
        self.ch = 16  # DFL channels
        self.nc = nc  # number of classes
        self.nl = len(filters)  # number of detection layers
        self.no = nc + self.ch * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        self.n_outputs = self.no
        self.reg_max = self.ch


        box = max(64, filters[0] // 4)
        cls = max(80, filters[0], self.nc)

        self.dfl = DFL(self.ch)

        self.box = torch.nn.ModuleList(
           torch.nn.Sequential(Conv(x, box,torch.nn.SiLU(), k=3, p=1),
           Conv(box, box,torch.nn.SiLU(), k=3, p=1),
           torch.nn.Conv2d(box, out_channels=4 * self.ch,kernel_size=1)) for x in filters)

        self.cls = torch.nn.ModuleList(
            torch.nn.Sequential(Conv(x, x, torch.nn.SiLU(), k=3, p=1, g=x),
            Conv(x, cls, torch.nn.SiLU()),
            Conv(cls, cls, torch.nn.SiLU(), k=3, p=1, g=cls),
            Conv(cls, cls, torch.nn.SiLU()),
            torch.nn.Conv2d(cls, out_channels=self.nc,kernel_size=1)) for x in filters)

    def forward(self, x):
        shape = x[0].shape
        for i, (box, cls) in enumerate(zip(self.box, self.cls)):
            x[i] = torch.cat(tensors=(box(x[i]), cls(x[i])), dim=1)
        if self.training:
            return x
        if self.shape != shape:
            self.shape = shape
            self.anchors, self.strides = (i.transpose(0, 1) for i in make_anchors(x, self.stride))


        x = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], dim=2)
        box, cls = x.split(split_size=(4*self.ch, self.nc), dim=1)
        box = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), dim=1)


        return torch.cat(tensors=(box * self.strides, cls.sigmoid()), dim=1)

    def initialize_biases(self):
        for box, cls, s in zip(self.box, self.cls, self.stride):
            # box
            box[-1].bias.data[:] = 1.0
            # cls
            cls[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)
