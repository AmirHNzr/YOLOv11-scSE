from carafe import carafe
import torch
import torch.nn as nn
import torch.nn.functional as F
from YOLOv11_scSE.Blocks import Conv, C3K2, ChannelSpatialSELayer


# CARAFE Module
# ATTENTION:
# To use CARAFE-modified model you need to use GPU as your device. 
# The backpropagation is not supported on CPU.
class CARAFE(torch.nn.Module):
    """A unified package of CARAFE upsampler that contains: 1) channel
    compressor 2) content encoder 3) CARAFE op.

    Official implementation of ICCV 2019 paper
    `CARAFE: Content-Aware ReAssembly of FEatures
    <https://arxiv.org/abs/1905.02188>`_.

    Args:
        channels (int): input feature channels
        scale_factor (int): upsample ratio
        up_kernel (int): kernel size of CARAFE op
        up_group (int): group size of CARAFE op
        encoder_kernel (int): kernel size of content encoder
        encoder_dilation (int): dilation of content encoder
        compressed_channels (int): output channels of channels compressor

    Returns:
        upsampled feature map
    """

    def __init__(self,
                 channels: int,
                 scale_factor: int,
                 up_kernel: int = 5,
                 up_group: int = 1,
                 encoder_kernel: int = 3,
                 encoder_dilation: int = 1,
                 compressed_channels: int = 64):
        super().__init__()
        self.channels = channels
        self.scale_factor = scale_factor
        self.up_kernel = up_kernel
        self.up_group = up_group
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
        self.compressed_channels = compressed_channels
        self.channel_compressor = nn.Conv2d(channels, self.compressed_channels,1)
        self.content_encoder = nn.Conv2d(
            self.compressed_channels,
            self.up_kernel * self.up_kernel * self.up_group *
            self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, val=0.0)
        nn.init.normal_(self.content_encoder.weight, mean=0.0, std=0.001)
        if hasattr(self.content_encoder, 'bias') and self.content_encoder.bias is not None:
            nn.init.constant_(self.content_encoder.bias, val=0.0)

    def kernel_normalizer(self, mask):
        mask = F.pixel_shuffle(mask, self.scale_factor)
        n, mask_c, h, w = mask.size()
        # use float division explicitly,
        # to void inconsistency while exporting to onnx
        mask_channel = int(mask_c / float(self.up_kernel**2))
        mask = mask.view(n, mask_channel, -1, h, w)

        mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        mask = mask.view(n, mask_c, h, w).contiguous()

        return mask

    def feature_reassemble(self, x, mask):
        x = carafe(x, mask, self.up_kernel, self.up_group, self.scale_factor)
        return x

    def forward(self, x):
        compressed_x = self.channel_compressor(x)
        mask = self.content_encoder(compressed_x)
        mask = self.kernel_normalizer(mask)

        x = self.feature_reassemble(x, mask)
        return x
    
class Neck(torch.nn.Module):
    def __init__(self, width, depth, csp):
        super().__init__()
        self.up1 = CARAFE(channels=width[5], scale_factor=2, up_kernel=9, up_group=1)
        self.h1 = C3K2(width[4] + width[5], width[4], depth[5], csp[0], r=2)
        self.up2 = CARAFE(channels=width[4], scale_factor=2, up_kernel=9, up_group=1)
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
        p4 = self.h1(torch.cat(tensors=[self.up1(p5), p4], dim=1))
        p3 = self.h2(torch.cat(tensors=[self.up2(p4), p3], dim=1))
        p4 = self.h4(torch.cat(tensors=[self.h3(p3), p4], dim=1))
        p5 = self.h6(torch.cat(tensors=[self.h5(p4), p5], dim=1))

        p3 = self.attn1(p3)
        p4 = self.attn2(p4)
        p5 = self.attn3(p5)

        return p3, p4, p5
