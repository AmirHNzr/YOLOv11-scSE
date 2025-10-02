import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Any, Union


def xywh2xyxy(xywh:torch.Tensor):
    """
    Convert bounding box coordinates from (xywh) to (xyxy)
    """
    xy, wh = torch.chunk(xywh, 2, dim=-1)
    return torch.cat((xy - wh / 2, xy + wh / 2), dim=-1)

def xyxy2xywh(xyxy:torch.Tensor):
    """
    Convert bounding box coordinates from (xyxy) to (xywh)
    """
    xy_lt, xy_rb = torch.chunk(xyxy, 2, dim=-1)
    return torch.cat(((xy_lt + xy_rb) / 2, xy_rb - xy_lt), dim=-1)

def dist2bbox(distance:torch.Tensor, anchor_points:torch.Tensor, xywh:bool=True, dim:int=-1):
    """
    Transform distance in (ltrb) to bounding box (xywh) or (xyxy)
    """

    lt, rb = torch.chunk(distance, 2, dim=dim)
    xy_lt = anchor_points - lt
    xy_rb = anchor_points + rb

    if xywh:
        center = (xy_lt + xy_rb) / 2
        wh = xy_rb - xy_lt
        return torch.cat((center, wh), dim=dim)

    return torch.cat((xy_lt, xy_rb), dim=dim)

def bbox2dist(bbox:torch.Tensor, anchor_points:torch.Tensor, reg_max:int):
    """
    Transform bounding box (xyxy) to distance (ltrb)
    """
    xy_lt, xy_rb = torch.chunk(bbox, 2, dim=-1)
    lt = anchor_points - xy_lt
    rb = xy_rb - anchor_points
    return torch.cat((lt, rb), dim=-1).clamp(max=reg_max-0.01)

def make_anchors(x, strides, offset=0.5):
    assert x is not None
    anchor_tensor, stride_tensor = [], []
    dtype, device = x[0].dtype, x[0].device
    for i, stride in enumerate(strides):
        _, _, h, w = x[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + offset  # shift y
        sy, sx = torch.meshgrid(sy, sx)
        anchor_tensor.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_tensor), torch.cat(stride_tensor)

def pad_to(x:torch.Tensor, stride:int=None, shape:Tuple[int,int]=None):
    """
    Pads an image with zeros to make it divisible by stride
    (Pads both top/bottom and left/right evenly) or pads to
    specified shape.

    Args:
        x (Tensor): image tensor of shape (..., h, w)
        stride (optional, int): stride of model
        shape (optional, Tuple[int,int]): shape to pad image to
    """
    h, w = x.shape[-2:]

    if stride is not None:
        h_new = h if h % stride == 0 else h + stride - h % stride
        w_new = w if w % stride == 0 else w + stride - w % stride
    elif shape is not None:
        h_new, w_new = shape

    t, b = int((h_new-h) / 2), int(h_new-h) - int((h_new-h) / 2)
    l, r = int((w_new-w) / 2), int(w_new-w) - int((w_new-w) / 2)
    pads = (l, r, t, b)

    x_padded = F.pad(x, pads, "constant", 0)

    return x_padded, pads

def unpad(x:torch.Tensor, pads:tuple):
    l, r, t, b = pads
    return x[..., t:-b, l:-r]

def pad_xyxy(xyxy:Union[np.ndarray, torch.Tensor], pads:Tuple[int, int, int, int], im_size:Tuple[int, int]=None, return_norm:bool=False):
    """
    Add padding to the bounding boxes based on image padding

    Args:
        xyxy: The bounding boxes in the format of `(x_min, y_min, x_max, y_max)`.
            if `im_size` is provided, assume this is normalized coordinates
        pad: The padding added to the image in the format
            of `(left, right, top, bottom)`.
        im_size: The size of the original image in the format of `(height, width)`.
        return_norm: Whether to return normalized coordinates
    """
    l, r, t, b = pads
    if return_norm and im_size is None:
        raise ValueError("im_size must be provided if return_norm is True")

    if im_size is not None:
        h, w = im_size
        hpad, wpad = h+b+t, w+l+r

    if isinstance(xyxy, np.ndarray):
        xyxy_unnorm = xyxy * np.array([w, h, w, h], dtype=xyxy.dtype) if im_size else xyxy
        padded = xyxy_unnorm + np.array([l, t, l, t], dtype=xyxy.dtype)
        if return_norm:
            padded /= np.array([wpad, hpad, wpad, hpad], dtype=xyxy.dtype)
        return padded

    xyxy_unnorm = xyxy * torch.tensor([w, h, w, h], dtype=xyxy.dtype, device=xyxy.device) if im_size else xyxy
    padded = xyxy_unnorm + torch.tensor([l, t, l, t], dtype=xyxy.dtype, device=xyxy.device)
    if return_norm:
        padded /= torch.tensor([wpad, hpad, wpad, hpad], dtype=xyxy.dtype, device=xyxy.device)
    return padded

def pad_xywh(xywh:Union[np.ndarray, torch.Tensor], pads:Tuple[int, int, int, int], im_size:Tuple[int, int]=None, return_norm:bool=False):
    """
    Add padding to the bounding boxes based on image padding

    Args:
        xywh: The bounding boxes in the format of `(x, y, w, h)`.
            if `im_size` is provided, assume this is normalized coordinates
        pad: The padding added to the image in the format
            of `(left, right, top, bottom)`.
        im_size: The size of the original image in the format of `(height, width)`.
        return_norm: Whether to return normalized coordinates
    """
    l, r, t, b = pads
    if return_norm and im_size is None:
        raise ValueError("im_size must be provided if return_norm is True")

    if im_size is not None:
        h, w = im_size
        hpad, wpad = h+b+t, w+l+r

    if isinstance(xywh, np.ndarray):
        xywh_unnorm = xywh * np.array([w, h, w, h], dtype=xywh.dtype) if im_size else xywh
        padded = xywh_unnorm + np.array([l, t, 0, 0], dtype=xywh.dtype)
        if return_norm:
            padded /= np.array([wpad, hpad, wpad, hpad], dtype=xywh.dtype)
        return padded

    xywh_unnorm = xywh * torch.tensor([w, h, w, h], dtype=xywh.dtype, device=xywh.device) if im_size else xywh
    padded = xywh_unnorm + torch.tensor([l, t, 0, 0], dtype=xywh.dtype, device=xywh.device)
    if return_norm:
        padded /= torch.tensor([wpad, hpad, wpad, hpad], dtype=xywh.dtype, device=xywh.device)
    return padded

def unpad_xyxy(xyxy:Union[np.ndarray, torch.Tensor], pads:Tuple[int, int, int, int]):
    """
    Remove padding from the bounding boxes based on image padding

    Args:
        pad: The padding added to the image in the format
            of `(left, right, top, bottom)`.
    """
    l, r, t, b = pads
    if isinstance(xyxy, np.ndarray):
        return xyxy - np.array([l, t, l, t], dtype=xyxy.dtype)
    return xyxy - torch.tensor([l, t, l, t], dtype=xyxy.dtype, device=xyxy.device)

def box_iou_batch(gt_boxes: np.ndarray, pred_boxes: np.ndarray) -> np.ndarray:
    """
    Compute Intersection over Union (IoU) of two sets of bounding boxes -
        `gt_boxes` and `pred_boxes`. Both sets
        of boxes are expected to be in `(xyxy)` format.

    Args:
        gt_boxes (np.ndarray): 2D `np.ndarray` representing ground-truth boxes.
            `shape = (N, 4)` where `N` is number of true objects.
        pred_boxes (np.ndarray): 2D `np.ndarray` representing detection boxes.
            `shape = (M, 4)` where `M` is number of detected objects.

    Returns:
        np.ndarray: Pairwise IoU of boxes from `gt_boxes` and `pred_boxes`.
            `shape = (N, M)` where `N` is number of true objects and
            `M` is number of detected objects.
    """

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area_true = box_area(gt_boxes.T)
    area_detection = box_area(pred_boxes.T)

    top_left = np.maximum(gt_boxes[:, None, :2], pred_boxes[:, :2])
    bottom_right = np.minimum(gt_boxes[:, None, 2:], pred_boxes[:, 2:])

    area_inter = np.prod(np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)
    return area_inter / (area_true[:, None] + area_detection - area_inter)

def non_max_suppression(predictions: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """
    Perform Non-Maximum Suppression (NMS) on object detection predictions.

    Args:
        predictions (np.ndarray): An array of object detection predictions in
            the format of `(x_min, y_min, x_max, y_max, score)`
            or `(x_min, y_min, x_max, y_max, score, class)`.
        iou_threshold (float, optional): The intersection-over-union threshold
            to use for non-maximum suppression.

    Returns:
        np.ndarray: A boolean array indicating which predictions to keep after n
            on-maximum suppression.

    Raises:
        AssertionError: If `iou_threshold` is not within the
            closed range from `0` to `1`.
    """
    assert 0 <= iou_threshold <= 1, (
        "Value of `iou_threshold` must be in the closed range from 0 to 1, "
        f"{iou_threshold} given."
    )
    rows, columns = predictions.shape

    # add column #5 - category filled with zeros for agnostic nms
    if columns == 5:
        predictions = np.c_[predictions, np.zeros(rows)]

    # sort predictions column #4 - score
    sort_index = np.flip(predictions[:, 4].argsort())
    predictions = predictions[sort_index]

    boxes = predictions[:, :4]
    categories = predictions[:, 5]
    ious = box_iou_batch(boxes, boxes)
    ious = ious - np.eye(rows)

    keep = np.ones(rows, dtype=bool)

    for index, (iou, category) in enumerate(zip(ious, categories)):
        if not keep[index]:
            continue

        # drop detections with iou > iou_threshold and
        # same category as current detections
        condition = (iou > iou_threshold) & (categories == category)
        keep = keep & ~condition

    return keep[sort_index.argsort()]



##########################################
# Needed Components for Loss (YOLOv8 Loss)
##########################################

def bbox_iou(box1:torch.Tensor, box2:torch.Tensor, xywh:bool=True, eps:float=1e-10):
    """
    Calculate IoU between two bounding boxes

    Args:
        box1: (Tensor) with shape (..., 1 or n, 4)
        box2: (Tensor) with shape (..., n, 4)
        xywh: (bool) True if box coordinates are in (xywh) else (xyxy)

    Returns:
        iou: (Tensor) with IoU
    """
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, dim=-1), box2.chunk(4, dim=-1)
        b1_x1, b1_y1, b1_x2, b1_y2 = x1 - w1 / 2, y1 - h1 / 2, x1 + w1 / 2, y1 + h1 / 2
        b2_x1, b2_y1, b2_x2, b2_y2 = x2 - w2 / 2, y2 - h2 / 2, x2 + w2 / 2, y2 + h2 / 2
    else:
        (b1_x1, b1_y1, b1_x2, b1_y2), (b2_x1, b2_y1, b2_x2, b2_y2) = box1.chunk(4, dim=-1), box2.chunk(4, dim=-1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    intersection = (torch.minimum(b1_x2, b2_x2) - torch.maximum(b1_x1, b2_x1)).clamp(min=0) * \
                   (torch.minimum(b1_y2, b2_y2) - torch.maximum(b1_y1, b2_y1)).clamp(min=0)

    union = w1 * h1 + w2 * h2 - intersection + eps

    iou = intersection / union

    return iou

def df_loss(pred_box_dist:torch.Tensor, targets:torch.Tensor):
    """
    Sum of left and right DFL losses
    """
    target_left = targets.long()
    target_right = target_left + 1
    weight_left = target_right - targets
    weight_right = 1 - weight_left

    dfl_left = F.cross_entropy(pred_box_dist, target_left.view(-1), reduction='none').view(target_left.shape) * weight_left
    dfl_right = F.cross_entropy(pred_box_dist, target_right.view(-1), reduction='none').view(target_right.shape) * weight_right

    return torch.mean(dfl_left + dfl_right, dim=-1, keepdim=True)

def anchors_in_gt_boxes(anchor_points:torch.Tensor, gt_boxes:torch.Tensor, eps:float=1e-8):
    """
    Returns mask for positive anchor centers that are in GT boxes

    Args:
        anchor_points (Tensor): Anchor points of shape (n_anchors, 2)
        gt_boxes (Tensor): GT boxes of shape (batch_size, n_boxes, 4)

    Returns:
        mask (Tensor): Mask of shape (batch_size, n_boxes, n_anchors)
    """
    n_anchors = anchor_points.shape[0]
    batch_size, n_boxes, _ = gt_boxes.shape
    lt, rb = gt_boxes.view(-1, 1, 4).chunk(chunks=2, dim=2)
    box_deltas = torch.cat((anchor_points.unsqueeze(0) - lt, rb - anchor_points.unsqueeze(0)), dim=2).view(batch_size, n_boxes, n_anchors, -1)
    return torch.amin(box_deltas, dim=3) > eps

def select_highest_iou(mask:torch.Tensor, ious:torch.Tensor, num_max_boxes:int):
    """
    Select GT box with highest IoU for each anchor

    Args:
        mask (Tensor): Mask of shape (batch_size, num_max_boxes, n_anchors)
        ious (Tensor): IoU of shape (batch_size, num_max_boxes, n_anchors)

    Returns:
        target_gt_box_idxs (Tensor): Index of GT box with highest IoU for each anchor of shape (batch_size, n_anchors)
        fg_mask (Tensor): Mask of shape (batch_size, n_anchors) where 1 indicates positive anchor
        mask (Tensor): Mask of shape (batch_size, num_max_boxes, n_anchors) where 1 indicates positive anchor
    """
    # sum over n_max_boxes dim to get num GT boxes assigned to each anchor
    # (batch_size, num_max_boxes, n_anchors) -> (batch_size, n_anchors)
    fg_mask = mask.sum(dim=1)

    if fg_mask.max() > 1:
        # If 1 anchor assigned to more than one GT box, select the one with highest IoU
        max_iou_idx = ious.argmax(dim=1)  # (batch_size, n_anchors)

        # mask for where there are more than one GT box assigned to anchor
        multi_gt_mask = (fg_mask.unsqueeze(1) > 1).expand(-1, num_max_boxes, -1)  # (batch_size, num_max_boxes, n_anchors)

        # mask for GT box with highest IoU
        max_iou_mask = torch.zeros_like(mask, dtype=torch.bool)
        max_iou_mask.scatter_(dim=1, index=max_iou_idx.unsqueeze(1), value=1)

        mask = torch.where(multi_gt_mask, max_iou_mask, mask)
        fg_mask = mask.sum(dim=1)

    target_gt_box_idxs = mask.argmax(dim=1)  # (batch_size, n_anchors)
    return target_gt_box_idxs, fg_mask, mask

class TaskAlignedAssigner(nn.Module):
    """
    Task-aligned assigner for object detection
    """
    def __init__(self, topk:int=10, num_classes:int=80, alpha:float=1.0, beta:float=6.0, eps:float=1e-8, device:str='cuda'):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.device = device

        self.bg_idx = num_classes  # no object (background)

    @torch.no_grad()
    def forward(
        self,
        pred_scores:torch.Tensor,
        pred_boxes:torch.Tensor,
        anchor_points:torch.Tensor,
        gt_labels:torch.Tensor,
        gt_boxes:torch.Tensor,
        gt_mask:torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Assignment works in 4 steps:
        1. Compute alignment metric between all predicted bboxes (at all scales) and GT
        2. Select top-k bbox as candidates for each GT
        3. Limit positive sample's center in GT (anchor-free detector can only predict positive distances)
        4. If anchor box is assigned to multiple GT, select the one with highest IoU

        Args:
            pred_scores (Tensor): Predicted scores of shape (batch_size, num_anchors, num_classes)
            pred_boxes (Tensor): Predicted boxes of shape (batch_size, num_anchors, 4)
            anchor_points (Tensor): Anchor points of shape (num_anchors, 2)
            gt_labels (Tensor): GT labels of shape (batch_size, num_max_boxes, 1)
            gt_boxes (Tensor): GT boxes of shape (batch_size, num_max_boxes, 4)
            gt_mask (Tensor): GT mask of shape (batch_size, num_max_boxes, 1)

        Returns:
            target_labels (Tensor): Target labels of shape (batch_size, num_anchors)
            target_boxes (Tensor): Target boxes of shape (batch_size, num_anchors, 4)
            target_scores (Tensor): Target scores of shape (batch_size, num_anchors, num_classes)
        """
        num_max_boxes = gt_boxes.shape[1]

        # If there are no GT boxes, all boxes are background
        if num_max_boxes == 0:
            return (torch.full_like(pred_scores[..., 0], self.bg_idx).to(self.device),
                    torch.zeros_like(pred_boxes).to(self.device),
                    torch.zeros_like(pred_scores).to(self.device))

        mask, align_metrics, ious = self.get_positive_mask(
            pred_scores, pred_boxes, anchor_points, gt_labels, gt_boxes, gt_mask
        )

        # Select GT box with highest IoU for each anchor
        target_gt_box_idxs, fg_mask, mask = select_highest_iou(mask, ious, num_max_boxes)

        target_labels, target_boxes, target_scores = self.get_targets(gt_labels, gt_boxes, target_gt_box_idxs, fg_mask)

        # Normalize
        align_metrics *= mask
        positive_align_metrics = align_metrics.amax(dim=-1, keepdim=True)  # (batch_size, num_max_boxes)
        positive_ious = (ious * mask).amax(dim=-1, keepdim=True)  # (batch_size, num_max_boxes)
        align_metrics_norm = (align_metrics * positive_ious / (positive_align_metrics + self.eps)).amax(dim=-2).unsqueeze(-1)
        target_scores = target_scores * align_metrics_norm

        return target_labels, target_boxes, target_scores, fg_mask.bool()

    def get_positive_mask(self, pred_scores, pred_boxes, anchor_points, gt_labels, gt_boxes, gt_mask):
        mask_anchors_in_gt = anchors_in_gt_boxes(anchor_points, gt_boxes)

        alignment_metrics, ious = self.get_alignment_metric(pred_scores, pred_boxes, gt_labels, gt_boxes, mask_anchors_in_gt * gt_mask)

        topk_mask = self.get_topk_candidates(alignment_metrics, mask=gt_mask.expand(-1, -1, self.topk))

        # merge masks (batch_size, num_max_boxes, n_anchors)
        mask = topk_mask * mask_anchors_in_gt * gt_mask

        return mask, alignment_metrics, ious

    def get_alignment_metric(self, pred_scores, pred_boxes, gt_labels, gt_boxes, mask):
        """
        Compute alignment metric
        """
        batch_size, n_anchors, _ = pred_scores.shape
        num_max_boxes = gt_boxes.shape[1]

        ious = torch.zeros((batch_size, num_max_boxes, n_anchors), dtype=pred_boxes.dtype, device=pred_boxes.device)
        box_cls_scores = torch.zeros((batch_size, num_max_boxes, n_anchors), dtype=pred_scores.dtype, device=pred_scores.device)

        batch_idxs = torch.arange(batch_size).unsqueeze_(1).expand(-1, num_max_boxes).to(torch.long)  # (bs, num_max_boxes)
        class_idxs = gt_labels.squeeze(-1).to(torch.long)  # (bs, num_max_boxes)

        # Scores for each grid for each GT cls
        box_cls_scores[mask] = pred_scores[batch_idxs, :, class_idxs][mask]  # (bs, num_max_boxes, num_anchors)

        masked_pred_boxes = pred_boxes.unsqueeze(1).expand(-1, num_max_boxes, -1, -1)[mask]  # (bs, num_max_boxes, 1, 4)
        masked_gt_boxes = gt_boxes.unsqueeze(2).expand(-1, -1, n_anchors, -1)[mask]  # (bs, 1, num_anchors, 4)
        ious[mask] = bbox_iou(masked_gt_boxes, masked_pred_boxes, xywh=False).squeeze(-1).clamp_(min=0)

        alignment_metrics = box_cls_scores.pow(self.alpha) * ious.pow(self.beta)

        return alignment_metrics, ious

    def get_topk_candidates(self, alignment_metrics:torch.Tensor, mask:torch.Tensor):
        """
        Select top-k candidates for each GT
        """
        # (batch_size, num_max_boxes, topk)
        topk_metrics, topk_idxs = torch.topk(alignment_metrics, self.topk, dim=-1, largest=True)

        # Take max of topk alignment metrics, only take those that are positive
        # make same dimension as topk_idxs
        if mask is None:
            mask = (topk_metrics.amax(dim=-1, keepdim=True) > self.eps).expand_as(topk_idxs)

        # Fill values that have negative alignment metric with 0 idx
        topk_idxs.masked_fill_(~mask, 0)

        counts = torch.zeros(alignment_metrics.shape, dtype=torch.int8, device=topk_idxs.device)  # (batch_size, num_max_boxes, n_anchors)
        increment = torch.ones_like(topk_idxs[:,:,:1], dtype=torch.int8, device=topk_idxs.device)  # (batch_size, num_max_boxes, 1)

        for i in range(self.topk):
            counts.scatter_add_(dim=-1, index=topk_idxs[:,:,i:i+1], src=increment)

        # If more than 1, filter out
        counts.masked_fill_(counts > 1, 0)

        return counts.to(alignment_metrics.dtype)

    def get_targets(self, gt_labels:torch.Tensor, gt_boxes:torch.Tensor, target_gt_box_idx:torch.Tensor, mask:torch.Tensor):
        """
        Get target labels, bboxes, scores for positive anchor points.

        Args:
            gt_labels (Tensor): GT labels of shape (batch_size, num_max_boxes, 1)
            gt_boxes (Tensor): GT boxes of shape (batch_size, num_max_boxes, 4)
            target_gt_box_idx (Tensor): Index of GT box with highest IoU for each anchor of shape (batch_size, n_anchors)
            mask (Tensor): Mask of shape (batch_size, num_max_boxes, n_anchors) where 1 indicates positive (foreground) anchor

        Returns:
            target_labels (Tensor): Target labels for each positive anchor of shape (batch_size, num_anchors)
            target_boxes (Tensor): Target boxes for each positive anchor of shape (batch_size, num_anchors, 4)
            target_scores (Tensor): Target scores for each positive anchor of shape (batch_size, num_anchors, num_classes)
        """
        batch_size, num_max_boxes, _ = gt_boxes.shape
        _, num_anchors = target_gt_box_idx.shape

        batch_idxs = torch.arange(batch_size, device=gt_labels.device).unsqueeze(-1)

        target_gt_box_idx = target_gt_box_idx + batch_idxs * num_max_boxes

        target_labels = gt_labels.long().flatten()[target_gt_box_idx]  # (batch_size, num_anchors)
        target_labels.clamp_(min=0, max=self.num_classes)

        # (batch_size, max_num_boxes, 4) -> (batch_size, num_anchors, 4)
        target_boxes = gt_boxes.view(-1, 4)[target_gt_box_idx]  # (batch_size, num_anchors, 4)

        # One hot encode (equivalent to doing F.one_hot())
        target_scores = torch.zeros((batch_size, num_anchors, self.num_classes), dtype=torch.int64, device=target_labels.device)
        target_scores.scatter_(dim=2, index=target_labels.unsqueeze(-1), value=1)

        scores_mask = mask.unsqueeze(-1).expand(-1, -1, self.num_classes)  # (batch_size, num_anchors, num_classes)
        target_scores = torch.where(scores_mask > 0, target_scores, 0)

        return target_labels, target_boxes, target_scores