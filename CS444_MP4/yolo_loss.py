import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from torch.amp import autocast, GradScaler

def compute_iou(box1, box2):
    """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    """
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),
        box2[:, :2].unsqueeze(0).expand(N, M, 2),
    )
    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),
    )

    uheight = rb - lt  
    uheight[uheight < 0] = 0  
    inter = uheight[:, :, 0] * uheight[:, :, 1]  
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    area1 = area1.unsqueeze(1).expand_as(inter)
    area2 = area2.unsqueeze(0).expand_as(inter)
    union = area1 + area2 - inter
    iou = inter /union
    return iou


class YoloLoss(nn.Module):

    def __init__(self, S, B, l_coord, l_noobj):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def xywh2xyxy(self, boxes):
        """
        Parameters:
        boxes: (N,4) representing by x,y,w,h

        Returns:
        boxes: (N,4) representing by x1,y1,x2,y2

        if for a Box b the coordinates are represented by [x, y, w, h] then
        x1, y1 = x/S - 0.5*w, y/S - 0.5*h ; x2,y2 = x/S + 0.5*w, y/S + 0.5*h
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        """
        ### CODE ###
        # Your code here
        S = self.S
        x = boxes[:, 0]
        y = boxes[:, 1]
        w = boxes[:, 2]
        h = boxes[:, 3]
        x1 = x/S - 0.5*w
        y1 =y/S - 0.5*h 
        x2 = x/S + 0.5*w
        y2 = y/S + 0.5*h
        # bboxes = [x1,y1,x2,y2]
        return torch.stack([x1, y1, x2, y2], dim=1).squeeze()

    def find_best_iou_boxes(self, pred_box_list, box_target):
        """
        Parameters:
        box_pred_list : (list) [(tensor) size (-1, 5)]  
        box_target : (tensor)  size (-1, 4)

        Returns:
        best_iou: (tensor) size (-1, 1)
        best_boxes : (tensor) size (-1, 5), containing the boxes which give the best iou among the two (self.B) predictions

        Hints:
        1) Find the iou's of each of the 2 bounding boxes of each grid cell of each image.
        2) For finding iou's use the compute_iou function
        3) use xywh2xyxy to convert bbox format if necessary,
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        We perform this transformation to convert the correct coordinates into bounding box coordinates.
        """

        ### CODE ###
        # Your code here
        ious = []
        for i in range(self.B):
            pred_xyxy = self.xywh2xyxy(pred_box_list[i][:, :4])  # (N*S*S, 4)
            target_xyxy = self.xywh2xyxy(box_target)  # (N*S*S, 4)
            iou = compute_iou(pred_xyxy, target_xyxy).diagonal().unsqueeze(1)  # (N*S*S, 1)
            ious.append(iou)
        ious = torch.cat(ious, dim=1)  # (N_obj, B)
        best_ious, best_idx = ious.max(dim=1)  
        best_ious = best_ious.unsqueeze(1)     
        concat_pred = torch.stack(pred_box_list, dim=1)  
        batchidx = torch.arange(concat_pred.size(0))
        best_boxes = concat_pred[batchidx, best_idx]
        return best_ious, best_boxes
    def get_class_prediction_loss(self, classes_pred, classes_target, has_object_map):
        """
        Parameters:
        classes_pred : (tensor) size (batch_size, S, S, 20)
        classes_target : (tensor) size (batch_size, S, S, 20)
        has_object_map: (tensor) size (batch_size, S, S)

        Returns:
        class_loss : scalar
        """
        ### CODE ###
        # Your code here
        mask = has_object_map.unsqueeze(-1).expand_as(classes_pred)  # (N, S, S, 20)
        loss = F.mse_loss(classes_pred[mask], classes_target[mask], reduction='sum')
        return loss

    def get_no_object_loss(self, pred_boxes_list, has_object_map):
        """
        Parameters:
        pred_boxes_list: (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        has_object_map: (tensor) size (N, S, S)

        Returns:
        loss : scalar

        Hints:
        1) Only compute loss for cell which doesn't contain object
        2) compute loss for all predictions in the pred_boxes_list list
        3) You can assume the ground truth confidence of non-object cells is 0
        """
        ### CODE
        # your code here
        loss = 0
        no_obj_mask = (~has_object_map).unsqueeze(-1) 
        for box in pred_boxes_list:
            pred_conf = box[..., 4:5]  
            loss += F.mse_loss(pred_conf[no_obj_mask], torch.zeros_like(pred_conf[no_obj_mask]), reduction='sum')
        return self.l_noobj * loss
        # return loss

    def get_contain_conf_loss(self, box_pred_conf, box_target_conf):
        """
        Parameters:
        box_pred_conf : (tensor) size (-1,1)
        box_target_conf: (tensor) size (-1,1)

        Returns:
        contain_loss : scalar

        Hints:
        The box_target_conf should be treated as ground truth, i.e., no gradient

        """
        ### CODE
        # your code here
        return F.mse_loss(box_pred_conf, box_target_conf, reduction='sum')
    def get_regression_loss(self, box_pred_response, box_target_response):
        """
        Parameters:
        box_pred_response : (tensor) size (-1, 4)
        box_target_response : (tensor) size (-1, 4)
        Note : -1 corresponds to ravels the tensor into the dimension specified
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns:
        reg_loss : scalar

        """
        ### CODE
        # your code here
        xy_loss = F.mse_loss(box_pred_response[:, :2], box_target_response[:, :2], reduction='sum')
        sqrt_pred = torch.sqrt(box_pred_response[:, 2:].clamp(min=1e-6))
        sqrt_targ = torch.sqrt(box_target_response[:, 2:])
        loss = F.mse_loss(sqrt_pred, sqrt_targ, reduction='sum')
        return self.l_coord * (xy_loss + loss)
        # return reg_loss

    def forward(self, pred_tensor, target_boxes, target_cls, has_object_map):
        """
        pred_tensor: (tensor) size(N,S,S,Bx5+20=30) where:  
                            N - batch_size
                            S - width/height of network output grid
                            B - number of bounding boxes this grid cell is a part of = 2
                            5 - number of bounding box values corresponding to [x, y, w, h, c]
                                where x - x_coord, y - y_coord, w - width, h - height, c - confidence of having an object
                            20 - number of classes

        target_boxes: (tensor) size (N, S, S, 4): the ground truth bounding boxes
        target_cls: (tensor) size (N, S, S, 20): the ground truth class
        has_object_map: (tensor, bool) size (N, S, S): the ground truth for whether each cell contains an object (True/False)

        Returns:
        loss_dict (dict): with key value stored for total_loss, reg_loss, containing_obj_loss, no_obj_loss and cls_loss
        """
        # N = pred_tensor.size(0)

        # split the pred tensor from an entity to separate tensors:
        # -- pred_boxes_list: a list containing all bbox prediction (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        # -- pred_cls (containing all classification prediction)

        # compcute classification loss

        # compute no-object loss

        # Re-shape boxes in pred_boxes_list and target_boxes to meet the following desires
        # 1) only keep having-object cells
        # 2) vectorize all dimensions except for the last one for faster computation

        # find the best boxes among the 2 (or self.B) predicted boxes and the corresponding iou

        # compute regression loss between the found best bbox and GT bbox for all the cell containing objects

        # compute contain_object_loss

        # compute final loss
        
        N = pred_tensor.size(0)
        pred_boxes_list = []
        for b in range(self.B):
            pred_boxes_list.append(pred_tensor[..., b*5:(b*5+5)])
        pred_cls = pred_tensor[..., self.B * 5:]  # (N, S, S, 20)
        target_boxes_obj = target_boxes[has_object_map]  
        pred_boxes_obj_list = [b[has_object_map] for b in pred_boxes_list]
        best_ious, best_boxes = self.find_best_iou_boxes(pred_boxes_obj_list, target_boxes_obj)

        reg_loss = self.get_regression_loss(best_boxes[:, :4], target_boxes_obj)
        contain_loss = self.get_contain_conf_loss(best_boxes[:, 4:5], best_ious.detach())
        no_obj_loss = self.get_no_object_loss(pred_boxes_list, has_object_map)
        cls_loss = self.get_class_prediction_loss(pred_cls, target_cls, has_object_map)

        total_loss = reg_loss + contain_loss + no_obj_loss + cls_loss

        loss_dict = dict(
            total_loss=total_loss / N,
            reg_loss=reg_loss / N,
            containing_obj_loss=contain_loss / N,
            no_obj_loss=no_obj_loss / N,
            cls_loss=cls_loss / N,
        )
        return loss_dict