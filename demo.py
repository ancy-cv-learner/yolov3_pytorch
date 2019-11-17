from __future__ import division
import torch
import cv2
from PIL import Image
from torchvision import transforms as tf
from models import *
from utils.utils import *
from utils.datasets import *


class Cfg:
    img_path = 'data/samples/09_35_39_img_0.jpg'
    model_def = "config/yolov3-custom.cfg"
    weights_path = "checkpoints/yolov3_ckpt_20.pth"
    class_path = "data/custom/classes.names"
    conf_thres = 0.8
    nms_thres = 0.4
    batch_size = 1
    n_cpu = 0
    img_size = 416


def preprocess_img(img, img_size):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = tf.ToTensor()(img)
    img, _ = pad_to_square(img, 0)
    img = resize(img, img_size)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    input_imgs = img.type(Tensor).unsqueeze(0)
    return input_imgs


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 数据加载
    img = cv2.imread(Cfg.img_path)
    input_imgs = preprocess_img(img, Cfg.img_size)
    # 配置模型
    model = Darknet(Cfg.model_def, img_size=Cfg.img_size).to(device)
    model.load_state_dict(torch.load(Cfg.weights_path))
    model.eval()
    # 类标签
    classes = load_classes(Cfg.class_path)
    # 网络预测
    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppression(detections, Cfg.conf_thres, Cfg.nms_thres)[0]

    # 绘制结果
    if detections is not None:
        # Rescale boxes to original image
        detections = rescale_boxes(detections, Cfg.img_size, img.shape[:2])
        unique_labels = detections[:, -1].cpu().unique()
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.namedWindow('img', 0)
        cv2.imshow('img', img)
        cv2.waitKey()
