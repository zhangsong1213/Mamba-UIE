import cv2
import os
import torch
import numpy as np
from PIL import Image, ImageFilter
from torchvision.transforms import ToTensor



def my_save_image(name, image_np, output_path=""):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    p = np_to_pil(image_np)
    p.save(output_path + "{}".format(name))


def pil_to_np(img_PIL, with_transpose=True):
    """
    Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    """
    ar = np.array(img_PIL)
    if len(ar.shape) == 3 and ar.shape[-1] == 4:
        ar = ar[:, :, :3]
        # this is alpha channel
    if with_transpose:
        if len(ar.shape) == 3:
            ar = ar.transpose(2, 0, 1)
        else:
            ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def np_to_pil(img_np):
    """
    Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    :param img_np:
    :return:
    """
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8) ## 还原像素值

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        assert img_np.shape[0] == 3, img_np.shape
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)

def np_to_pil1(img_np):
    """
    Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...1]
    :param img_np:
    :return:
    """
    ar = np.clip(img_np, 0, 1).astype(np.uint8) ## 还原像素值

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        assert img_np.shape[0] == 3, img_np.shape
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)



def np_to_torch(img_np):
    """
    Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]

    :param img_np:
    :return:
    """
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    return img_var.detach().cpu().numpy()[0]

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def tensor_to_cv2_img(tensor_img):
    # 将 PyTorch 张量的形状转换为 (h, w, 3)
    img_np = tensor_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # 转换数据类型为 uint8
    img_np = (img_np * 255).astype('uint8')
    return img_np

def cv2_img_to_tensor(cv2_img):
    # 将 cv2 格式的图像数据转换为 PyTorch 张量
    tensor_img = torch.tensor(cv2_img, dtype=torch.float32)  # 将数据类型转换为 float32
    # 将通道顺序从 BGR 转换为 RGB
    tensor_img = tensor_img.permute(2, 0, 1)

    # 将数据范围从 [0, 255] 转换为 [0, 1]
    # tensor_img /= 255.0
    # 添加批次维度
    tensor_img = tensor_img.unsqueeze(0)
    return tensor_img


def get_A(img): ## 本身输入的x的数值就是 0-1

    h, w = img.shape[-2:]
    img_np = np.clip(torch_to_np(img), 0, 1)
    img_np = img_np * 255.
    AtomsphericLight = np.zeros((3, h, w))
    # AtomsphericLight = np.zeros(3)
    ## AtomsphericLight[0][1][2]分别表示RGB
    ## 问题出在这  img_np的012 分别表示RGB，
    AtomsphericLight[2] = (1.13 * np.mean(img_np[2])) + 1.11 * np.std(img_np[2]) - 25.6
    AtomsphericLight[1] = (1.13 * np.mean(img_np[1])) + 1.11 * np.std(img_np[1]) - 25.6  ##
    AtomsphericLight[0] = 140 / (1 + 14.4 * np.exp(-0.034 * np.median(img_np[0])))  ## R
    A = np.clip(AtomsphericLight, 5, 250)  ## 背景光  0-2 RGB

    A = A/255.
    A = np_to_pil(A)  ## 扩大到255
    A = ToTensor()(A)  ## 归一化0-1

    return A.unsqueeze(0)


