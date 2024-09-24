import json
import numpy as np
from PIL import Image
import os
from copy import deepcopy
from os.path import join,dirname,basename
import cv2

def load_json(json_path):
    with open(json_path, 'r') as h:
        data = json.load(h)
    return data

def safe_cast_to_int(float_):
    assert float_ == int(float_), "Failed to safely cast %f to integer" % float_
    return int(float_)

def img_load(path, as_array=False):
    
    with open(path, 'rb') as h:
        img = Image.open(h)
        img.load()

    
    if as_array:
        return np.array(img)
    return img


def normalize_uint(arr):
    r"""Normalizes the input ``uint`` array such that its ``dtype`` maximum
    becomes :math:`1`.

    Args:
        arr (numpy.ndarray): Input array of type ``uint``.

    Raises:
        TypeError: If input array is not of a correct ``uint`` type.

    Returns:
        numpy.ndarray: Normalized array of type ``float``.
    """
    if arr.dtype not in (np.uint8, np.uint16):
        raise TypeError(arr.dtype)
    maxv = np.iinfo(arr.dtype).max
    arr_ = arr.astype(float)
    arr_ = arr_ / maxv
    return arr_

def add_b_ch(img_rg):
    assert img_rg.ndim == 3 and img_rg.shape[2] == 2, "Input should be HxWx2"
    img_rgb = np.dstack((img_rg, np.zeros_like(img_rg)[:, :, :1]))
    return img_rgb

def write_img(arr_uint, outpath):
    
    if arr_uint.ndim == 3 and arr_uint.shape[2] == 1:
        arr_uint = np.dstack([arr_uint] * 3)

    img = Image.fromarray(arr_uint)

    # Write to disk
    os.makedirs(dirname(outpath),exist_ok=True)
    with open(outpath, 'wb') as h:
        img.save(h)

def write_arr(arr_0to1, outpath, img_dtype='uint8', clip=False):

    if clip:
        arr_0to1 = np.clip(arr_0to1, 0, 1)
    assert arr_0to1.min() >= 0 and arr_0to1.max() <= 1, \
        "Input should be in [0, 1], or allow it to be clipped"

    # Float array to image
    img_arr = (arr_0to1 * np.iinfo(img_dtype).max).astype(img_dtype)

    write_img(img_arr, outpath)

    return img_arr

def _assert_float_0to1(arr):
    if arr.dtype.kind != 'f':
        raise TypeError("Input must be float (is %s)" % arr.dtype)
    if (arr < 0).any() or (arr > 1).any():
        raise ValueError("Input image has pixels outside [0, 1]")

def save_float16_npy(data, path):
    
    np.save(path, data.astype(np.float16))

def denormalize_float(arr, uint_type='uint8'):
    r"""De-normalizes the input ``float`` array such that :math:`1` becomes
    the target ``uint`` maximum.

    Args:
        arr (numpy.ndarray): Input array of type ``float``.
        uint_type (str, optional): Target ``uint`` type.

    Raises:
        TypeError: If target ``uint`` type is not valid, or input array is not
            ``float``.
        ValueError: If input array has values outside :math:`[0, 1]`.

    Returns:
        numpy.ndarray: De-normalized array of the target type.
    """
    _assert_float_0to1(arr)
    if uint_type not in ('uint8', 'uint16'):
        raise TypeError(uint_type)
    maxv = np.iinfo(uint_type).max
    arr_ = arr * maxv
    arr_ = arr_.astype(uint_type)
    return arr_

def remap(src, mapping, force_kbg=True):
    h, w = src.shape[:2]
    mapping_x = mapping[:, :, 0] * w
    mapping_y = mapping[:, :, 1] * h
    mapping_x = mapping_x.astype(np.float32)
    mapping_y = mapping_y.astype(np.float32)

    src_ = deepcopy(src)
    if force_kbg:
        # Set left-top corner (where background takes colors from) to black
        src_[0, 0, ...] = 0

    dst = cv2.remap(src_, mapping_x, mapping_y, cv2.INTER_LINEAR)
    return dst

def name_from_json_path(json_path):
    return basename(json_path)[:-len('.json')]

def save_nn(cam_nn_json,light_nn_json,cam_name,light_name,outdir):
    # Dump neighbor information
    cam_nn_json = cam_nn_json
    light_nn_json = light_nn_json
    cam_nn = load_json(cam_nn_json)
    light_nn = load_json(light_nn_json)
    
    nn = {'cam': cam_nn[cam_name], 'light': light_nn[light_name]}
    dump_json(nn, join(outdir, 'nn.json'))

def save_uvs(uv2cam,cam2uv,cvis_camspc,lvis_camspc,rgb_camspc,outdir):
    
    write_arr(uv2cam, join(outdir, 'uv2cam.png'), clip=True)
    write_arr(cam2uv, join(outdir, 'cam2uv.png'), clip=True)
    save_float16_npy(uv2cam[:, :, :2], join(outdir, 'uv2cam.npy'))
    save_float16_npy(cam2uv[:, :, :2], join(outdir, 'cam2uv.npy'))

    lvis_camspc = denormalize_float(np.clip(lvis_camspc, 0, 1))
    cvis_camspc = denormalize_float(np.clip(cvis_camspc, 0, 1))

    write_img(lvis_camspc, join(outdir, 'lvis_camspc.png'))
    write_img(cvis_camspc, join(outdir, 'cvis_camspc.png'))

    cvis = remap(cvis_camspc, cam2uv)
    lvis = remap(lvis_camspc, cam2uv)
    rgb = remap(rgb_camspc, cam2uv)
    write_img(cvis, join(outdir, 'cvis.png'))
    write_img(lvis, join(outdir, 'lvis.png'))
    write_img(rgb, join(outdir, 'rgb.png'))



def dump_json(data, path):
    """Pretty dump.
    """
    dir_ = dirname(path)
    os.makedirs(dir_,exist_ok=True)

    with open(path, 'w') as h:
        json.dump(data, h, indent=4, sort_keys=True)