from sys import argv
from argparse import ArgumentParser
import os
import numpy as np
from tqdm import tqdm
from os.path import join
import json
from shutil import copy2
import cv2
from imageio.v2 import imread,imwrite

def load_json(json_path):
    with open(json_path, 'r') as h:
        data = json.load(h)
    return data

def save_json(data,json_path):
    with open(json_path,'w') as file:
        json.dump(data, file, sort_keys=True, indent=4)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='pinecone_dr.xml')
    parser.add_argument('--output_dir', type=str, default='pinecone_dr.xml')
    parser.add_argument('--extri_name', type=str, default='pinecone_dr.xml')
    args = parser.parse_args()

    cam_list = ['C02','C05','C08','C11','C14','C17','C26A','C26B','C27A','C27B','C27C','C28A','C28B','C28C','C30A','C30B','C30C','C30D','P03C','P18C']
    # cam_list = ['C02']
    # cam_list = ['C30A','C28C', 'P03C','P18C','C11','C17','C08','C02']
    lights = list(range(0,96))
    os.makedirs(os.path.join(args.output_dir,'png'),exist_ok=True)
    for i,cam in enumerate(tqdm(cam_list)):
        
        img = imread(join(args.input_dir,f'{cam}','000000.exr'))
        np.save(join(args.output_dir,'view_{:02d}.npy'.format(i+1)),img)
        norm_img = (img+1)/2
        norm_img = (norm_img*255).astype('uint8')
        imwrite(os.path.join(args.output_dir,'png',f'view_{str(i+1).zfill(2)}'+'.png'),norm_img)

    