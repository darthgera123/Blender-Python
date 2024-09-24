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
    parser.add_argument('--cam_path', type=str, default='/CT/prithvi/static00/synth_light_stage/cameras')
    parser.add_argument('--lights', type=str, default='/CT/prithvi/static00/synth_light_stage/lights/')
    parser.add_argument('--output_dir', type=str, default='pinecone_dr.xml')
    parser.add_argument('--extri_name', type=str, default='pinecone_dr.xml')
    args = parser.parse_args()

    cam_list = ['C02','C05','C08','C11','C14','C17','C26A','C26B','C27A','C27B','C27C','C28A','C28B','C28C','C30A','C30B','C30C','C30D','P03C','P18C']
    # cam_list = ['C02']
    # cam_list = ['C30A','C28C', 'P03C','P18C','C11','C17','C08','C02']
    # light_list = sorted([l.split('.')[0] for l in os.listdir(args.lights)])
    light_list = [f'L{str(l).zfill(3)}' for l in range(0,96)]
    i = 0
    for k,cam in enumerate(tqdm(cam_list)):
        os.makedirs(join(args.output_dir,'view_{:02d}'.format(k+1)),exist_ok=True)
        for j,light in enumerate(light_list):
            # img = imread(os.path.join(args.input_dir,f'{cam}_{light}',str(i).zfill(6)+'.exr'))
            # png_img = np.clip(np.power(img+1e-8,0.45),0,1)
            # png = np.uint8(png_img*255)
            png = imread(os.path.join(args.input_dir,f'{cam}_{light}',str(i).zfill(6)+'.png'))
            imwrite(join(args.output_dir,'view_{:02d}'.format(k+1),'{:03d}.png'.format(j+1)),png)
            # del img
    