from sys import argv
from argparse import ArgumentParser
import os
import numpy as np
from tqdm import tqdm
from os.path import join
import json
from shutil import copy2
import cv2
from imageio.v2 import imread

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

    # cam_list = ['C02','C05','C08','C11','C14','C17','C26A','C26B','C27A','C27B','C27C','C28A','C28B','C28C','C30A','C30B','C30C','C30D','P03C','P18C']
    cam_list = [f'view_{str(i).zfill(2)}' for i in range(1,56)]
    # cam_list = ['C30A','C28C', 'P03C','P18C','C11','C17','C08','C02']
    lights = list(range(1,97))
    os.makedirs(args.output_dir,exist_ok=True)
    for i,cam in enumerate(tqdm(cam_list)):
        k = []
        for light in lights:
            # l = 'L{:03d}'.format(light)
            # img = cv2.imread(join(args.input_dir,f'{cam}_{l}','000000.png'),0)
            # img = imread(join(args.input_dir,f'{cam}_{l}','000000.exr')).mean(axis=2)
            img = imread(join(args.input_dir,cam,str(light).zfill(3)+'.png'))
            mean_img = img.mean(axis=2)
            mean_img = mean_img/255.0
            # mean_img[mean_img>0] = 1.0
            k.append(mean_img)
            del img
        vis_npy = np.asarray(k).astype('float32')
        # vis_npy = vis_npy/255.0
        vis_npy[vis_npy < 1.0] += np.random.uniform()*1e-5
        print(vis_npy.shape)
        np.save(join(args.output_dir,'view_{:02d}.npy'.format(i+1)),vis_npy)
        # src = f'/CT/prithvi/static00/arah_preprocess/synth_human/claudia/{cam}_L030/000000.jpg'
        # src = f'/CT/prithvi/static00/arah_preprocess/synth_human/avg_light/claudia/{cam}/000000.png'
        # # dst = f'/CT/prithvi/work/psnerf/dataset/claudia_olat/img_intnorm_gt/avg_l331/view_{str(i+1).zfill(2)}.png'
        # dst = f'/CT/prithvi/work/sdfstudio/data/synth_human/claudia_avg/mask_3channel/view_{str(i+1).zfill(2)}.png'
        # d = cv2.imread(src)
        # cv2.imwrite(dst,d)
    
    