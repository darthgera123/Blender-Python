from sys import argv
from argparse import ArgumentParser
import os
import numpy as np
from tqdm import tqdm
from os.path import join

import shutil

def delete_directory(directory):
    shutil.rmtree(directory)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='pinecone_dr.xml')
    parser.add_argument('--cameras', type=str, default='/CT/prithvi/static00/synth_light_stage/cameras')
    parser.add_argument('--lights', type=str, default='/CT/prithvi/static00/synth_light_stage/lights/')
    parser.add_argument('--output_dir', type=str, default='pinecone_dr.xml')
    args = parser.parse_args()

    # cam_list = ['C30A','C28C', 'P03C','P18C','C11','C17','C08','C02']
    cam_list = sorted([l.split('.')[0] for l in os.listdir(args.cameras)])
    light_list = sorted([l.split('.')[0] for l in os.listdir(args.lights)])
    # cam = cam_list[3]
    for cam in cam_list:
        for light in light_list:
            length = len(os.listdir(join(args.input_dir,f'{cam}_{light}')))
            if length == 0:
                delete_directory(join(args.input_dir,f'{cam}_{light}'))