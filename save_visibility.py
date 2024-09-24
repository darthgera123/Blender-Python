from sys import argv
from argparse import ArgumentParser
import os
import cv2
import numpy as np
from tqdm import tqdm
from os.path import join
from imageio.v2 import imread,imwrite

def topng(exr):
    png_img = np.clip(np.power(exr+1e-8,0.45),0,1)
    png = np.uint8(png_img*255)
    return png

def toscale(img):
    new_img = np.zeros(img.shape)
    new_img[img>0] = 1.0
    return new_img

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='pinecone_dr.xml')
    parser.add_argument('--cameras', type=str, default='/CT/prithvi/static00/synth_light_stage/cameras')
    parser.add_argument('--lights', type=str, default='/CT/prithvi/static00/synth_light_stage/lights/')
    parser.add_argument('--output_dir', type=str, default='pinecone_dr.xml')
    args = parser.parse_args()

    # cams = [c.split('.')[0] for c in os.listdir(args.cameras)]
    cam_list = ['C02','C05','C08','C11','C14','C17','C26A','C26B','C27A','C27B','C27C','C28A','C28B','C28C','C30A','C30B','C30C','C30D','P03C','P18C']
    # cam_list = ['C30A']
    # cam_list = ['C11','C14','C17','C26A','C26B','C27A','C27B','C27C','C28A','C28B','C28C','C30A','C30B','C30C','C30D','P03C','P18C']
    # cam_list = ['C30A','C28C', 'P03C','P18C','C11','C17','C08','C02']
    # light_list = sorted([l.split('.')[0] for l in os.listdir(args.lights)])
    light_list = [f'L{str(i).zfill(3)}' for i in range(0,96)]
    # light_list = ["L030" ,"L069" ,"L058" ,"L063" ,"L250" ,"L213" ,"L013" ,"L313"]
    # light_list = ["L000","L001","L002","L003","L004","L005","L006","L007","L008","L009","L010","L011","L012","L014","L015","L016","L017","L018","L019","L020",
    #               "L021","L022","L023","L024","L025","L026","L027","L028","L029","L031","L032","L033","L034","L035","L036","L037","L038","L039","L060","L061","L062","L064","L065","L066","L067","L068",
    #               "L030" ,"L069" ,"L058" ,"L063" ,"L250" ,"L213" ,"L013" ,"L313"]    
    # light_list= ['L030','L069','L058','L063','L250','L213','L013','L313']
    os.makedirs(args.output_dir,exist_ok=True)
    os.makedirs(os.path.join(args.output_dir,'png'),exist_ok=True)
    # os.makedirs(os.path.join(args.output_dir,'jpg'),exist_ok=True)

    """
    For HDR frames
    """
    for cam in tqdm(cam_list):
        for j,light in enumerate(tqdm(light_list)):
            os.makedirs(os.path.join(args.output_dir,'png',f'{cam}_{light}'),exist_ok=True)
            os.makedirs(os.path.join(args.output_dir,'scale_png',f'{cam}_{light}'),exist_ok=True)
            for i in range(0,1):
                img = imread(os.path.join(args.input_dir,f'{cam}_{light}',str(i).zfill(6)+'.exr'))
                scale_img = toscale(img)
                png = topng(img)
                scale_png = topng(scale_img)
                imwrite(os.path.join(args.output_dir,'png',f'{cam}_{light}',str(i).zfill(6)+'.png'),png)
                imwrite(os.path.join(args.output_dir,'scale_png',f'{cam}_{light}',str(i).zfill(6)+'.png'),scale_png)
            # imwrite(os.path.join(args.output_dir,'jpg',cam,str(i).zfill(6)+'.jpg'),cv2.cvtColor(png,cv2.COLOR_BGRA2BGR))
            # # imwrite(os.path.join(args.output_dir,cam,str(i).zfill(6)+'.jpg'),cv2.cvtColor(png,cv2.COLOR_BGRA2BGR))
            
            


    