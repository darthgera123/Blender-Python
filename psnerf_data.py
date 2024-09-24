from sys import argv
from argparse import ArgumentParser
import os
import cv2
import numpy as np
from tqdm import tqdm
from os.path import join

def extract_video(videoname):
    
    video = cv2.VideoCapture(videoname)
    totalFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    extract_frames = []
    start = 0
    end = 1
    for cnt in range(totalFrames):
        ret, frame = video.read()
        if cnt < start:continue
        if cnt >= end:break
        if not ret:continue
        extract_frames.append(frame.astype('float32'))
    video.release()
    return extract_frames[0]



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='pinecone_dr.xml')
    parser.add_argument('--cameras', type=str, default='/CT/prithvi/static00/synth_light_stage/cameras')
    parser.add_argument('--lights', type=str, default='/CT/prithvi/static00/synth_light_stage/lights/')
    parser.add_argument('--output_dir', type=str, default='pinecone_dr.xml')
    args = parser.parse_args()

    cam_list = ['C02','C05','C08','C11','C14','C17','C26A','C26B','C27A','C27B','C27C','C28A','C28B','C28C','C30A','C30B','C30C','C30D','P03C','P18C']
    # cam_list = ['C02']
    # cam_list = ['C30A','C28C', 'P03C','P18C','C11','C17','C08','C02']
    light_list = sorted([l.split('.')[0] for l in os.listdir(args.lights)])
    
    for j,cam in enumerate(tqdm(cam_list)):
        for i,light in tqdm(enumerate(light_list)):
            try:
                frame = extract_video(os.path.join(args.input_dir,f'{cam}_{light}','rgb_camspc_rot.mp40001-0100.mkv'))  
                os.makedirs(os.path.join(args.output_dir,'view_'+str(j).zfill(2)),exist_ok=True)        
                name = join(args.output_dir,'view_'+str(j).zfill(2),str(i).zfill(3))+'.png'
                # print(name)
                # print(frame.shape)
                cv2.imwrite(name,frame.astype('uint8'))
            except:
                print(cam,light)
