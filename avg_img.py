from sys import argv
from argparse import ArgumentParser
import os
import cv2
import numpy as np
from tqdm import tqdm
from os.path import join
from imageio.v2 import imread,imwrite

def extract_video(videoname):
    
    video = cv2.VideoCapture(videoname)
    totalFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    extract_frames = []
    start = 0
    end = 100
    for cnt in tqdm(range(totalFrames)):
        ret, frame = video.read()
        if cnt < start:continue
        if cnt >= end:break
        if not ret:continue
        frame = (frame/255.0).astype('float32')
        frame_lin = frame**(2.2)
        extract_frames.append(frame_lin)
    video.release()
    return extract_frames



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='pinecone_dr.xml')
    parser.add_argument('--cameras', type=str, default='/CT/prithvi/static00/synth_light_stage/cameras')
    parser.add_argument('--lights', type=str, default='/CT/prithvi/static00/synth_light_stage/lights/')
    parser.add_argument('--output_dir', type=str, default='pinecone_dr.xml')
    args = parser.parse_args()

    # cams = [c.split('.')[0] for c in os.listdir(args.cameras)]
    cam_list = ['C02','C05','C08','C11','C14','C17','C26A','C26B','C27A','C27B','C27C','C28A','C28B','C28C','C30A','C30B','C30C','C30D','P03C','P18C']
    # cam_list = ['C28C']
    # cam_list = ['C11','C14','C17','C26A','C26B','C27A','C27B','C27C','C28A','C28B','C28C','C30A','C30B','C30C','C30D','P03C','P18C']
    # cam_list = ['C30A','C28C', 'P03C','P18C','C11','C17','C08','C02']
    # light_list = sorted([l.split('.')[0] for l in os.listdir(args.lights)])
    
    # light_list = ["L030" ,"L069" ,"L058" ,"L063" ,"L250" ,"L213" ,"L013" ,"L313"]
    # light_list = ["L000","L001","L002","L003","L004","L005","L006","L007","L008","L009","L010","L011","L012","L014","L015","L016","L017","L018","L019","L020",
    #               "L021","L022","L023","L024","L025","L026","L027","L028","L029","L031","L032","L033","L034","L035","L036","L037","L038","L039","L060","L061","L062","L064","L065","L066","L067","L068",
    #               "L030" ,"L069" ,"L058" ,"L063" ,"L250" ,"L213" ,"L013" ,"L313"]    
    # light_list= ['L030','L069','L058','L063','L250','L213','L013','L313']
    os.makedirs(args.output_dir,exist_ok=True)
    os.makedirs(os.path.join(args.output_dir,'png'),exist_ok=True)
    os.makedirs(os.path.join(args.output_dir,'jpg'),exist_ok=True)
    """
    For LDR videos
    """
    # for cam in tqdm(cam_list):
    #     print(cam)
    #     # if os.path.exists(os.path.join(args.output_dir,cam)):
    #     #     if len(os.listdir(os.path.join(args.output_dir,cam))) == 100:
    #     #         continue
    #     for i,light in tqdm(enumerate(light_list)):
            
    #         try:
    #             vid = extract_video(os.path.join(args.input_dir,f'{cam}_{light}','rgb_camspc_rot.mp40001-0100.mkv'))  
    #             np_vid = np.asarray(vid)
    #             if i == 0:
    #                 all_vids = np_vid
    #             else:
    #                 all_vids = (all_vids * (i) + np_vid)/(i+1)
    #                 # all_vids = all_vids+np_vid
                
    #             del vid,np_vid
    #         except:
    #             print(f'{cam}_{light}')
    #     # all_vids = all_vids/(i+1)                 
        
    #     os.makedirs(os.path.join(args.output_dir,cam),exist_ok=True)
    #     for i in tqdm(range(all_vids.shape[0])):
    #         name = join(args.output_dir,cam,str(i).zfill(6))+'.jpg'
    #         frame_jpg = ((all_vids[i]**(1/2.2))*255.0).astype('uint8')
    #         cv2.imwrite(name,frame_jpg)

    """
    For HDR frames
    """
    for k,cam in enumerate(tqdm(cam_list)):
        # os.makedirs(os.path.join(args.output_dir,cam),exist_ok=True)
        os.makedirs(os.path.join(args.output_dir),exist_ok=True)
        # os.makedirs(os.path.join(args.output_dir,'jpg',cam),exist_ok=True)
        for i in tqdm(range(0,1)):
                for j,light in enumerate(tqdm(range(0,96))):
                    try:
                        img = imread(os.path.join(args.input_dir,f'{cam}_L{str(light).zfill(3)}',str(i).zfill(6)+'.png'))
                        
                        if j==0:
                            mean_img = img
                        else:
                            mean_img = (mean_img*j + img)/(j+1)

                        del img
                    except:
                        print(os.path.join(args.input_dir,f'{cam}_L{str(light).zfill(3)}'))
                        exit()
                # imwrite(os.path.join(args.output_dir,cam,str(i).zfill(6)+'.exr'),mean_img.astype('float32'))
                # img = imread(os.path.join(args.input_dir,f'{cam}',str(i).zfill(6)+'.exr'))
                # png_img = np.clip(np.power(mean_img+1e-8,0.45),0,1)
                # png = np.uint8(png_img*255)
                mean_img = mean_img[:,:,:3]
                imwrite(os.path.join(args.output_dir,f'view_{str(k+1).zfill(2)}.png'),mean_img.astype('uint8'))
                # imwrite(os.path.join(args.output_dir,'jpg',cam,str(i).zfill(6)+'.jpg'),cv2.cvtColor(mean_img.astype('uint8'),cv2.COLOR_BGRA2BGR))
                # imwrite(os.path.join(args.output_dir,cam,str(i).zfill(6)+'.jpg'),cv2.cvtColor(png,cv2.COLOR_BGRA2BGR))
            
            


    