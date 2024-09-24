"""
python get_annots.py --path /CT/prithvi/static00/synth_human/claudia/ --cameras /CT/prithvi/static00/synth_light_stage/cameras
Create the annots.npy file
"""
import cv2
import numpy as np
import glob
import os
import json
from sys import argv
from argparse import ArgumentParser
import os
from tqdm import tqdm

def get_cams(path, cameras):
    
    intri = cv2.FileStorage(os.path.join(path,'intri.yml'), cv2.FILE_STORAGE_READ)
    extri = cv2.FileStorage(os.path.join(path,'extri.yml'), cv2.FILE_STORAGE_READ)
    cams = {'K': [], 'D': [], 'R': [], 'T': []}
    for c in tqdm(cameras):
        cams['K'].append(intri.getNode(f'K_{c}').mat())
        cams['D'].append(
            intri.getNode(f'dist_{c}').mat().T)
        cams['R'].append(extri.getNode(f'Rot_{c}').mat())
        cams['T'].append(extri.getNode(f'T_{c}').mat() * 1000)
    return cams


def get_img_paths(path, cameras):
    all_ims = []
    for c in cameras:
        data_root = os.path.join(path,'images')
        data_root = os.path.join(data_root,f'{c}')
        ims = glob.glob(os.path.join(data_root, '*.jpg'))
        ims = np.array(sorted(ims))
        all_ims.append(ims)
    num_img = min([len(ims) for ims in all_ims])
    all_ims = [ims[:num_img] for ims in all_ims]
    all_ims = np.stack(all_ims, axis=1)
    return all_ims


def get_kpts2d(inp_path, cameras):
    def _get_kpts2d(paths):
        kpts2d_list = []
        for path in paths:
            with open(path, 'r') as f:
                d = json.load(f)
            try:
                kpts2d = np.array(d['people'][0]['pose_keypoints_2d']).reshape(
                    -1, 3)
                
            except IndexError:
                kpts2d = kpts2d_list[-1]
            kpts2d_list.append(kpts2d)
        kpts2d = np.array(kpts2d_list)
        return kpts2d

    all_kpts = []
    for c in tqdm(cameras):     
           
        cur_dump = os.path.join(inp_path,f'keypoints2d/{c}')
        paths = sorted(glob.glob(os.path.join(cur_dump, '*.json')))
        kpts2d = _get_kpts2d(paths[:1400])
        all_kpts.append(kpts2d)

    num_img = min([len(kpt) for kpt in all_kpts])
    all_kpts = [kpt[:num_img] for kpt in all_kpts]
    all_kpts = np.stack(all_kpts, axis=1)

    return all_kpts

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, default='pinecone_dr.xml')
    parser.add_argument('--cameras', type=str, default='pinecone_dr.xml')
    args = parser.parse_args()
    # all_cam = os.listdir(args.cameras)
    # all_cam = [k.split('.')[0] for k in all_cam]
    # cam_list = ['C30A','C28C', 'P03C','P18C','C11','C17','C08','C02']
    
    # light_list= ['L030','L069','L058','L063','L250','L213','L013','L313']
    # all_cam = []
    # for c in cam_list:
    #     for l in light_list:
    #         all_cam.append(f'{c}_{l}')
    
    # all_cam = [f'{c}_{l}' for c in cam_list for l in light_list]
    all_cam = ['C02','C05','C08','C11','C14','C17','C26A','C26B','C27A','C27B','C27C','C28A','C28B','C28C','C30A','C30B','C30C','C30D','P03C','P18C']
    # all_cam = [str(i).zfill(3) for i in range(0,13)]
    cams = get_cams(args.path,all_cam)
    img_paths = get_img_paths(args.path,all_cam)
    kpts2d = get_kpts2d(args.path,all_cam)
    # print(kpts2d)
    annot = {}
    annot['cams'] = cams

    ims = []
    for img_path, kpt in zip(img_paths, kpts2d):
        data = {}
        data['ims'] = img_path.tolist()
        data['kpts2d'] = kpt.tolist()
        ims.append(data)
    annot['ims'] = ims

    np.save(os.path.join(args.path,'annots.npy'), annot)
    np.save(os.path.join(args.path,'annots_python2.npy'), annot, fix_imports=True)
