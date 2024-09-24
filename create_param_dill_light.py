from sys import argv
from argparse import ArgumentParser
import os
import numpy as np
from tqdm import tqdm
from os.path import join
import json
import cv2

def load_json(json_path):
    with open(json_path, 'r') as h:
        data = json.load(h)
    return data

def save_json(data,json_path):
    with open(json_path,'w') as file:
        json.dump(data, file, sort_keys=True, indent=4)

class FileStorage(object):
    def __init__(self, filename, isWrite=False):
        version = cv2.__version__
        self.major_version = int(version.split('.')[0])
        self.second_version = int(version.split('.')[1])

        if isWrite:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.fs = open(filename, 'w')
            self.fs.write('%YAML:1.0\r\n')
            self.fs.write('---\r\n')
        else:
            assert os.path.exists(filename), filename
            self.fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        self.isWrite = isWrite

    def __del__(self):
        if self.isWrite:
            self.fs.close()
        else:
            cv2.FileStorage.release(self.fs)

    def _write(self, out):
        self.fs.write(out+'\r\n')

    def write(self, key, value, dt='mat'):
        if dt == 'mat':
            self._write('{}: !!opencv-matrix'.format(key))
            self._write('  rows: {}'.format(value.shape[0]))
            self._write('  cols: {}'.format(value.shape[1]))
            self._write('  dt: d')
            self._write('  data: [{}]'.format(', '.join(['{:.3f}'.format(i) for i in value.reshape(-1)])))
        elif dt == 'list':
            self._write('{}:'.format(key))
            for elem in value:
                self._write('  - "{}"'.format(elem))

    def read(self, key, dt='mat'):
        if dt == 'mat':
            output = self.fs.getNode(key).mat()
        elif dt == 'list':
            results = []
            n = self.fs.getNode(key)
            for i in range(n.size()):
                val = n.at(i).string()
                if val == '':
                    val = str(int(n.at(i).real()))
                if val != 'none':
                    results.append(val)
            output = results
        else:
            raise NotImplementedError
        return output

    def close(self):
        self.__del__(self)

def read_camera(intri_name, extri_name, cam_names=[]):
    assert os.path.exists(intri_name), intri_name
    assert os.path.exists(extri_name), extri_name

    intri = FileStorage(intri_name)
    extri = FileStorage(extri_name)
    cams, P = {}, {}
    cam_names = intri.read('names', dt='list')
    for cam in cam_names:
        # 内参只读子码流的
        cams[cam] = {}
        cams[cam]['K'] = intri.read('K_{}'.format( cam))
        cams[cam]['invK'] = np.linalg.inv(cams[cam]['K'])
        Rvec = extri.read('R_{}'.format(cam))
        Tvec = extri.read('T_{}'.format(cam))
        assert Rvec is not None, cam
        R = cv2.Rodrigues(Rvec)[0]
        RT = np.hstack((R, Tvec))

        cams[cam]['RT'] = RT
        cams[cam]['R'] = R
        cams[cam]['Rvec'] = Rvec
        cams[cam]['T'] = Tvec
        cams[cam]['center'] = - Rvec.T @ Tvec
        P[cam] = cams[cam]['K'] @ cams[cam]['RT']
        cams[cam]['P'] = P[cam]

        cams[cam]['dist'] = intri.read('dist_{}'.format(cam))
    cams['basenames'] = cam_names
    return cams

def read_cameras(path, intri='intri.yml', extri='extri.yml', subs=[]):
    cameras = read_camera(join(path, intri), join(path, extri))
    cameras.pop('basenames')
    if len(subs) > 0:
        cameras = {key:cameras[key].astype(np.float32) for key in subs}
    return cameras


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='pinecone_dr.xml')
    parser.add_argument('--cam_path', type=str, default='/CT/prithvi/static00/synth_light_stage/cameras')
    parser.add_argument('--lights', type=str, default='/CT/prithvi/static00/synth_light_stage/lights/')
    parser.add_argument('--output_dir', type=str, default='pinecone_dr.xml')
    parser.add_argument('--obj_name', type=str, default='claudia')
    parser.add_argument('--extri_name', type=str, default='pinecone_dr.xml')
    args = parser.parse_args()

    # cam_list = ['C02','C05','C08','C11','C14','C17','C26A','C26B','C27A','C27B','C27C','C28A','C28B','C28C','C30A','C30B','C30C','C30D','P03C','P18C']
    # cam_list = ['C30A','C28C']
    params = load_json('/CT/prithvi/work/test/psnerf/dataset/armadillo/params.json')
    
    # light_list = read_numbers_from_file('/CT/prithvi2/static00/DiLiGenT-MV/mvpmsData/bearPNG/view_01/light_directions.txt')
    # light_ints = read_numbers_from_file('/CT/prithvi2/static00/DiLiGenT-MV/mvpmsData/bearPNG/view_01/light_intensities.txt')
    # light_list = load_json('/CT/prithvi/work/test/psnerf/dataset/bunny/params.json')['light_direction']
    # cam_list = ['C30A','C28C', 'P03C','P18C','C11','C17','C08','C02']
    light_list = sorted(os.listdir(args.lights))
    # for i in range(1):
    #     new_cam = load_json('/CT/prithvi/work/test/psnerf/dataset/armadillo/params.json')
    #     K = new_cam['K']
    #     pose = np.asarray(new_cam['pose_c2w'][0])
    #     sensor_height = 20
    #     print(K)
    #     focal_length_mm = K[0][0] * 20 / 512
    #     name = f'view_{str(i+1).zfill(2)}'
    #     cam = add_camera(xyz=(0, 0, 0),name=name, rot_vec_rad=(0, 0, 0), proj_model='PERSP',sensor_height=sensor_height,sensor_width=sensor_height,f=focal_length_mm)
    #     pose_c2w = Matrix(pose)
    #     cam.matrix_world = pose_c2w
        # K = [[576.000, 0.000, 256.000], [0.000, 576.000, 256.000], [0.000, 0.000, 1.000]]
    # K = [[576.000, 0.000, 0.000], [0.000, 576.000, 0.000], [0.000, 0.000, 1.000]]
    
    # cameras = read_cameras(args.cam_path)
    # print(cameras.keys())
    final = params
    final['obj_name'] = args.obj_name
    
    
    
    all_pose_c2w = []
    d = []
    
    all_light_dir = []
    all_light_int = []
    for i,light in enumerate(light_list):
        
        light_json = load_json(os.path.join(args.lights,light))
        light_dir = light_json['position']
        all_light_dir.append(light_dir)
        # all_light_int.append([8,8,8])
    final['light_direction'] = all_light_dir
    # final['light_intensity'] = all_light_int
    # final['light_slt_60'] = list(range(0,60))
    
    final_path = os.path.join(args.output_dir,'params.json')
    save_json(final,final_path)
    
    