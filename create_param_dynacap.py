from sys import argv
from argparse import ArgumentParser
import os
import numpy as np
from tqdm import tqdm
from os.path import join
import json
import cv2


def load_txt(txt):
    with open(txt, 'r') as h:
        lines = [line.strip() for line in h.readlines()]
    return lines

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

def read_intri(intri_name):
    assert os.path.exists(intri_name), intri_name
    intri = FileStorage(intri_name)
    camnames = intri.read('names', dt='list')
    cameras = {}
    for key in camnames:
        cam = {}
        cam['K'] = intri.read('K_{}'.format(key))
        cam['invK'] = np.linalg.inv(cam['K'])
        cam['dist'] = intri.read('dist_{}'.format(key))
        cameras[key] = cam
    return cameras

def write_intri(intri_name, cameras):
    if not os.path.exists(os.path.dirname(intri_name)):
        os.makedirs(os.path.dirname(intri_name))
    intri = FileStorage(intri_name, True)
    results = {}
    camnames = list(cameras.keys())
    intri.write('names', camnames, 'list')
    for key_, val in cameras.items():
        key = key_.split('.')[0]
        K, dist = val['K'], val['dist']
        assert K.shape == (3, 3), K.shape
        assert dist.shape == (1, 5) or dist.shape == (5, 1) or dist.shape == (1, 4) or dist.shape == (4, 1), dist.shape
        intri.write('K_{}'.format(key), K)
        intri.write('dist_{}'.format(key), dist.flatten()[None])

def write_extri(extri_name, cameras):
    if not os.path.exists(os.path.dirname(extri_name)):
        os.makedirs(os.path.dirname(extri_name))
    extri = FileStorage(extri_name, True)
    results = {}
    camnames = list(cameras.keys())
    extri.write('names', camnames, 'list')
    for key_, val in cameras.items():
        key = key_.split('.')[0]
        extri.write('R_{}'.format(key), val['Rvec'])
        extri.write('Rot_{}'.format(key), val['R'])
        extri.write('T_{}'.format(key), val['T'])
    return 0


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

def write_camera(camera, path):
    from os.path import join
    intri_name = join(path, 'intri.yml')
    extri_name = join(path, 'extri.yml')
    intri = FileStorage(intri_name, True)
    extri = FileStorage(extri_name, True)
    results = {}
    camnames = []
    # for light in lights:
    #     camnames += [key_.split('.')[0]+'_'+light for key_ in camera.keys()]
    camnames = [key_.split('.')[0] for key_ in camera.keys()]
    intri.write('names', camnames, 'list')
    extri.write('names', camnames, 'list')
    for key_, val in camera.items():
        if key_ == 'basenames':
            continue
        # key = key_.split('.')[0]+'_'+light
        key = key_.split('.')[0]
        intri.write('K_{}'.format(key), val['K'])
        intri.write('dist_{}'.format(key), val['dist'])
        if 'Rvec' not in val.keys():
            val['Rvec'] = cv2.Rodrigues(val['R'])[0]
        extri.write('R_{}'.format(key), val['Rvec'])
        extri.write('Rot_{}'.format(key), val['R'])
        extri.write('T_{}'.format(key), val['T'])

def read_cameras(path, intri='intri.yml', extri='extri.yml', subs=[]):
    cameras = read_camera(join(path, intri), join(path, extri))
    cameras.pop('basenames')
    if len(subs) > 0:
        cameras = {key:cameras[key].astype(np.float32) for key in subs}
    return cameras
def list2np(string_array):
    new_list = [np.fromstring(line,dtype=np.float32, sep=' ') for line in string_array]
    return np.asarray(new_list)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='pinecone_dr.xml')
    parser.add_argument('--intri_path', type=str, default='/CT/prithvi/static00/synth_light_stage/cameras')
    parser.add_argument('--extri_path', type=str, default='pinecone_dr.xml')
    parser.add_argument('--output_dir', type=str, default='pinecone_dr.xml')
    args = parser.parse_args()

    
    os.makedirs(args.output_dir,exist_ok=True)
    cam_names = [str(i).zfill(3) for i in range(0,13)]
    cameras = {}
    for i in range(0,13):
        if i==9:
            continue
        intri_lines = load_txt(os.path.join(args.intri_path,f'0_train_{str(i).zfill(4)}.txt'))
        intri = list2np(intri_lines)
        extri_lines = load_txt(os.path.join(args.extri_path,f'0_train_{str(i).zfill(4)}.txt'))
        
        extri = list2np(extri_lines)
        temp = {}
        temp['K'] = intri
        temp['dist'] = np.array([[0,0,0,0,0]])
        temp['R'] = extri[:3,:3]/1000
        temp['T'] = extri[:3,3:]/1000
        cameras[cam_names[i]] = temp
    
    write_camera(cameras,args.output_dir)

