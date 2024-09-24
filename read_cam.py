"""
Run:
$BLENDER --background --python /CT/prithvi/work/neural-light-transport/data_2.8/read_cam.py -- \
--scene=/CT/prithvi/static00/render_people/claudia/claudia_cards.blend \
--cameras=/CT/prithvi/static00/synth_light_stage/cameras \
--output_path=/CT/prithvi/static00/synth_light_stage/cameras/
"""
import cv2
import numpy as np
import os
from os.path import join
dir = os.path.dirname('/CT/prithvi/work/neural-light-transport/data_2.8/')
import sys
from sys import argv
from argparse import ArgumentParser
if not dir in sys.path:
    sys.path.append(dir)
from mathutils import Matrix
import bpy

from util import load_json, safe_cast_to_int
from camera import add_camera,get_camera_matrix
from blender_render import  easyset

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

def read_cameras(path, intri='intri.yml', extri='extri.yml', subs=[]):
    cameras = read_camera(join(path, intri), join(path, extri))
    cameras.pop('basenames')
    if len(subs) > 0:
        cameras = {key:cameras[key].astype(np.float32) for key in subs}
    return cameras

def write_camera(camera, path,lights):
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
        for light in lights:
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

def reset():
    for o in bpy.data.objects:
        if o.type == 'LIGHT' or o.type == 'CAMERA':
            o.select_set(True)
        else:
            o.select_set(False)
    bpy.ops.object.delete()

def run(args):
    bpy.ops.wm.open_mainfile(filepath=args.scene)
    
    # Remove existing lights and cameras
    reset()
    cam_all = args.cameras
    # cam_list = os.listdir(cam_all)
    cam_list = ['C02','C05','C08','C11','C14','C17','C26A','C26B','C27A','C27B','C27C','C28A','C28B','C28C','C30A','C30B','C30C','C30D','P03C','P18C']
    cam_list = [c+'.json' for c in cam_list]
    
    cameras = {}
    for i in range(20):
        new_cam = load_json('/CT/prithvi/work/test/psnerf/dataset/bunny/params.json')
        K1 = new_cam['K']
        pose = np.asarray(new_cam['pose_c2w'][i])
        focal_length_mm = K1[0][0] * 16 / 512
        cam = add_camera(xyz=(0, 0, 0), rot_vec_rad=(0, 0, 0), proj_model='PERSP',sensor_height=16,sensor_width=16,f=focal_length_mm)
        pose_c2w = Matrix(pose)
        cam.matrix_world = pose_c2w
        # focal_length_mm = K1[0][0] * cam.data.sensor_width / 512
        # cam.data.lens = focal_length_mm
        
        imh = 512
        imw = imh / cam.data.sensor_width * cam.data.sensor_height
        imw = safe_cast_to_int(imw)
        easyset(n_samples=256,color_mode='RGB',h=imh,w=imw)
        bpy.context.scene.camera=cam
        temp = {}
        cam_mat, K, dist, ext = get_camera_matrix(cam)
        r = ext[:,:3]
        t = ext[:,3:]
        temp['K'] = K
        temp['dist'] = dist
        temp['R'] = r
        temp['T'] = t
        cam_name = f'view_{str(i+1).zfill(2)}'
        cameras[cam_name] = temp

    # for cam_name in cam_list:
    #     cam = load_json(os.path.join(cam_all,cam_name))
    #     cam_obj = add_camera(
    #             xyz=cam['position'], rot_vec_rad=cam['rotation'],
    #             name=cam['name'], f=cam['focal_length'],
    #             sensor_width=cam['sensor_width'], sensor_height=cam['sensor_height'],
    #             clip_start=cam['clip_start'], clip_end=cam['clip_end'])
    #     imh = 512
    #     imw = imh / cam['sensor_height'] * cam['sensor_width']
    #     imw = safe_cast_to_int(imw)
        
    #     easyset(n_samples=256,color_mode='RGB',h=imh,w=imw)
    #     bpy.context.scene.camera=cam_obj
    #     temp = {}
    #     cam_mat, K, dist, ext = get_camera_matrix(cam_obj)
    #     r = ext[:,:3]
    #     t = ext[:,3:]
    #     temp['K'] = K
    #     temp['dist'] = dist
    #     temp['R'] = r
    #     temp['T'] = t
    #     cameras[cam_name] = temp

    lights_all = args.lights
    # light_list = os.listdir(lights_all)
    # light_list = sorted([l.split('.')[0] for l in light_list])
    # light_list= ['L030','L069','L058','L063','L250','L213','L013','L313']
    light_list=['']
    write_camera(cameras,args.output_path,light_list)

if __name__ == '__main__':
    # Blender-Python binary
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    parser = ArgumentParser(description="")
    parser.add_argument(
        '--scene', type=str, required=True, help="path to the .blend scene")
    parser.add_argument(
        '--cameras', type=str, required=True, help="all the cameras")
    parser.add_argument(
        '--lights', type=str, default='', help="all the cameras")
    parser.add_argument(
        '--output_path', type=str, required=True, help="output directory")
    run(parser.parse_args(argv))


