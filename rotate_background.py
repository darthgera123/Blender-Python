import sys
from sys import argv
from argparse import ArgumentParser
from os.path import join
import os 
from shutil import copyfile
from mathutils import Matrix
import numpy as np

dir = os.path.dirname('/CT/prithvi/work/neural-light-transport/data_2.8/')
if not dir in sys.path:
    sys.path.append(dir)

import bpy

from util import load_json, safe_cast_to_int, normalize_uint, name_from_json_path,\
                img_load, add_b_ch, save_nn, save_uvs
from camera import add_camera, backproject_to_3d
from blender_render import render, render_alpha, easyset, render_visibility
from blender_util import add_light_point,add_light_env




def run(args):
    bpy.ops.wm.open_mainfile(filepath=args.scene)

    # Remove existing lights and cameras
    
    camera = args.cam_json
    lights = args.light_json
    # for o in bpy.data.objects:
    #     if o.type == 'MESH' or o.type == 'CAMERA':
    #         o.select_set(True)
    #     else:
    #         o.select_set(False)
    #     bpy.ops.object.delete()
    bpy.ops.object.select_all(action='DESELECT')

    # Loop through all objects in bpy.data.objects and delete them
    for obj in bpy.data.objects:
        obj.select_set(True)  # Select the object
        bpy.ops.object.delete()

    # Clear the selection
    bpy.ops.object.select_all(action='DESELECT')

    # image_tex = args.image_tex
    # normal_tex = args.normal_tex
    # #  for appearance, choose mesh
    # # for pose choose armature
    
    # # obj = bpy.data.objects['object']
    # obj_mesh = bpy.data.objects['object']
    # obj_arm = bpy.data.objects['object.001'] # obj.type==armature
    
    cameras = load_json(args.cam_json)
    
    N = len(cameras['frames'])
    envname = (args.envmap).split('.')[0]
    outdir = join(args.output_path,f'{envname}')

    for pix in range(0,512,8):
        envmap = join(args.env,args.envmap+f'_{pix}.png')
        # add_light_point((1.5,1.5,1.5))
        add_light_env(env =envmap)
        
        easyset(n_samples=100,color_mode='RGB',h=1440,w=810)
        # Create scene
        
        for i,frame in enumerate(cameras['frames'][:1]):
            poses = np.asarray(frame['transform_matrix'])
            poses[:3, 1:3] *= -1
            
            focal_length_mm = np.asarray(frame['fl_x'])
            name = f'view_{str(i+1).zfill(2)}'
            cam = add_camera(xyz=(0, 0, 0),name=name, rot_vec_rad=(0, 0, 0), proj_model='PERSP',sensor_height=frame['h'],sensor_width=frame['w'],f=focal_length_mm)
            
        #    cam = add_camera(xyz=(0, 0, 0),name=name, rot_vec_rad=(0, 0, 0), proj_model='PERSP',sensor_height=512,sensor_width=512,f=focal_length_mm)
            pose_c2w = Matrix(poses)
            cam.matrix_world = pose_c2w
            output_name = f"Cam_{str(i).zfill(2)}_{pix}"
            # bpy.context.scene.render.filepath = os.path.join(outdir, output_name)
            print("Output",os.path.join(outdir, output_name))
            # bpy.ops.render.render(write_still=True)
            render(outpath=os.path.join(outdir, output_name))
            for o in bpy.data.objects:
                if o.type == 'MESH' or o.type == 'CAMERA':
                    o.select_set(True)
                else:
                    o.select_set(False)
                bpy.ops.object.delete()

    
    
    

if __name__ == '__main__':
    # Blender-Python binary
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    parser = ArgumentParser(description="")
    parser.add_argument(
        '--scene', type=str, required=True, help="path to the .blend scene")
    parser.add_argument(
        '--cam_json', type=str, required=True, help="path to the camera .json")
    parser.add_argument(
        '--light_json', type=str, help="path to the light .json")
    parser.add_argument(
        '--env', type=str, help="path to the light .json")
    parser.add_argument(
        '--output_path', type=str, required=True, help="output directory")
    parser.add_argument(
        '--envmap', type=str, required=True, help="path to the camera .json")
    run(parser.parse_args(argv))