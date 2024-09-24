import sys
from sys import argv
from argparse import ArgumentParser
from os.path import join
import os 
from shutil import copyfile
import numpy as np

dir = os.path.dirname('/CT/prithvi/work/neural-light-transport/data_2.8/')
if not dir in sys.path:
    sys.path.append(dir)

import bpy
import mathutils
from mathutils import Matrix

from util import load_json, safe_cast_to_int, normalize_uint, name_from_json_path,\
                img_load, add_b_ch, save_nn, save_uvs
from camera import add_camera, backproject_to_3d
from blender_render import render, render_alpha, easyset, render_visibility, render_normal, render_diffuse, render_position
from blender_util import add_light_point


def create_mat(path):
    mat = bpy.data.materials.new(name="Normal")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
    texImage.image = bpy.data.images.load(path)
    mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
    # create normal
    normal = mat.node_tree.nodes.new('ShaderNodeNormalMap')
    mat.node_tree.links.new(bsdf.inputs['Normal'], normal.outputs['Normal'])
    return mat 

def read_numbers_from_file(filename):                          
     numbers_list = []                                          
     with open(filename, 'r') as file:                          
             for line in file:                          
                     numbers = line.strip().split()     
                     numbers = [float(n) for n in numbers]        
                     numbers_list.append(numbers)          
     return numbers_list

def swap_first_two_elements(lst):
    if len(lst) >= 2:
        lst[1], lst[2] = -lst[2], lst[1]
    return lst

def run(args):
    bpy.ops.wm.open_mainfile(filepath=args.scene)

    # Remove existing lights and cameras
    for o in bpy.data.objects:
        if o.type == 'LIGHT' or o.type == 'CAMERA':
            o.select_set(True)
        else:
            o.select_set(False)
    bpy.ops.object.delete()

    

    
    # imh = 512
    # imw = imh / cam['sensor_height'] * cam['sensor_width']
    # imw = safe_cast_to_int(imw)
    
    # easyset(n_samples=256,color_mode='RGB',h=imh,w=imw)


    # initialize
    bpy.context.scene.frame_end = 1
    bpy.context.scene.frame_set(1)
    for i in range(21,56):
        obj_params = load_json('/CT/prithvi/work/test/psnerf/dataset/armadillo/params.json')
        K = obj_params['K']
        pose = np.asarray(obj_params['pose_c2w'][i])
        sensor_height = 20
        focal_length_mm = K[0][0] * 20 / 512
        cam_name = f'view_{str(i+1).zfill(2)}'
        cam = add_camera(xyz=(0, 0, 0),name=cam_name, rot_vec_rad=(0, 0, 0), proj_model='PERSP',sensor_height=sensor_height,sensor_width=sensor_height,f=focal_length_mm)
        pose_c2w = Matrix(pose)
        cam.matrix_world = pose_c2w
        cam.data.lens = focal_length_mm

        lights = obj_params['light_direction']


        camera_matrix = cam.matrix_world
        camera_matrix_4x4 = camera_matrix.to_4x4()
        # camera_matrix_inv = camera_matrix_4x4.inverted()
        for num,l in enumerate(lights):
            lpose = np.asarray(camera_matrix)[:3,:3]@l
            light_obj = add_light_point(xyz=lpose, size=0.3,energy=8)
            light_name = f'L{str(num).zfill(3)}'
            if args.rgb_video:
                outdir = join(args.output_path,f'img/{cam_name}')
                os.makedirs(outdir,exist_ok=True)
                rgb_camspc_f = join(outdir, f'{str(num+1).zfill(3)}.png')
                render(rgb_camspc_f)
            if args.visibility:
                visible_dir = join(args.output_path,'visibility',f'{cam_name}')
                os.makedirs(visible_dir,exist_ok=True)
                visible = join(visible_dir, f'{str(num+1).zfill(3)}.png')
                render_visibility(visible, samples=256,exr=False)
            for o in bpy.data.objects:
                if o.type == 'LIGHT':
                    o.select_set(True)
                else:
                    o.select_set(False)
            bpy.ops.object.delete()
    
    
    
    
        if args.normal:
            normal_dir = join(args.output_path,'normal')
            os.makedirs(normal_dir,exist_ok=True)
            normal = join(normal_dir, cam_name)
            render_normal(normal, samples=256,exr=True)
        if args.diffuse:
            diffuse_dir = join(args.output_path,'diffuse')
            os.makedirs(diffuse_dir,exist_ok=True)
            diffuse = join(diffuse_dir, cam_name)
            render_diffuse(diffuse, samples=256,exr=True)
    
        if args.mask:
            alpha_dir = join(args.output_path,'mask')
            os.makedirs(alpha_dir,exist_ok=True)
            alpha_f = join(alpha_dir, f'{cam_name}.png')
            render_alpha(alpha_f, samples=256)
        for o in bpy.data.objects:
            if o.type == 'CAMERA':
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
        '--cam_json', type=str,  help="path to the camera .json")
    parser.add_argument(
        '--light_json', type=str, help="path to the light .json")
    parser.add_argument(
        '--output_path', type=str, required=True, help="output directory")
    parser.add_argument(
        '--rgb_video', action='store_true',  help="save_rgb_video")
    parser.add_argument(
        '--mask', action='store_true',  help="save_rgb_video")
    parser.add_argument(
        '--visibility', action='store_true',  help="save_rgb_video")
    parser.add_argument(
        '--normal', action='store_true',  help="save_rgb_video")
    parser.add_argument(
        '--diffuse', action='store_true',  help="save_rgb_video")
    parser.add_argument(
        '--position', action='store_true',  help="save_rgb_video")
    run(parser.parse_args(argv))