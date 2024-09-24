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

from util import load_json, safe_cast_to_int, normalize_uint, name_from_json_path,\
                img_load, add_b_ch, save_nn, save_uvs
from camera import add_camera, backproject_to_3d
from blender_render import render, render_alpha, easyset
from blender_util import add_light_point
import math
import mathutils
# from uv_render import calc_bidir_mapping,calc_light_cosines,calc_view_cosines

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

def rotate_bone(bone, rotation_vector):
    # Convert the rotation vector to a quaternion
    quat = mathutils.Quaternion(rotation_vector)

    # Set the bone's rotation mode to quaternion
    bone.rotation_mode = 'QUATERNION'

    # Set the bone's rotation quaternion
    bone.rotation_quaternion = quat

def translate_bone(bone, translation):
    # Create a translation matrix
    matrix = mathutils.Matrix.Translation(translation)

    # Multiply the bone's matrix by the translation matrix
    bone.matrix = bone.matrix @ matrix

def unreal2blender():
    unreal = ['pelvis','neck_01','lowerarm_twist_01_r','lowerarm_twist_01_l',
        'upperarm_twist_01_r','upperarm_twist_01_l','clavicle_r','clavicle_l',
            'calf_twist_01_r','calf_twist_01_l','thigh_twist_01_r','thigh_twist_01_l',
            'calf_r','calf_l','thigh_r','thigh_l']
    blender = ['hip','neck','lowerarm_twist_r','lowerarm_twist_l','upperarm_twist_r',
    'upperarm_twist_l','shoulder_r','shoulder_l','lowerleg_twist_r','lowerleg_twist_l',
    'upperleg_twist_r','upperleg_twist_l','lowerleg_r','lowerleg_l','upperleg_r','upperleg_l']
    u2b = dict(zip(unreal,blender))
    return u2b
    
def animate_bone(bone, rotations, translations, frame):
    # Set the bone's rotation mode to Quaternion
    bone.rotation_mode = 'QUATERNION'

    # Set the current frame to 1
    bpy.context.scene.frame_set(frame)

    # Set the bone's initial rotation and location
    bone.rotation_quaternion = mathutils.Quaternion(rotations)
    bone.location = translations

    # Insert keyframes for the bone's rotation and location at frame 1
    bone.keyframe_insert(data_path="rotation_quaternion", index=-1)
    bone.keyframe_insert(data_path="location", index=-1)



def run(args):
    bpy.ops.wm.open_mainfile(filepath=args.scene)

    # Remove existing lights and cameras
    for o in bpy.data.objects:
        if o.type == 'LIGHT' or o.type == 'CAMERA':
            o.select_set(True)
        else:
            o.select_set(False)
    bpy.ops.object.delete()

    camera = args.cam_json
    lights = args.light_json
    
    image_tex = args.image_tex
    normal_tex = args.normal_tex
    #  for appearance, choose mesh
    # for pose choose armature
    
    # obj = bpy.data.objects['object']
    obj_mesh = bpy.data.objects['object']
    obj_arm = bpy.data.objects['object.001'] # obj.type==armature
    
    cam_name = name_from_json_path(camera)
    light_name = name_from_json_path(lights)
    cam = load_json(camera)
    light = load_json(lights)
    outdir = join(args.output_path,f'{cam_name}_{light_name}')
    os.makedirs(outdir,exist_ok=True)

    # Create scene
    cam_obj = add_camera(
            xyz=cam['position'], rot_vec_rad=cam['rotation'],
            name=cam['name'], f=cam['focal_length'],
            sensor_width=cam['sensor_width'], sensor_height=cam['sensor_height'],
            clip_start=cam['clip_start'], clip_end=cam['clip_end'])
    
    light_obj = add_light_point(xyz=light['position'], name=light['name'], size=light['size'])
    
    imh = args.imh
    imw = imh / cam['sensor_height'] * cam['sensor_width']
    imw = safe_cast_to_int(imw)
    
    easyset(n_samples=args.spp,color_mode='RGB',h=imh,w=imw)

    bpy.context.view_layer.objects.active = obj_arm
    bpy.ops.object.mode_set(mode='POSE')
    bones = obj_arm.pose.bones
    
    motion_path = '/CT/UnrealEgo2/static00/UnrealEgoData/ArchVisInterior_ArchVis_RT/Day/rp_claudia_rigged_002_ue4/SKM_MenReadingGlasses_Shape_01/017/Victory_Idle/json'
    motion = f'{motion_path}/frame_{0}.json'
    mot = load_json(motion)
    u2b = unreal2blender()

    # render frame 0
    # render frame 1
    # all xy translations are w.r.t the first motion 

    # initialize
    start = obj_arm.location
    start_mot = mot 
    bpy.context.scene.frame_set(1)
    for key in mot['joints'].keys():
        pitch,yaw,roll = mot['joints'][key]['rot']
        x,y,z = mot['joints'][key]['trans']
        if key in bones.keys():
            bone = bpy.context.object.pose.bones[key]
        else:
            bone = bpy.context.object.pose.bones[u2b[key]]
        rotation_vector = (math.radians(pitch),math.radians(yaw),math.radians(roll))
        rotate_bone(bone,rotation_vector)
        translation = (x/100.0, y/100.0, z/100.0) # Create the translation vector
        #             # UE4 is in cm but blender is in m
        translation = mathutils.Vector(translation) - start
        translate_bone(bone,translation)
        bone.keyframe_insert(data_path="rotation_quaternion", index=-1)
        bone.keyframe_insert(data_path="matrix", index=-1)
        # animate_bone(bone,rotation_vector,translation,1)
    
    # rgb_camspc_f = join(outdir, 'rgb_camspc_rot4.png')
    # render(rgb_camspc_f)


    for i in range(1,5):
        motion = f'{motion_path}/frame_{i}.json'
        mot = load_json(motion)
        bpy.context.scene.frame_set(5*i+1)
        for key in mot['joints'].keys():
            pitch,yaw,roll = mot['joints'][key]['rot']
            x,y,z = mot['joints'][key]['trans']
            x_s,y_s,z_s = start_mot['joints'][key]['trans']
            x,y = x-x_s, y-y_s
            if key in bones.keys():
                bone = bpy.context.object.pose.bones[key]
            else:
                bone = bpy.context.object.pose.bones[u2b[key]]
            rotation_vector = (math.radians(pitch),math.radians(yaw),math.radians(roll))
            rotate_bone(bone,rotation_vector)
            translation = (x/100.0, y/100.0, z/100.0) # Create the translation vector
                        # UE4 is in cm but blender is in m
            translation = mathutils.Vector(translation) - start
            translate_bone(bone,translation)
            bone.keyframe_insert(data_path="rotation_quaternion", index=-1)
            bone.keyframe_insert(data_path="matrix", index=-1)
        # animate_bone(bone,rotation_vector,translation,10)
    # rgb_camspc_f = join(outdir, 'rgb_camspc_rot5.png')
    # render(rgb_camspc_f)

    # for i in range(1,19):
    #     bpy.context.scene.frame_set(2*i+1)
    #     motion = f'{motion_path}/frame_{i}.json'
    #     mot = load_json(motion)
        
            
    #         # translate_bone(bone, translation)# Translate a bone by (1, 0, 0)
    #     bone.keyframe_insert(data_path="rotation_quaternion", index=-1)
    #     # bone.keyframe_insert(data_path="rotation_quaternion", index=-1)
    

    # frames = [0,10,20]
    # bone = bpy.context.object.pose.bones['upperleg_l']
    # rotations = [(math.radians(45), 0, 0), (math.radians(90), 0, 0)]  # Set the rotations for the animation
    # translations = [(1, 0, 0), (0, 1, 0)]  # Set the translations for the animation
    # frame_rates = [10, 20]  # Set the frame rates for the animation
    # animate_bone(bone, rotations, translations, frame_rates)
    
#     bpy.context.scene.frame_set(1)

#     # Rotate the bone by 45 degrees around the x-axis
    
    # bone = bpy.context.object.pose.bones['head']
    # bone.rotation_mode = 'QUATERNION'
    # bone.rotation_quaternion = (math.sqrt(2) / 2, math.sqrt(2) / 2, 0, 0)

#     # Insert a keyframe for the bone's rotation at frame 1


#     # Set the current frame to 30
#     bpy.context.scene.frame_set(5)

#     # Rotate the bone by 45 degrees around the y-axis
#     bone.rotation_quaternion = (math.sqrt(2) / 2, 0, math.sqrt(2) / 2, 0)

#     # Insert a keyframe for the bone's rotation at frame 30
#     bone.keyframe_insert(data_path="rotation_quaternion", index=-1)

#     # Set the current frame to 60
#     bpy.context.scene.frame_set(10)

#     # Rotate the bone by 45 degrees around the z-axis
#     bone.rotation_quaternion = (0, 0, math.sqrt(2) / 2, math.sqrt(2) / 2)

#     # Insert a keyframe for the bone's rotation at frame 60
#     bone.keyframe_insert(data_path="rotation_quaternion", index=-1)

    bpy.context.scene.frame_end = 30

    bpy.context.scene.render.filepath = join(outdir, 'rgb_camspc_rot.mp4')
    bpy.context.scene.render.image_settings.file_format = 'FFMPEG'

# # # Render the animation
    bpy.context.scene.camera=cam_obj
    bpy.ops.render.render(animation=True)
    
    

    # Render rgb image
    # obj.data.materials[0] = create_mat(image_tex)
    # rgb_camspc_f = join(outdir, 'rgb_camspc_rot1.png')
    # render(rgb_camspc_f)
    
    # # Render Alpha map
    # alpha_f = join(outdir, 'alpha.png')
    # render_alpha(alpha_f, samples=args.spp)
    
    # obj.data.materials[0] = create_mat(normal_tex)
    # rgb_camspc_f = join(outdir, 'normal_camspc.png')
    # render(rgb_camspc_f)
    

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
        '--light_json', type=str, required=True, help="path to the light .json")
    parser.add_argument(
        '--cam_nn_json', type=str, required=True,
        help="path to the .json of mapping from camera to its nearest neighbor")
    parser.add_argument(
        '--light_nn_json', type=str, required=True,
        help="path to the .json of mapping from light to its nearest neighbor")
    parser.add_argument(
        '--imh', type=int, default=512,
        help="image height (width derived from camera's sensor height and width)")
    parser.add_argument(
        '--uvs', type=int, default=5, help="size of (square) texture map")
    parser.add_argument(
        '--spp', type=int, default=256, help="samples per pixel for rendering")
    parser.add_argument(
        '--output_path', type=str, required=True, help="output directory")
    parser.add_argument(
        '--image_tex', type=str, required=True,help="image texture")
    parser.add_argument(
        '--normal_tex', type=str, required=True,help="normal map")
    # parser.add_argument(
    #     '--type', type=str, required=True, help="trainvali or test")
    parser.add_argument(
        '--debug', type=bool, default=False,
        help="whether to dump additional outputs for debugging")
    run(parser.parse_args(argv))