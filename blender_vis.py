import bpy
import json
import numpy as np
from mathutils import Matrix
import mathutils
from PIL import Image
def add_camera(xyz=(0, 0, 0),
               rot_vec_rad=(0, 0, 0),
               name=None,
               proj_model='PERSP',
               f=35,
               sensor_fit='HORIZONTAL',
               sensor_width=32,
               sensor_height=18,
               clip_start=0.1,
               clip_end=100):
    """Adds a camera to  the current scene.

    Args:
        xyz (tuple, optional): Location. Defaults to ``(0, 0, 0)``.
        rot_vec_rad (tuple, optional): Rotations in radians around x, y and z.
            Defaults to ``(0, 0, 0)``.
        name (str, optional): Camera object name.
        proj_model (str, optional): Camera projection model. Must be
            ``'PERSP'``, ``'ORTHO'``, or ``'PANO'``. Defaults to ``'PERSP'``.
        f (float, optional): Focal length in mm. Defaults to 35.
        sensor_fit (str, optional): Sensor fit. Must be ``'HORIZONTAL'`` or
            ``'VERTICAL'``. See also :func:`get_camera_matrix`. Defaults to
            ``'HORIZONTAL'``.
        sensor_width (float, optional): Sensor width in mm. Defaults to 32.
        sensor_height (float, optional): Sensor height in mm. Defaults to 18.
        clip_start (float, optional): Near clipping distance. Defaults to 0.1.
        clip_end (float, optional): Far clipping distance. Defaults to 100.

    Returns:
        bpy_types.Object: Camera added.
    """
    bpy.ops.object.camera_add()
    cam = bpy.context.active_object

    if name is not None:
        cam.name = name

    cam.location = xyz
    cam.rotation_euler = rot_vec_rad

    cam.data.type = proj_model
    cam.data.lens = f
    cam.data.sensor_fit = sensor_fit
    cam.data.sensor_width = sensor_width
    cam.data.sensor_height = sensor_height
    cam.data.clip_start = clip_start
    cam.data.clip_end = clip_end

    print(f"Camera {cam.name} added")

    return cam

def easyset(cam,
            xyz=None,
            rot_vec_rad=None,
            name=None,
            proj_model=None,
            f=None,
            sensor_fit=None,
            sensor_width=None,
            sensor_height=None):
    """Sets camera parameters more easily.

    See :func:`add_camera` for arguments. ``None`` will result in no change.
    """
    if name is not None:
        cam.name = name

    if xyz is not None:
        cam.location = xyz

    if rot_vec_rad is not None:
        cam.rotation_euler = rot_vec_rad

    if proj_model is not None:
        cam.data.type = proj_model

    if f is not None:
        cam.data.lens = f

    if sensor_fit is not None:
        cam.data.sensor_fit = sensor_fit

    if sensor_width is not None:
        cam.data.sensor_width = sensor_width

    if sensor_height is not None:
        cam.data.sensor_height = sensor_height

def add_light_point(
        xyz=(0, 0, 0), name=None, size=0, color=(1, 1, 1), energy=100):
    
    bpy.ops.object.light_add(type='POINT', location=xyz)
    point = bpy.context.active_object
    
    if name is not None:
        point.name = name

    if len(color) == 3:
        color += (1.,)

    point.data.shadow_soft_size = size

    # Strength
    engine = bpy.context.scene.render.engine
    if engine == 'CYCLES':
        point.data.use_nodes = True
        point.data.node_tree.nodes['Emission'].inputs[
            'Strength'].default_value = energy
        point.data.node_tree.nodes['Emission'].inputs[
            'Color'].default_value = color
    else:
        raise NotImplementedError(engine)

    print(f"Light {point.name} added")

    return point

def load_json(json_path):
    with open(json_path, 'r') as h:
        data = json.load(h)
    return data

for o in bpy.data.objects:
    if o.type == 'LIGHT' or o.type == 'CAMERA':
        o.select_set(True)
    else:
        o.select_set(False)
bpy.ops.object.delete()

camera = '/CT/prithvi/static00/synth_light_stage/cameras/'
#lights = '/CT/prithvi/static00/nlt/trainvali_lights/L030.json'
lights = '/CT/prithvi/static00/synth_light_stage/lights/'
#cam = load_json(camera)
import os
#light = load_json(lights)
#all_cams = ['C30A.json','C28C.json','P03C.json','P18C.json']
all_cams = ['C02','C05','C08','C11','C14','C17','C26A','C26B','C27A','C27B','C27C','C28A','C28B','C28C','C30A','C30B','C30C','C30D','P03C','P18C']
#all_cams = ['C30D']
def swap_first_two_elements(lst):
    if len(lst) >= 2:
        lst[0], lst[2] = lst[2], -lst[0]
    return lst
def read_numbers_from_file(filename):                          
     numbers_list = []                                          
     with open(filename, 'r') as file:                          
             for line in file:                          
                     numbers = line.strip().split()     
                     numbers = [float(n) for n in numbers]        
                     numbers_list.append(numbers)          
     return numbers_list
cam_id = 15
for i in range(1):
    new_cam = load_json('/CT/prithvi/work/test/psnerf/dataset/armadillo/params.json')
    K = new_cam['K']
    pose = np.asarray(new_cam['pose_c2w'][cam_id])
    sensor_height = 20
    print(K)
    focal_length_mm = K[0][0] * 20 / 512
    name = f'view_{str(i+1).zfill(2)}'
    cam = add_camera(xyz=(0, 0, 0),name=name, rot_vec_rad=(0, 0, 0), proj_model='PERSP',sensor_height=sensor_height,sensor_width=sensor_height,f=focal_length_mm)
    pose_c2w = Matrix(pose)
    cam.matrix_world = pose_c2w
    
    cam.data.lens = focal_length_mm
#    print(focal_length_mm)



#all_cams = [c+'.json' for c in all_cams]
#for c in all_cams:
#    cam = load_json(os.path.join(camera,c))
#    cam_obj = add_camera(
#                xyz=cam['position'], rot_vec_rad=cam['rotation'],
#                name=cam['name'], f=cam['focal_length'],
#                sensor_width=cam['sensor_width'], sensor_height=cam['sensor_height'],
#                clip_start=cam['clip_start'], clip_end=cam['clip_end'])



#all_lights=['L030.json','L069.json','L058.json','L063.json','L250.json','L213.json','L013.json','L313.json']
#light_list = read_numbers_from_file('/CT/prithvi2/static00/DiLiGenT-MV/mvpmsData/bearPNG/view_01/light_directions.txt')
all_lights = os.listdir(lights)
lighty = load_json('/CT/prithvi/work/test/psnerf/dataset/claudia_blender/params.json')['light_direction']

#all_lights=['L030.json','L069.json','L058.json','L063.json']
camera_matrix = cam.matrix_world

#ldirs = np.load('/CT/prithvi/work/test/psnerf/dataset/claudia_dill_light/sdps_out_l331/light_direction_pred.npy')
ldirs = np.load('/CT/prithvi/work/test/psnerf/dataset/claudia_blender/blender_light_dir.npy')
lcam = ldirs[cam_id]
#for num,l in enumerate(lighty):
#    lpos_d = np.asarray(camera_matrix)[:3,:3]@l
#    light_name = f'L{str(num).zfill(3)}'
#    light_obj = add_light_point(xyz=lpos_d,name=light_name, size=0.001,energy=8)

#print(lpos_d) 

for num,l in enumerate(lcam[10:11]):
    lpos_d = np.asarray(camera_matrix)[:3,:3]@l
#    lpos_d = l
    light_obj = add_light_point(xyz=lpos_d, size=0.001,energy=1.95)
    light_name = f'L{str(num).zfill(3)}'
#print(lpos_d)
#for l in all_lights:
#    light = load_json(os.path.join(lights,l))
#    light_obj = add_light_point(xyz=light['position'], name=light['name'], size=0.01,energy=10)

