import bpy
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree

import numpy as np
from blender_util import get_bmesh, raycast


def get_calibration_matrices(camera):
    # Get camera intrinsic matrix
    focal_length = camera.data.lens
    sensor_width = camera.data.sensor_width
    sensor_height = camera.data.sensor_height
    pixel_aspect_ratio = camera.data.pixel_aspect_x / camera.data.pixel_aspect_y
    principal_point_x = camera.data.shift_x * sensor_width
    principal_point_y = camera.data.shift_y * sensor_height

    K = np.array([[focal_length, 0, principal_point_x],
                  [0, focal_length * pixel_aspect_ratio, principal_point_y],
                  [0, 0, 1]])

    # Get camera distortion coefficients
    k1 = camera.data.lens_distortion_k1
    k2 = camera.data.lens_distortion_k2
    k3 = camera.data.lens_distortion_k3
    p1 = camera.data.lens_distortion_p1
    p2 = camera.data.lens_distortion_p2

    distortion_coeffs = np.array([k1, k2, k3, p1, p2])

    return K, distortion_coeffs


def get_extrinsic_matrix(camera):
    # Get camera rotation and translation vectors
    rotation_vector, translation_vector = camera.matrix_world.decompose()[1:]

    # Convert to 3x3 rotation matrix
    rotation_matrix = rotation_vector.to_matrix().to_4x4()

    # Convert to 4x4 transformation matrix
    transformation_matrix = np.identity(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = translation_vector

    return transformation_matrix


def intrinsics_compatible_with_scene(cam, eps=1e-6):
    r"""Checks if camera intrinsic parameters are comptible with the current
    scene.

    Intrinsic parameters include sensor size and pixel aspect ratio, and scene
    parameters refer to render resolutions and their scale. The entire sensor is
    assumed active.

    Args:
        cam (bpy_types.Object): Camera object
        eps (float, optional): :math:`\epsilon` for numerical comparison.
            Considered equal if :math:`\frac{|a - b|}{b} < \epsilon`.

    Returns:
        bool: Check result.
    """
    # Camera
    sensor_width_mm = cam.data.sensor_width
    sensor_height_mm = cam.data.sensor_height

    # Scene
    scene = bpy.context.scene
    w = scene.render.resolution_x
    h = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100.
    pixel_aspect_ratio = \
        scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
    
    # Do these parameters make sense together?
    mm_per_pix_horizontal = sensor_width_mm / (w * scale)
    mm_per_pix_vertical = sensor_height_mm / (h * scale)
    if abs(mm_per_pix_horizontal / mm_per_pix_vertical - pixel_aspect_ratio) \
            / pixel_aspect_ratio < eps:
        return True

    print((
        "Render resolutions (w_pix = %d; h_pix = %d), active sensor size "
        "(w_mm = %f; h_mm = %f), and pixel aspect ratio (r = %f) don't make "
        "sense together. This could cause unexpected behaviors later. "
        "Consider running correct_sensor_height()"
    ), w, h, sensor_width_mm, sensor_height_mm, pixel_aspect_ratio)
    return False

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


def get_camera_matrix(cam, keep_disparity=False):
    r"""Gets camera matrix, intrinsics, and extrinsics from a camera.

    You can ask for a 4-by-4 projection that projects :math:`(x, y, z, 1)` to
    :math:`(x, y, 1, d)`, where :math:`d` is the disparity, reciprocal of
    depth.

    ``cam_mat.dot(pts)`` gives you projections in the following convention:

    .. code-block:: none

        +------------>
        |       proj[:, 0]
        |
        |
        v proj[:, 1]

    Args:
        cam (bpy_types.Object): Camera.
        keep_disparity (bool, optional): Whether or not the matrices keep
            disparity.

    Raises:
        ValueError: If render settings and camera intrinsics mismatch. Run
            :func:`intrinsics_compatible_with_scene` for advice.

    Returns:
        tuple:
            - **cam_mat** (*mathutils.Matrix*) -- Camera matrix, product of
              intrinsics and extrinsics. 4-by-4 if ``keep_disparity``; else,
              3-by-4.
            - **int_mat** (*mathutils.Matrix*) -- Camera intrinsics. 4-by-4 if
              ``keep_disparity``; else, 3-by-3.
            - **ext_mat** (*mathutils.Matrix*) -- Camera extrinsics. 4-by-4 if
              ``keep_disparity``; else, 3-by-4.
    """
    # Necessary scene update
    scene = bpy.context.scene
    bpy.context.view_layer.update() 
    # Check if camera intrinsic parameters comptible with render settings
    if not intrinsics_compatible_with_scene(cam):
        raise ValueError(
            ("Render settings and camera intrinsic parameters mismatch. "
             "Such computed matrices will not make sense. Make them "
             "consistent first. See error message from "
             "intrinsics_compatible_with_scene() above for advice"))

    # Intrinsics

    f_mm = cam.data.lens
    sensor_width_mm = cam.data.sensor_width
    sensor_height_mm = cam.data.sensor_height
    w = scene.render.resolution_x
    h = scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100.
    pixel_aspect_ratio = \
        scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

    if cam.data.sensor_fit == 'VERTICAL':
        # h times pixel height must fit into sensor_height_mm
        # w / pixel_aspect_ratio times pixel width will then fit into
        # sensor_width_mm
        s_y = h * scale / sensor_height_mm
        s_x = w * scale / pixel_aspect_ratio / sensor_width_mm
    else: # 'HORIZONTAL' or 'AUTO'
        # w times pixel width must fit into sensor_width_mm
        # h * pixel_aspect_ratio times pixel height will then fit into
        # sensor_height_mm
        s_x = w * scale / sensor_width_mm
        s_y = h * scale * pixel_aspect_ratio / sensor_height_mm

    skew = 0 # only use rectangular pixels

    if keep_disparity:
        # 4-by-4
        int_mat = Matrix((
            (s_x * f_mm, skew, w * scale / 2, 0),
            (0, s_y * f_mm, h * scale / 2, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1)))
    else:
        # 3-by-3
        int_mat = Matrix((
            (s_x * f_mm, skew, w * scale / 2),
            (0, s_y * f_mm, h * scale / 2),
            (0, 0, 1)))
    

    distortion_coeffs = np.array([[0,0,0,0,0]])

    # Extrinsics

    # Three coordinate systems involved:
    #   1. World coordinates "world"
    #   2. Blender camera coordinates "cam":
    #        - x is horizontal
    #        - y is up
    #        - right-handed: negative z is look-at direction
    #   3. Desired computer vision camera coordinates "cv":
    #        - x is horizontal
    #        - y is down (to align to the actual pixel coordinates)
    #        - right-handed: positive z is look-at direction

    rotmat_cam2cv = Matrix((
        (1, 0, 0),
        (0, -1, 0),
        (0, 0, -1)))

    # matrix_world defines local-to-world transformation, i.e.,
    # where is local (x, y, z) in world coordinate system?
    t, rot_euler = cam.matrix_world.decompose()[0:2]

    # World to Blender camera
    rotmat_world2cam = rot_euler.to_matrix().transposed() # same as inverse
    t_world2cam = rotmat_world2cam @ -t

    # World to computer vision camera
    rotmat_world2cv = rotmat_cam2cv @ rotmat_world2cam
    t_world2cv = rotmat_cam2cv @ t_world2cam

    if keep_disparity:
        # 4-by-4
        ext_mat = Matrix((
            rotmat_world2cv[0][:] + (t_world2cv[0],),
            rotmat_world2cv[1][:] + (t_world2cv[1],),
            rotmat_world2cv[2][:] + (t_world2cv[2],),
            (0, 0, 0, 1)))
    else:
        # 3-by-4
        ext_mat = Matrix((
            rotmat_world2cv[0][:] + (t_world2cv[0],),
            rotmat_world2cv[1][:] + (t_world2cv[1],),
            rotmat_world2cv[2][:] + (t_world2cv[2],)))

    # Camera matrix
    cam_mat = int_mat @ ext_mat

    print(f"Done computing camera matrix for {cam.name}")

    return cam_mat, np.array(int_mat), distortion_coeffs, np.array(ext_mat)  

def from_homo(pts, axis=None):
    """Converts from homogeneous to non-homogeneous coordinates.

    Args:
        pts (numpy.ndarray or mathutils.Vector): NumPy array of N-D point(s),
            or Blender vector of a single N-D point.
        axis (int, optional): The last slice of which dimension holds the
            :math:`w` values. Optional for 1D inputs.

    Raises:
        TypeError: If the input is neither a NumPy array nor a Blender vector.
        ValueError: If the provided ``axis`` value doesn't make sense for
            input point(s).

    Returns:
        numpy.ndarray or mathutils.Vector: Non-homogeneous coordinates of the
        input point(s).
    """
    if isinstance(pts, Vector):
        if axis not in (None, 0):
            raise ValueError(("axis must be either None (auto) or "
                              "0 for a Blender vector input"))
        pts_nonhomo = Vector(x / pts[-1] for x in pts[:-1])

    elif isinstance(pts, np.ndarray):
        if axis is None:
            if pts.ndim == 1:
                axis = 0
            else:
                raise ValueError(("When pts has more than one dimension, "
                                  "axis must be specified"))
        arr = np.take(pts, range(pts.shape[axis] - 1), axis=axis)
        w = np.take(pts, -1, axis=axis)
        pts_nonhomo = np.divide(arr, w) # by broadcasting

    else:
        raise TypeError(pts)

    return pts_nonhomo

def backproject_to_3d(xys, cam, obj_names=None, world_coords=False):
    """Backprojects 2D coordinates to 3D.

    Since a 2D point could have been projected from any point on a 3D line,
    this function will return the 3D point at which this line (ray)
    intersects with an object for the first time.

    Args:
        xys (array_like): XY coordinates of length 2 or shape N-by-2,
            in the following convention:

            .. code-block:: none

                (0, 0)
                +------------> (w, 0)
                |           x
                |
                |
                |
                v y (0, h)

        cam (bpy_types.Object): Camera.
        obj_names (str or list(str), optional): Name(s) of object(s) of
            interest. ``None`` means considering all objects.
        world_coords (bool, optional): Whether to return world or the object's
            local coordinates.

    Returns:
        tuple:
            - **ray_tos** (*mathutils.Vector or list(mathutils.Vector)*) --
              Location(s) at which each ray points in the world coordinates,
              regardless of ``world_coords``. This and the (shared) ray origin
              (``cam.location``) determine the rays.
            - **xyzs** (*mathutils.Vector or list(mathutils.Vector)*) --
              Intersection coordinates specified in either the world or the
              object's local coordinates, depending on ``world_coords``.
              ``None`` means no intersection.
            - **intersect_objnames** (*str or list(str)*) -- Name(s) of
              object(s) responsible for intersections. ``None`` means no
              intersection.
            - **intersect_facei** (*int or list(int)*) -- Index/indices of the
              face(s), where the intersection happens.
            - **intersect_normals** (*mathutils.Vector or
              list(mathutils.Vector)*) -- Normal vector(s) at the
              intersection(s) specified in the same space as ``xyzs``.
    """
    # Standardize inputs
    xys = np.array(xys).reshape(-1, 2)
    objs = bpy.data.objects
    if isinstance(obj_names, str):
        obj_names = [obj_names]
    elif obj_names is None:
        obj_names = [o.name for o in objs if o.type == 'MESH']
    z_c = 1 # any depth in the camera space, so long as not infinity

    scene = bpy.context.scene
    w, h = scene.render.resolution_x, scene.render.resolution_y
    scale = scene.render.resolution_percentage / 100.

    # Get 4-by-4 invertible camera matrix
    cam_mat, _, _ = get_camera_matrix(cam, keep_disparity=True)
    cam_mat_inv = cam_mat.inverted() # pixel space to world

    # Precompute BVH trees and world-to-object transformations
    trees, world2objs = {}, {}
    for obj_name in obj_names:
        world2objs[obj_name] = objs[obj_name].matrix_world.inverted()
        obj = objs[obj_name]
        bm = get_bmesh(obj)
        trees[obj_name] = BVHTree.FromBMesh(bm)

    ray_tos = [None] * xys.shape[0]
    xyzs = [None] * xys.shape[0]
    intersect_objnames = [None] * xys.shape[0]
    intersect_facei = [None] * xys.shape[0]
    intersect_normals = [None] * xys.shape[0]

    ray_from_world = cam.location

    # TODO: vectorize for performance
    for i in range(xys.shape[0]):

        # Compute any point on the line passing camera center and
        # projecting to (x, y)
        xy = xys[i, :]
        xy1d = np.append(xy, [1, 1 / z_c]) # with disparity
        xyzw = cam_mat_inv @ Vector(xy1d) # world

        # Ray start and direction in world coordinates
        ray_to_world = from_homo(xyzw)
        ray_tos[i] = ray_to_world

        first_intersect = None
        first_intersect_objname = None
        first_intersect_facei = None
        first_intersect_normal = None
        dist_min = np.inf

        # Test intersections with each object of interest
        for obj_name, tree in trees.items():
            obj2world = objs[obj_name].matrix_world
            world2obj = world2objs[obj_name]

            # Ray start and direction in local coordinates
            ray_from = world2obj @ ray_from_world
            ray_to = world2obj @ ray_to_world

            # Ray tracing
            loc, normal, facei, _ = raycast(tree, ray_from, ray_to)
            # Not using the returned ray distance as that's local
            dist = None if loc is None else (
                obj2world @ loc - ray_from_world).length

            # See if this intersection is closer to camera center than
            # previous intersections with other objects
            if (dist is not None) and (dist < dist_min):
                first_intersect = obj2world @ loc if world_coords else loc
                first_intersect_objname = obj_name
                first_intersect_facei = facei
                first_intersect_normal = \
                    obj2world.to_3x3() @ normal if world_coords else normal
                first_intersect_normal.normalize()
                # Re-normalize in case transforming to world coordinates has
                # ruined the unit length
                dist_min = dist

        xyzs[i] = first_intersect
        intersect_objnames[i] = first_intersect_objname
        intersect_facei[i] = first_intersect_facei
        intersect_normals[i] = first_intersect_normal

    assert None not in ray_tos, \
        ("No matter whether a ray is a hit or not, we must have a "
         "\"look-at\" for it")

    print(f"Backprojection done with camera {cam.name}")
    

    ret = (
        ray_tos, xyzs, intersect_objnames, intersect_facei, intersect_normals)
    if xys.shape[0] == 1:
        return tuple(x[0] for x in ret)
    return ret