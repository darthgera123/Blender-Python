import cv2
import numpy as np
import pickle as pk
from tqdm import tqdm

from mathutils import Vector
from mathutils.bvhtree import BVHTree
from blender_util import get_bmesh,raycast



def grid_query_unstruct(uvs, values, grid_res, method=None):
    r"""Grid queries unstructured data given by coordinates and their values.

    If you are looking to grid query structured data, such as an image, check
    out :func:`grid_query_img`.

    This function interpolates values on a rectangular grid given some sparse,
    unstrucured samples. One use case is where you have some UV locations and
    their associated colors, and you want to "paint the colors" on a UV canvas.

    Args:
        uvs (numpy.ndarray): N-by-2 array of UV coordinates where we have
            values (e.g., colors). See
            :func:`xiuminglib.blender.object.smart_uv_unwrap` for the UV
            coordinate convention.
        values (numpy.ndarray): N-by-M array of M-D values at the N UV
            locations, or N-array of scalar values at the N UV locations.
            Channels are interpolated independently.
        grid_res (array_like): Resolution (height first; then width) of
            the query grid.
        method (dict, optional): Dictionary of method-specific parameters.
            Implemented methods and their default parameters:

            .. code-block:: python

                # Default
                method = {
                    'func': 'griddata',
                    # Which SciPy function to call.

                    'func_underlying': 'linear',
                    # Fed to `griddata` as the `method` parameter.

                    'fill_value': (0,), # black
                    # Will be used to fill in pixels outside the convex hulls
                    # formed by the UV locations, and if `max_l1_interp` is
                    # provided, also the pixels whose interpolation is too much
                    # of a stretch to be trusted. In the context of "canvas
                    # painting," this will be the canvas' base color.

                    'max_l1_interp': np.inf, # trust/accept all interpolations
                    # Maximum L1 distance, which we can trust in interpolation,
                    # to pixels that have values. Interpolation across a longer
                    # range will not be trusted, and hence will be filled with
                    # `fill_value`.
                }

            .. code-block:: python

                method = {
                    'func': 'rbf',
                    # Which SciPy function to call.

                    'func_underlying': 'linear',
                    # Fed to `Rbf` as the `method` parameter.

                    'smooth': 0, # no smoothing
                    # Fed to `Rbf` as the `smooth` parameter.
                }

    Returns:
        numpy.ndarray: Interpolated values at query locations, of shape
        ``grid_res`` for single-channel input or ``(grid_res[0], grid_res[1],
        values.shape[2])`` for multi-channel input.
    """
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    assert values.ndim == 2 and values.shape[0] == uvs.shape[0]

    if method is None:
        method = {'func': 'griddata'}

    h, w = grid_res
    # Generate query coordinates
    grid_x, grid_y = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    # +---> x
    # |
    # v y
    grid_u, grid_v = grid_x, 1 - grid_y
    # ^ v
    # |
    # +---> u

    if method['func'] == 'griddata':
        from scipy.interpolate import griddata

        func_underlying = method.get('func_underlying', 'linear')
        fill_value = method.get('fill_value', (0,))
        max_l1_interp = method.get('max_l1_interp', np.inf)

        fill_value = np.array(fill_value)
        if len(fill_value) == 1:
            fill_value = np.tile(fill_value, values.shape[1])
        assert len(fill_value) == values.shape[1]

        if max_l1_interp is None:
            max_l1_interp = np.inf # trust everything

        # Figure out which pixels can be trusted
        has_value = np.zeros((h, w), dtype=np.uint8)
        ri = ((1 - uvs[:, 1]) * (h - 1)).astype(int).ravel()
        ci = (uvs[:, 0] * (w - 1)).astype(int).ravel()
        in_canvas = np.logical_and.reduce(
            (ri >= 0, ri < h, ci >= 0, ci < w)) # to ignore out-of-canvas points
        has_value[ri[in_canvas], ci[in_canvas]] = 1
        dist2val = cv2.distanceTransform(1 - has_value, cv2.DIST_L1, 3)
        trusted = dist2val <= max_l1_interp

        # Process each color channel separately
        interps = []
        for ch_i in range(values.shape[1]):
            v_fill = fill_value[ch_i]
            v = values[:, ch_i]
            interp = griddata(uvs, v, (grid_u, grid_v),
                              method=func_underlying,
                              fill_value=v_fill)
            interp[~trusted] = v_fill
            interps.append(interp)
        interps = np.dstack(interps)

    elif method['func'] == 'rbf':
        from scipy.interpolate import Rbf

        func_underlying = method.get('func_underlying', 'linear')
        smooth = method.get('smooth', 0)

        # Process each color channel separately
        interps = []
        for ch_i in range(values.shape[1]):
            v = values[:, ch_i]
            rbfi = Rbf(uvs[:, 0], uvs[:, 1], v,
                       function=func_underlying,
                       smooth=smooth)
            interp = rbfi(grid_u, grid_v)
            interps.append(interp)
        interps = np.dstack(interps)

    else:
        raise NotImplementedError(method['func'])

    if interps.shape[2] == 1:
        return interps[:, :, 0].squeeze()
    return interps

def calc_bidir_mapping(
        cached_unwrap, obj_name, xys, intersect, uvs, max_l1_interp=4):
    imw = xys[:, 0].max() + 1
    imh = xys[:, 1].max() + 1

    # Load the UV unwrapping by Blender
    with open(cached_unwrap, 'rb') as h:
        fi_li_vi_u_v = pk.load(h)

    # UV convention:
    # (0, 1)
    #   ^ v
    #   |
    #   |
    #   +------> (1, 0)
    # (0, 0)   u

    # Collect locations and their associated values
    uv2cam_locs, uv2cam_vals = [], []
    cam2uv_locs, cam2uv_vals = [], []
    for xy, oname, fi in tqdm(
            zip(xys, intersect['obj_names'], intersect['face_i']),
            total=xys.shape[0], desc="Filling camera-UV mappings"):
        if fi is None or oname != obj_name:
            continue

        uv = fi_li_vi_u_v[fi][:, 2:]

        # Collect locations and values for UV to camera
        camspc_loc = (xy[0] / float(imw), 1 - xy[1] / float(imh))
        uvspc_loc = np.hstack((uv[:, :1], 1 - uv[:, 1:]))
        uv2cam_locs.append(np.vstack([camspc_loc] * uvspc_loc.shape[0]))
        uv2cam_vals.append(uvspc_loc)

        # Now for camera to UV
        uvspc_loc = uv
        camspc_loc = (xy[0] / float(imw), xy[1] / float(imh))
        cam2uv_locs.append(uvspc_loc)
        cam2uv_vals.append(np.vstack([camspc_loc] * uvspc_loc.shape[0]))

    # Location convention for xm.img.grid_query_unstruct():
    # (0, 1)
    #     ^ v
    #     |
    #     +------> (1, 0)
    # (0, 0)      u

    # Value convention for xm.img.grid_query_unstruct(), for use by remap():
    # (0, 0)
    # +--------> (1, 0)
    # |           x
    # |
    # v y (0, 1)

    interp_method = {
        'func': 'griddata',
        'func_underlying': 'nearest',
        'fill_value': (0,), # black
        'max_l1_interp': max_l1_interp}

    # UV to camera space: interpolate unstructured values into an image
    locs = np.vstack(uv2cam_locs)
    vals = np.vstack(uv2cam_vals)
    uv2cam = grid_query_unstruct(
        locs, vals, (imh, imw), method=interp_method)

    # Camera to UV space: interpolate unstructured values into an image
    locs = np.vstack(cam2uv_locs)
    vals = np.vstack(cam2uv_vals)
    cam2uv = grid_query_unstruct(
        locs, vals, (uvs, uvs), method=interp_method)

    return uv2cam, cam2uv



def calc_view_cosines(cam_loc, xys, intersect, obj_name):
    imw = xys[:, 0].max() + 1
    imh = xys[:, 1].max() + 1

    view_cosines = np.zeros((imh, imw))

    for oname, xy, loc, normal in tqdm(
            zip(
                intersect['obj_names'], xys, intersect['locs'],
                intersect['normals']),
            total=xys.shape[0], desc="Filling view cosines"):
        if loc is None or oname != obj_name:
            continue

        p2c = (cam_loc - loc).normalized()
        normal = normal.normalized()

        view_cosines[xy[1], xy[0]] = p2c.dot(normal)

    return view_cosines


def calc_light_cosines(light_loc, xys, cam_intersect, obj):
    """Self-occlusion is considered here, so pixels in cast shadow have 0
    cosine values.
    """
    light_loc = Vector(light_loc)

    # Cast rays from the light to determine occlusion
    bm = get_bmesh(obj)
    tree = BVHTree.FromBMesh(bm)
    world2obj = obj.matrix_world.inverted()
    occluded = [False] * len(cam_intersect['locs'])
    for i, loc in enumerate(cam_intersect['locs']):
        if loc is not None:
            ray_from = world2obj @ light_loc
            ray_to = world2obj @ loc
            _, _, _, ray_dist = raycast(
                tree, ray_from, ray_to)
            if ray_dist is None:
                # Numerical issue, but hey, the ray is not blocked
                occluded[i] = False
            else:
                reach = np.isclose(ray_dist, (ray_to - ray_from).magnitude)
                occluded[i] = not reach

    imw = xys[:, 0].max() + 1
    imh = xys[:, 1].max() + 1

    light_cosines = np.zeros((imh, imw))

    for oname, xy, loc, normal, occlu in tqdm(
            zip(
                cam_intersect['obj_names'], xys, cam_intersect['locs'],
                cam_intersect['normals'], occluded),
            total=xys.shape[0], desc="Filling light cosines"):
        if loc is None or oname != obj.name:
            continue

        if occlu:
            continue

        p2l = (Vector(light_loc) - loc).normalized()
        normal = normal.normalized()

        light_cosines[xy[1], xy[0]] = p2l.dot(normal)

    return light_cosines