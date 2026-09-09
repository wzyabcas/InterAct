import os
from pathlib import Path

# Select the offscreen backend before importing OpenGL via MeshViewer.
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')

import numpy as np
import trimesh
import math
# from render.mesh_utils import MeshViewer
# from render.utils import colors
from .mesh_utils import MeshViewer
from .utils import colors
import imageio
import pyrender
from PIL import Image
from PIL import ImageDraw 

def c2rgba(c):
    if len(c) == 3:
        c.append(1)
    c = [c_i/255 for c_i in c[:3]]

    return c

def visualize_body_objs(body_verts, body_face, obj_verts, obj_faces, save_path,
                        multi_angle=False, h=1024, w=1024, bg_color='white',
                        show_frame=False, fps=30):
    """Write a Y-up body and any number of objects to an MP4 animation.

    ``body_verts`` is (T, V, 3), and ``body_face`` is (F, 3).
    ``obj_verts`` is a list of world-space arrays, each (T, V_i, 3),
    or (V_i, 3) for a static object. ``obj_faces`` is the matching list
    of triangle arrays (F_i, 3), indexed into each object's own vertices.
    Empty object lists render the body alone. No input arrays are modified.

    A fixed camera fits every mesh over the whole sequence. ``multi_angle``
    adds a second view 90 degrees away, doubling the output width. ``h`` and
    ``w`` specify each view's dimensions; ``fps`` sets playback speed.
    Frames are streamed to disk and the OpenGL renderer is always released.
    The output directory is created if necessary. Returns ``str(save_path)``.
    """
    body_verts = np.asarray(body_verts)
    if body_verts.ndim != 3 or body_verts.shape[-1] != 3:
        raise ValueError('body_verts must have shape (T, V, 3)')
    if body_verts.shape[0] == 0 or body_verts.shape[1] == 0:
        raise ValueError('The body sequence must contain frames and vertices')
    if not np.isfinite(body_verts).all():
        raise ValueError('body_verts contains nonfinite coordinates')
    if len(obj_verts) != len(obj_faces):
        raise ValueError('obj_verts and obj_faces must have the same length')
    if any(not isinstance(size, (int, np.integer)) or size <= 0 or size % 2
           for size in (h, w)):
        raise ValueError('h and w must be positive even integers for MP4 output')
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be positive and finite')

    def checked_faces(faces, vertex_count, name):
        faces = np.asarray(faces)
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
            raise ValueError(f'{name} must contain triangles with shape (F, 3)')
        if not np.issubdtype(faces.dtype, np.integer):
            raise ValueError(f'{name} must contain integer vertex indices')
        if faces.min() < 0 or faces.max() >= vertex_count:
            raise ValueError(f'{name} contains an out-of-range vertex index')
        return faces

    body_face = checked_faces(body_face, body_verts.shape[1], 'body_face')
    frame_count = len(body_verts)
    objects = []
    lower = body_verts.min(axis=(0, 1)).astype(float)
    upper = body_verts.max(axis=(0, 1)).astype(float)
    for index, (verts, faces) in enumerate(zip(obj_verts, obj_faces)):
        verts = np.asarray(verts)
        if verts.ndim not in (2, 3) or verts.shape[-1] != 3 or verts.shape[-2] == 0:
            raise ValueError(f'obj_verts[{index}] must have shape (T, V, 3) or (V, 3)')
        if verts.ndim == 3 and len(verts) != frame_count:
            raise ValueError(f'obj_verts[{index}] must have {frame_count} frames')
        if not np.isfinite(verts).all():
            raise ValueError(f'obj_verts[{index}] contains nonfinite coordinates')
        faces = checked_faces(faces, verts.shape[-2], f'obj_faces[{index}]')
        objects.append((verts, faces))
        lower = np.minimum(lower, verts.reshape(-1, 3).min(axis=0))
        upper = np.maximum(upper, verts.reshape(-1, 3).max(axis=0))

    background = np.asarray(colors[bg_color] if isinstance(bg_color, str)
                            else bg_color, dtype=float).copy()
    if background.max() > 1:
        background /= 255.0
    scene = pyrender.Scene(bg_color=background, ambient_light=[0.3, 0.3, 0.3])
    center = (lower + upper) / 2.0
    radius = max(np.linalg.norm(upper - lower) / 2.0, 0.1)
    yfov = np.radians(45.0)
    half_angle = min(yfov / 2.0, np.arctan(np.tan(yfov / 2.0) * w / h))
    distance = radius / np.sin(half_angle) * 1.15

    def camera_pose(azimuth, elevation=20):
        azimuth, elevation = np.radians([azimuth, elevation])
        backward = np.array([np.sin(azimuth) * np.cos(elevation),
                             np.sin(elevation), np.cos(azimuth) * np.cos(elevation)])
        right = np.cross([0.0, 1.0, 0.0], backward)
        right /= np.linalg.norm(right)
        pose = np.eye(4)
        pose[:3, :3] = np.column_stack([right, np.cross(backward, right), backward])
        pose[:3, 3] = center + distance * backward
        return pose

    camera = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=w / h,
                                       znear=max(distance - 2 * radius, 0.001),
                                       zfar=distance + 4 * radius)
    camera_node = scene.add(camera, pose=camera_pose(35))
    for azimuth in (45, 165, 285):
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=1.6)
        scene.add(light, pose=camera_pose(azimuth, elevation=45))

    # Preserve the physical ground at world y=0 and all relative mesh positions.
    floor = trimesh.creation.box(extents=[max(upper[0] - lower[0], 1.0) * 1.5,
                                         0.004, max(upper[2] - lower[2], 1.0) * 1.5])
    floor.apply_translation([center[0], -0.002, center[2]])
    floor_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.88, 0.9, 0.93, 1.0], metallicFactor=0.0, roughnessFactor=1.0)
    scene.add(pyrender.Mesh.from_trimesh(floor, material=floor_material, smooth=False))
    palette = [[0.38, 0.64, 0.82, 1.0], [0.87, 0.47, 0.43, 1.0],
               [0.46, 0.7, 0.54, 1.0], [0.69, 0.55, 0.78, 1.0]]
    body_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.88, 0.74, 0.57, 1.0], metallicFactor=0.0,
        roughnessFactor=0.8, doubleSided=True)
    object_materials = [pyrender.MetallicRoughnessMaterial(
        baseColorFactor=palette[i % len(palette)], metallicFactor=0.0,
        roughnessFactor=0.8, doubleSided=True) for i in range(len(objects))]

    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)
    nodes = []
    try:
        with imageio.get_writer(str(output), fps=fps, codec='libx264',
                                pixelformat='yuv420p', macro_block_size=1) as writer:
            for frame_index in range(frame_count):
                for node in nodes:
                    scene.remove_node(node)
                body = trimesh.Trimesh(vertices=body_verts[frame_index], faces=body_face,
                                       process=False)
                nodes = [scene.add(pyrender.Mesh.from_trimesh(body, material=body_material))]
                for (verts, faces), material in zip(objects, object_materials):
                    mesh = trimesh.Trimesh(vertices=verts[frame_index] if verts.ndim == 3 else verts,
                                           faces=faces, process=False)
                    nodes.append(scene.add(pyrender.Mesh.from_trimesh(mesh, material=material)))
                views = []
                for azimuth in ([35, 125] if multi_angle else [35]):
                    scene.set_pose(camera_node, pose=camera_pose(azimuth))
                    color, _ = renderer.render(scene)
                    views.append(color)
                frame = np.concatenate(views, axis=1)
                if show_frame:
                    frame_image = Image.fromarray(frame)
                    ImageDraw.Draw(frame_image).text((8, 8), f'{frame_index:04d}',
                                                    fill=(30, 30, 30))
                    frame = np.asarray(frame_image)
                writer.append_data(frame)
    finally:
        renderer.delete()
    return str(output)


def visualize_body_obj(body_verts, body_face, obj_verts, obj_face, save_path,
                       multi_angle=False, h=256, w=256, bg_color='white', show_frame=False):
    """[summary]

    Args:
        rec (torch.tensor): [description]
        inp (torch.tensor, optional): [description]. Defaults to None.
        multi_angle (bool, optional): Whether to use different angles. Defaults to False.

    Returns:
        np.array :   Shape of output (view_angles, seqlen, 3, im_width, im_height, )

    """
    import os
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

    im_height = h
    im_width = w
    seqlen = len(body_verts)

    mesh_rec = body_verts
    obj_mesh_rec = obj_verts
    
    minx, _, miny = mesh_rec.min(axis=(0, 1))
    maxx, _, maxy = mesh_rec.max(axis=(0, 1))
    minsxy = (minx, maxx, miny, maxy)
    height_offset = np.min(mesh_rec[:, :, 1])  # Min height

    mesh_rec = mesh_rec.copy()
    obj_mesh_rec = obj_mesh_rec.copy()
    # mesh_rec[:, :, 1] -= height_offset
    # obj_mesh_rec[:, :, 1] -= height_offset
    mesh_rec[:, :, 0] -= (minx + maxx) / 2
    obj_mesh_rec[:, :, 0] -= (minx + maxx) / 2
    mesh_rec[:, :, 2] -= (miny + maxy) / 2
    obj_mesh_rec[:, :, 2] -= (miny + maxy) / 2

    mv = MeshViewer(width=im_width, height=im_height,
                    add_ground_plane=True, plane_mins=minsxy,
                    use_offscreen=True,
                    bg_color=bg_color)

    mv.render_wireframe = False

    if multi_angle:
        video = np.zeros([seqlen, 1 * im_width, 2 * im_height, 3])
    else:
        video = np.zeros([seqlen, im_width, im_height, 3])

    for i in range(seqlen):

        obj_mesh_color = np.tile(c2rgba(colors['pink']), (obj_mesh_rec.shape[1], 1))

        obj_m_rec = trimesh.Trimesh(vertices=obj_mesh_rec[i],
                                    faces=obj_face,
                                    vertex_colors=obj_mesh_color)

        mesh_color = np.tile(c2rgba(colors['yellow_pale']), (mesh_rec.shape[1], 1))

        m_rec = trimesh.Trimesh(vertices=mesh_rec[i],
                                faces=body_face,
                                vertex_colors=mesh_color)
        all_meshes = []

        all_meshes = all_meshes + [obj_m_rec, m_rec]
        mv.set_meshes(all_meshes, group_name='static')
        video_i = mv.render()

        if multi_angle:
            video_views = [video_i]
            for _ in range(1):
                all_meshes = []
                Ry = trimesh.transformations.rotation_matrix(math.radians(90), [0, 1, 0])
                obj_m_rec.apply_transform(Ry)
                m_rec.apply_transform(Ry)
                all_meshes = all_meshes + [obj_m_rec, m_rec]
                mv.set_meshes(all_meshes, group_name='static')
                
                video_views.append(mv.render())
            # video_i = np.concatenate((np.concatenate((video_views[0], video_views[1]), axis=1),
            #                           np.concatenate((video_views[3], video_views[2]), axis=1)), axis=1)
            video_i = np.concatenate((video_views[0], video_views[1]), axis=1)
        video[i] = video_i

    video_writer = imageio.get_writer(save_path, fps=30)
    video = video.astype(np.uint8)
    for i in range(seqlen):
        frame = video[i]
        pil_image = Image.fromarray(frame)
        if show_frame:
            draw = ImageDraw.Draw(pil_image)
            text = f"{i}".zfill(4)
            draw.text((5, 5), text,fill='red')
        frame_with_text = np.array(pil_image).astype(np.uint8)
        video_writer.append_data(frame_with_text)
    video_writer.close()
    del mv

def points_to_spheres(points, radius=0.01):
    spheres = []
    for p in points:
        sphere = trimesh.creation.icosphere(radius=radius)
        sphere.apply_translation(p)
        spheres.append(sphere)
    return trimesh.util.concatenate(spheres)

def visualize_points_obj(m_pcd, obj_verts, obj_face, save_path,
                       multi_angle=False, h=256, w=256, bg_color='white', show_frame=False):
    """[summary]

    Args:
        rec (torch.tensor): [description]
        inp (torch.tensor, optional): [description]. Defaults to None.
        multi_angle (bool, optional): Whether to use different angles. Defaults to False.

    Returns:
        np.array :   Shape of output (view_angles, seqlen, 3, im_width, im_height, )

    """
    import os
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

    im_height = h
    im_width = w
    seqlen = len(m_pcd)

    mesh_rec = m_pcd
    obj_mesh_rec = obj_verts
    
    minx, _, miny = mesh_rec.min(axis=(0, 1))
    maxx, _, maxy = mesh_rec.max(axis=(0, 1))
    minsxy = (minx, maxx, miny, maxy)
    height_offset = np.min(mesh_rec[:, :, 1])  # Min height

    mesh_rec = mesh_rec.copy()
    obj_mesh_rec = obj_mesh_rec.copy()
    # mesh_rec[:, :, 1] -= height_offset
    # obj_mesh_rec[:, :, 1] -= height_offset
    mesh_rec[:, :, 0] -= (minx + maxx) / 2
    obj_mesh_rec[:, :, 0] -= (minx + maxx) / 2
    mesh_rec[:, :, 2] -= (miny + maxy) / 2
    obj_mesh_rec[:, :, 2] -= (miny + maxy) / 2

    mv = MeshViewer(width=im_width, height=im_height,
                    add_ground_plane=True, plane_mins=minsxy,
                    use_offscreen=True,
                    bg_color=bg_color)

    mv.render_wireframe = False

    if multi_angle:
        video = np.zeros([seqlen, 1 * im_width, 2 * im_height, 3])
    else:
        video = np.zeros([seqlen, im_width, im_height, 3])

    for i in range(seqlen):

        obj_mesh_color = np.tile(c2rgba(colors['pink']), (obj_mesh_rec.shape[1], 1))

        obj_m_rec = trimesh.Trimesh(vertices=obj_mesh_rec[i],
                                    faces=obj_face,
                                    vertex_colors=obj_mesh_color)

        mesh_color = np.tile(c2rgba(colors['yellow_pale']), (mesh_rec.shape[1], 1))

        # m_rec = trimesh.points.PointCloud(mesh_rec[i])
        m_rec = points_to_spheres(mesh_rec[i])

        all_meshes = []

        all_meshes = all_meshes + [obj_m_rec, m_rec]
        mv.set_meshes(all_meshes, group_name='static')
        video_i = mv.render()

        if multi_angle:
            video_views = [video_i]
            for _ in range(1):
                all_meshes = []
                Ry = trimesh.transformations.rotation_matrix(math.radians(90), [0, 1, 0])
                obj_m_rec.apply_transform(Ry)
                m_rec.apply_transform(Ry)
                all_meshes = all_meshes + [obj_m_rec, m_rec]
                mv.set_meshes(all_meshes, group_name='static')
                
                video_views.append(mv.render())
            # video_i = np.concatenate((np.concatenate((video_views[0], video_views[1]), axis=1),
            #                           np.concatenate((video_views[3], video_views[2]), axis=1)), axis=1)
            video_i = np.concatenate((video_views[0], video_views[1]), axis=1)
        video[i] = video_i

    video_writer = imageio.get_writer(save_path, fps=30)
    video = video.astype(np.uint8)
    for i in range(seqlen):
        frame = video[i]
        pil_image = Image.fromarray(frame)
        if show_frame:
            draw = ImageDraw.Draw(pil_image)
            text = f"{i}".zfill(4)
            draw.text((5, 5), text,fill='red')
        frame_with_text = np.array(pil_image).astype(np.uint8)
        video_writer.append_data(frame_with_text)
    video_writer.close()
    del mv
