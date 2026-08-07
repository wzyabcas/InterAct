import numpy as np
import trimesh
import math
import imageio

from PIL import Image, ImageDraw
from text2interaction.render.mesh_viz import MeshViewer
from text2interaction.render.utils import colors

class Debugger:
    @staticmethod
    def visualize_body_obj_and_axes(
        body_verts_faces,
        obj_verts_faces,
        save_path,
        *,
        axis_length=0.5,
        multi_angle=False,
        h=512,
        w=512,
        bg_color="white",
        show_frame=False,
    ):
        """Render body, objects, and a fixed XYZ world-axis marker.

        ``body_verts_faces`` and ``obj_verts_faces`` are ``(vertices, faces)``
        pairs, so callers can pass mesh-producing function results directly.

        The marker follows the standard XYZ/RGB convention: +X is red, +Y is
        green, and +Z is blue.  Its white origin is placed at world (0, 0, 0);
        the renderer's ground plane is the XZ plane at y=0.

        This is intentionally local to the HUMOTO visualizer so the shared
        ``visualize_body_obj`` implementation remains unchanged.
        """
        body_verts, body_faces = body_verts_faces
        obj_verts, obj_faces = obj_verts_faces

        body_verts = np.asarray(body_verts)
        body_faces = np.asarray(body_faces)
        obj_verts = np.asarray(obj_verts)
        obj_faces = np.asarray(obj_faces)

        if axis_length <= 0:
            raise ValueError(f"axis_length must be positive, got {axis_length}")
        if body_verts.ndim != 3 or body_verts.shape[-1] != 3:
            raise ValueError(f"Expected body vertices (T, V, 3), got {body_verts.shape}")
        if body_faces.ndim != 2 or body_faces.shape[-1] != 3:
            raise ValueError(f"Expected body faces (F, 3), got {body_faces.shape}")
        if obj_verts.ndim != 3 or obj_verts.shape[-1] != 3:
            raise ValueError(f"Expected object vertices (T, V, 3), got {obj_verts.shape}")
        if obj_faces.ndim != 2 or obj_faces.shape[-1] != 3:
            raise ValueError(f"Expected object faces (F, 3), got {obj_faces.shape}")
        if body_verts.shape[0] != obj_verts.shape[0]:
            raise ValueError(
                "Body and object sequences must have the same frame count, got "
                f"{body_verts.shape[0]} and {obj_verts.shape[0]}"
            )

        # Match visualize_body_obj: center the sequence horizontally, but preserve
        # its Y coordinates so y=0 remains the actual world ground plane.
        min_x, _, min_z = body_verts.min(axis=(0, 1))
        max_x, _, max_z = body_verts.max(axis=(0, 1))
        center_x = (min_x + max_x) / 2
        center_z = (min_z + max_z) / 2

        body_verts = body_verts.copy()
        obj_verts = obj_verts.copy()
        body_verts[:, :, 0] -= center_x
        body_verts[:, :, 2] -= center_z
        obj_verts[:, :, 0] -= center_x
        obj_verts[:, :, 2] -= center_z

        # Since the scene was centered above, this marker represents the translated
        # world origin.  It remains fixed for every frame.
        axis_transform = np.eye(4)
        axis_transform[0, 3] = -center_x
        axis_transform[2, 3] = -center_z
        axis_mesh = trimesh.creation.axis(
            origin_size=axis_length * 0.06,
            axis_radius=axis_length * 0.02,
            axis_length=axis_length,
            transform=axis_transform,
        )

        viewer = MeshViewer(
            width=w,
            height=h,
            add_ground_plane=True,
            plane_mins=(min_x, max_x, min_z, max_z),
            use_offscreen=True,
            bg_color=bg_color,
        )
        viewer.render_wireframe = False

        object_rgb = np.asarray(colors["pink"][:3], dtype=np.float32) / 255.0
        body_rgb = np.asarray(colors["yellow_pale"][:3], dtype=np.float32) / 255.0
        rotate_y_90 = trimesh.transformations.rotation_matrix(
            math.radians(90), [0, 1, 0]
        )

        writer = imageio.get_writer(save_path, fps=30)
        try:
            for frame_idx in range(body_verts.shape[0]):
                object_mesh = trimesh.Trimesh(
                    vertices=obj_verts[frame_idx],
                    faces=obj_faces,
                    vertex_colors=np.tile(object_rgb, (obj_verts.shape[1], 1)),
                    process=False,
                )
                body_mesh = trimesh.Trimesh(
                    vertices=body_verts[frame_idx],
                    faces=body_faces,
                    vertex_colors=np.tile(body_rgb, (body_verts.shape[1], 1)),
                    process=False,
                )

                # Keep the same number of meshes passed to MeshViewer as the shared
                # renderer: the axes are merged into the object mesh for rendering.
                object_and_axes = trimesh.util.concatenate(
                    [object_mesh, axis_mesh.copy()]
                )
                viewer.set_meshes([object_and_axes, body_mesh], group_name="static")
                rendered_views = [viewer.render()]

                if multi_angle:
                    rotated_object_and_axes = object_and_axes.copy()
                    rotated_body = body_mesh.copy()
                    rotated_object_and_axes.apply_transform(rotate_y_90)
                    rotated_body.apply_transform(rotate_y_90)
                    viewer.set_meshes(
                        [rotated_object_and_axes, rotated_body],
                        group_name="static",
                    )
                    rendered_views.append(viewer.render())

                frame = np.concatenate(rendered_views, axis=1)
                image = Image.fromarray(frame.astype(np.uint8))
                draw = ImageDraw.Draw(image)
                if show_frame:
                    draw.text((5, 5), f"{frame_idx:04d}", fill="red")

                legend_y = max(5, image.height - 15)
                draw.text((5, legend_y), "+X", fill=(255, 0, 0))
                draw.text((23, legend_y), "+Y", fill=(0, 160, 0))
                draw.text((41, legend_y), "+Z", fill=(0, 0, 255))
                draw.text((62, legend_y), "world axes", fill=(40, 40, 40))
                writer.append_data(np.asarray(image, dtype=np.uint8))
        finally:
            writer.close()
            del viewer
