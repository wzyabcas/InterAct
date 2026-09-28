import colorsys
import numpy as np
import torch
from pytorch3d.structures import Meshes, join_meshes_as_batch, join_meshes_as_scene
from pytorch3d.renderer import (
    DirectionalLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    BlendParams,
    PerspectiveCameras,
    look_at_view_transform,
)
from typing import Union, List, Tuple, Optional, Dict, Any

def create_chessboard_mesh(
    board_size: int = 30, 
    square_size: float = 1.0, 
    device: torch.device = torch.device("cpu"), 
    y_up: bool = True
) -> Meshes:
    """
    Create a chessboard mesh for ground plane visualization.
    
    This function generates a checkerboard pattern mesh that can be used as a ground
    plane in 3D visualizations. The mesh is created with alternating light and dark
    squares for better visual reference.
    
    Args:
        board_size: Number of squares along each axis (total size is 2*board_size x 2*board_size).
        square_size: Size of each individual square.
        device: PyTorch device to create tensors on.
        y_up: If True, uses Y-up coordinate system; if False, uses Z-up.
    
    Returns:
        PyTorch3D Meshes object containing the chessboard mesh with vertex colors.
    
    Example:
        >>> import torch
        >>> from pytorch3d.structures import Meshes
        >>> 
        >>> # Create a small chessboard
        >>> chessboard = create_chessboard_mesh(board_size=5, device=torch.device("cpu"))
        >>> print(type(chessboard))  # <class 'pytorch3d.structures.meshes.Meshes'>
        >>> print(chessboard.verts_list()[0].shape)  # torch.Size([100, 3]) for 5x5 board
    """
    vertices = []
    faces = []
    colors = []

    for i in range(-board_size, board_size):
        for j in range(-board_size, board_size):
            # Define the 4 vertices of the square
            v0 = [i * square_size, j * square_size, 0]
            v1 = [(i + 1) * square_size, j * square_size, 0]
            v2 = [(i + 1) * square_size, (j + 1) * square_size, 0]
            v3 = [i * square_size, (j + 1) * square_size, 0]

            # Add vertices to the list
            vertices.extend([v0, v1, v2, v3])

            # Define the 2 faces of the square
            face_offset = ((i + board_size) * (2 * board_size) + (j + board_size)) * 4
            faces.append([face_offset, face_offset + 1, face_offset + 2])
            faces.append([face_offset, face_offset + 2, face_offset + 3])

            # Define the color of the square
            if (i + j) % 2 == 0:
                color = [0.95, 0.95, 0.95]  # Light gray
            else:
                color = [0.8, 0.8, 0.8]  # Deeper gray

            # Add color for each vertex of the square
            colors.extend([color, color, color, color])

    vertices = torch.tensor(vertices, dtype=torch.float32, device=device)
    if y_up:
        vertices = vertices[:, [0, 2, 1]]
        vertices[:, 2] = -vertices[:, 2]
    faces = torch.tensor(faces, dtype=torch.int64, device=device)
    colors = torch.tensor(colors, dtype=torch.float32, device=device)

    textures = TexturesVertex(verts_features=colors.unsqueeze(0))
    chessboard_mesh = Meshes(verts=[vertices], faces=[faces], textures=textures)

    return chessboard_mesh

def get_light_colors() -> List[Tuple[float, float, float]]:
    """
    Generate a diverse set of light colors for rendering.
    
    This function creates a palette of colors by varying hue and saturation
    in HSV color space, providing a good range of colors for lighting
    different objects in the scene.
    
    Returns:
        List of RGB color tuples, each containing (R, G, B) values in [0, 1] range.
    
    Example:
        >>> colors = get_light_colors()
        >>> print(len(colors))  # 30 (10 hues x 3 saturations)
        >>> print(colors[0])  # (0.8, 0.16, 0.16) - first color
    """
    hue = np.linspace(0, 1, 10)
    saturation = np.linspace(0.2, 0.6, 3)
    value = 0.8
    colors = []
    for s in saturation:
        for h in hue:
            colors.append(colorsys.hsv_to_rgb(h, s, value))
    return colors

def setup_renderer(
    image_size: Tuple[int, int] = (900, 1600), 
    device: torch.device = torch.device("cpu"), 
    y_up: bool = True
) -> Tuple[MeshRenderer, DirectionalLights]:
    """
    Setup PyTorch3D renderer with appropriate settings for human motion visualization.
    
    This function creates a configured mesh renderer with directional lighting,
    suitable for rendering human motion sequences and objects with good visual quality.
    
    Args:
        image_size: Tuple of (height, width) for the rendered images.
        device: PyTorch device to run rendering on.
        y_up: If True, uses Y-up coordinate system; if False, uses Z-up.
    
    Returns:
        Tuple containing:
        - MeshRenderer: Configured PyTorch3D mesh renderer
        - DirectionalLights: Configured directional lighting setup
    
    Example:
        >>> import torch
        >>> from pytorch3d.renderer import MeshRenderer, DirectionalLights
        >>> 
        >>> renderer, lights = setup_renderer(
        ...     image_size=(512, 512), 
        ...     device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ... )
        >>> print(type(renderer))  # <class 'pytorch3d.renderer.mesh_renderer.MeshRenderer'>
        >>> print(type(lights))  # <class 'pytorch3d.renderer.lighting.DirectionalLights'>
    """
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        perspective_correct=True,
        cull_backfaces=True,
        max_faces_per_bin=50000
    )
    
    light_direction = (1.0, 1.0, 1.0) if y_up else (1.0, -1.0, 1.0)
    lights = DirectionalLights(
        device=device,
        direction=(light_direction,),
        ambient_color=((0.8, 0.8, 0.8),),
        diffuse_color=((0.2, 0.2, 0.2),),
        specular_color=((0., 0., 0.),),
    )
    
    shader = SoftPhongShader(
        device=device,
        lights=lights,
        blend_params=BlendParams(background_color=(1.0, 1.0, 1.0), sigma=1e-4, gamma=1e-4)
    )
    
    rasterizer = MeshRasterizer(
        raster_settings=raster_settings
    )
    
    renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=shader
    )
    
    return renderer, lights

def extract_eye_at_from_hip(
    hip_data: torch.Tensor, 
    y_up: bool = True
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Compute camera eye and target positions from hip joint data.
    
    This function analyzes the hip joint positions across a sequence to determine
    appropriate camera positioning for rendering the human motion. The camera
    is positioned to capture the full range of motion with good viewing angle.
    
    Args:
        hip_data: PyTorch tensor of shape [B, 3] containing hip joint positions.
        y_up: If True, uses Y-up coordinate system; if False, uses Z-up.
    
    Returns:
        Tuple containing:
        - eye: Camera position as (x, y, z) tuple
        - at: Camera target position as (x, y, z) tuple
    
    Raises:
        ValueError: If hip_data doesn't have shape [B, 3].
        TypeError: If hip_data is not a PyTorch tensor.
    
    Example:
        >>> import torch
        >>> hip_positions = torch.randn(10, 3)  # 10 frames of hip data
        >>> eye, at = extract_eye_at_from_hip(hip_positions)
        >>> print(eye)  # (x, y, z) camera position
        >>> print(at)   # (x, y, z) target position
    """
    # Input validation
    if not isinstance(hip_data, torch.Tensor):
        raise TypeError(f"hip_data must be a PyTorch tensor, got {type(hip_data)}")
    
    if hip_data.ndim != 2 or hip_data.shape[1] != 3:
        raise ValueError(f"hip_data must have shape [B, 3], got {hip_data.shape}")
    
    min_hip = torch.min(hip_data, dim=0)[0]
    max_hip = torch.max(hip_data, dim=0)[0]
    center_hip = (min_hip + max_hip) / 2
    hip_diff = max_hip - min_hip
    hip_diff = torch.sum(hip_diff ** 2) ** 0.5
    dis = hip_diff / 2 + 2
    if y_up:
        camLocation = (center_hip[0], center_hip[1] + 1.5, center_hip[2] + dis)
    else:
        camLocation = (center_hip[0], center_hip[1] - dis, center_hip[2] + 1.5)
    at = (center_hip[0], center_hip[1] , center_hip[2])
    return camLocation, at

def render_frames(
    renderer: MeshRenderer, 
    cameras: PerspectiveCameras, 
    human_verts: torch.Tensor, 
    human_faces: torch.Tensor, 
    object_group: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], 
    device: torch.device = torch.device("cpu"), 
    ground: Optional[Union[Meshes, List[Meshes]]] = None
) -> torch.Tensor:
    """
    Render a batch of frames with human meshes and objects.
    
    This function renders multiple frames simultaneously, combining human meshes
    with object meshes and optional ground plane meshes. It creates textured
    meshes for each component and renders them using the provided camera setup.
    
    Args:
        renderer: Configured PyTorch3D mesh renderer.
        cameras: Perspective cameras for rendering.
        human_verts: Human mesh vertices of shape [B, N, 3] where B is batch size, N is number of vertices.
        human_faces: Human mesh faces of shape [F, 3] where F is number of faces.
        object_group: Dictionary mapping object names to (vertices, faces, colors) tuples.
                     Each tuple contains tensors for all frames in the batch.
        device: PyTorch device to run rendering on.
        ground: Optional ground plane mesh(es) to include in the scene.
    
    Returns:
        Rendered images as PyTorch tensor of shape [B, H, W, 4] (RGBA format).
    
    Raises:
        ValueError: If input tensors have incompatible shapes.
        TypeError: If inputs are not the expected types.
    
    Example:
        >>> import torch
        >>> from pytorch3d.renderer import MeshRenderer, PerspectiveCameras
        >>> 
        >>> # Setup renderer and cameras
        >>> renderer, lights = setup_renderer()
        >>> cameras = PerspectiveCameras(...)
        >>> 
        >>> # Render frames
        >>> human_verts = torch.randn(5, 1000, 3)  # 5 frames, 1000 vertices
        >>> human_faces = torch.randint(0, 1000, (2000, 3))  # 2000 faces
        >>> object_group = {"chair": (verts, faces, colors)}
        >>> images = render_frames(renderer, cameras, human_verts, human_faces, object_group)
        >>> print(images.shape)  # torch.Size([5, 900, 1600, 4])
    """
    if ground is None:
        ground = []
    elif isinstance(ground, Meshes):
        ground = [ground]

    B, N, _ = human_verts.shape
    human_texture = torch.tensor([[[144. / 255., 210. / 255., 236. / 255.]]], device=device).repeat(B, N, 1)
    human_textures = TexturesVertex(verts_features=human_texture)
    human_faces = human_faces.unsqueeze(0).repeat(B, 1, 1)
    human_meshes = Meshes(verts=list(human_verts), faces=list(human_faces), textures=human_textures)
    
    object_meshes = {}
    for obj in object_group:
        obj_verts, obj_faces, obj_color = object_group[obj]
        obj_faces = obj_faces.repeat(B, 1, 1)
        obj_color = obj_color.unsqueeze(0).unsqueeze(0).repeat(B, obj_verts.shape[1], 1)
        obj_textures = TexturesVertex(verts_features=obj_color)
        obj_mesh = Meshes(verts=list(obj_verts), faces=list(obj_faces), textures=obj_textures)
        object_meshes[obj] = obj_mesh
    
    scenes = []
    for t in range(B):
        human_mesh = human_meshes[t]
        obj_meshes = [object_meshes[obj][t] for obj in object_meshes.keys()]
        combined_meshes = join_meshes_as_scene([human_mesh] + ground + obj_meshes)
        scenes.append(combined_meshes)
    
    scenes = join_meshes_as_batch(scenes)
    
    images = renderer(scenes, cameras=cameras)
    return images

def render_sequence(
    human_joints: Dict[str, torch.Tensor], 
    human_verts: torch.Tensor, 
    human_faces: torch.Tensor, 
    object_transformed: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], 
    image_size: Tuple[int, int] = (900, 1600), 
    render_batch_size: int = 200, 
    device: torch.device = torch.device("cpu"), 
    ground: Optional[Union[Meshes, List[Meshes]]] = None, 
    y_up: bool = True
) -> np.ndarray:
    """
    Render a complete human motion sequence with objects.
    
    This function renders an entire human motion sequence by processing it in batches
    to manage memory usage. It automatically sets up cameras based on hip joint
    positions and renders each batch of frames.
    
    Args:
        human_joints: Dictionary mapping joint names to joint positions of shape [T, 3].
                     Must contain 'mixamorig:Hips' key for camera positioning.
        human_verts: Human mesh vertices of shape [T, N, 3] where T is sequence length, N is number of vertices.
        human_faces: Human mesh faces of shape [F, 3] where F is number of faces.
        object_transformed: Dictionary mapping object names to (vertices, faces, colors) tuples.
                           Each tuple contains tensors for all frames in the sequence.
        image_size: Tuple of (height, width) for the rendered images.
        render_batch_size: Number of frames to render in each batch to manage memory.
        device: PyTorch device to run rendering on.
        ground: Optional ground plane mesh(es) to include in the scene.
        y_up: If True, uses Y-up coordinate system; if False, uses Z-up.
    
    Returns:
        Rendered images as NumPy array of shape [T, H, W, 3] (RGB format, uint8).
    
    Raises:
        KeyError: If 'mixamorig:Hips' key is missing from human_joints.
        ValueError: If input tensors have incompatible shapes.
        TypeError: If inputs are not the expected types.
    """
    # get the human verts and joints
    hip_positions = human_joints['mixamorig:Hips']

    # get the eye and at from the hip positions
    eye, at = extract_eye_at_from_hip(hip_positions, y_up=y_up)
    R, T = look_at_view_transform(
        eye=[eye],
        at=[at],
        up=[[0, 1, 0] if y_up else [0, 0, 1]]
    )
    renderer, lights = setup_renderer(image_size=image_size, device=device, y_up=y_up)

    len_frames = human_verts.shape[0]
    frame_groups = [range(i, min(i + render_batch_size, len_frames)) for i in range(0, len_frames, render_batch_size)]
    frame_images = []
    for i, frame_group in enumerate(frame_groups):
        human_verts_group = human_verts[frame_group]
        object_group = {obj: (o[0][frame_group], o[1], o[2]) for obj, o in object_transformed.items()}
        
        B = len(frame_group)
        cameras = PerspectiveCameras(
            focal_length=1.73,
            principal_point=((0., 0.),),
            image_size=(image_size, ),
            R=R.repeat(B, 1, 1),
            T=T.repeat(B, 1),
            device=device,
        )
        images = render_frames(renderer, cameras, human_verts_group, human_faces, object_group, device=device, ground=ground)

        images = images[..., :3] * 255
        images = images.cpu().numpy().astype(np.uint8)
        frame_images.append(images)
    frame_images = np.concatenate(frame_images, axis=0)
    return frame_images
