import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Union

import torch
import torch.nn.functional as F
try:
    import pytorch3d.transforms as transforms
    PYTORCH3D_AVAILABLE = True
except ImportError:
    transforms = None
    PYTORCH3D_AVAILABLE = False

def quaternion_to_matrix_numpy(quaternion: np.ndarray) -> np.ndarray:
    """
    Convert quaternion(s) to rotation matrix with optional translation using NumPy.
    
    This function handles both quaternion-only and quaternion+translation inputs,
    automatically detecting the input format and returning the appropriate matrix type.
    
    Args:
        quaternion: NumPy array of shape [..., 4] (quaternion only) or [..., 7] (quaternion + translation).
                   Quaternion should be in [w, x, y, z] format (scalar-first convention).
    
    Returns:
        NumPy array of shape [..., 3, 3] (rotation only) or [..., 4, 4] (rotation + translation).
        The output shape depends on the input shape:
        - Input [..., 4] -> Output [..., 3, 3] (rotation matrix)
        - Input [..., 7] -> Output [..., 4, 4] (transformation matrix)
    
    Raises:
        ValueError: If input shape is not [..., 4] or [..., 7].
        TypeError: If input is not a NumPy array.
    
    Example:
        >>> import numpy as np
        >>> quat = np.array([1, 0, 0, 0])  # Identity quaternion
        >>> rot_mat = quaternion_to_matrix_numpy(quat)
        >>> print(rot_mat.shape)  # (3, 3)
        
        >>> quat_trans = np.array([1, 0, 0, 0, 1, 2, 3])  # Quaternion + translation
        >>> transform_mat = quaternion_to_matrix_numpy(quat_trans)
        >>> print(transform_mat.shape)  # (4, 4)
    """
    # Input validation
    if not isinstance(quaternion, np.ndarray):
        raise TypeError(f"Input must be a NumPy array, got {type(quaternion)}")
    
    if quaternion.ndim < 1:
        raise ValueError("Input must have at least 1 dimension")
    
    batch_size = quaternion.shape[:-1]
    last_dim = quaternion.shape[-1]
    
    if last_dim == 4:
        # Only quaternion, return 3x3 rotation matrices
        quat_flat = quaternion.reshape(-1, 4)
        rotation = R.from_quat(quat=quat_flat, scalar_first=True)
        rot_mats = rotation.as_matrix().reshape(*batch_size, 3, 3)
        return rot_mats
    
    elif last_dim == 7:
        # Quaternion + translation, return 4x4 transformation matrices
        quat = quaternion[..., :4].reshape(-1, 4)
        translation = quaternion[..., 4:7].reshape(-1, 3)
        
        rotation = R.from_quat(quat=quat, scalar_first=True)
        rotation_matrices = rotation.as_matrix()
        
        # Create 4x4 transformation matrices
        transform_matrices = np.zeros((rotation_matrices.shape[0], 4, 4), dtype=rotation_matrices.dtype)
        transform_matrices[:, :3, :3] = rotation_matrices
        transform_matrices[:, :3, 3] = translation
        transform_matrices[:, 3, 3] = 1.0
        
        return transform_matrices.reshape(*batch_size, 4, 4)
    
    else:
        raise ValueError(f"Input must have shape [..., 4] for quaternion only or [..., 7] for quaternion + translation, got [..., {last_dim}]")

def matrix_to_quaternion_numpy(matrix: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix with optional translation to quaternion(s) using NumPy.
    
    This function handles both rotation-only and transformation matrices,
    automatically detecting the input format and returning the appropriate quaternion type.
    
    Args:
        matrix: NumPy array of shape [..., 3, 3] (rotation only) or [..., 4, 4] (rotation + translation).
               3x3 matrices are treated as pure rotation, 4x4 matrices as transformation.
    
    Returns:
        NumPy array of shape [..., 4] (quaternion only) or [..., 7] (quaternion + translation).
        Quaternion is returned in [w, x, y, z] format (scalar-first convention).
        The output shape depends on the input shape:
        - Input [..., 3, 3] -> Output [..., 4] (quaternion only)
        - Input [..., 4, 4] -> Output [..., 7] (quaternion + translation)
    
    Raises:
        ValueError: If input shape is not [..., 3, 3] or [..., 4, 4].
        TypeError: If input is not a NumPy array.
    
    Example:
        >>> import numpy as np
        >>> rot_mat = np.eye(3)  # Identity rotation matrix
        >>> quat = matrix_to_quaternion_numpy(rot_mat)
        >>> print(quat.shape)  # (4,)
        
        >>> transform_mat = np.eye(4)  # Identity transformation matrix
        >>> quat_trans = matrix_to_quaternion_numpy(transform_mat)
        >>> print(quat_trans.shape)  # (7,)
    """
    # Input validation
    if not isinstance(matrix, np.ndarray):
        raise TypeError(f"Input must be a NumPy array, got {type(matrix)}")
    
    if matrix.ndim < 2:
        raise ValueError("Input must have at least 2 dimensions")
    
    batch_size = matrix.shape[:-2]
    last_two_dims = matrix.shape[-2:]
    
    if last_two_dims == (3, 3):
        # Only rotation matrix, return quaternion only
        mats_flat = matrix.reshape(-1, 3, 3)
        rotation = R.from_matrix(mats_flat)
        quat = rotation.as_quat(scalar_first=True)  # Returns [w, x, y, z]
        return quat.reshape(*batch_size, 4)
    
    elif last_two_dims == (4, 4):
        # Transformation matrix, return quaternion + translation
        rotation_matrices = matrix[..., :3, :3].reshape(-1, 3, 3)
        translation = matrix[..., :3, 3].reshape(-1, 3)
        
        rotation = R.from_matrix(rotation_matrices)
        quat = rotation.as_quat(scalar_first=True)  # Returns [w, x, y, z]
        
        # Combine quaternion and translation
        result = np.zeros((rotation_matrices.shape[0], 7), dtype=quat.dtype)
        result[:, :4] = quat
        result[:, 4:7] = translation
        
        return result.reshape(*batch_size, 7)
    
    else:
        raise ValueError(f"Input must have shape [..., 3, 3] for rotation only or [..., 4, 4] for rotation + translation, got [..., {last_two_dims[0]}, {last_two_dims[1]}]")

def quaternion_to_matrix_torch(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion(s) to rotation matrix with optional translation using PyTorch3D.
    
    This function preserves gradients for backpropagation and handles both quaternion-only
    and quaternion+translation inputs, automatically detecting the input format.
    
    Args:
        quaternion: PyTorch tensor of shape [..., 4] (quaternion only) or [..., 7] (quaternion + translation).
                   Quaternion should be in [w, x, y, z] format (scalar-first convention).
    
    Returns:
        PyTorch tensor of shape [..., 3, 3] (rotation only) or [..., 4, 4] (rotation + translation).
        The output shape depends on the input shape:
        - Input [..., 4] -> Output [..., 3, 3] (rotation matrix)
        - Input [..., 7] -> Output [..., 4, 4] (transformation matrix)
    
    Raises:
        ImportError: If PyTorch or PyTorch3D are not available.
        ValueError: If input shape is not [..., 4] or [..., 7].
        TypeError: If input is not a PyTorch tensor.
    
    Example:
        >>> import torch
        >>> quat = torch.tensor([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        >>> rot_mat = quaternion_to_matrix_torch(quat)
        >>> print(rot_mat.shape)  # torch.Size([3, 3])
        
        >>> quat_trans = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0])  # Quaternion + translation
        >>> transform_mat = quaternion_to_matrix_torch(quat_trans)
        >>> print(transform_mat.shape)  # torch.Size([4, 4])
    """
    if not PYTORCH3D_AVAILABLE:
        raise ImportError("PyTorch3D are required for quaternion_to_matrix_torch")
    
    # Input validation
    if not isinstance(quaternion, torch.Tensor):
        raise TypeError(f"Input must be a PyTorch tensor, got {type(quaternion)}")
    
    if quaternion.ndim < 1:
        raise ValueError("Input must have at least 1 dimension")
    
    batch_shape = quaternion.shape[:-1]
    device = quaternion.device
    dtype = quaternion.dtype
    last_dim = quaternion.shape[-1]
    
    if last_dim == 4:
        # Only quaternion, return 3x3 rotation matrices
        # pytorch3d expects [w, x, y, z] format
        rotation_matrix = transforms.quaternion_to_matrix(quaternion)
        return rotation_matrix
    
    elif last_dim == 7:
        # Quaternion + translation, return 4x4 transformation matrices
        quat = quaternion[..., :4]
        translation = quaternion[..., 4:7]
        
        # Get rotation matrix using pytorch3d
        rotation_matrix = transforms.quaternion_to_matrix(quat)
        
        # Create 4x4 transformation matrices
        transform_matrix = torch.zeros(*batch_shape, 4, 4, device=device, dtype=dtype)
        transform_matrix[..., :3, :3] = rotation_matrix
        transform_matrix[..., :3, 3] = translation
        transform_matrix[..., 3, 3] = 1.0
        
        return transform_matrix
    
    else:
        raise ValueError(f"Input must have shape [..., 4] for quaternion only or [..., 7] for quaternion + translation, got [..., {last_dim}]")

def matrix_to_quaternion_torch(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix with optional translation to quaternion(s) using PyTorch3D.
    
    This function preserves gradients for backpropagation and handles both rotation-only
    and transformation matrices, automatically detecting the input format.
    
    Args:
        matrix: PyTorch tensor of shape [..., 3, 3] (rotation only) or [..., 4, 4] (rotation + translation).
               3x3 matrices are treated as pure rotation, 4x4 matrices as transformation.
    
    Returns:
        PyTorch tensor of shape [..., 4] (quaternion only) or [..., 7] (quaternion + translation).
        Quaternion is returned in [w, x, y, z] format (scalar-first convention).
        The output shape depends on the input shape:
        - Input [..., 3, 3] -> Output [..., 4] (quaternion only)
        - Input [..., 4, 4] -> Output [..., 7] (quaternion + translation)
    
    Raises:
        ImportError: If PyTorch or PyTorch3D are not available.
        ValueError: If input shape is not [..., 3, 3] or [..., 4, 4].
        TypeError: If input is not a PyTorch tensor.
    
    Example:
        >>> import torch
        >>> rot_mat = torch.eye(3)  # Identity rotation matrix
        >>> quat = matrix_to_quaternion_torch(rot_mat)
        >>> print(quat.shape)  # torch.Size([4])
        
        >>> transform_mat = torch.eye(4)  # Identity transformation matrix
        >>> quat_trans = matrix_to_quaternion_torch(transform_mat)
        >>> print(quat_trans.shape)  # torch.Size([7])
    """
    if not PYTORCH3D_AVAILABLE:
        raise ImportError("PyTorch and PyTorch3D are required for matrix_to_quaternion_torch")
    
    # Input validation
    if not isinstance(matrix, torch.Tensor):
        raise TypeError(f"Input must be a PyTorch tensor, got {type(matrix)}")
    
    if matrix.ndim < 2:
        raise ValueError("Input must have at least 2 dimensions")
    
    device = matrix.device
    dtype = matrix.dtype
    last_two_dims = matrix.shape[-2:]
    
    if last_two_dims == (3, 3):
        # Only rotation matrix, return quaternion only
        rotation_matrix = matrix
        
        # Use pytorch3d for matrix to quaternion conversion
        quat = transforms.matrix_to_quaternion(rotation_matrix)
        return quat
    
    elif last_two_dims == (4, 4):
        # Transformation matrix, return quaternion + translation
        rotation_matrix = matrix[..., :3, :3]
        translation = matrix[..., :3, 3]
        
        # Get quaternion from rotation part using pytorch3d
        quat = transforms.matrix_to_quaternion(rotation_matrix)
        
        # Combine quaternion and translation
        result = torch.zeros(*matrix.shape[:-2], 7, device=device, dtype=dtype)
        result[..., :4] = quat
        result[..., 4:7] = translation
        
        return result
    
    else:
        raise ValueError(f"Input must have shape [..., 3, 3] for rotation only or [..., 4, 4] for rotation + translation, got [..., {last_two_dims[0]}, {last_two_dims[1]}]")

def quaternion_to_matrix(quaternion: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert quaternion(s) to rotation matrix with optional translation.
    
    This is a convenience function that automatically chooses the appropriate implementation
    (NumPy or PyTorch) based on the input type. It handles both quaternion-only and
    quaternion+translation inputs.
    
    Args:
        quaternion: NumPy array or PyTorch tensor of shape [..., 4] or [..., 7].
                   Quaternion should be in [w, x, y, z] format (scalar-first convention).
    
    Returns:
        NumPy array or PyTorch tensor of shape [..., 3, 3] or [..., 4, 4].
        The output type matches the input type, and the shape depends on the input:
        - Input [..., 4] -> Output [..., 3, 3] (rotation matrix)
        - Input [..., 7] -> Output [..., 4, 4] (transformation matrix)
    
    Raises:
        NotImplementedError: If input type is not supported or required dependencies are missing.
        ValueError: If input shape is not [..., 4] or [..., 7].
        TypeError: If input is not a NumPy array or PyTorch tensor.
    
    Example:
        >>> import numpy as np
        >>> import torch
        >>> 
        >>> # NumPy example
        >>> quat_np = np.array([1, 0, 0, 0])
        >>> rot_mat_np = quaternion_to_matrix(quat_np)
        >>> print(type(rot_mat_np), rot_mat_np.shape)  # <class 'numpy.ndarray'> (3, 3)
        >>> 
        >>> # PyTorch example
        >>> quat_torch = torch.tensor([1.0, 0.0, 0.0, 0.0])
        >>> rot_mat_torch = quaternion_to_matrix(quat_torch)
        >>> print(type(rot_mat_torch), rot_mat_torch.shape)  # <class 'torch.Tensor'> torch.Size([3, 3])
    """
    if isinstance(quaternion, torch.Tensor):
        return quaternion_to_matrix_torch(quaternion)
    elif isinstance(quaternion, np.ndarray):
        return quaternion_to_matrix_numpy(quaternion)
    else:
        raise NotImplementedError(f"Input must be a torch tensor or numpy array, got {type(quaternion)}. PyTorch3D is required for PyTorch tensors.")

def matrix_to_quaternion(matrix: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert rotation matrix with optional translation to quaternion(s).
    
    This is a convenience function that automatically chooses the appropriate implementation
    (NumPy or PyTorch) based on the input type. It handles both rotation-only and
    transformation matrices.
    
    Args:
        matrix: NumPy array or PyTorch tensor of shape [..., 3, 3] or [..., 4, 4].
               3x3 matrices are treated as pure rotation, 4x4 matrices as transformation.
    
    Returns:
        NumPy array or PyTorch tensor of shape [..., 4] or [..., 7].
        Quaternion is returned in [w, x, y, z] format (scalar-first convention).
        The output type matches the input type, and the shape depends on the input:
        - Input [..., 3, 3] -> Output [..., 4] (quaternion only)
        - Input [..., 4, 4] -> Output [..., 7] (quaternion + translation)
    
    Raises:
        NotImplementedError: If input type is not supported or required dependencies are missing.
        ValueError: If input shape is not [..., 3, 3] or [..., 4, 4].
        TypeError: If input is not a NumPy array or PyTorch tensor.
    
    Example:
        >>> import numpy as np
        >>> import torch
        >>> 
        >>> # NumPy example
        >>> rot_mat_np = np.eye(3)
        >>> quat_np = matrix_to_quaternion(rot_mat_np)
        >>> print(type(quat_np), quat_np.shape)  # <class 'numpy.ndarray'> (4,)
        >>> 
        >>> # PyTorch example
        >>> rot_mat_torch = torch.eye(3)
        >>> quat_torch = matrix_to_quaternion(rot_mat_torch)
        >>> print(type(quat_torch), quat_torch.shape)  # <class 'torch.Tensor'> torch.Size([4])
    """
    if isinstance(matrix, torch.Tensor):
        return matrix_to_quaternion_torch(matrix)
    elif isinstance(matrix, np.ndarray):
        return matrix_to_quaternion_numpy(matrix)
    else:
        raise NotImplementedError(f"Input must be a torch tensor or numpy array, got {type(matrix)}. PyTorch3D is required for PyTorch tensors.")
    
def normalize_rot6d_torch(rot: torch.Tensor) -> torch.Tensor:
    """
    Normalize 6D rotation representation using PyTorch.
    
    This function normalizes the two 3D vectors in the 6D rotation representation
    to be orthonormal, ensuring they form a valid rotation matrix basis.
    
    Args:
        rot: PyTorch tensor of shape [..., 6] or [..., 2, 3] representing 6D rotation.
    
    Returns:
        PyTorch tensor of the same shape with normalized 6D rotation representation.
    
    Raises:
        ValueError: If input shape is not compatible with 6D rotation representation.
        TypeError: If input is not a PyTorch tensor.
    
    Note:
        The 6D rotation representation uses two 3D vectors that should be orthonormal.
        This function ensures they are properly normalized and orthogonal.
    """
    if not isinstance(rot, torch.Tensor):
        raise TypeError(f"Input must be a PyTorch tensor, got {type(rot)}")
    
    if rot.shape[-1] == 3:
        unflatten = True
        rot = rot.flatten(-2, -1)
    else:
        unflatten = False
    
    if rot.shape[-1] != 6:
        raise ValueError(f"Input must have shape [..., 6] or [..., 2, 3], got [..., {rot.shape[-1]}]")
    
    a1, a2 = rot[..., :3], rot[..., 3:]
    b1 = F.normalize(a1, p=2, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, p=2, dim=-1)
    rot = torch.cat([b1, b2], dim=-1)  # back to [..., 6]
    if unflatten:
        rot = rot.unflatten(-1, (2, 3))
    return rot

def normalize_np(x: np.ndarray) -> np.ndarray:
    """
    Normalize vectors along the last dimension using NumPy.
    
    This is a helper function that normalizes vectors to unit length with numerical
    stability by clipping the norm to avoid division by zero.
    
    Args:
        x: NumPy array of vectors to normalize, shape [..., N] where N is the vector dimension.
    
    Returns:
        NumPy array of the same shape with normalized vectors.
    
    Raises:
        TypeError: If input is not a NumPy array.
        ValueError: If input has no dimensions.
    
    Note:
        The minimum norm is clipped to 1e-8 to avoid division by zero.
    """
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Input must be a NumPy array, got {type(x)}")
    
    if x.ndim == 0:
        raise ValueError("Input must have at least 1 dimension")
    
    x_n = np.linalg.norm(x, axis=-1, keepdims=True)
    x_n = x_n.clip(min=1e-8)
    x = x / x_n
    return x

def normalize_rot6d_numpy(rot: np.ndarray) -> np.ndarray:
    """
    Normalize 6D rotation representation using NumPy.
    
    This function normalizes the two 3D vectors in the 6D rotation representation
    to be orthonormal, ensuring they form a valid rotation matrix basis.
    
    Args:
        rot: NumPy array of shape [..., 6] or [..., 2, 3] representing 6D rotation.
    
    Returns:
        NumPy array of the same shape with normalized 6D rotation representation.
    
    Raises:
        ValueError: If input shape is not compatible with 6D rotation representation.
        TypeError: If input is not a NumPy array.
    
    Note:
        The 6D rotation representation uses two 3D vectors that should be orthonormal.
        This function ensures they are properly normalized and orthogonal.
    """
    if not isinstance(rot, np.ndarray):
        raise TypeError(f"Input must be a NumPy array, got {type(rot)}")
    
    if rot.shape[-1] == 3:
        unflatten = True
        undim = True
        ori_shape = rot.shape[:-2]
        p = np.prod(ori_shape)
        rot = rot.reshape(p, 6)
    elif len(rot.shape) > 2:
        unflatten = False
        undim = True
        ori_shape = rot.shape[:-1]
        p = np.prod(ori_shape)
        rot = rot.reshape(p, 6)
    else:
        unflatten = False
        undim = False
        ori_shape = None
    
    if rot.shape[-1] != 6:
        raise ValueError(f"Input must have shape [..., 6] or [..., 2, 3], got [..., {rot.shape[-1]}]")
    
    a1, a2 = rot[:, :3], rot[:, 3:]
    b1 = normalize_np(a1)
    b2 = a2 - (b1 * a2).sum(axis=-1, keepdims=True) * b1
    b2 = normalize_np(b2)
    rot = np.concatenate([b1, b2], axis=-1)  # back to [..., 6]
    if unflatten:
        rot = rot.reshape(ori_shape + (2, 3))
    elif undim:
        rot = rot.reshape(ori_shape + (6, ))
    return rot

def normalize_rot6d(rot: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Normalize 6D rotation representation.
    
    This is a convenience function that automatically chooses the appropriate implementation
    (NumPy or PyTorch) based on the input type.
    
    Args:
        rot: NumPy array or PyTorch tensor of shape [..., 6] or [..., 2, 3] representing 6D rotation.
    
    Returns:
        NumPy array or PyTorch tensor of the same shape with normalized 6D rotation representation.
        The output type matches the input type.
    
    Raises:
        NotImplementedError: If input type is not supported.
        ValueError: If input shape is not compatible with 6D rotation representation.
        TypeError: If input is not a NumPy array or PyTorch tensor.
    
    Example:
        >>> import numpy as np
        >>> import torch
        >>> 
        >>> # NumPy example
        >>> rot_np = np.random.randn(3, 6)
        >>> normalized_np = normalize_rot6d(rot_np)
        >>> print(type(normalized_np), normalized_np.shape)  # <class 'numpy.ndarray'> (3, 6)
        >>> 
        >>> # PyTorch example
        >>> rot_torch = torch.randn(3, 6)
        >>> normalized_torch = normalize_rot6d(rot_torch)
        >>> print(type(normalized_torch), normalized_torch.shape)  # <class 'torch.Tensor'> torch.Size([3, 6])
    """
    if isinstance(rot, torch.Tensor):
        return normalize_rot6d_torch(rot)
    elif isinstance(rot, np.ndarray):
        return normalize_rot6d_numpy(rot)
    else:
        raise NotImplementedError(f"Input must be a torch tensor or numpy array, got {type(rot)}")
    
def normalize_vector(v: torch.Tensor) -> torch.Tensor:
    """
    Normalize vectors along the last dimension for arbitrary leading shapes using PyTorch.
    
    This function normalizes vectors to unit length with numerical stability by
    clipping the norm to avoid division by zero.
    
    Args:
        v: PyTorch tensor of vectors to normalize, shape [..., N] where N is the vector dimension.
    
    Returns:
        PyTorch tensor of the same shape with normalized vectors.
    
    Raises:
        TypeError: If input is not a PyTorch tensor.
        ValueError: If input has no dimensions.
    
    Note:
        The minimum norm is clipped to 1e-8 to avoid division by zero.
    """
    if not isinstance(v, torch.Tensor):
        raise TypeError(f"Input must be a PyTorch tensor, got {type(v)}")
    
    if v.ndim == 0:
        raise ValueError("Input must have at least 1 dimension")
    
    v_mag = torch.linalg.norm(v, dim=-1, keepdim=True)
    v_mag = torch.clamp(v_mag, min=1e-8)
    return v / v_mag

def cross_product(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Cross product along the last dimension for arbitrary leading shapes using PyTorch.
    
    This function computes the cross product of two 3D vectors, supporting arbitrary
    leading dimensions for batch processing.
    
    Args:
        u: PyTorch tensor of shape [..., 3] representing the first vector.
        v: PyTorch tensor of shape [..., 3] representing the second vector.
    
    Returns:
        PyTorch tensor of shape [..., 3] representing the cross product u × v.
    
    Raises:
        TypeError: If inputs are not PyTorch tensors.
        ValueError: If inputs don't have shape [..., 3] or have incompatible shapes.
    
    Example:
        >>> import torch
        >>> u = torch.tensor([1.0, 0.0, 0.0])
        >>> v = torch.tensor([0.0, 1.0, 0.0])
        >>> result = cross_product(u, v)
        >>> print(result)  # tensor([0., 0., 1.])
    """
    if not isinstance(u, torch.Tensor) or not isinstance(v, torch.Tensor):
        raise TypeError("Both inputs must be PyTorch tensors")
    
    if u.shape[-1] != 3 or v.shape[-1] != 3:
        raise ValueError("Both inputs must have shape [..., 3]")
    
    if u.shape != v.shape:
        raise ValueError("Input tensors must have the same shape")
    
    return torch.cross(u, v, dim=-1)

def robust_compute_rotation_matrix_from_ortho6d_torch(poses: torch.Tensor) -> torch.Tensor:
    """
    Compute rotation matrices from 6D orthonormal representation using PyTorch.
    
    This function uses a robust method to compute rotation matrices from 6D vectors
    by creating an orthonormal basis that takes into account both predicted directions
    equally, rather than simply making the second vector orthogonal to the first.
    
    Args:
        poses: PyTorch tensor of shape [..., 6] representing 6D rotation vectors.
              The first 3 elements represent the first direction, the last 3 the second.
    
    Returns:
        PyTorch tensor of shape [..., 3, 3] representing rotation matrices.
    
    Raises:
        TypeError: If input is not a PyTorch tensor.
        ValueError: If input doesn't have shape [..., 6].
    
    Note:
        This method is more robust than simple Gram-Schmidt orthogonalization
        as it considers both input directions equally when constructing the basis.
    """
    if not isinstance(poses, torch.Tensor):
        raise TypeError(f"Input must be a PyTorch tensor, got {type(poses)}")
    
    if poses.shape[-1] != 6:
        raise ValueError(f"Input must have shape [..., 6], got [..., {poses.shape[-1]}]")
    
    x_raw = poses[..., 0:3]
    y_raw = poses[..., 3:6]

    x = normalize_vector(x_raw)
    y = normalize_vector(y_raw)
    middle = normalize_vector(x + y)
    orthmid = normalize_vector(x - y)
    x = normalize_vector(middle + orthmid)
    y = normalize_vector(middle - orthmid)
    # Their scalar product should be small !
    # assert torch.einsum("ij,ij->i", [x, y]).abs().max() < 0.00001
    z = normalize_vector(cross_product(x, y))

    x = x.unsqueeze(-1)
    y = y.unsqueeze(-1)
    z = z.unsqueeze(-1)
    matrix = torch.cat((x, y, z), dim=-1)
    # Check for reflection in matrix ! If found, flip last vector TODO
    # assert (torch.stack([torch.det(mat) for mat in matrix ])< 0).sum() == 0
    return matrix


def normalize_vector_numpy(v: np.ndarray) -> np.ndarray:
    """
    Normalize vectors along the last dimension using NumPy with numerical stability.
    
    This function normalizes vectors to unit length with numerical stability by
    clipping the norm to avoid division by zero.
    
    Args:
        v: NumPy array of vectors to normalize, shape [..., N] where N is the vector dimension.
    
    Returns:
        NumPy array of the same shape with normalized vectors.
    
    Raises:
        TypeError: If input is not a NumPy array.
        ValueError: If input has no dimensions.
    
    Note:
        The minimum norm is clipped to 1e-8 to avoid division by zero.
    """
    if not isinstance(v, np.ndarray):
        raise TypeError(f"Input must be a NumPy array, got {type(v)}")
    
    if v.ndim == 0:
        raise ValueError("Input must have at least 1 dimension")
    
    v_mag = np.linalg.norm(v, axis=-1, keepdims=True)
    v_mag = np.clip(v_mag, a_min=1e-8, a_max=None)
    return v / v_mag


def robust_compute_rotation_matrix_from_ortho6d_numpy(poses: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrices from 6D orthonormal representation using NumPy.
    
    This function uses a robust method to compute rotation matrices from 6D vectors
    by creating an orthonormal basis that takes into account both predicted directions
    equally, rather than simply making the second vector orthogonal to the first.
    This mirrors the PyTorch implementation.
    
    Args:
        poses: NumPy array of shape [..., 6] representing 6D rotation vectors.
              The first 3 elements represent the first direction, the last 3 the second.
    
    Returns:
        NumPy array of shape [..., 3, 3] representing rotation matrices.
    
    Raises:
        TypeError: If input is not a NumPy array.
        ValueError: If input doesn't have shape [..., 6].
    
    Note:
        This method is more robust than simple Gram-Schmidt orthogonalization
        as it considers both input directions equally when constructing the basis.
    """
    if not isinstance(poses, np.ndarray):
        raise TypeError(f"Input must be a NumPy array, got {type(poses)}")
    
    if poses.shape[-1] != 6:
        raise ValueError(f"Input must have shape [..., 6], got [..., {poses.shape[-1]}]")
    
    x_raw = poses[..., 0:3]
    y_raw = poses[..., 3:6]

    x = normalize_vector_numpy(x_raw)
    y = normalize_vector_numpy(y_raw)
    middle = normalize_vector_numpy(x + y)
    orthmid = normalize_vector_numpy(x - y)
    x = normalize_vector_numpy(middle + orthmid)
    y = normalize_vector_numpy(middle - orthmid)
    z = normalize_vector_numpy(np.cross(x, y))

    x = np.expand_dims(x, axis=-1)
    y = np.expand_dims(y, axis=-1)
    z = np.expand_dims(z, axis=-1)
    matrix = np.concatenate((x, y, z), axis=-1)
    return matrix


def rotation6d_to_matrix(poses: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute rotation matrices from 6D orthonormal representation.
    
    This is a convenience function that automatically chooses the appropriate implementation
    (NumPy or PyTorch) based on the input type. It handles both 6D rotation-only and
    6D+translation inputs.
    
    Args:
        poses: NumPy array or PyTorch tensor of shape [..., 6] or [..., 9].
              - [..., 6]: 6D rotation representation
              - [..., 9]: 6D rotation + 3D translation
    
    Returns:
        NumPy array or PyTorch tensor of shape [..., 3, 3] or [..., 4, 4].
        The output type matches the input type, and the shape depends on the input:
        - Input [..., 6] -> Output [..., 3, 3] (rotation matrix)
        - Input [..., 9] -> Output [..., 4, 4] (transformation matrix)
    
    Raises:
        NotImplementedError: If input type is not supported.
        ValueError: If input shape is not [..., 6] or [..., 9].
        TypeError: If input is not a NumPy array or PyTorch tensor.
    
    Example:
        >>> import numpy as np
        >>> import torch
        >>> 
        >>> # NumPy example - 6D rotation
        >>> poses_np = np.random.randn(3, 6)
        >>> rot_mat_np = rotation6d_to_matrix(poses_np)
        >>> print(type(rot_mat_np), rot_mat_np.shape)  # <class 'numpy.ndarray'> (3, 3, 3)
        >>> 
        >>> # PyTorch example - 6D + translation
        >>> poses_torch = torch.randn(3, 9)
        >>> transform_mat_torch = rotation6d_to_matrix(poses_torch)
        >>> print(type(transform_mat_torch), transform_mat_torch.shape)  # <class 'torch.Tensor'> torch.Size([3, 4, 4])
    """
    if isinstance(poses, torch.Tensor):
        if poses.shape[-1] == 6:
            return robust_compute_rotation_matrix_from_ortho6d_torch(poses)
        elif poses.shape[-1] == 9:
            Rm = robust_compute_rotation_matrix_from_ortho6d_torch(poses[..., :6])
            t = poses[..., 6:]
            # Build [...,4,4]
            eye_pad = torch.zeros((*Rm.shape[:-2], 4, 4), device=Rm.device, dtype=Rm.dtype)
            eye_pad[..., 3, 3] = 1.0
            eye_pad[..., :3, :3] = Rm
            eye_pad[..., :3, 3] = t
            return eye_pad
        else:
            raise ValueError(f"Input must have shape [..., 6] or [..., 9] (6D + translation), got [..., {poses.shape[-1]}]")
    elif isinstance(poses, np.ndarray):
        if poses.shape[-1] == 6:
            return robust_compute_rotation_matrix_from_ortho6d_numpy(poses)
        elif poses.shape[-1] == 9:
            Rm = robust_compute_rotation_matrix_from_ortho6d_numpy(poses[..., :6])
            t = poses[..., 6:]
            leading = Rm.shape[:-2]
            T = np.zeros((*leading, 4, 4), dtype=Rm.dtype)
            T[..., 3, 3] = 1.0
            T[..., :3, :3] = Rm
            T[..., :3, 3] = t
            return T
        else:
            raise ValueError(f"Input must have shape [..., 6] or [..., 9] (6D + translation), got [..., {poses.shape[-1]}]")
    else:
        raise NotImplementedError(f"Input must be a torch tensor or numpy array, got {type(poses)}")

def matrix_to_rotation6d_torch(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotation matrix to 6D rotation representation using PyTorch.
    
    This function extracts the first two rows of a rotation matrix to create
    a 6D rotation representation.
    
    Args:
        matrix: PyTorch tensor of shape [..., 3, 3] or [..., 4, 4] representing rotation/transformation matrices.
    
    Returns:
        PyTorch tensor of shape [..., 6] or [..., 9] representing 6D rotation or 6D+translation.
        The output shape depends on the input shape:
        - Input [..., 3, 3] -> Output [..., 6] (6D rotation)
        - Input [..., 4, 4] -> Output [..., 9] (6D rotation + 3D translation)
    
    Raises:
        TypeError: If input is not a PyTorch tensor.
        ValueError: If input doesn't have shape [..., 3, 3] or [..., 4, 4].
    
    Note:
        The 6D representation uses the first two rows of the rotation matrix.
        This is a lossy conversion as the third row can be computed from the first two.
    """
    if not isinstance(matrix, torch.Tensor):
        raise TypeError(f"Input must be a PyTorch tensor, got {type(matrix)}")
    
    if matrix.shape[-2:] == (3, 3):
        return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)
    elif matrix.shape[-2:] == (4, 4):
        rot_6d = matrix[..., :2, :3].clone().reshape(*matrix.size()[:-2], 6)
        translation = matrix[..., :3, 3]
        return torch.cat([rot_6d, translation], dim=-1)
    else:
        raise ValueError(f"Input must have shape [..., 3, 3] or [..., 4, 4], got [..., {matrix.shape[-2]}, {matrix.shape[-1]}]")

def matrix_to_rotation6d_numpy(matrix: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to 6D rotation representation using NumPy.
    
    This function extracts the first two rows of a rotation matrix to create
    a 6D rotation representation.
    
    Args:
        matrix: NumPy array of shape [..., 3, 3] or [..., 4, 4] representing rotation/transformation matrices.
    
    Returns:
        NumPy array of shape [..., 6] or [..., 9] representing 6D rotation or 6D+translation.
        The output shape depends on the input shape:
        - Input [..., 3, 3] -> Output [..., 6] (6D rotation)
        - Input [..., 4, 4] -> Output [..., 9] (6D rotation + 3D translation)
    
    Raises:
        TypeError: If input is not a NumPy array.
        ValueError: If input doesn't have shape [..., 3, 3] or [..., 4, 4].
    
    Note:
        The 6D representation uses the first two rows of the rotation matrix.
        This is a lossy conversion as the third row can be computed from the first two.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError(f"Input must be a NumPy array, got {type(matrix)}")
    
    if matrix.shape[-2:] == (3, 3):
        return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)
    elif matrix.shape[-2:] == (4, 4):
        rot_6d = matrix[..., :2, :3].reshape(*matrix.shape[:-2], 6)
        translation = matrix[..., :3, 3]
        return np.concatenate([rot_6d, translation], axis=-1)
    else:
        raise ValueError(f"Input must have shape [..., 3, 3] or [..., 4, 4], got [..., {matrix.shape[-2]}, {matrix.shape[-1]}]")

def matrix_to_rotation6d(matrix: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert rotation matrix to 6D rotation representation.
    
    This is a convenience function that automatically chooses the appropriate implementation
    (NumPy or PyTorch) based on the input type. It handles both rotation-only and
    transformation matrices.
    
    Args:
        matrix: NumPy array or PyTorch tensor of shape [..., 3, 3] or [..., 4, 4].
               3x3 matrices are treated as pure rotation, 4x4 matrices as transformation.
    
    Returns:
        NumPy array or PyTorch tensor of shape [..., 6] or [..., 9].
        The output type matches the input type, and the shape depends on the input:
        - Input [..., 3, 3] -> Output [..., 6] (6D rotation)
        - Input [..., 4, 4] -> Output [..., 9] (6D rotation + 3D translation)
    
    Raises:
        NotImplementedError: If input type is not supported.
        ValueError: If input shape is not [..., 3, 3] or [..., 4, 4].
        TypeError: If input is not a NumPy array or PyTorch tensor.
    
    Example:
        >>> import numpy as np
        >>> import torch
        >>> 
        >>> # NumPy example - 3x3 rotation matrix
        >>> rot_mat_np = np.eye(3)
        >>> rot6d_np = matrix_to_rotation6d(rot_mat_np)
        >>> print(type(rot6d_np), rot6d_np.shape)  # <class 'numpy.ndarray'> (6,)
        >>> 
        >>> # PyTorch example - 4x4 transformation matrix
        >>> transform_mat_torch = torch.eye(4)
        >>> rot6d_trans_torch = matrix_to_rotation6d(transform_mat_torch)
        >>> print(type(rot6d_trans_torch), rot6d_trans_torch.shape)  # <class 'torch.Tensor'> torch.Size([9])
    """
    if isinstance(matrix, torch.Tensor):
        if matrix.shape[-2:] == (3, 3):
            return matrix_to_rotation6d_torch(matrix)
        elif matrix.shape[-2:] == (4, 4):
            rot6 = matrix_to_rotation6d_torch(matrix[..., :3, :3])
            t = matrix[..., :3, 3]
            return torch.cat([rot6, t], dim=-1)
        else:
            raise ValueError(f"Matrix must have shape [..., 3, 3] or [..., 4, 4], got [..., {matrix.shape[-2]}, {matrix.shape[-1]}]")
    elif isinstance(matrix, np.ndarray):
        if matrix.shape[-2:] == (3, 3):
            return matrix_to_rotation6d_numpy(matrix)
        elif matrix.shape[-2:] == (4, 4):
            rot6 = matrix_to_rotation6d_numpy(matrix[..., :3, :3])
            t = matrix[..., :3, 3]
            return np.concatenate([rot6, t], axis=-1)
        else:
            raise ValueError(f"Matrix must have shape [..., 3, 3] or [..., 4, 4], got [..., {matrix.shape[-2]}, {matrix.shape[-1]}]")
    else:
        raise NotImplementedError(f"Input must be a torch tensor or numpy array, got {type(matrix)}")

def matrix3x3_to_matrix4x4(matrix: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert 3x3 rotation matrix to 4x4 transformation matrix.
    
    This function pads a 3x3 rotation matrix with zeros and a 1 in the bottom-right
    corner to create a 4x4 transformation matrix with no translation.
    
    Args:
        matrix: NumPy array or PyTorch tensor of shape [..., 3, 3] representing rotation matrices.
    
    Returns:
        NumPy array or PyTorch tensor of shape [..., 4, 4] representing transformation matrices.
        The output type matches the input type.
    
    Raises:
        NotImplementedError: If input type is not supported.
        ValueError: If input doesn't have shape [..., 3, 3].
        TypeError: If input is not a NumPy array or PyTorch tensor.
    
    Example:
        >>> import numpy as np
        >>> import torch
        >>> 
        >>> # NumPy example
        >>> rot_mat_np = np.eye(3)
        >>> transform_mat_np = matrix3x3_to_matrix4x4(rot_mat_np)
        >>> print(type(transform_mat_np), transform_mat_np.shape)  # <class 'numpy.ndarray'> (4, 4)
        >>> 
        >>> # PyTorch example
        >>> rot_mat_torch = torch.eye(3)
        >>> transform_mat_torch = matrix3x3_to_matrix4x4(rot_mat_torch)
        >>> print(type(transform_mat_torch), transform_mat_torch.shape)  # <class 'torch.Tensor'> torch.Size([4, 4])
    """
    if isinstance(matrix, torch.Tensor):
        if matrix.shape[-2:] != (3, 3):
            raise ValueError(f"Input must have shape [..., 3, 3], got [..., {matrix.shape[-2]}, {matrix.shape[-1]}]")
        T = torch.zeros((*matrix.shape[:-2], 4, 4), device=matrix.device, dtype=matrix.dtype)
        T[..., :3, :3] = matrix
        T[..., 3, 3] = 1.0
        return T
    elif isinstance(matrix, np.ndarray):
        if matrix.shape[-2:] != (3, 3):
            raise ValueError(f"Input must have shape [..., 3, 3], got [..., {matrix.shape[-2]}, {matrix.shape[-1]}]")
        T = np.zeros((*matrix.shape[:-2], 4, 4), dtype=matrix.dtype)
        T[..., :3, :3] = matrix
        T[..., 3, 3] = 1.0
        return T
    else:
        raise NotImplementedError(f"Input must be a torch tensor or numpy array, got {type(matrix)}")

def matrix4x4_to_matrix3x3(matrix: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert 4x4 transformation matrix to 3x3 rotation matrix.
    
    This function extracts the top-left 3x3 submatrix from a 4x4 transformation matrix,
    which represents the rotation component.
    
    Args:
        matrix: NumPy array or PyTorch tensor of shape [..., 4, 4] representing transformation matrices.
    
    Returns:
        NumPy array or PyTorch tensor of shape [..., 3, 3] representing rotation matrices.
        The output type matches the input type.
    
    Raises:
        ValueError: If input doesn't have shape [..., 4, 4].
        TypeError: If input is not a NumPy array or PyTorch tensor.
    
    Example:
        >>> import numpy as np
        >>> import torch
        >>> 
        >>> # NumPy example
        >>> transform_mat_np = np.eye(4)
        >>> rot_mat_np = matrix4x4_to_matrix3x3(transform_mat_np)
        >>> print(type(rot_mat_np), rot_mat_np.shape)  # <class 'numpy.ndarray'> (3, 3)
        >>> 
        >>> # PyTorch example
        >>> transform_mat_torch = torch.eye(4)
        >>> rot_mat_torch = matrix4x4_to_matrix3x3(transform_mat_torch)
        >>> print(type(rot_mat_torch), rot_mat_torch.shape)  # <class 'torch.Tensor'> torch.Size([3, 3])
    """
    if matrix.shape[-2:] != (4, 4):
        raise ValueError(f"Input must have shape [..., 4, 4], got [..., {matrix.shape[-2]}, {matrix.shape[-1]}]")
    
    return matrix[..., :3, :3]

