import torch
import numpy as np
from typing import Any, Dict, Union

def try_to_device(d: Any, device: Union[str, torch.device]) -> Any:
    """
    Safely move data to the specified device.
    
    This function attempts to move PyTorch tensors to the specified device,
    but gracefully handles non-tensor data by returning it unchanged.
    
    Args:
        d: Data to move to device (tensor, array, or other data).
        device: Target device (string or torch.device).
    
    Returns:
        Data moved to device if it's a tensor, otherwise unchanged.
    """
    try:
        d = d.to(device)
    except AttributeError:
        pass
    return d

def try_to_cuda(d: Any) -> Any:
    """
    Safely move data to CUDA device.
    
    This function attempts to move PyTorch tensors to CUDA, but gracefully
    handles non-tensor data by returning it unchanged.
    
    Args:
        d: Data to move to CUDA (tensor, array, or other data).
    
    Returns:
        Data moved to CUDA if it's a tensor, otherwise unchanged.
    """
    try:
        d = d.cuda()
    except AttributeError:
        pass
    return d

def try_to_detach(d: Any) -> Any:
    """
    Safely detach data from computation graph.
    
    This function attempts to detach PyTorch tensors from the computation graph,
    but gracefully handles non-tensor data by returning it unchanged.
    
    Args:
        d: Data to detach (tensor, array, or other data).
    
    Returns:
        Detached data if it's a tensor, otherwise unchanged.
    """
    try:
        d = d.detach()
    except AttributeError:
        pass
    return d

def try_to_cpu(d: Any) -> Any:
    """
    Safely move data to CPU and detach from computation graph.
    
    This function attempts to move PyTorch tensors to CPU and detach them,
    but gracefully handles non-tensor data by returning it unchanged.
    
    Args:
        d: Data to move to CPU (tensor, array, or other data).
    
    Returns:
        Data moved to CPU and detached if it's a tensor, otherwise unchanged.
    """
    try:
        d = d.detach().cpu()
    except AttributeError:
        pass
    return d

def try_to_numpy(d: Any) -> Any:
    """
    Safely convert data to NumPy array.
    
    This function attempts to convert PyTorch tensors to NumPy arrays,
    but gracefully handles non-tensor data by returning it unchanged.
    
    Args:
        d: Data to convert to NumPy (tensor, array, or other data).
    
    Returns:
        NumPy array if input is a tensor, otherwise unchanged.
    """
    try:
        d = d.detach().cpu().numpy()
    except AttributeError:
        pass
    except TypeError:
        pass
    return d

def try_to_torch(d: Any) -> Any:
    """
    Safely convert data to PyTorch tensor.
    
    This function attempts to convert NumPy arrays to PyTorch tensors,
    but gracefully handles non-array data by returning it unchanged.
    
    Args:
        d: Data to convert to PyTorch tensor (array, tensor, or other data).
    
    Returns:
        PyTorch tensor if input is a NumPy array, otherwise unchanged.
    """
    try:
        d = torch.from_numpy(d)
    except AttributeError:
        pass
    except TypeError:
        pass
    return d
        
def dict_to_device(data_dict: Dict[str, Any], device: Union[str, torch.device]) -> Dict[str, Any]:
    """
    Recursively move all tensors in a dictionary to the specified device.
    
    This function traverses a nested dictionary structure and moves all PyTorch
    tensors to the specified device while preserving the structure.
    
    Args:
        data_dict: Dictionary containing tensors and other data.
        device: Target device (string or torch.device).
    
    Returns:
        Dictionary with all tensors moved to the specified device.
    """
    ret_dict = dict()
    for k, v in data_dict.items():
        if isinstance(v, list):
            v_device = [try_to_device(t, device) for t in v]
            ret_dict[k] = v_device
        elif isinstance(v, tuple):
            v_device = tuple(try_to_device(t, device) for t in v)
            ret_dict[k] = v_device
        elif isinstance(v, dict):
            v_device = dict_to_device(v, device)
            ret_dict[k] = v_device
        else:
            v_device = try_to_device(v, device)
            ret_dict[k] = v_device
    return ret_dict

def dict_to_cuda(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively move all tensors in a dictionary to CUDA.
    
    This function traverses a nested dictionary structure and moves all PyTorch
    tensors to CUDA while preserving the structure.
    
    Args:
        data_dict: Dictionary containing tensors and other data.
    
    Returns:
        Dictionary with all tensors moved to CUDA.
    """
    ret_dict = dict()
    for k, v in data_dict.items():
        if isinstance(v, list):
            v_device = [try_to_cuda(t) for t in v]
            ret_dict[k] = v_device
        elif isinstance(v, tuple):
            v_device = tuple(try_to_cuda(t) for t in v)
            ret_dict[k] = v_device
        elif isinstance(v, dict):
            v_device = dict_to_cuda(v)
            ret_dict[k] = v_device
        else:
            v_device = try_to_cuda(v)
            ret_dict[k] = v_device
    return ret_dict

def dict_to_detach(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively detach all tensors in a dictionary from computation graph.
    
    This function traverses a nested dictionary structure and detaches all PyTorch
    tensors from the computation graph while preserving the structure.
    
    Args:
        data_dict: Dictionary containing tensors and other data.
    
    Returns:
        Dictionary with all tensors detached from computation graph.
    """
    ret_dict = dict()
    for k, v in data_dict.items():
        if isinstance(v, list):
            v_device = [try_to_detach(t) for t in v]
            ret_dict[k] = v_device
        elif isinstance(v, tuple):
            v_device = tuple(try_to_detach(t) for t in v)
            ret_dict[k] = v_device
        elif isinstance(v, dict):
            v_device = dict_to_detach(v)
            ret_dict[k] = v_device
        else:
            v_device = try_to_detach(v)
            ret_dict[k] = v_device
    return ret_dict

def dict_to_cpu(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively move all tensors in a dictionary to CPU and detach.
    
    This function traverses a nested dictionary structure and moves all PyTorch
    tensors to CPU while detaching them from the computation graph.
    
    Args:
        data_dict: Dictionary containing tensors and other data.
    
    Returns:
        Dictionary with all tensors moved to CPU and detached.
    """
    ret_dict = dict()
    for k, v in data_dict.items():
        if isinstance(v, list):
            v_device = [try_to_cpu(t) for t in v]
            ret_dict[k] = v_device
        elif isinstance(v, tuple):
            v_device = tuple(try_to_cpu(t) for t in v)
            ret_dict[k] = v_device
        elif isinstance(v, dict):
            v_device = dict_to_cpu(v)
            ret_dict[k] = v_device
        else:
            v_device = try_to_cpu(v)
            ret_dict[k] = v_device
    return ret_dict

def dict_to_numpy(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively convert all tensors in a dictionary to NumPy arrays.
    
    This function traverses a nested dictionary structure and converts all PyTorch
    tensors to NumPy arrays while preserving the structure.
    
    Args:
        data_dict: Dictionary containing tensors and other data.
    
    Returns:
        Dictionary with all tensors converted to NumPy arrays.
    """
    ret_dict = dict()
    for k, v in data_dict.items():
        if isinstance(v, list):
            v_device = [try_to_numpy(t) for t in v]
            ret_dict[k] = v_device
        elif isinstance(v, dict):
            v_device = dict_to_numpy(v)
            ret_dict[k] = v_device
        elif isinstance(v, tuple):
            v_device = tuple(try_to_numpy(t) for t in v)
            ret_dict[k] = v_device
        else:
            v_device = try_to_numpy(v)
            ret_dict[k] = v_device
    return ret_dict


def dict_to_torch(data_dict: Dict[str, Any], device: Union[str, torch.device] = 'cpu') -> Dict[str, Any]:
    """
    Recursively convert all arrays in a dictionary to PyTorch tensors and move to device.
    
    This function traverses a nested dictionary structure and converts all NumPy
    arrays to PyTorch tensors, then moves them to the specified device.
    
    Args:
        data_dict: Dictionary containing arrays and other data.
        device: Target device for the tensors (default: 'cpu').
    
    Returns:
        Dictionary with all arrays converted to PyTorch tensors on the specified device.
    """
    ret_dict = dict()
    for k, v in data_dict.items():
        if isinstance(v, list):
            v_device = [try_to_torch(t) for t in v]
            ret_dict[k] = v_device
        elif isinstance(v, tuple):
            v_device = tuple(try_to_torch(t) for t in v)
            ret_dict[k] = v_device
        elif isinstance(v, dict):
            v_device = dict_to_torch(v)
            ret_dict[k] = v_device
        else:
            v_device = try_to_torch(v)
            ret_dict[k] = v_device
    ret_dict = dict_to_device(ret_dict, device)
    return ret_dict

