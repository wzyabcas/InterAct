import torch
import numpy as np

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]
        
    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    if 'seq_name' in notnone_batches[0]:
        seq_name = [b['seq_name']for b in notnone_batches]
        cond['y'].update({'seq_name': seq_name})
    
    if 'obj_points' in notnone_batches[0]:
        obj_points = [b['obj_points']for b in notnone_batches]
        obj_points = torch.from_numpy(np.array(obj_points))
        if len(obj_points.shape) < 3:
            obj_points = obj_points.unsqueeze(0)
        cond['y'].update({'obj_points': obj_points})  #  this part is changed for prompt-based generation

    if 'afford_labels' in notnone_batches[0]:    # For prediciton
        afford_labels = [b['afford_labels']for b in notnone_batches]
        afford_labels = torch.from_numpy(np.array(afford_labels))
        cond['y'].update({'afford_labels': afford_labels})  #  this part is changed for prompt-based generation

    if 'obj_normals' in notnone_batches[0]:
        obj_normals = [b['obj_normals']for b in notnone_batches]
        # print(f"==== {obj_faces.shape}")
        obj_normals = torch.from_numpy(np.array(obj_normals))
        if len(obj_normals.shape) < 3:
            obj_normals = obj_normals.unsqueeze(0)
        cond['y'].update({'obj_normals': obj_normals})  #  this part is changed for prompt-based generation

    if 'obj_bps' in notnone_batches[0]:
        obj_bps = [b['obj_bps']for b in notnone_batches]
        obj_bps = torch.from_numpy(np.array(obj_bps))
        if len(obj_bps.shape) < 3:
            obj_bps = obj_bps.unsqueeze(0)
        cond['y'].update({'obj_bps': obj_bps})  #  this part is changed for prompt-based generation
    
    if 'gt_afford_data' in notnone_batches[0]:
        gt_afford_data = [b['gt_afford_data']for b in notnone_batches]
        gt_afford_data = torch.as_tensor(gt_afford_data)
        cond['y'].update({'gt_afford_data': gt_afford_data})  #  this part is changed for prompt-based generation

    return motion, cond

# an adapter to our collate func
def t2m_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2], #b[0]['caption']
        'tokens': b[6],
        'lengths': b[5],
        'hint': b[-1],
        # 'seq_mask': b[-2],
        # 'joint_id': b[-1],
    } for b in batch]
    return collate(adapted_batch)

# an adapter to our collate func
def t2m_behave_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2], #b[0]['caption']
        'tokens': b[6],
        'lengths': b[5],
        'seq_name': b[7],
        'obj_points': b[8],
        'obj_normals': b[9],
        # 'gt_afford_data': b[10]
    } for b in batch]
    return collate(adapted_batch)

def t2m_behave_bps_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2], #b[0]['caption']
        'tokens': b[6],
        'lengths': b[5],
        'seq_name': b[7],
        'obj_points': b[8],
        'obj_normals': b[9],
        'obj_bps': b[10],
        # 'gt_afford_data': b[10]
    } for b in batch]
    return collate(adapted_batch)

# an adapter to our collate func
def t2m_omomo_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2], #b[0]['caption']
        'tokens': b[6],
        'lengths': b[5],
        'seq_name': b[7],
        'obj_points': b[8],
        'obj_normals': b[9]
    } for b in batch]
    return collate(adapted_batch)



def afford_collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]

    databatchTensor = collate_tensors(databatch)

    motion = databatchTensor
    cond = {'y': {'mask': None}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})


    if 'seq_name' in notnone_batches[0]:
        seq_name = [b['seq_name']for b in notnone_batches]
        cond['y'].update({'seq_name': seq_name})
    
    if 'obj_points' in notnone_batches[0]:
        obj_points = [b['obj_points']for b in notnone_batches]
        obj_points = torch.as_tensor(obj_points)
        if len(obj_points.shape) < 3:
            obj_points = obj_points.unsqueeze(0)
        cond['y'].update({'obj_points': obj_points})  #  this part is changed for prompt-based generation

    # if 'obj_faces' in notnone_batches[0]:
    #     obj_faces = [b['obj_faces']for b in notnone_batches]
    #     obj_faces = torch.as_tensor(obj_normals)
    #     if len(obj_faces.shape) < 3:
    #         obj_faces = obj_faces.unsqueeze(0)
    #     cond['y'].update({'obj_faces': obj_faces})  #  this part is changed for prompt-based generation
    
    return motion, cond


# an adapter to our collate func
def t2m_contact_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2], #b[0]['caption']
        'tokens': b[5],
        'seq_name': b[6],
        'obj_points': b[7],
        # 'obj_faces':b[8]
    } for b in batch]
    return afford_collate(adapted_batch)

