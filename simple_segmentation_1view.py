from cellpose.models import CellposeModel
import tifffile as tiff
import numpy as np
import scipy.ndimage as ndi
from cellpose.dynamics import compute_masks
import cellpose
from importlib.metadata import version as _getv
import pkg_resources


def segment_3D_stack(image_stack, config, model):

    shape = image_stack.shape
    print(shape)
    flowsx_stack = np.zeros_like(image_stack)
    flowsy_stack = np.zeros_like(image_stack)
    flowsz_stack = np.zeros_like(image_stack)
    cell_prob_stack = np.zeros_like(image_stack)


    for i in range(shape[0]):
        image = image_stack[i, :, :]
        masks, flows, styles = model.eval(image, diameter=None, channels=[0, 0],batch_size=config['batch_size'], do_3D=False, min_size=config['min_size'])

        flowsx_stack[i, :, :] = flows[1][1]
        flowsy_stack[i, :, :] = flows[1][0]
        cell_prob_stack[i, :, :] = flows[2]
    
    dP = np.array([ flowsy_stack, flowsx_stack])

    return dP, cell_prob_stack

def segment_zstack_1view(vid_frame, model, view, cellpose_config_dict=None):
    '''
    segment a single timepoint 3D frame with cellpose.

    Inputs:
    --------------------------------------------------
    vid_frame: 3D numpy array of the frame to segment (shape: [z, y, x])
    model: Cellpose model object to use for segmentation
    cellpose_config_dict: dictionary containing cellpose configuration parameters (optional)
    Outputs:
    --------------------------------------------------
    masks: 3D numpy array of the segmentation masks for the frame (shape: [z, y, x])
    '''

    default_config = {
        'model' : model,
        'batch_size': 256,
        'do_3D': False,
        'diameter': None,
        'min_size': 100,
        'z_axis': 0,
        'gamma': 1.0,
        'cell_prob_threshold': 0.0,
        'use_gpu': True
    }

    config = {**default_config, **(cellpose_config_dict or {})}
    img_xy = vid_frame                          #video in shape (z,y,x)

    print('image shapes:', img_xy.shape)

    dP , cell_prob = segment_3D_stack(img_xy, config, model)

    cell_prob_blur = ndi.gaussian_filter(cell_prob, sigma=2)
    cell_prob_blur = np.clip(cell_prob_blur, -6, 6)
    dP_blur = ndi.gaussian_filter(dP, sigma=(0,2,2,2))

    print('computing masks...')
    print(dP_blur.shape , cell_prob_blur.shape)
    return dP_blur , cell_prob_blur


