from cellpose_omni.models import CellposeModel
import tifffile as tiff
import numpy as np
import scipy.ndimage as ndi
import omnipose
from importlib.metadata import version as _getv
import pkg_resources

def segment_3D_stack(image_stack, config, view):

    shape = image_stack.shape
    print(shape)
    flowsx_stack = np.zeros_like(image_stack)
    flowsy_stack = np.zeros_like(image_stack)
    flowsz_stack = np.zeros_like(image_stack)
    cell_prob_stack = np.zeros_like(image_stack)

    model = config['model']

    if view == 'XY':
        for i in range(shape[0]):
            image = image_stack[i, :, :]
            masks, flows, styles = model.eval(image, diameter=config.get('diameter', None), channels=None, batch_size=config['batch_size'], do_3D=False, min_size=config['min_size'], 
                                             cluster=config.get('cluster', False), omni=config.get('omni', True) )

            flowsx_stack[i, :, :] = flows[1][1]
            flowsy_stack[i, :, :] = flows[1][0]
            cell_prob_stack[i, :, :] = flows[2]

            print(type(flows), [f.shape if hasattr(f,'shape') else None for f in flows])

        tiff.imwrite('cellprob_xy_raw.tif', cell_prob_stack.astype(np.float32))

    elif view == 'XZ':
        for i in range(shape[0]):
            image = image_stack[i, :, :]
            masks, flows, styles = model.eval(image, diameter=config.get('diameter', None), channels=None, batch_size=config['batch_size'], do_3D=False, min_size=config['min_size'],
                                             cluster=config.get('cluster', False), omni=config.get('omni', True))

            flowsx_stack[i, :, :] = flows[1][1]
            flowsz_stack[i, :, :] = flows[1][0]
            cell_prob_stack[i, :, :] = flows[2]

        #transpose to be consistent with XY view
        flowsz_stack = np.transpose(flowsz_stack, (1, 0, 2))
        flowsx_stack = np.transpose(flowsx_stack, (1, 0, 2))
        flowsy_stack = np.transpose(flowsy_stack, (1, 0, 2))
        cell_prob_stack = np.transpose(cell_prob_stack, (1, 0, 2))

        print(type(flows), [f.shape if hasattr(f,'shape') else None for f in flows])

        print('after transpose', flowsz_stack.shape)

        tiff.imwrite('cellprob_xz_raw.tif', cell_prob_stack.astype(np.float32))

    elif view == 'YZ':
        for i in range(shape[0]):
            image = image_stack[i, :, :]
            masks, flows, styles = model.eval(image, diameter=config.get('diameter', None), channels=None, batch_size=config['batch_size'], do_3D=False, min_size=config['min_size'],
                                             cluster=config.get('cluster', False), omni=config.get('omni', True))

            flowsy_stack[i, :, :] = flows[1][1]
            flowsz_stack[i, :, :] = flows[1][0]
            cell_prob_stack[i, :, :] = flows[2]
            print(type(flows), [f.shape if hasattr(f,'shape') else None for f in flows])

        #transpose to be consistent with XY view
        flowsy_stack = np.transpose(flowsy_stack, (1, 2, 0))
        flowsz_stack = np.transpose(flowsz_stack, (1, 2, 0))
        flowsx_stack = np.transpose(flowsx_stack, (1, 2, 0))
        cell_prob_stack = np.transpose(cell_prob_stack, (1, 2, 0))

        print('after transpose', flowsy_stack.shape)
        tiff.imwrite('cellprob_yz_raw.tif', cell_prob_stack.astype(np.float32))

    return flowsx_stack, flowsy_stack, flowsz_stack, cell_prob_stack


def segment_3views(image_xy, image_xz, image_yz, config):

    _, flowsy_yz, flowsz_yz, cell_prob_yz = segment_3D_stack(image_yz, config, view='YZ')
    flowsx_xz, _, flowsz_xz, cell_prob_xz = segment_3D_stack(image_xz, config, view='XZ')
    flowsx_xy, flowsy_xy, _, cell_prob_xy = segment_3D_stack(image_xy, config, view='XY')

    #average the flows and cell probabilities from different views
    flowsx = (flowsx_xy + flowsx_xz)
    flowsy = (flowsy_xy + flowsy_yz) 
    flowsz = (flowsz_xz + flowsz_yz) 
    cell_prob = (cell_prob_xy + cell_prob_xz + cell_prob_yz)

    dP = np.array([flowsz, flowsy, flowsx])
    print(dP.shape)
    print(cell_prob.shape)
    return dP, cell_prob

def segment_zstack_3views(vid_frame_3views, model, omnipose_config_dict=None):
    '''
    segment a single timepoint 3D frame with omnipose.

    Inputs:
    --------------------------------------------------
    vid_frame: 3D numpy array of the frame to segment (shape: [z, y, x])
    model: Omnipose model object to use for segmentation
    omnipose_config_dict: dictionary containing omnipose configuration parameters (optional)
    Outputs:
    --------------------------------------------------
    masks: 3D numpy array of the segmentation masks for the frame (shape: [z, y, x])
    '''

    default_config = {
        'model' : model,
        'batch_size': 256,
        'do_3D': False,
        'diameter': 40,
        'min_size': 100,
        'z_axis': 0,
        'gamma': 1.0,
        'mask_threshold': 8.0,
        'use_gpu': True,
        'cluster': False,
        'omni': True
    }

    config = {**default_config, **(omnipose_config_dict or {})}

    img_xy = np.transpose(vid_frame_3views[0,...], (0, 1, 2))   #shape (z,y,x)
    img_xz = np.transpose(vid_frame_3views[1,...], (1 ,0, 2))   #transpose to (y,z,x)
    img_yz = np.transpose(vid_frame_3views[2,...], (2, 0, 1))   #transpose to (x,z,y) 

    print('image shapes:', img_xy.shape, img_xz.shape, img_yz.shape)

    dP , cell_prob = segment_3views(img_xy, img_xz, img_yz, config)

    cell_prob_blur = ndi.gaussian_filter(cell_prob, sigma=2)
    cell_prob_blur = np.clip(cell_prob_blur, -6, 12)
    dP_blur = ndi.gaussian_filter(dP, sigma=(0,2,2,2))

    print('computing masks...')
    print(dP_blur.shape , cell_prob_blur.shape)
    return dP_blur , cell_prob_blur


if __name__ == "__main__":
    try:
        version = getattr(omnipose, '__version__', None)
        if not version:
            try:
                version = _getv('omnipose')
            except Exception:
                try:
                    version = pkg_resources.get_distribution('omnipose').version
                except Exception:
                    version = 'unknown'
        print('Omnipose version:', version)
    except Exception as e:
        print('Could not determine Omnipose version:', e)

    pretrained_model_path = r'/users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/models/omnipose_model'
    model = CellposeModel(gpu=True, pretrained_model=pretrained_model_path, nchan=1, nclasses=3)
    img_path    = r"/path/to/your/T0_32bit.tif"
    output_path = r"/path/to/your/T0_32bit_segmented.tif"
    
    # Load 3-view stack (shape should be [3, z, y, x])
    vid = tiff.imread(img_path)
    print('video shape:', vid.shape)

    # Normalize the video to range 0-1
    vid = vid.astype(np.float32)
    vid = (vid - np.min(vid)) / (np.max(vid) - np.min(vid))

    dP_blur, cell_prob_blur = segment_zstack_3views(vid, model)


