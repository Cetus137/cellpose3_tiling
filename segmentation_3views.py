from cellpose.models import CellposeModel
import tifffile as tiff
import numpy as np
import scipy.ndimage as ndi
from cellpose.dynamics import compute_masks
import cellpose
from importlib.metadata import version as _getv
import pkg_resources
import time

def segment_3D_stack(image_stack, config, view):

    shape = image_stack.shape
    print(shape)

    has_channels = image_stack.ndim == 4
    if has_channels:
        if shape[0] <= 4:
            # (channels, slices, y, x)
            stack_for_eval = image_stack
        else:
            raise ValueError(
                f"Channel-aware input must be channel-first (channels, slices, y, x); got {shape}."
            )
        _, n_slices, dim_1, dim_2 = stack_for_eval.shape
    elif image_stack.ndim == 3:
        stack_for_eval = image_stack
        n_slices, dim_1, dim_2 = shape
    else:
        raise ValueError(
            f"Expected image_stack with shape (slices, y, x) or channel-first (channels, slices, y, x), got {shape}"
        )

    flowsx_stack = np.zeros((n_slices, dim_1, dim_2), dtype=np.float32)
    flowsy_stack = np.zeros((n_slices, dim_1, dim_2), dtype=np.float32)
    flowsz_stack = np.zeros((n_slices, dim_1, dim_2), dtype=np.float32)
    cell_prob_stack = np.zeros((n_slices, dim_1, dim_2), dtype=np.float32)

    model = config['model']

    channels = config.get('channels', [0, 0])

    use_gpu    = config.get('use_gpu', False)
    batch_size = config['batch_size'] if use_gpu else 1
    eval_kwargs = dict(
        channels=channels,
        batch_size=batch_size,
        do_3D=False,
        min_size=config['min_size'],
        cellprob_threshold=config.get('cell_prob_threshold', 0.0),
        diameter=config.get('diameter', None),
    )

    def _extract_flows(flows_list, x_idx, y_idx):
        flowsx = np.stack([f[1][x_idx] for f in flows_list])
        flowsy = np.stack([f[1][y_idx] for f in flows_list])
        cellprob = np.stack([f[2]       for f in flows_list])
        return flowsx, flowsy, cellprob

    if view == 'XY':
        t0 = time.time()
        print('Segmenting view XY')
        images = [stack_for_eval[:, i, :, :] if has_channels else stack_for_eval[i]
                  for i in range(n_slices)]
        _, flows_list, _ = model.eval(images, **eval_kwargs)
        flowsx_stack, flowsy_stack_tmp, cell_prob_stack = _extract_flows(flows_list, 1, 0)
        flowsy_stack = flowsy_stack_tmp
        print(f'XY done in {time.time()-t0:.1f}s')

    elif view == 'XZ':
        t0 = time.time()
        print('Segmenting view XZ')
        images = [stack_for_eval[:, i, :, :] if has_channels else stack_for_eval[i]
                  for i in range(n_slices)]
        _, flows_list, _ = model.eval(images, **eval_kwargs)
        flowsx_stack, flowsz_stack, cell_prob_stack = _extract_flows(flows_list, 1, 0)

        flowsz_stack   = np.transpose(flowsz_stack,   (1, 0, 2))
        flowsx_stack   = np.transpose(flowsx_stack,   (1, 0, 2))
        flowsy_stack   = np.transpose(flowsy_stack,   (1, 0, 2))
        cell_prob_stack = np.transpose(cell_prob_stack, (1, 0, 2))
        print(f'XZ done in {time.time()-t0:.1f}s')

    elif view == 'YZ':
        t0 = time.time()
        print('Segmenting view YZ')
        images = [stack_for_eval[:, i, :, :] if has_channels else stack_for_eval[i]
                  for i in range(n_slices)]
        _, flows_list, _ = model.eval(images, **eval_kwargs)
        flowsy_stack, flowsz_stack, cell_prob_stack = _extract_flows(flows_list, 1, 0)

        flowsy_stack   = np.transpose(flowsy_stack,   (1, 2, 0))
        flowsz_stack   = np.transpose(flowsz_stack,   (1, 2, 0))
        flowsx_stack   = np.transpose(flowsx_stack,   (1, 2, 0))
        cell_prob_stack = np.transpose(cell_prob_stack, (1, 2, 0))
        print(f'YZ done in {time.time()-t0:.1f}s')

    return flowsx_stack, flowsy_stack, flowsz_stack, cell_prob_stack


def segment_3views(image_xy, image_xz, image_yz, config):
    flowsx_xy, flowsy_xy, _, cell_prob_xy = segment_3D_stack(image_xy, config, view='XY')
    _, flowsy_yz, flowsz_yz, cell_prob_yz = segment_3D_stack(image_yz, config, view='YZ')
    flowsx_xz, _, flowsz_xz, cell_prob_xz = segment_3D_stack(image_xz, config, view='XZ')

    #average the flows and cell probabilities from different views
    flowsx = (flowsx_xy + flowsx_xz)
    flowsy = (flowsy_xy + flowsy_yz) 
    flowsz = (flowsz_xz + flowsz_yz) 
    cell_prob = (cell_prob_xy + cell_prob_xz + cell_prob_yz)

    dP = np.array([flowsz, flowsy, flowsx])
    print(dP.shape)
    print(cell_prob.shape)
    return dP, cell_prob

def segment_zstack_3views(vid_frame_3views, model, cellpose_config_dict=None):
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
        'channels': [0, 0],
        'z_axis': 0,
        'gamma': 1.0,
        'cell_prob_threshold': 8.0,
        'use_gpu': True
    }

    config = {**default_config, **(cellpose_config_dict or {})}

    if vid_frame_3views.ndim == 4 and vid_frame_3views.shape[0] == 3:
        img_xy = np.transpose(vid_frame_3views[0, ...], (0, 1, 2))
        img_xz = np.transpose(vid_frame_3views[1, ...], (1, 0, 2))
        img_yz = np.transpose(vid_frame_3views[2, ...], (2, 0, 1))
    elif vid_frame_3views.ndim == 5 and vid_frame_3views.shape[0] == 3:
        # input shape: (views, channels, z, y, x)
        img_xy = vid_frame_3views[0, ...]
        img_xz = np.transpose(vid_frame_3views[1, ...], (0, 2, 1, 3))
        img_yz = np.transpose(vid_frame_3views[2, ...], (0, 3, 1, 2))
    elif vid_frame_3views.ndim == 4:
        # input shape: (channels, z, y, x) -> build XY/XZ/YZ with channels retained
        img_xy = vid_frame_3views
        img_xz = np.transpose(vid_frame_3views, (0, 2, 1, 3))
        img_yz = np.transpose(vid_frame_3views, (0, 3, 1, 2))
    else:
        raise ValueError(
            "Expected input shape (3, z, y, x), (3, channels, z, y, x), or (channels, z, y, x); "
            f"got {vid_frame_3views.shape}"
        )

    print('image shapes:', img_xy.shape, img_xz.shape, img_yz.shape)
    print(config)

    t_total = time.time()
    dP , cell_prob = segment_3views(img_xy, img_xz, img_yz, config)
    print(f'3-view inference done in {time.time()-t_total:.1f}s')

    cell_prob_blur = ndi.gaussian_filter(cell_prob, sigma=2)
    cell_prob_blur = np.clip(cell_prob_blur, -6, 12)
    dP_blur = ndi.gaussian_filter(dP, sigma=(0,2,2,2))

    print(dP_blur.shape , cell_prob_blur.shape)
    return dP_blur , cell_prob_blur


if __name__ == "__main__":
    try:
        version = getattr(cellpose, '__version__', None)
        if not version:
            try:
                version = _getv('cellpose')
            except Exception:
                try:
                    version = pkg_resources.get_distribution('cellpose').version
                except Exception:
                    version = 'unknown'
        print('Cellpose version:', version)
    except Exception as e:
        print('Could not determine Cellpose version:', e)

    pretrained_model_path = r'/Users/ewheeler/.cellpose/models/CP_20250430_181517'
    model = CellposeModel(gpu=True , pretrained_model=pretrained_model_path)
    img_path    = r"/Users/ewheeler/cellpose3_testing/data/T0_32bit_xy.tif"
    output_path = r"/Users/ewheeler/cellpose3_testing/data/T0_32bit_xy_segmented.tif"
    vid = tiff.imread(img_path)
    print('video shape:', vid.shape)

    #first normalize the video to range 0-1
    vid = vid.astype(np.float32)
    vid = (vid - np.min(vid)) / (np.max(vid) - np.min(vid))

    segment_zstack(vid, model)


