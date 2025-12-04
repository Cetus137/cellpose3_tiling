from cellpose import io, models, train
import tifffile as tiff
io.logger_setup()

def train_model(train_dir):
    """
    Train a Cellpose model using images and labels from the specified directory.

    Parameters:
    --------------------------------------------------
    train_dir: str
        Path to the parent of directory containing training images and labels.

    Returns:
    --------------------------------------------------
    model_path: str
        Path to the saved trained model.
    train_losses: list
        Training loss values over epochs.
    test_losses: list
        Testing loss values over epochs.
    """

    ### print all *.tif files in the train_dir using os module
    import os
    tif_files = [f for f in os.listdir(train_dir) if f.endswith('.tif')]
    print("TIF files in training directory:")
    for f in tif_files:
        img = tiff.imread(os.path.join(train_dir, f))
        if img.ndim != 2:
            print(f" - {f}, shape: {img.shape}, 3D image")
   

    output = io.load_train_test_data(train_dir,
                                     look_one_level_down=True , mask_filter = "_masks")
    images, labels, image_names, test_images, test_labels, image_names_test = output

    print(image_names)

    model = models.CellposeModel(gpu=True)

    model_path, train_losses, test_losses = train.train_seg(model.net, train_data=images,
                                train_labels=labels,
                                batch_size=16,
                                channels=[0,0],
                                save_every = 10,
                                test_data=test_images, test_labels=test_labels,
                                weight_decay=0.1, learning_rate=1e-5,
                                n_epochs=100, model_name="cp3_addnoise_model_node2_cellpose3")
    

if __name__ == "__main__":
    train_dir = r'/users/kir-fritzsche/aif490/devel/tissue_analysis/segmentation_scripts/for_training/train_directory_node2'
    train_model(train_dir)