import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from monai import transforms


def plot_intensity_histogram(img):
    # Convert SimpleITK image to NumPy array
    img_array = sitk.GetArrayFromImage(img)

    plt.hist(img_array.flatten(), bins=128, density=True)
    plt.show()


# Calculate parameters low and high from window and level
def wl_to_lh(window, level):
    low = level - window/2
    high = level + window/2
    return low, high


def display_image(img, x=None, y=None, z=None, window=None, level=None, colormap='gray', crosshair=False):
    # Convert SimpleITK image to NumPy array
    img_array = sitk.GetArrayFromImage(img)
    
    # Get image dimensions in millimetres
    size = img.GetSize()
    spacing = img.GetSpacing()
    width  = size[0] * spacing[0]
    height = size[1] * spacing[1]
    depth  = size[2] * spacing[2]
    
    if x is None:
        x = np.floor(size[0]/2).astype(int)
    if y is None:
        y = np.floor(size[1]/2).astype(int)
    if z is None:
        z = np.floor(size[2]/2).astype(int)
    
    if window is None:
        window = np.max(img_array) - np.min(img_array)
    
    if level is None:
        level = window / 2 + np.min(img_array)
    
    low, high = wl_to_lh(window,level)

    # Display the orthogonal slices
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 4))

    ax1.imshow(img_array[z,:,:], cmap=colormap, clim=(low, high), extent=(0, width, height, 0))
    ax2.imshow(img_array[:,y,:], origin='lower', cmap=colormap, clim=(low, high), extent=(0, width,  0, depth))
    ax3.imshow(img_array[:,:,x], origin='lower', cmap=colormap, clim=(low, high), extent=(0, height, 0, depth))

    # Additionally display crosshairs
    if crosshair:
        ax1.axhline(y * spacing[1], lw=1)
        ax1.axvline(x * spacing[0], lw=1)
        ax2.axhline(z * spacing[2], lw=1)
        ax2.axvline(x * spacing[0], lw=1)
        ax3.axhline(z * spacing[2], lw=1)
        ax3.axvline(y * spacing[1], lw=1)

    plt.show()


def display_image_and_labels(img, lbls, x=None, y=None, z=None, window=None, level=None, colormap='gray'):
    # Convert SimpleITK image to NumPy array
    img_array = sitk.GetArrayFromImage(img)
    lbls_array = sitk.GetArrayFromImage(lbls)
    
    # Get image dimensions in millimetres
    size = img.GetSize()
    spacing = img.GetSpacing()
    width  = size[0] * spacing[0]
    height = size[1] * spacing[1]
    depth  = size[2] * spacing[2]
    
    if x is None:
        x = np.floor(size[0]/2).astype(int)
    if y is None:
        y = np.floor(size[1]/2).astype(int)
    if z is None:
        z = np.floor(size[2]/2).astype(int)
    
    if window is None:
        window = np.max(img_array) - np.min(img_array)
    
    if level is None:
        level = window / 2 + np.min(img_array)
    
    low, high = wl_to_lh(window,level)

    window_lbls = np.max(lbls_array) - np.min(lbls_array)
    level_lbls = window_lbls / 2 + np.min(lbls_array)
    low_lbls, high_lbls = wl_to_lh(window_lbls,level_lbls)

    # Display the orthogonal slices
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(10, 8))

    ax1.imshow(img_array[z,:,:], cmap=colormap, clim=(low, high), extent=(0, width, height, 0))
    ax2.imshow(img_array[:,y,:], origin='lower', cmap=colormap, clim=(low, high), extent=(0, width,  0, depth))
    ax3.imshow(img_array[:,:,x], origin='lower', cmap=colormap, clim=(low, high), extent=(0, height, 0, depth))
    ax4.imshow(lbls_array[z,:,:], cmap=colormap, clim=(low_lbls, high_lbls), extent=(0, width, height, 0))
    ax5.imshow(lbls_array[:,y,:], origin='lower', cmap=colormap, clim=(low_lbls, high_lbls), extent=(0, width,  0, depth))
    ax6.imshow(lbls_array[:,:,x], origin='lower', cmap=colormap, clim=(low_lbls, high_lbls), extent=(0, height, 0, depth))

    plt.show()


def display_2d_tensor(img_tensor, lbls_tensor, x=None, y=None, window=None, level=None, colormap='gray'): 
    width  = img_tensor.shape[0]
    height = img_tensor.shape[1]
    
    if x is None:
        x = np.floor(img_tensor.shape[0]/2).astype(int)
    if y is None:
        y = np.floor(img_tensor.shape[1]/2).astype(int)

    array_transform = transforms.ToNumpy()
    img_array = array_transform(img_tensor)
    lbls_array = array_transform(lbls_tensor)
    
    if window is None:
        window = np.max(img_array) - np.min(img_array)
    
    if level is None:
        level = window / 2 + np.min(img_array)
    
    low, high = wl_to_lh(window,level)

    window_lbls = np.max(lbls_array) - np.min(lbls_array)
    level_lbls = window_lbls / 2 + np.min(lbls_array)
    low_lbls, high_lbls = wl_to_lh(window_lbls,level_lbls)

    # Display the slice
    fig, ((ax1), (ax2)) = plt.subplots(2, 1, figsize=(3.5, 8))

    ax1.imshow(np.transpose(img_array), cmap=colormap, clim=(low, high), extent=(0, width, height, 0))
    ax2.imshow(np.transpose(lbls_array), cmap=colormap, clim=(low_lbls, high_lbls), extent=(0, width, height, 0))

    plt.show()


def plot_save_predictions(x, y_pred, y_true, image_index, args, modality):
    width = x.shape[0]
    height = x.shape[1]

    window = np.max(x) - np.min(x)
    level = window / 2 + np.min(x)
    low, high = wl_to_lh(window,level)

    window_lbls = np.max(y_true) - np.min(y_true)
    level_lbls = window_lbls / 2 + np.min(y_true)
    low_lbls, high_lbls = wl_to_lh(window_lbls,level_lbls)

    fig, ((ax1), (ax2)) = plt.subplots(1, 2, figsize=(8, 3.5))

    ax1.imshow(np.transpose(x), cmap='gray', clim=(low, high), extent=(0, width, height, 0))
    ax1.imshow(np.transpose(y_true), cmap='viridis', clim=(low_lbls, high_lbls), alpha=0.5, extent=(0, width, height, 0))

    ax2.imshow(np.transpose(x), cmap='gray', clim=(low, high), extent=(0, width, height, 0))
    ax2.imshow(np.transpose(y_pred), cmap='viridis', clim=(low_lbls, high_lbls), alpha=0.5, extent=(0, width, height, 0))

    plt.savefig(fname=(args.pretrained_dir + modality + "_prediction_" + str(image_index) + ".png"))


if __name__ == "__main__":
    image_filename = "/Users/joannaye/Documents/_Imperial_AI_MSc/1_Individual_project/AMOS_dataset/amos22/imagesVa/amos_0008.nii.gz"
    image = sitk.ReadImage(image_filename)
    # display_image(image, window=350, level=50) # For CT images
    # display_image(image, window=600, level=200) # For MRI images
    print(f"Image size: {image.GetSize()}")
    print(f"Image spacing: {image.GetSpacing()}")

    labels_filename = "/Users/joannaye/Documents/_Imperial_AI_MSc/1_Individual_project/AMOS_dataset/amos22/labelsVa/amos_0008.nii.gz"
    labels = sitk.ReadImage(labels_filename)
    # display_image(sitk.LabelToRGB(labels))

    display_image_and_labels(image, sitk.LabelToRGB(labels), window=350, level=50)
    plot_intensity_histogram(image)