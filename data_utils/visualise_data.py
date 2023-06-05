import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from monai import transforms


organ_cmap = ListedColormap(["black", "red", "orange", "gold", "darkgreen", "lawngreen", "blue", "deepskyblue", "mediumslateblue",
                             "plum", "chocolate", "goldenrod", "yellow", "darkturquoise", "darkviolet", "pink"])
organ_cmap2 = ListedColormap(["gray", "red", "orange", "gold", "darkgreen", "lawngreen", "blue", "deepskyblue", "mediumslateblue",
                             "plum", "chocolate", "goldenrod", "yellow", "darkturquoise", "darkviolet", "pink"])


def plot_intensity_histogram(img):
    # Convert SimpleITK image to NumPy array
    img_array = sitk.GetArrayFromImage(img)

    plt.hist(img_array.flatten(), bins=128, density=True)
    plt.show()


def plot_intensity_histogram_ct_mri(img_ct, img_mri):
    # Convert SimpleITK image to NumPy array
    img_array_ct = sitk.GetArrayFromImage(img_ct)
    img_array_mri = sitk.GetArrayFromImage(img_mri)

    plt.hist(img_array_ct.flatten(), bins=128, density=True, alpha=0.5, label='CT')
    plt.hist(img_array_mri.flatten(), bins=128, density=True, alpha=0.5, label='MRI')
    plt.legend(loc='upper right')
    plt.ylim([0, 0.01])
    plt.show()


def plot_intensity_histogram_from_tensor(img):
    array_transform = transforms.ToNumpy()
    img_array = array_transform(img)

    plt.hist(img_array.flatten(), bins=128, density=True)
    plt.show()


def plot_intensity_histogram_from_tensor_ct_mri(img_ct, img_mri):
    array_transform = transforms.ToNumpy()
    img_array_ct = array_transform(img_ct)
    img_array_mri = array_transform(img_mri)

    plt.hist(img_array_ct.flatten(), bins=128, density=True, alpha=0.5, label='CT')
    plt.hist(img_array_mri.flatten(), bins=128, density=True, alpha=0.5, label='MRI')
    plt.legend(loc='upper right')
    plt.ylim([0, 2])
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


def display_image_and_labels_2d(img, lbls, z=None, window=None, level=None, colormap='gray'):
    # Convert SimpleITK image to NumPy array
    img_array = sitk.GetArrayFromImage(img)
    lbls_array = sitk.GetArrayFromImage(lbls)
    
    # Get image dimensions in millimetres
    size = img.GetSize()
    spacing = img.GetSpacing()
    width  = size[0] * spacing[0]
    height = size[1] * spacing[1]
    
    if z is None:
        z = np.floor(size[2]/2).astype(int)
    
    if window is None:
        window = np.max(img_array) - np.min(img_array)
    
    if level is None:
        level = window / 2 + np.min(img_array)
    
    low, high = wl_to_lh(window,level)

    # Display the orthogonal slices
    fig, ((ax1), (ax2)) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.imshow(img_array[z,:,:], cmap=colormap, clim=(low, high), extent=(0, width, height, 0))
    im2 = ax2.imshow(lbls_array[z,:,:], cmap=organ_cmap, clim=(-0.5, 15.5), extent=(0, width, height, 0))

    for ax in [ax1, ax2]:
        ax.set_xticks([])
        ax.set_xticks([], minor=True)
        ax.set_yticks([])
        ax.set_yticks([], minor=True)

    cax = fig.add_axes([ax2.get_position().x1+0.01, ax2.get_position().y0, 0.015, ax2.get_position().height])
    cbar = fig.colorbar(mappable=im2, cax=cax, ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    cbar.ax.set_yticklabels(["Background", "Spleen", "Right kidney", "Left kidney", "Gall bladder", "Esophagus", "Liver",
                             "Stomach", "Aorta", "Postcava", "Pancreas", "Right adrenal gland", "Left adrenal gland",
                             "Duodenum", "Bladder", "Prostrate/uterus"])
    cbar.ax.tick_params(labelsize=8)

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

    # Display the orthogonal slices
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(10, 8))

    ax1.imshow(img_array[z,:,:], cmap=colormap, clim=(low, high), extent=(0, width, height, 0))
    ax2.imshow(img_array[:,y,:], origin='lower', cmap=colormap, clim=(low, high), extent=(0, width,  0, depth))
    ax3.imshow(img_array[:,:,x], origin='lower', cmap=colormap, clim=(low, high), extent=(0, height, 0, depth))
    im4 = ax4.imshow(lbls_array[z,:,:], cmap=organ_cmap, clim=(-0.5, 15.5), extent=(0, width, height, 0))
    ax5.imshow(lbls_array[:,y,:], origin='lower', cmap=organ_cmap, clim=(-0.5, 15.5), extent=(0, width,  0, depth))
    ax6.imshow(lbls_array[:,:,x], origin='lower', cmap=organ_cmap, clim=(-0.5, 15.5), extent=(0, height, 0, depth))

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.set_xticks([])
        ax.set_xticks([], minor=True)
        ax.set_yticks([])
        ax.set_yticks([], minor=True)

    cax = fig.add_axes([ax6.get_position().x1+0.01, ax6.get_position().y0, 0.015, ax6.get_position().height])
    cbar = fig.colorbar(mappable=im4, cax=cax, ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    cbar.ax.set_yticklabels(["Background", "Spleen", "Right kidney", "Left kidney", "Gall bladder", "Esophagus", "Liver",
                             "Stomach", "Aorta", "Postcava", "Pancreas", "Right adrenal gland", "Left adrenal gland",
                             "Duodenum", "Bladder", "Prostrate/uterus"])
    cbar.ax.tick_params(labelsize=8)

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


def plot_save_predictions(x, y_pred, y_true, image_index, sample_no, args, modality):
    width = x.shape[0]
    height = x.shape[1]

    window = np.max(x) - np.min(x)
    level = window / 2 + np.min(x)
    low, high = wl_to_lh(window,level)

    fig, ((ax1), (ax2)) = plt.subplots(1, 2, figsize=(14, 3.5))

    ax1.imshow(np.transpose(x), cmap='gray', clim=(low, high), extent=(0, width, height, 0))
    ax1.imshow(np.transpose(y_true), cmap=organ_cmap2, clim=(-0.5, 15.5), alpha=0.5, extent=(0, width, height, 0))

    ax2.imshow(np.transpose(x), cmap='gray', clim=(low, high), extent=(0, width, height, 0))
    im2 = ax2.imshow(np.transpose(y_pred), cmap=organ_cmap2, clim=(-0.5, 15.5), alpha=0.5, extent=(0, width, height, 0))

    for ax in [ax1, ax2]:
        ax.set_xticks([])
        ax.set_xticks([], minor=True)
        ax.set_yticks([])
        ax.set_yticks([], minor=True)

    cax = fig.add_axes([ax2.get_position().x1+0.01, ax2.get_position().y0, 0.015, ax2.get_position().height])
    cbar = fig.colorbar(mappable=im2, cax=cax, ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    cbar.ax.set_yticklabels(["Background", "Spleen", "Right kidney", "Left kidney", "Gall bladder", "Esophagus", "Liver",
                             "Stomach", "Aorta", "Postcava", "Pancreas", "Right adrenal gland", "Left adrenal gland",
                             "Duodenum", "Bladder", "Prostrate/uterus"])
    cbar.ax.tick_params(labelsize=8)

    plt.savefig(fname=(args.pretrained_dir + modality + "_prediction_" + str(image_index) + "_" + str(sample_no) + ".png"))


if __name__ == "__main__":
    image_filename = "/Users/joannaye/Documents/_Imperial_AI_MSc/1_Individual_project/AMOS_dataset/amos22/imagesTr/amos_0508.nii.gz"
    image = sitk.ReadImage(image_filename)
    # display_image(image, window=350, level=50) # For CT images
    # display_image(image, window=600, level=200) # For MRI images
    # print(f"Image size: {image.GetSize()}")
    # print(f"Image spacing: {image.GetSpacing()}")

    labels_filename = "/Users/joannaye/Documents/_Imperial_AI_MSc/1_Individual_project/AMOS_dataset/amos22/labelsTr/amos_0508.nii.gz"
    labels = sitk.ReadImage(labels_filename)
    # display_image(sitk.LabelToRGB(labels))

    # display_image_and_labels(image, sitk.LabelToRGB(labels), window=350, level=50)
    display_image_and_labels(image, labels, window=600, level=200)
    # display_image_and_labels_2d(image, labels, window=600, level=200)
    # plot_intensity_histogram(image)

    # image_filename_ct = "/Users/joannaye/Documents/_Imperial_AI_MSc/1_Individual_project/AMOS_dataset/amos22/imagesTr/amos_0083.nii.gz"
    # image_ct = sitk.ReadImage(image_filename_ct)

    # image_filename_mri = "/Users/joannaye/Documents/_Imperial_AI_MSc/1_Individual_project/AMOS_dataset/amos22/imagesTr/amos_0600.nii.gz"
    # image_mri = sitk.ReadImage(image_filename_mri)

    # plot_intensity_histogram_ct_mri(image_ct, image_mri)