import matplotlib.pyplot as plt


def plot_3d_image(ret, cmap=None):
    # Plot axial slice
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)

    if cmap is not None:
        plt.imshow(ret[:, ret.shape[1] // 2, :, :].permute(1, 2, 0), cmap=cmap)
    else:
        plt.imshow(ret[:, ret.shape[1] // 2, :, :].permute(1, 2, 0))

    plt.title("Axial")
    plt.axis("off")

    # Plot sagittal slice
    plt.subplot(1, 3, 2)

    if cmap is not None:
        plt.imshow(ret[:, :, ret.shape[2] // 2, :].permute(1, 2, 0), cmap=cmap)
    else:
        plt.imshow(ret[:, :, ret.shape[2] // 2, :].permute(1, 2, 0))
    plt.title("Coronal")
    plt.axis("off")

    # Plot coronal slice
    plt.subplot(1, 3, 3)
    if cmap is not None:
        plt.imshow(ret[:, :, :, ret.shape[3] // 2].permute(1, 2, 0), cmap=cmap)
    else:
        plt.imshow(ret[:, :, :, ret.shape[3] // 2].permute(1, 2, 0))
    plt.title("Sagittal")

    plt.axis("off")
    plt.show()
