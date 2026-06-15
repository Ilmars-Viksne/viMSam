import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
# Set backend to Agg to prevent display issues in loops/threads
matplotlib.use('Agg')

STABLE_PALETTE = np.array([
    [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]
], dtype=np.uint8)

def create_visualization(image, segmentation_result, prompts=None, save_combined=False):
    """
    Generates a visualization (Mask or Combined) and returns it as a numpy array.
    Does NOT save to disk.
    """
    # 1. Standardize Input Image
    image = np.squeeze(image)
    if image.ndim == 2:
        image = np.stack((image,) * 3, axis=-1)

    # Normalize image for display/overlay
    if image.dtype != np.uint8:
        image_disp = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
    else:
        image_disp = image

    h, w = image.shape[:2]
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # 2. Process Masks
    masks = []
    if isinstance(segmentation_result, list):
        for m in segmentation_result: masks.append(m['segmentation'])
    elif isinstance(segmentation_result, np.ndarray):
        if segmentation_result.ndim == 3:
            for i in range(segmentation_result.shape[0]): masks.append(segmentation_result[i])
        else: masks.append(segmentation_result)

    for i, mask in enumerate(masks):
        color = STABLE_PALETTE[(i % 4) + 1]
        colored_mask[mask > 0] = color

    # --- BRANCHING LOGIC ---

    # CASE A: Default - Return ONLY the mask on black background
    if not save_combined:
        return colored_mask

    # CASE B: Combined - Return Original | Mask | Overlay

    # Create Overlay
    overlay = image_disp.copy()
    mask_indices = np.any(colored_mask > 0, axis=-1)
    overlay[mask_indices] = (overlay[mask_indices] * 0.6 + colored_mask[mask_indices] * 0.4).astype(np.uint8)

    # Setup Matplotlib Figure
    fig = plt.figure(figsize=(18, 6))

    # Panel 1: Original
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(image_disp)
    ax1.set_title("Original")
    ax1.axis('off')

    # Panel 2: Mask
    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(colored_mask)
    ax2.set_title("Segmentation Mask")
    ax2.axis('off')

    # Panel 3: Overlay
    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(overlay)
    ax3.set_title("Overlay")
    ax3.axis('off')

    # Draw Prompts
    if prompts:
        if prompts['type'] == 'point':
            pts = prompts['data']
            if pts.ndim == 1: pts = pts[None, :]
            ax3.scatter(pts[:, 0], pts[:, 1], marker='x', c='white', s=100, linewidth=2)
        elif prompts['type'] == 'box':
            b = prompts['data']
            rect = patches.Rectangle((b[0], b[1]), b[2]-b[0], b[3]-b[1], linewidth=2, edgecolor='white', facecolor='none')
            ax3.add_patch(rect)

    plt.tight_layout()

    # Convert Figure to Numpy Array (Robust Method for Matplotlib 3.8+)
    fig.canvas.draw()

    # buffer_rgba returns a memoryview of RGBA pixels
    buf = fig.canvas.buffer_rgba()
    data = np.asarray(buf)

    # Convert RGBA to RGB (drop alpha channel)
    im_array = data[:, :, :3]

    plt.close(fig)
    return im_array

# Backward compatibility alias
save_masks_as_image = create_visualization
