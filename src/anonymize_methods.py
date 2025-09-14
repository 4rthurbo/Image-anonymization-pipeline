import cv2
import numpy as np

def create_ellipse(image, x1, y1, x2, y2, threshold=100):

    #first grayscale image and threshold
    region = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold
    coords = np.column_stack(np.where(mask))

    # back to rectangular anonymization if <5 pixels
    if len(coords) < 5:
        full_mask = np.zeros(image.shape[:2], np.uint8)
        full_mask[y1:y2, x1:x2] = 1
        return full_mask.astype(bool)

    # PCA
    center = coords.mean(axis=0)     
    coords_centered = coords-center
    cov = np.cov(coords_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    a = 2 * np.sqrt(eigenvalues[0])
    b = 2 * np.sqrt(eigenvalues[1])

    # calculate angles
    vy, vx = eigenvectors[:, 0]
    angle = np.degrees(np.arctan2(-vy, -vx)) % 180

    #generate mask
    mask_full = np.zeros(image.shape[:2], np.uint8)
    center_int = (int(center[1]+x1), int(center[0]+y1))
    axes = (int(a), int(b))
    cv2.ellipse(mask_full, center_int, axes, angle, 0, 360, 1, -1)
    
    return mask_full.astype(bool)

def blur_region(image, x1, y1, x2, y2, k_rel=0.3, oriented=False):
    h, w = y2 - y1, x2 - x1
    #calculate kernel size
    k = max(3, int((max(h, w) * k_rel) //2*2+1))
    region = image[y1:y2, x1:x2]

    blurred = cv2.GaussianBlur(region, (k, k), 0)
    if oriented:
        mask = create_ellipse(image, x1, y1, x2, y2)
        local = mask[y1:y2, x1:x2][:, :, None]
        region[local.repeat(3, axis=2)] = blurred[local.repeat(3, axis=2)]
    else:
        image[y1:y2, x1:x2] = blurred
    return image

def pixelate_region(image, x1, y1, x2, y2, n_rel=0.1, oriented=False):
    h, w = y2 - y1, x2 - x1
    #calculate number of nxn
    n = max(4, int(min(h, w) * n_rel))
    region = image[y1:y2, x1:x2]

    small = cv2.resize(region, (n, n), interpolation=cv2.INTER_AREA)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    if oriented:
        mask = create_ellipse(image, x1, y1, x2, y2)
        local = mask[y1:y2, x1:x2][:, :, None]
        region[local.repeat(3, axis=2)] = pixelated[local.repeat(3, axis=2)]
    else:
        image[y1:y2, x1:x2] = pixelated

    return image

def mask_region(image, x1, y1, x2, y2, color=(254, 254, 254), oriented=False):

    region = image[y1:y2, x1:x2]
    if oriented:
        mask = create_ellipse(image, x1, y1, x2, y2)
        image[mask] = color
    else:
        region_masked = np.empty_like(region)
        region_masked[:] = color
        image[y1:y2, x1:x2] = color
    return image

