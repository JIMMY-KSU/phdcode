import numpy as np


def generate_circle_image(image_width, image_height, radius, x, y):
    rr,cc = np.meshgrid(np.arange(image_width), np.arange(image_height))
    return (np.sqrt((rr-x)**2 + (cc-y)**2) <= (radius-1)).astype(float)


def generate_square_image(image_width, image_height, width, x, y):
    rr,cc = np.meshgrid(np.arange(image_width), np.arange(image_height))
    return ( (np.abs(rr-x)<=(width/2)) & (np.abs(cc-y) <= (width/2)) ).astype(float)


def generate_gaussian_image(image_width, image_height, centers, sigmas, scales):
    rr,cc = np.meshgrid(np.arange(image_width), np.arange(image_height))

    image = np.zeros((image_height, image_width))
    for i in range(centers.shape[0]):
        image += scales[i]*np.exp(-((rr-centers[i,0])**2/(2.0*sigmas[i,0]**2) + (cc-centers[i,1])**2/(2.0*sigmas[i,1]**2)))
        # image += scales[i]*np.exp(-((np.sqrt((rr-centers[i,0])**2 + (cc-centers[i,1])**2))**2 / (2.0*sigmas[i]**2)))
    return image
