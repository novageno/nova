import cv2
import numpy as np
import scipy

def rescale_patient_images(image,org_spacing,target_spacing=(1,1,1)):
    
    resize_factor = org_spacing / target_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_size_factor = new_shape / image.shape
    new_spacing = org_spacing / real_size_factor
    
    image = scipy.ndimage.interpolation.zoom(image,real_size_factor,mode='nearest')
    
    return image
    