"""
Super class of dicom CTs,all dataset using dicom should inherit this class
Author:Lilhope
"""

import dicom
import numpy as np

class DCM():
    def __init__(self):
        pass
    
    def get_pixel_hu(self,slices):
        """convert dicom file to hounsfiled unit"""
        
        image = np.stack([s.pixel_array for s in slices])
        image.astype(np.int16)
        
        for slice_number in range(len(slices)):
            intercept = slices[slice_number].RescaleIntercept
            slope = slices[slice_number].RescaleSlope
            
            if slope != 1:
                image[slice_number] = slope * image[slice_number].astype(np.float16)
                image[slice_number] = image[slice_number].astype(np.int32)
            image[slice_number] += np.int16(intercept)
            
        return np.array(img,dtype=np.int16), np.array(slices[0].SliceThickness + slices[0].PixelSpacing,dtype=np.float32)
    
    def parse_dicom_single():