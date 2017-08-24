import ntpath

import numpy as np
import SimpleITK as sitk
import data_utils
from config import config


class MHD():
    def __ini__(self,name):
        """
        params:name:dataset name
        """
        self.name = name
    def process_image(self,src_path):
        """
        image preprocess to fit the ML/DL model,see the document
        """
        patient_id = ntpath.basename(src_path).replace(".mhd","")
        print('Patient ID: {}'.format(patient_id))
        
        itk_img = sitk.ReadImage(src_path)
        img_array = sitk.GetArrayFromImage(itk_img)
        print('Img array:{}'.format(img_arrag.shape))
        
        origin = np.array(itk_image.GetOrigin())
        print("Origin (x,y,z):",origin)
        
        direction = np.array(itk_img.GetDirection())
        print('Direction:',direction)
        
        spacing = np.array(itk_img.GetSpacing())
        print("Spacing (x,y,z):",spacing)
        
        rescale = spacing / config.TARGET_VOXEL_MM
        print("Rescale: ",rescale)
        
        #rescale to target voxel
        img_array = data_utils.rescale_patient_images(img_array,spacing,config.TARGET_VOXEL_MM)
        img_array = self.normalize(img_array)
        img_array = img_array * 255.
        return img_array,patient_id
    
    def normalize(self,image):
        """Normlization the image"""
        
        MIN_BOUND = -1000.0
        MAX_BOUND = 400.0
        image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        image[image > 1] = 1.
        image[image < 0] = 0.
        return image
    
    def worldToVoxelCoord(self,worldCoord,origin,spacing):
        """convert world coordination to voxel coordination"""
        stretchedVoxelCoord = np.absolute(worldCoord - origin)
        voxelCoord = stretchedVoxelCoord / spacing
        return voxelCoord

    def anns_parser(self,anns_file_path):
        """annotation parser for csv file"""
        pass
    def vis_nodule(self):        