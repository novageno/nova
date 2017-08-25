import ntpath

import numpy as np
import SimpleITK as sitk
import data_utils
import cv2
#from config import config


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
        print('Img array:{}'.format(img_array.shape))
        
        origin = np.array(itk_img.GetOrigin())
        print("Origin (x,y,z):",origin)
        
        direction = np.array(itk_img.GetDirection())
        print('Direction:',direction)
        
        spacing = np.array(itk_img.GetSpacing())
        print("Spacing (x,y,z):",spacing)
        
        rescale = spacing / 1.00
        print("Rescale: ",rescale)
        
        #rescale to target voxel
        img_array = self.rescale_patient_images(img_array,spacing,1.00)
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
    def rescale_patient_images(self,images_zyx, org_spacing_xyz, target_voxel_mm, is_mask_image=False, verbose=False):
        if verbose:
            print("Spacing: ", org_spacing_xyz)
            print("Shape: ", images_zyx.shape)
    
        # print "Resizing dim z"
        resize_x = 1.0
        resize_y = float(org_spacing_xyz[2]) / float(target_voxel_mm)
        interpolation = cv2.INTER_NEAREST if is_mask_image else cv2.INTER_LINEAR
        res = cv2.resize(images_zyx, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)  # opencv assumes y, x, channels umpy array, so y = z pfff
        # print "Shape is now : ", res.shape
    
        res = res.swapaxes(0, 2)
        res = res.swapaxes(0, 1)
        # print "Shape: ", res.shape
        resize_x = float(org_spacing_xyz[0]) / float(target_voxel_mm)
        resize_y = float(org_spacing_xyz[1]) / float(target_voxel_mm)
    
        # cv2 can handle max 512 channels..
        if res.shape[2] > 512:
            res = res.swapaxes(0, 2)
            res1 = res[:256]
            res2 = res[256:]
            res1 = res1.swapaxes(0, 2)
            res2 = res2.swapaxes(0, 2)
            res1 = cv2.resize(res1, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)
            res2 = cv2.resize(res2, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)
            res1 = res1.swapaxes(0, 2)
            res2 = res2.swapaxes(0, 2)
            res = numpy.vstack([res1, res2])
            res = res.swapaxes(0, 2)
        else:
            res = cv2.resize(res, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)
    
        # channels = cv2.split(res)
        # resized_channels = []
        # for channel in  channels:
        #     channel = cv2.resize(channel, dsize=None, fx=resize_x, fy=resize_y)
        #     resized_channels.append(channel)
        # res = cv2.merge(resized_channels)
        # print "Shape after resize: ", res.shape
        res = res.swapaxes(0, 2)
        res = res.swapaxes(2, 1)
        if verbose:
            print("Shape after: ", res.shape)
        return res
    def anns_parser(self,anns_file_path):
        """annotation parser for csv file"""
        pass
    def vis_nodule(self):
        pass