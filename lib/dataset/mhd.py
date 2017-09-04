import ntpath

import numpy as np
import SimpleITK as sitk
import data_utils
import cv2

from collections import defaultdict
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import disk, dilation, binary_erosion, binary_closing
from skimage.filters import roberts, sobel
from scipy import ndimage as ndi
from glob import glob


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
        img_array = self.rescale_patient_images(img_array,spacing,1.00,verbose=True)
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
    
        print "Resizing dim z"
        resize_x = 1.0
        resize_y = float(org_spacing_xyz[2]) / float(target_voxel_mm)
        interpolation = cv2.INTER_NEAREST if is_mask_image else cv2.INTER_LINEAR
        res = cv2.resize(images_zyx, dsize=None, fx=resize_x, fy=resize_y, interpolation=interpolation)  # opencv assumes y, x, channels umpy array, so y = z pfff
        print "Shape is now : ", res.shape
    
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
            res = np.vstack([res1, res2])
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
    def get_segmented_lungs(self,im, plot=False):
        """get the lung segmentation"""
        # Step 1: Convert into a binary image.
        binary = im < -400
        # Step 2: Remove the blobs connected to the border of the image.
        cleared = clear_border(binary)
        # Step 3: Label the image.
        label_image = label(cleared)
        # Step 4: Keep the labels with 2 largest areas.
        areas = [r.area for r in regionprops(label_image)]
        areas.sort()
        if len(areas) > 2:
            for region in regionprops(label_image):
                if region.area < areas[-2]:
                    for coordinates in region.coords:
                           label_image[coordinates[0], coordinates[1]] = 0
        binary = label_image > 0
        # Step 5: Erosion operation with a disk of radius 2. This operation is seperate the lung nodules attached to the blood vessels.
        selem = disk(2)
        binary = binary_erosion(binary, selem)
        # Step 6: Closure operation with a disk of radius 10. This operation is    to keep nodules attached to the lung wall.
        selem = disk(10) # CHANGE BACK TO 10
        binary = binary_closing(binary, selem)
        # Step 7: Fill in the small holes inside the binary mask of lungs.
        edges = roberts(binary)
        binary = ndi.binary_fill_holes(edges)
        # Step 8: Superimpose the binary mask on the input image.
        get_high_vals = binary == 0
        im[get_high_vals] = -2000
        return im, binary
    def load_patient_images(self,patient_id,base_dir):
        src_dir = base_dir + '/' + patient_id + "/"
        src_img_paths = glob(src_dir+'*.png')
        src_img_paths.sort()
        images = [cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) for img_path in src_img_paths]
        images = [im.reshape((1, ) + im.shape) for im in images]
        res = np.vstack(images)
        return res
    def anns_parser(self,anns_file_path):
        """annotation parser for csv file"""
        pass
        def vis_nodule(self,img,records):
        """visualize nodules in img for debug usage
        :prams img CT image(after preprocess),order(z,y,x)
        :records index,coord_x,coord_y,coord_z,diameter,mal_score
        """
        for rec in records:
            cen_x = int(rec[1] * img.shape[2])
            cen_y = int(rec[2] * img.shape[1])
            cen_z = int(rec[3] * img.shape[0])
            diam = int(rec[4] * img.shape[1])
            for i in range(cen_z - diam/2,cen_z + diam/2):
                ax = plt.gca()
                plt.figure()
                im = img[i,:,:]
                ax.imshow(im,cmap=plt.cm.gray)
                gt_box = Rectangle((cen_x - diam/2,cen_y - diam/2),diam+10,diam+10,fill=False,edgecolor='red',linewidth=2)
                ax.add_patch(gt_box)
                plt.show()
    