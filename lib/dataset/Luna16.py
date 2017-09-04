"""
Luna16 Dataset 
"""
import os
import cv2
import dicom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mhd import MHD 
import multiprocessing 
from glob import glob
import SimpleITK as sitk
from dicom.dataset import Dataset,FileDataset
from matplotlib.patches import Polygon,Rectangle


class Luna16(MHD):
    def __init__(self,data_path,anns_path,segment_path,result_path,mhd=True,img=True):
        """
        params:data_path:path to luna16 raw data path
        segment_path:path to luna16 lung segment file
        anns_path:path to luna16 annotations file
        result_path: path to prorcessed imgs
        subsetdirs: subset0-subset9 used for corss validation
        label_path: path to the preprocessed labels
        img_path: path to the preprocessed imgs
        """
        #super(Luna16,self).__init__("Luna16")
        self.data_path = data_path
        self.segment_path = segment_path
        self.anns_path = anns_path
        self.result_path = result_path
        self.subsetdirs = os.listdir(self.data_path)
        self.label_path = os.path.join(self.result_path,'luna_label')
        if img:
            self.img = True
            self.img_path = os.path.join(self.result_path,'img','data')
            self.seg_img_path = os.path.join(self.result_path,'img','seg')
        if mhd:
            self.mhd = True
            self.mhd_path = os.path.join(self.result_path,'mhd','data')
            self.seg_mhd_path = os.path.join(self.result_path,'mhd','seg')
        self.attrs = dict()
        
    def get_img(self,num_threads=None,img=True,dcm=True):
        print("Extracting images from LUNA16 Dataset")
        
        for subdir in self.subsetdirs:
            src_dir = os.path.join(self.data_path,subdir)
            src_paths = glob(src_dir+'/*.mhd')
        
            if num_threads:
                pool = multiprocessing.pool(num_threads)
                pool.map(self.process_image,src_paths)
                
            else:
                for src_path in src_paths:
                    print("src Path: ",src_path)
                    img_array,patient_id = self.process_image(src_path)
                    if self.img:
                        #print(self.img_path)
                        #dst_dir = os.path.join(self.img_path,str(patient_id))
                        dst_dir = self.img_path + '/' + str(patient_id) + '/'
                        self.writetoimg(img_array,dst_dir)
                    if self.mhd:
                        dst_dir = self.mhd_path + '/' + str(patient_id)
                        self.writetomhd(img_array,dst_dir)
    def parse_pos_anns(self):
        """get all the nodule annotations for luna16"""
        print("parsing annotations for LUNA16 Dataset")
        
        for subdir in self.subsetdirs:
            src_dir = os.path.join(self.data_path,subdir)
            src_paths = glob(src_dir+'/*.mhd')
            for src_path in src_paths:
                print('src_path:',src_path)
                patient_id = src_path.split('/')[-1].replace('.mhd','')
                print(patient_id)
                self.process_pos_annotations_patient(src_path,patient_id)
                        
    def cube_builder(self,cube_size,perc):
        """
        build 3d cubes for training
        params:cube_size:size of cube,e.g.:48*48*48
        perc: The percentage of cubes that don't contains any nodule
        """
        pass
    
    
        
    def to_npy(self,img):
        """save the preprocess data to npy file"""
        pass
    
    def writetoimg(self,img_array,dst_dir):
        """
        params:
            img_array:shape(z,x,y)
        """
        print('write images to:',dst_dir)
        #dst_path = os.path.join(self.img_path,patient_id)
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        for i in range(img_array.shape[0]):
            img = img_array[i]
            cv2.imwrite(dst_dir + 'img_'+str(i).rjust(4,'0') + ".png",img)
    def writetomhd(self,img_array,dst_dir):
        """
        write the prepocess data to dcm file for easy vis(using itk-snap)
        params:
            img_array:shape(z,x,y)"""
        
        print('write dcm to:',dst_dir)
        
        #dst_path = os.path.join(self.dcm_path,patient_id)
        mhd_file = sitk.GetImageFromArray(img_array)
        #.raw file will generate automatically
        sitk.WriteImage(mhd_file,dst_dir+'.mhd')
    
    
    def get_lung_seg(self):
        """get the lung segmention"""
        
        seg_files = glob(self.segment_path+'/*.mhd')
        for seg_file in seg_files:
            patient_id = seg_file.split('/')[-1].replace('.mhd','')
            print(patient_id)
            seg_im = sitk.ReadImage(seg_file)
            seg_array = sitk.GetArrayFromImage(seg_im)
            #left lung is 3
            seg_array[seg_array==3] = 1
            #right lung is 4
            seg_array[seg_array==4] = 1
            seg_array[seg_array!=1] = 0
            if self.img:
                #dst_dir = os.path.join(self.seg_img_path,patient_id)
                print(self.seg_img_path)
                dst_dir = self.seg_img_path + '/' + str(patient_id) + '/'
                #print(dst_dir)
                self.writetoimg(seg_array,dst_dir)
            if self.mhd:
                dst_dir = self.seg_mhd_path + '/' + str(patient_id)
                self.writetomhd(seg_array,dst_dir)
                
    

        
                
        
    def process_pos_annotations_patient(self,src_path, patient_id,vis=False):
        """get positive nodule annotations,positive nodules was labeled by 3 doctors""" 
        df_node = pd.read_csv(self.anns_path)
    
        itk_img = sitk.ReadImage(src_path)
        img_array = sitk.GetArrayFromImage(itk_img)
        print("Img array: ", img_array.shape)
        df_patient = df_node[df_node["seriesuid"] == patient_id]
        print("Annos: ", len(df_patient))
    
        num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
        origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
        print("Origin (x,y,z): ", origin)
        spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
        print("Spacing (x,y,z): ", spacing)
        rescale = spacing / 1.
        print("Rescale: ", rescale)
    
        direction = np.array(itk_img.GetDirection())      # x,y,z  Origin in world coordinates (mm)
        print("Direction: ", direction)
        flip_direction_x = False
        flip_direction_y = False
        if round(direction[0]) == -1:
            origin[0] *= -1
            direction[0] = 1
            flip_direction_x = True
            print("Swappint x origin")
        if round(direction[4]) == -1:
            origin[1] *= -1
            direction[4] = 1
            flip_direction_y = True
            print("Swappint y origin")
        print("Direction: ", direction)
        assert abs(sum(direction) - 3) < 0.01
    
        patient_imgs = self.load_patient_images(patient_id,self.img_path)
    
        pos_annos = []
        df_patient = df_node[df_node["seriesuid"] == patient_id]
        #print(df_patient)
        anno_index = 0
        for index, annotation in df_patient.iterrows():
            node_x = annotation["coordX"]
            if flip_direction_x:
                node_x *= -1
            node_y = annotation["coordY"]
            if flip_direction_y:
                node_y *= -1
            node_z = annotation["coordZ"]
            diam_mm = annotation["diameter_mm"]
            print("Node org (x,y,z,diam): ", (round(node_x, 2), round(node_y, 2), round(node_z, 2), round(diam_mm, 2)))
            center_float = np.array([node_x, node_y, node_z])
            center_int = np.rint((center_float-origin) / spacing)
            # center_int = numpy.rint((center_float - origin) )
            print("Node tra (x,y,z,diam): ", (center_int[0], center_int[1], center_int[2]))
            # center_int_rescaled = numpy.rint(((center_float-origin) / spacing) * rescale)
            center_float_rescaled = (center_float - origin) / 1.
            center_float_percent = center_float_rescaled / patient_imgs.swapaxes(0, 2).shape
            # center_int = numpy.rint((center_float - origin) )
            print("Node sca (x,y,z,diam): ", (center_float_rescaled[0], center_float_rescaled[1], center_float_rescaled[2]))
            diameter_pixels = diam_mm / 1.
            diameter_percent = diameter_pixels / float(patient_imgs.shape[1])
    
            pos_annos.append([anno_index, round(center_float_percent[0], 4), round(center_float_percent[1], 4), round(center_float_percent[2], 4), round(diameter_percent, 4), 1])
            anno_index += 1
        if vis:
            self.vis_nodule(patient_imgs,pos_annos)
        df_annos = pd.DataFrame(pos_annos, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"])
        df_annos.to_csv(self.label_path+'/'+str(patient_id) + "_annos_pos.csv", index=False)
        return [patient_id, spacing[0], spacing[1], spacing[2]]

        
    
if __name__=="__main__":
    data_path = '/home/genonova/luna16'
    anns_path = '/home/genonova/luna16/annotations.csv'
    segment_path = '/home/genonova/seg-lungs-LUNA16'
    result_path = '/home/genonova/data'
    dataset= Luna16(data_path,anns_path,segment_path,result_path)
    #dataset.get_img()
    dataset.parse_pos_anns()
    #dataset.get_lung_seg()