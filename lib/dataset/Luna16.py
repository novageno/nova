"""
Luna16 Dataset 
"""
import os
import cv2
import dicom
import numpy as np
import pandas as pd
from mhd import MHD 
import multiprocessing 
from glob import glob
import SimpleITK as sitk
from dicom.dataset import Dataset,FileDataset

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
        seg_files = glob(self.segment_path+'*.mhd')
        for seg_file in seg_files:
            patient_id = seg_file.replace('.mhd','')
            seg_im = sitk.ReadImage(seg_file)
            seg_array = sitk.GetArrayFromImage(seg_im)
            #left lung is 3
            seg_array[seg_array==3] = 1
            #right lung is 4
            seg_array[seg_array==4] = 1
            seg_array[seg_array!=1] = 0
            if self.img:
                dst_dir = os.path.join(self.seg_img_path,patient_id)
                self.writetoimg(seg_array,dst_dir)
            if self.mhd:
                dst_dir = self.seg_img_path + str(patient_id)
                self.writetomhd(seg_array,dst_dir)
                
    
    def gt_roidb(self):
        """get the ground-truth roidb"""
        pass
    
if __name__=="__main__":
    data_path = '/home/genonova/luna16'
    anns_path = None
    segment_path = '/home/genonova/seg-lungs-LUNA16'
    result_path = '/home/genonova/data'
    dataset= Luna16(data_path,anns_path,segment_path,result_path)
    dataset.get_img()
    dataset.get_lung_seg()