import os
import dicom #pip install dicom
import numpy as np
from dcm import DCM

class DSB3():
    def __init__(self,data_path,ann_path,result_path,mhd=True,img=True):
        self.data_path = data_path
        self.ann_path = ann_path
        self.result_path = result_path
        self.img = img
        if self.img:
            self.img_path = os.path.join(self.result_path,'img','data')
            self.seg_img_path = os.path.join(self.result_path,'img','seg')
        self.mhd = mhd
        if self.mhd:
            self.mhd_path = os.path.join(self.result_path,'mhd','data')
            self.seg_mhd_path = os.path.join(self.result_path,'mhd','seg')
        self.attrs = dict()
        
    def get_img(self):
        print('extracting imgs from dsb3 dataset')
        