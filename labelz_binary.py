import cv2
import numpy as np
from tqdm import tqdm
from scipy import misc
import os

def dirlist(path, allfile):  
    filelist =  os.listdir(path)  
    for filename in filelist:  
        filepath = os.path.join(path, filename)  
        if os.path.isdir(filepath):  
            dirlist(filepath, allfile)  
        else:  
            allfile.append(filepath)  
    return allfile

def gt_operate(pic):
	b_mask, g_mask, r_mask, a_mask = cv2.split(pic)
	ret, binary = cv2.threshold(a_mask, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
	#print("threshold value %s"%ret)
	pic_alpha = binary/255
	return pic_alpha

#size_limit = 200 #
img_srciptParentPath = "/home/igs/Documents/soiling_dataset-all/test/rgbImages/"
gt_srciptParentPath = "/home/igs/Documents/soiling_dataset-all/test/gtLabels/"
save_path = '/home/igs/Documents/soiling_dataset-all/test/binaryLabels/'
gt_results = dirlist(gt_srciptParentPath, [])

for img_results_single in gt_results:
	filename = img_results_single.split('/')[-1]
	gt = cv2.imread(img_results_single)
	gt[gt!=3]=0
	gt[gt==3]=255
	cv2.imwrite(save_path+filename,gt)


