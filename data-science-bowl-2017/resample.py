#!/usr/bin/python

import sys

INPUT_FOLDER = '/Volumes/MS-1TB/Data/stage1/'
#INPUT_FOLDER = './sample_images/'

import numpy as np
import pandas as pd
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

patients = os.listdir(INPUT_FOLDER)
patients.sort()

def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


'''
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.int16)
    
    image[image == -2000] = 0
    
    for slice_number in range(len(slices)):
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
    
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
        
    return np.array(image, dtype=np.int16)

def resample(image, scan, new_spacing=[1,1,1]):
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)
    
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
#     return image, new_spacing
    return image


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)
    
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    
    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None     


def segment_lung_mask(image, fill_lung_structures=True):
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    background_label = labels[0,0,0]
    
    binary_image[background_label == labels] = 2
    
    if fill_lung_structures:
        for i, axial_slice in enumerate(binary_image):
            axial_slicee = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None:
                binary_image[i][labeling != l_max] = 1
                
                
    binary_image -= 1
    binary_image = 1 - binary_image
    
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    
    if l_max is not None:
        binary_image[labels != l_max] = 0
    
    return binary_image

'''
start = int(sys.argv[1])
end = int(sys.argv[2])

#print(start)
#print(end)

# scans = [load_scan(INPUT_FOLDER + p) for p in patients]
scans = []
for i in range(start, end):
    scans.append(load_scan(INPUT_FOLDER + patients[i]))

for scan in scans:
    k = len(scan)
    patientname = scan[0].PatientsName
    print(patientname)
    for i in range(0, k):
        plt.figure(figsize=(3,3))
        plt.axes([0,0,1,1])
        plt.axis('off')
        plt.imshow(scan[i].pixel_array, cmap=plt.cm.gray)
        filename = "{0:0=3d}".format(i)
        plt.savefig("stage1_data/" + patientname + "/" + filename + ".jpg",bbox_inches=0)
        plt.close()

'''
scan_pixels = [get_pixels_hu(s) for s in scans]

scan_pixels_resampled = []
n = len(scans)
for i in range(0, n):
    scan_pixels_resampled.append(resample(scan_pixels[i], scans[i]))


print("Shape before resampling\t", scan_pixels_resampled[3].shape)
print("Shape after resampling\t", scan_pixels_resampled[3].shape)

scan_pixels_seg_lungs = [segment_lung_mask(sr, False) for sr in scan_pixels_resampled]

scan_pixels_seg_lungs_fill = [segment_lung_mask(sr, True) for sr in scan_pixels_resampled]
'''

print("Finished!")
