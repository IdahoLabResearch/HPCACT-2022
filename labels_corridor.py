"""
This file's purpose is to take the Corridor Test dataset which is labeled per frame and to label it per 30 frames.
This is done by taking the annotations provided and iterating through every 30 frames we allso provide a
changing likmit to change the ratio of normal to anomolous frames within the SITSR to allow for a varying
ratio that we can use to see how well the SITSR's do with mixed SITSR's

We also sample every 10 frames as in we capture 30 frames every 10 frames so each SITSR is 2/3 the same as the last.
It would be way too much data to sample every frame.

"""


import json
import math
import pandas as pd
import cv2
import os
import numpy as np





test_mp4_location = "/projects/activity_detection/biggbs/IITB_Corridor_mp4/Test"
IITB_Corridor_location = "/projects/activity_detection/biggbs"
filepath_save_to = "/projects/activity_detection/CVPR_2023/supplementary"



def list_files(filepath, filetype):
    paths = []
    for root, dirs, files in os.walk(filepath):
        for file in files:
            if file.lower().endswith(filetype.lower()):
                paths.append(os.path.join(root, file))
    paths.sort()
    return(paths)

paths = list_files(test_mp4_location, '.mp4')

frames = {}
for path in paths:
    cap = cv2.VideoCapture(path)
    totalframecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    file = os.path.basename(path).split(".")[0]
    frame_label = []
    for frame in range(0, totalframecount):
        frame_label.append(0)
    frames[file] = frame_label
    
    
citsr = {}
count = 0
for key,value in frames.items():
    frame_labels = np.load(f'{IITB_Corridor_location}/IITB_Corridor/Test_IITB_Corridor/Test_Annotation/Annotation/{key}/{key}.npy')
    if frame_labels[-1] == 0:
        frame_labels = np.append(frame_labels, 0)
    else:
        frame_labels = np.append(frame_labels, 1)
    citsr[key] = {}
#     print(key)
    for frame_index in range(0, len(frame_labels), 10):
        citsr_val = frame_labels[frame_index:frame_index+30]
        if len(citsr_val) == 30:
#             citsr[key][int(frame_index/10)] = np.count_nonzero(citsr_val == 1)/30
            citsr[key][int(frame_index/10)] = np.count_nonzero(citsr_val == 1)
            count += 1
        
        
# The second index is the amount of normal vids per SITSR
# limits = [[0, 30]]
# limits = [[0, 30]]


limits = [[0, 30]]
count = 0
one_count = 0
limit_of_anom = 6823
# limit_of_anom = 20000
for limit in limits:
    citsr_labels = []
    for key, val in citsr.items():
        for num, label in val.items():
            if label >= limit[1]:
                if count >= limit_of_anom:
                    citsr_labels.append(0)
                else:
                    citsr_labels.append(-1) # anom
                count += 1
            elif label <= limit[0]:
                citsr_labels.append(1) # Normal
            else:
                citsr_labels.append(0)
                one_count += 1


    data = {
        'labels': citsr_labels,
    }
    df = pd.DataFrame(data)
    df.to_csv(f"{filepath_save_to}/corridor_labels_limit_{limit[0]}_{limit[1]}.csv")