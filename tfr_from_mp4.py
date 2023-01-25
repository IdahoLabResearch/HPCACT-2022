"""
Usage: python3 tfr_from_mp4.py
This converts each mp4 in a folder into a large tfrecord that holds all of our data as uint8.
This way we can train the models without loading in all the data at once.
you will need to set some ENV variables with the filepaths for the MP4's and where you want to save the tfrecord
For example 
TRAINING_PATH  could be /IITB_Corridor_mp4/Test/
SAVE_FILE    could be /IITB_Corridor_tfrecord/corridor_test.tfrecords

"""

import cv2
import numpy as np
import os
import pandas as pd
import signal
import sys
import tensorflow as tf
from tqdm import tqdm
import itertools
from glob import glob

def list_files(filepath, filetype):
    paths = []
    for root, dirs, files in os.walk(filepath):
        for file in files:
            if file.lower().endswith(filetype.lower()):
                paths.append(os.path.join(root, file))
    paths.sort()
    return(paths)

def write_tfrecords(writer, img_stack):
    '''
    adds an image to a tf record file
    args:
        writer: the tf record writer. This is should be initialized with the desired path to the file
        img: the image_stack that will be written to the tf record file. This should be a (30, 512, 512) array and should
        be in grayscale. It should also be in uint8 format to help save space and speed things up.
    '''
    
    example = tf.train.Example(features=tf.train.Features(
        feature={
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_stack.tobytes()])),
            'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_stack.tobytes()])),
        }))
    writer.write(example.SerializeToString())

    
def iterate_through_video(path, writer, sampling):

    # define the video capture object
    vid = cv2.VideoCapture(path)
    
    #print(f'fps for datacenter = 20. So, sampling set to be every {sampling} frames.') 
    q_f = []
    
    #read in the frame
    more, frame = vid.read()

    while(more):
        #downsample if frame rate is too high
        #do necessary image augmentation
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (512,512))
        frame = np.reshape(frame, (512, 512))
        q_f.append(frame)
        
        """
        We iterate every 30 frames and concat the 30 with a sampling rate of 10.
        We iterate every 30 frames and concat the 30 with a sample taken every 10 frames 
        I.E we capture 30 frames every 10 frames so each SITSR is 2/3 the same as the last.
        It would be way too much data to sample every frame.
        """

        if len(q_f) == 30:
            write_tfrecords(writer, np.array(q_f).astype(np.uint8))
            q_f = q_f[sampling:]

        more, frame = vid.read()
    
if __name__ == "__main__":
    sampling = 10
    train_path = os.getenv('TRAINING_PATH')
    save_file = os.getenv('SAVE_FILE')

    paths = list_files(train_path, '.mp4')

    print('starting')
    writer = tf.io.TFRecordWriter(save_file)
    for path in paths:
        print(path)
        iterate_through_video(path, writer, sampling)

    print('\nConversion to tfrecord complete')
