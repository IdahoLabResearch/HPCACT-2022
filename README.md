# HPCACT-2022
The HPCACT-2022 dataset was created to help complement temporal action detection datasets by providing both normal
and anomalous activities where not all activities are human driven and where the same activity can be viewed from 
multiple camera angles and with varying levels of occlusion.  The HPCACT-2022 dataset incorporates elements from an
HPC datacenter setting where anomalies may arise from equipment or other non-human driven causes or an interaction
between multiple non-human driven causes.

**Note: To clone this repository, you will need to use git-lfs.**
e.g. git-lfs clone https://github.com/IdahoLabResearch/HPCACT-2022.git

Here is a list of the anomalous and normal activities represented in the dataset:

![Alt text](./img/list_activities.png?raw=true "List of anomalous and normal activities in the HPCACT-2022 dataset")

The dataset is broken into two sets of video clips:  anomalous and normal.  Here are some example frames from the dataset.

Credit for the annotations goes to:
Denver Conger, Ben Mahoney, Jacob Dickinson, Truman Hughes, Jaxon Lewis, Eva Roybal, Danielle Sleight, Miles Tudor
## Normal:

![Alt text](./img/chilled_door.png?raw=true "opening a chilled door")

![Alt text](./img/lift.png?raw=true "server lift")

## Anomalous:

![Alt text](./img/ladder.png?raw=true "ladder tipping over")

## Video Annotation Data Structure

This repository JSON files that can easily be converted to a YOLO labeling structure, specifically designed to make sense for activity labeled video data. Below is a description of the generated JSON data structure:
### JSON Structure

```json
{
    "video_id_1": {
        "filename": "path_to_video",
        "framesCount": 1000,
        "duration in seconds": 60,
        "calculated_fps": 16,
        "frames": {
            "frame_number_1": [
                {
                    "bounding_box": {
                        "x": 50.0,
                        "y": 50.0,
                        "width": 11.2,
                        "height": 12.3
                    },
                    "label": "activity"
                },
                {
                    "bounding_box": {
                        "x": 150,
                        "y": 150,
                        "width": 200,
                        "height": 200
                    },
                    "label": "activity"
                }
            ],
            "frame_number_2": [
                // Similar to frame_number_1 but for a different frame
            ]
            // Additional frames
        }
    },
    "video_id_2": {
        // Similar to video_id_1 but for a different video
    }
    // Additional video_ids
}
```

### Field Descriptions

- `video_id`: Unique identifier for the video.
    - `filename`: The local file path to the video.
    - `framesCount`: The total number of frames in the video.
    - `duration in seconds`: The duration of the video in seconds.
    - `calculated_fps`: The calculated frames-per-second (fps) for the video.
    - `frames`: A dictionary where the keys are the frame numbers and the values are lists of bounding boxes and labels for objects in that frame.
        - `frame_number`: The frame number within the video.
            - `bounding_box`: Contains the x, y coordinates and the width and height of the bounding box.
                - `x`: The x-coordinate of the top-left corner of the bounding box.
                - `y`: The y-coordinate of the top-left corner of the bounding box.
                - `width`: The width of the bounding box.
                - `height`: The height of the bounding box.
            - `label`: The label of the object within the bounding box.

The camera type used for all collection was an Amcrest 5 MP T1179EW. Each video was recorded at 2560 x 1920 at 20 frames per second.

Source code is provided for anomaly detection
and comparison against the IITB-Corridor dataset.

To Test the Corridor Trained OCNN model you will need to download the publicly available IITB-Corridor Dataset (instructions are here: https://rodrigues-royston.github.io/Multi-timescale_Trajectory_Prediction/).

1. To start you will need to convert the MP4 files from Corridor into a tfrecord file that iterates through each video and every 30 frames concatenates the video for a simple input into the Model that won't take up your entire memory.
This can be done by running tfr_from_mp4.py

2. The file labels_corridor.py will take the per frame annotations provided and create a label for each SITSR created in step 1. We provided a way to change the limits of the label seeing how it is a concat of 30 frames that may have different label values.

3. Inference

    This is done with the infer_quantize_model.py file.
    once you obtain the prediction to infer you must simply find the r value (the output) that splits the data the best by plotting or iterating through.
    That is the cutoff between your Normal and Anomolous predictions.

    The int8 model provided is trained on the corridor dataset and quantized on Vitis AI Docker image Version: 2.5.0.1260 and Git Hash: d2cd3d7eb.
    Please follow the steps provided at https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/Vitis-AI-Containers to install the container.

    Please use the tensorflow2 conda environment within the docker container.

    The model was quantized using the fast fine tune method following the steps provided in https://docs.xilinx.com/r/en-US/ug1414-vitis-ai

    To run this model you will need to be inside the Docker Container mentioned above and inside the conda environment tensorflow2.
    
    
   # Citation and Contact
If you use our dataset, please cite it:
```
@misc{
doecode_99634,
title = {Anomalous And Normal High Performance Computing Datacenter Activities},
author = {Anderson, Matthew and Sgambati, Matthew R. and Conger, Denver S. and Jacobson, Brendan G. and Petersen, Bryton J. and Biggs, Brandon S.},
abstractNote = {This code provides annotation and an interface to 10+ hours of video activities in a high performance datacenter with 20+ different types of anomalous activities. The purpose is to enable machine learning for video surveillance systems in high performance computing centers. This is the first code of this type addressing the space of high performance computing datacenters.},
doi = {10.11578/dc.20230125.2},
url = {https://doi.org/10.11578/dc.20230125.2},
howpublished = {[Computer Software] \url{https://doi.org/10.11578/dc.20230125.2}},
year = {2022},
month = {nov}
}
```
If you find our paper useful, please cite it:
```
@INPROCEEDINGS{,
  author={Conger, Denver and Anderson, Matthew and Sgambati, Matthew and Petersen, Bryton and Biggs, Brandon and Spencer, Damon},
  booktitle={}, 
  title={}, 
  year={},
  volume={},
  number={},
  pages={},
  doi={}}
```
For any questions or concerns, please contact `matthew.anderson2@inl.gov`
*** 
