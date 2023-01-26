# HPCACT-2022
The HPCACT-2022 dataset was created to help complement temporal action detection datasets by providing both normal
and anomalous activities where not all activities are human driven and where the same activity can be viewed from 
multiple camera angles and with varying levels of occlusion.  The HPCACT-2022 dataset incorporates elements from an
HPC datacenter setting where anomalies may arise from equipment or other non-human driven causes or an interaction
between multiple non-human driven causes..

Here is a list of the anomalous and normal activities represented in the dataset:

![Alt text](./img/list_activities.png?raw=true "List of anomalous and normal activities in the HPCACT-2022 dataset")

The dataset is broken into two sets of video clips:  anomalous and normal.  Here are some example frames from the dataset.

## Normal:

![Alt text](./img/chilled_door.png?raw=true "opening a chilled door")

![Alt text](./img/lift.png?raw=true "server lift")

## Anomalous:

![Alt text](./img/ladder.png?raw=true "ladder tipping over")

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
