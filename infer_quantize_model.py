
"""
Once you have processed the dataset accordingly you will need to run this file in the Docker Container Vitis A.I 2.5 with the tensorflow2 conda env

This does not save the prediction but can be saved as a npy array and than the AUC score can be computed compared to the truth values created by labels_corridor.py

"""


from tensorflow_model_optimization.quantization.keras import vitis_quantize
import keras
import keras.models as models
import tensorflow as tf
import os
import pandas as pd
from sklearn.metrics import classification_report
from sklearn import metrics
labels_csv = '/projects/activity_detection/CVPR_2023/supplementary/corridor_labels_limit_0_30.csv'
filename = "/projects/activity_detection/CVPR_2023/supplementary/corridor_test.tfrecords"
quantized_model = models.load_model('/projects/activity_detection/CVPR_2023/supplementary/quant_corridor_ocnn_SITSR_model.h5')
quantized_model.compile(loss='mse')
image_feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.string),
}

def _parse_image_function(example_proto):
    """
    Parse one tf.train.Example at a time - perform any transforms to the data here
    """
    features = tf.io.parse_single_example(example_proto, image_feature_description)
    image = tf.io.decode_raw(features['image'], tf.uint8)
    # image = tf.image.convert_image_dtype(image_int8, dtype=tf.uint8)
    image.set_shape([30 * 512 * 512])
    
    # image = tf.reshape(image, [30,512, 512])
    image = tf.reshape(image, [30,512, 512])
    
    #the transpose is necceasry due to the FPGA's implementation only allowing CONV2D with channels last within the encoder
    image = tf.transpose(image, [1, 2, 0])
    

    #when compiling ocnn model
    # label = tf.constant([1])
    
    #when compiling encoder only
    # label = image

    return image
def read_dataset(batch_size, filename):
    """
    Shuffle the data and extract by batch size
    """
    dataset = tf.data.TFRecordDataset(filename)

    dataset = dataset.map(_parse_image_function, num_parallel_calls=os.cpu_count()-4)
    dataset = dataset.prefetch(buffer_size = batch_size*4)
    dataset = dataset.batch(batch_size)

    return dataset



# CONVERT THE OCNN or the encoder if that is what we are moving

batch_size = 4
test_dataset = read_dataset(batch_size=batch_size, filename="/projects/activity_detection/CVPR_2023/supplementary/corridor_test.tfrecords")
print("starting predictions")
pred = quantized_model.predict(test_dataset)
print("done predicting")
pred = pred.flatten()

pred_norm = []
pred_anom = []
def seperate_results(df):
    count  = 0
    for true_label in zip(df['labels']):
        # print(true_label)
        true_label = true_label[0]
        if true_label != 0:
            if true_label == 1:
                pred_norm.append(pred[count])
            if true_label == -1:
                pred_anom.append(pred[count])
        count = count + 1
seperate_results(pd.read_csv(labels_csv))


w = 0
bestr = 0

"""
This is my slightly roundabout and slow way of finding the best r value. I didnt want to make anything fancy
"""
for r in range(600,800,1):
    # r = r/1000
    s_n = [pred_norm[i] -r >= 0 for i in range(len(pred_norm))]
    # print(s_n)
    frac_of_outliers = len([s for s in s_n if s == 0]) / len(s_n)


    # Simple Anomolous set Outlier prediction
    s_n = [pred_anom[i] -r >= 0 for i in range(len(pred_anom))]
    # print(s_n)
    frac_of_outliers2 = len([s for s in s_n if s == 0]) / len(s_n)
    if ((frac_of_outliers2 - (frac_of_outliers))>w):
        w = frac_of_outliers2 - frac_of_outliers
        bestr = r
r = bestr
print("The best r value is: ",r)
y_predicted = []
i = 0
for i in range(pred.shape[0]):
    if pred[i] >= r:
        y_predicted.append(1) # Normal
#         y_predicted.append(-1) # TEST - Normal
    elif pred[i] < r:
        y_predicted.append(-1) # Anom
#         y_predicted.append(1) # TEST - Anom
#         y_predicted.append(0)
    else:
        print('prediction failed')

def get_results(df):
    predicted = []
    truth = []
    for true_label, predicted_label in zip(df['labels'], y_predicted):
        if true_label != 0:
            predicted.append(predicted_label)
            truth.append(true_label)

    report = classification_report(truth, predicted)
    print(report)
    fpr, tpr, thresholds = metrics.roc_curve(truth, predicted)


    auc = metrics.auc(fpr, tpr)
    print("AUC score is: ",auc)
    

labels = pd.read_csv(labels_csv)
get_results(labels)