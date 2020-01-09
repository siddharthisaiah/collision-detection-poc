'''
importing the necessory depedencies. there are many modules like collections and utils which need to call from object
detection folder so dont forget to save the code in that folder.
'''
 
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow.compat.v1 as tf
import zipfile
from collections import defaultdict
from io import StringIO
from PIL import Image
import cv2
from utils import label_map_util
from utils import visualization_utils as vis_util
from input_video import video_name

# Name of video to use as input: all input videos must be placed in the `videos` folder
INPUT_VIDEO_PATH = os.path.join("../videos", video_name)

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = "./ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb"
 
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = "./data/mscoco_label_map.pbtxt"
 
NUM_CLASSES = 90
 
# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
 
 
# Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
 
 
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    cap=cv2.VideoCapture(INPUT_VIDEO_PATH)
    out = cv2.VideoWriter('./output/collision_detection.mp4',0, 30.0, (640,480))
    while(cap.isOpened()):
      ret, image_np = cap.read()
      if ret==True:
        image_np=cv2.resize(image_np,(420,220))
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
         
        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(image_np, np.squeeze(boxes),np.squeeze(classes).astype(np.int32),np.squeeze(scores),category_index,use_normalized_coordinates=True,line_thickness=1,skip_scores=True)
 

        for i,b in enumerate(boxes[0]):
          if classes[0][i] == 3 or classes[0][i] == 6 or classes[0][i] == 8:
            if scores[0][i] >= 0.5:
              mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
              mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
              apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**2),1)
              cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*420),int(mid_y*220)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
 
              if apx_distance <=0.4:
                if mid_x > 0.2 and mid_x < 0.8:
                  cv2.putText(image_np, 'WARNING!!!', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
         
        # out.write(image_np)
        image_np1=cv2.resize(image_np,(420,220))

        cv2.imshow('window',image_np1)

        if cv2.waitKey(25) & 0xFF == ord('q'):
             
            break
      else:
        break
 
 
cap.release()
out.release()
cv2.destroyAllWindows()
