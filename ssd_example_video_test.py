import time
import cv2
from sense_hat import SenseHat
import tflite_runtime.interpreter as tflite

#Define colors for Raspberry Pi SenseHat LED matrix

red =(255,0,0)
yellow=(255,255,0)
green = (0,255,0)
blue = (0,0,255)
grey = (0,0,0)

import numpy as np

label2string = \
{
    0:   "person",
    1:   "bicycle",
    2:   "car",
    3:   "motorcycle",
    4:   "airplane",
    5:   "bus",
    6:   "train",
    7:   "truck",
    8:   "boat",
    9:   "traffic light",
    10:  "fire hydrant",
    12:  "stop sign",
    13:  "parking meter",
    14:  "bench",
    15:  "bird",
    16:  "cat",
    17:  "dog",
    18:  "horse",
    19:  "sheep",
    20:  "cow",
    21:  "elephant",
    22:  "bear",
    23:  "zebra",
    24:  "giraffe",
    26:  "backpack",
    27:  "umbrella",
    30:  "handbag",
    31:  "tie",
    32:  "suitcase",
    33:  "frisbee",
    34:  "skis",
    35:  "snowboard",
    36:  "sports ball",
    37:  "kite",
    38:  "baseball bat",
    39:  "baseball glove",
    40:  "skateboard",
    41:  "surfboard",
    42:  "tennis racket",
    43:  "bottle",
    45:  "wine glass",
    46:  "cup",
    47:  "fork",
    48:  "knife",
    49:  "spoon",
    50:  "bowl",
    51:  "banana",
    52:  "apple",
    53:  "sandwich",
    54:  "orange",
    55:  "broccoli",
    56:  "carrot",
    57:  "hot dog",
    58:  "pizza",
    59:  "donut",
    60:  "cake",
    61:  "chair",
    62:  "couch",
    63:  "potted plant",
    64:  "bed",
    66:  "dining table",
    69:  "toilet",
    71:  "tv",
    72:  "laptop",
    73:  "mouse",
    74:  "remote",
    75:  "keyboard",
    76:  "cell phone",
    77:  "microwave",
    78:  "oven",
    79:  "toaster",
    80:  "sink",
    81:  "refrigerator",
    83:  "book",
    84:  "clock",
    85:  "vase",
    86:  "scissors",
    87:  "teddy bear",
    88:  "hair drier",
    89:  "toothbrush",
}


def detect_from_camera():
    #print ('Inside loop of detect from camera')
    sense = SenseHat()
    sense.clear()

    cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit('ERROR: Unable to read from webcam. Please verify your webcam settings.')

    # prepare input image
        start = time.time()
        img_org = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img_org, (300, 300))
        img = img.reshape(1, img.shape[0], img.shape[1],
                      img.shape[2])  # (1, 300, 300, 3)
        img = np.asarray(img)
        img = img.astype(np.uint8)
        #print ('img', img)

    # Overview of Object Detection: https://www.tensorflow.org/lite/examples/object_detection/overview
    # Load pretrained model:https://tfhub.dev/tensorflow/lite-model/ssd_mobilenet_v1/1/default/1
     
        interpreter = tflite.Interpreter(
            model_path="/home/pi/DAC-Demo/ssd_mobilenet_v1.tflite")
        
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
    
    # set input tensor
        interpreter.set_tensor(input_details[0]['index'], img)

    # run
        start = time.time()
        interpreter.invoke()
        stop = time.time()
    # get output tensor
   
        boxes = interpreter.get_tensor(output_details[0]['index'])
        boxes_shape = output_details[0]['shape_signature']
        labels = interpreter.get_tensor(output_details[1]['index'])
        scores = interpreter.get_tensor(output_details[2]['index'])
        num = interpreter.get_tensor(output_details[3]['index'])
        #print('Boxes:', boxes)
        #print('Boxes_shape:', boxes_shape)
        print('Labels', labels)
        #print('Labels', label2string[labels[0][0]])
        #print('Score:', scores)
        #print('Num:', num)
		
        #Detect Car
        if (2 in labels):   #
            sense.set_pixel(2,0,blue)
            sense.set_pixel(1,0,blue)
            sense.set_pixel(0,0,blue)
            sense.set_pixel(0,1,blue)
            sense.set_pixel(0,2,blue)
            sense.set_pixel(1,2,blue)
            sense.set_pixel(2,2,blue)
        else:
            sense.set_pixel(2,0,grey)
            sense.set_pixel(1,0,grey)
            sense.set_pixel(0,0,grey)
            sense.set_pixel(0,1,grey)
            sense.set_pixel(0,2,grey)
            sense.set_pixel(1,2,grey)
            sense.set_pixel(2,2,grey)
        #Detect Bicycle
        if ((1 in labels) or (3 in labels)):
            sense.set_pixel(0,4,blue)
            sense.set_pixel(0,5,blue)
            sense.set_pixel(0,6,blue)
            sense.set_pixel(0,7,blue)
            sense.set_pixel(1,6,blue)
            sense.set_pixel(2,6,blue)
            sense.set_pixel(1,7,blue)
            sense.set_pixel(2,7,blue)
        else:
            sense.set_pixel(0,4,grey)
            sense.set_pixel(0,5,grey)
            sense.set_pixel(0,6,grey)
            sense.set_pixel(0,7,grey)
            sense.set_pixel(1,6,grey)
            sense.set_pixel(2,6,grey)
            sense.set_pixel(1,7,grey)
            sense.set_pixel(2,7,grey)
        
            #Control light (Red or Green)

        if ((1 in labels) or (2 in labels) or (3 in labels)):
            sense.set_pixel(4,6,green)
            sense.set_pixel(4,4,grey)
        else:
            sense.set_pixel(4,4,red)
            sense.set_pixel(4,6,grey)

        ''' # Comment out section saving an image with bounding boxes

        for i in range(boxes.shape[1]):
            if scores[0, i] > 0.50:
                box = boxes[0, i, :]
                x0 = int(box[1] * img_org.shape[1])
                y0 = int(box[0] * img_org.shape[0])
                x1 = int(box[3] * img_org.shape[1])
                y1 = int(box[2] * img_org.shape[0])
                box = box.astype(np.int32)
                cv2.rectangle(img_org, (x0, y0), (x1, y1), (255, 0, 0), 2)
            #cv2.rectangle(img_org, (x0, y0), (x0 + 100, y0 - 30), (255, 0, 0), -1)
                cv2.putText(img_org, str(label2string[labels[0][i]]), (x0, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imwrite('object-detected.jpg', img_org)
        #print(f'time for inference is {stop-start:.2f} seconds')'''
    cap.release()


if __name__ == '__main__':
    detect_from_camera()
    
