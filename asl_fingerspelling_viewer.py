import os
import shutil
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import json
import mediapipe
import matplotlib
import matplotlib.pyplot as plt
import random

import cv2

from mediapipe.framework.formats import landmark_pb2

from matplotlib import animation, rc

print("Mediapipe v" + mediapipe.__version__)

mp_pose = mediapipe.solutions.pose
mp_hands = mediapipe.solutions.hands
mp_face = mediapipe.solutions.face_mesh
mp_drawing = mediapipe.solutions.drawing_utils 
mp_drawing_styles = mediapipe.solutions.drawing_styles

# Reference:
#   https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/face_mesh.py
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LEFT_EYE
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_LIPS
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_NOSE
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_RIGHT_EYE

# Top 130 landmarks
#
# Reference 1:
#   Isolated Sign Language : 1st place kept landmarks
#   https://github.com/hoyso48/Google---Isolated-Sign-Language-Recognition-1st-place-solution/blob/main/ISLR_1st_place_Hoyeol_Sohn.ipynb   
#
# Reference 2: 
#   Fingerspelling : 1st place kept landmarks
#   https://www.kaggle.com/code/darraghdog/asl-fingerspelling-preprocessing-train/notebook
#   input landmark concatenation : FACE+LHAND+POSE+RHAND
#   output landmark concatenation : LIP + LHAND + RHAND + NOSE + REYE + LEYE + LARMS + RARMS
#

NOSE=[
    1,2,98,327
]
LNOSE = [98]
RNOSE = [327]
LIP = [ 0, 
    61, 185, 40, 39, 37, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
LLIP = [84,181,91,146,61,185,40,39,37,87,178,88,95,78,191,80,81,82]
RLIP = [314,405,321,375,291,409,270,269,267,317,402,318,324,308,415,310,311,312]

POSE = [500, 502, 504, 501, 503, 505, 512, 513]
LPOSE = [513,505,503,501]
RPOSE = [512,504,502,500]

LARMS = [501, 503, 505, 507, 509, 511]
RARMS = [500, 502, 504, 506, 508, 510]

REYE = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    246, 161, 160, 159, 158, 157, 173,
]
LEYE = [
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    466, 388, 387, 386, 385, 384, 398,
]

LHAND = np.arange(468, 489).tolist()
RHAND = np.arange(522, 543).tolist()

POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE + LARMS + RARMS

# Function to Generate image of landmarks over time (use XYZ as BGR)
NLANDMARKS=130
TSAMPLES=128
holistic_xyz_frame  = np.zeros((NLANDMARKS,TSAMPLES,3))
def create_landmark_tframe( pose_landmarks, face_landmarks, left_hand_landmarks, right_hand_landmarks ):
        # Concatenate landmarks together
        # Pose :   33 landmarks =>  132 values (x,y,z,p)
        # Face :  468 landmarks => 1872 values (x,y,z,p)
        # Hand :   21 landmarks =>   84 values (x,y,z,p)
        # Hand :   21 landmarks =>   84 values (x,y,z,p)
        # Total:  543 landmarks => 2172 values (x,y,z,p)

        try:
            pose = pose_landmarks.landmark
            pose_xyz = np.array([[landmark.x, landmark.y, landmark.z] for landmark in pose])
        except:
            pose_xyz = np.zeros((33,3))
        
        try:
            face = face_landmarks.landmark
            face_xyz = np.array([[landmark.x, landmark.y, landmark.z] for landmark in face])
        except:
            face_xyz = np.zeros((468,3))
        
        try:
            lhand = left_hand_landmarks.landmark
            lhand_xyz = np.array([[landmark.x, landmark.y, landmark.z] for landmark in lhand])
        except:
            lhand_xyz = np.zeros((21,3))

        try:
            rhand = right_hand_landmarks.landmark
            rhand_xyz = np.array([[landmark.x, landmark.y, landmark.z] for landmark in rhand])
        except:
            rhand_xyz = np.zeros((21,3))

        lips_xyz = face_xyz[LIP,:]
        nose_xyz = face_xyz[NOSE,:]
        leye_xyz = face_xyz[LEYE,:]
        reye_xyz = face_xyz[REYE,:]

        larm_xyz = pose_xyz[[L-522 for L in LARMS],:]
        rarm_xyz = pose_xyz[[R-522 for R in RARMS],:]
        
        if False:
            print("")
            print("lips  xy/z min/max : ",lips_xyz[:,0:2].min(),lips_xyz[:,0:2].max(),lips_xyz[:,2].min(),lips_xyz[:,2].max())
            print("nose  xy/z min/max : ",nose_xyz[:,0:2].min(),nose_xyz[:,0:2].max(),nose_xyz[:,2].min(),nose_xyz[:,2].max())
            print("leye  xy/z min/max : ",leye_xyz[:,0:2].min(),leye_xyz[:,0:2].max(),leye_xyz[:,2].min(),leye_xyz[:,2].max())
            print("reye  xy/z min/max : ",reye_xyz[:,0:2].min(),reye_xyz[:,0:2].max(),reye_xyz[:,2].min(),reye_xyz[:,2].max())
            print("larm  xy/z min/max : ",larm_xyz[:,0:2].min(),larm_xyz[:,0:2].max(),larm_xyz[:,2].min(),larm_xyz[:,2].max())
            print("rarm  xy/z min/max : ",rarm_xyz[:,0:2].min(),rarm_xyz[:,0:2].max(),rarm_xyz[:,2].min(),rarm_xyz[:,2].max())
            print("lhand xy/z min/max : ",lhand_xyz[:,0:2].min(),lhand_xyz[:,0:2].max(),lhand_xyz[:,2].min(),lhand_xyz[:,2].max())
            print("rhand xy/z min/max : ",rhand_xyz[:,0:2].min(),rhand_xyz[:,0:2].max(),rhand_xyz[:,2].min(),rhand_xyz[:,2].max())
        
        # POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE + LARMS + RARMS
        holistic_xyz = cv2.vconcat([lips_xyz,lhand_xyz,rhand_xyz,nose_xyz,reye_xyz,leye_xyz,larm_xyz,rarm_xyz])
        #print(holistic_xyz.shape)
        
        holistic_xyz_frame[:,1:TSAMPLES,:] = holistic_xyz_frame[:,0:TSAMPLES-1,:]
        holistic_xyz_frame[:,0,:] = holistic_xyz

        return holistic_xyz_frame

#
# Parse asl-fingerspelling dataset to view landmarks
#
# Reference:
#    https://www.kaggle.com/competitions/asl-fingerspelling/data

#dataset_df = pd.read_csv('/kaggle/input/asl-fingerspelling/train.csv')
dataset_df = pd.read_csv('./asl-fingerspelling/train.csv')
#dataset_df = pd.read_csv('./asl-fingerspelling/supplemental_landmarks.csv')
print("Full train dataset shape is {}".format(dataset_df.shape))

dataset_df.head()

# 1:1 aspect ratio
#IMAGE_WIDTH  = NLANDMARKS*4
#IMAGE_HEIGHT = NLANDMARKS*4

# 4:3 aspect ratio
#IMAGE_WIDTH  = NLANDMARKS*3
#IMAGE_HEIGHT = NLANDMARKS*4

# 16:9 aspect ratio
IMAGE_WIDTH  = 290
IMAGE_HEIGHT = NLANDMARKS*4

TFRAME_WIDTH  = NLANDMARKS*4
TFRAME_HEIGHT = NLANDMARKS*4

nb_rows = dataset_df.shape[0]
print("Rows = ",nb_rows)
for row in range(0,nb_rows):
    # Fetch sequence_id, path, phrase from first row
    #sequence_id, file_id, phrase = dataset_df.iloc[0][['sequence_id', 'file_id', 'phrase']]
    path, sequence_id, file_id, phrase = dataset_df.iloc[row][['path', 'sequence_id', 'file_id', 'phrase']]
    print(f"path: {path}, sequence_id: {sequence_id}, file_id: {file_id}, phrase: {phrase}")

    # Fetch data from parquet file
    try:
        sample_sequence_df = pq.read_table(f"./asl-fingerspelling/{path}",
            filters=[[('sequence_id', '=', sequence_id)],]).to_pandas()    
        print("Full sequence dataset shape is {}".format(sample_sequence_df.shape))
        # Full sequence dataset shape is (123, 1630)
        # 1630 = 543*3 + 1
    except:
        continue
    # Contents of parquet
    #    x_face_0: float
    #    ...
    #    x_face_467: float
    #    x_left_hand_0: float
    #    ...
    #    x_left_hand_20: float
    #    x_pose_0: float
    #    ...    
    #    x_pose_32: float
    #    x_right_hand_0: float
    #    ...    
    #    x_right_hand_20: float
    #    y_face_0: float
    #    ...    
    #    y_face_467: float
    #    y_left_hand_0: float
    #    ...    
    #    y_left_hand_20: float
    #    y_pose_0: float
    #    ...
    #    y_pose_32: float
    #    y_right_hand_0: float
    #    ...    
    #    y_right_hand_20: float
    #    z_face_0: float
    #    ...
    #    z_face_467: float
    #    z_left_hand_0: float
    #    ...
    #    z_left_hand_20: float
    #    z_pose_0: float
    #    ...
    #    z_pose_32: float
    #    z_right_hand_0: float
    #    ...
    #    z_right_hand_20: float
    #    sequence_id: int64
    

    sample_sequence_df.head()


    # Get the images created using mediapipe apis
    #hand_images, hand_landmarks = get_hands(sample_sequence_df)
    #images, landmarks = get_landmarks(sample_sequence_df)
    seq_df = sample_sequence_df
    for seq_idx in range(len(seq_df)):

        image_landmarks = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3),np.uint8)
	
        # Extract pose landmarks        
        x_pose = seq_df.iloc[seq_idx].filter(regex="x_pose.*").values
        y_pose = seq_df.iloc[seq_idx].filter(regex="y_pose.*").values
        z_pose = seq_df.iloc[seq_idx].filter(regex="z_pose.*").values

        pose_landmarks = landmark_pb2.NormalizedLandmarkList()
        for x, y, z in zip(x_pose, y_pose, z_pose):
            pose_landmarks.landmark.add(x=x, y=y, z=z)

        # Draw pose landmarks
        mp_drawing.draw_landmarks(image_landmarks, pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                 landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        # Extract face landmarks
        x_face = seq_df.iloc[seq_idx].filter(regex="x_face.*").values
        y_face = seq_df.iloc[seq_idx].filter(regex="y_face.*").values
        z_face = seq_df.iloc[seq_idx].filter(regex="z_face.*").values
	
        face_landmarks = landmark_pb2.NormalizedLandmarkList()
        for x, y, z in zip(x_face, y_face, z_face):
            face_landmarks.landmark.add(x=x, y=y, z=z)

        # Draw face landmarks	
        #mp_drawing.draw_landmarks(image_landmarks, face_landmarks, FACEMESH_TESSELATION,
        mp_drawing.draw_landmarks(image_landmarks, face_landmarks, FACEMESH_LIPS,
                                 landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,256,121), thickness=2, circle_radius=2)
                                 )
        mp_drawing.draw_landmarks(image_landmarks, face_landmarks, FACEMESH_NOSE,
                                 landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,256,121), thickness=2, circle_radius=2)
                                 )
        mp_drawing.draw_landmarks(image_landmarks, face_landmarks, FACEMESH_LEFT_EYE,
                                 landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,256,121), thickness=2, circle_radius=2)
                                 )
        mp_drawing.draw_landmarks(image_landmarks, face_landmarks, FACEMESH_RIGHT_EYE,
                                 landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,256,121), thickness=2, circle_radius=2)
                                 ) 

        # Extract Left Hand landmarks 
        x_hand = seq_df.iloc[seq_idx].filter(regex="x_left_hand.*").values
        y_hand = seq_df.iloc[seq_idx].filter(regex="y_left_hand.*").values
        z_hand = seq_df.iloc[seq_idx].filter(regex="z_left_hand.*").values
	
        left_hand_landmarks = landmark_pb2.NormalizedLandmarkList()
        for x, y, z in zip(x_hand, y_hand, z_hand):
            left_hand_landmarks.landmark.add(x=x, y=y, z=z)

        # Draw Left Hand landmarks                
        mp_drawing.draw_landmarks(image_landmarks, left_hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                 landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )                 

        # Extract Right hand landmarks
        x_hand = seq_df.iloc[seq_idx].filter(regex="x_right_hand.*").values
        y_hand = seq_df.iloc[seq_idx].filter(regex="y_right_hand.*").values
        z_hand = seq_df.iloc[seq_idx].filter(regex="z_right_hand.*").values

        right_hand_landmarks = landmark_pb2.NormalizedLandmarkList()
	
        for x, y, z in zip(x_hand, y_hand, z_hand):
            right_hand_landmarks.landmark.add(x=x, y=y, z=z)

        # Draw Right hand landmarks
        mp_drawing.draw_landmarks(image_landmarks, right_hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                 landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )                
	
        image_landmarks = cv2.putText(image_landmarks, phrase, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)                 
        image_landmarks = image_landmarks.astype(np.float64)/256.0
	
        tframe = create_landmark_tframe( pose_landmarks, face_landmarks, left_hand_landmarks, right_hand_landmarks )
        tframe = cv2.resize(tframe,dsize=[TFRAME_WIDTH, TFRAME_HEIGHT])

        #print(image_landmarks.shape,image_landmarks.dtype)
        #print(tframe.shape,tframe.dtype)
        
        #cv2.imshow(phrase,image_landmarks)
        #cv2.imshow('asl_fingerspelling_viewer',image_landmarks)
        #cv2.imshow('130 top landmarks (lips, hands, nose, eyes, arms)',tframe)
        
        output = cv2.hconcat([image_landmarks,tframe])
        cv2.imshow('asl_fingerspelling_viewer',output)
	
        #c = cv2.waitKey(100) # x1 time scale
        c = cv2.waitKey(50) # x2 times faster
        #print(c)
        if c == 99: #'c'
        	break
        if c == 110: #'n'
        	break
        if c == 113: #'q':
        	break

    if c == 99: #'c'
        continue
    if c == 110: #'n'
        continue
    if c == 113: #'q':
        break

    # Fetch and show the data for right hand
    #create_animation(np.array(hand_images)[:, 0])
    #create_animation(np.array(images)[:, 0])


    
    
