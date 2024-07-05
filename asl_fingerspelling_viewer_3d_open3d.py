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

import open3d as o3d

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
            print(f" pose min xyz = {np.min( pose_xyz, axis=0)}, max xyz = {np.max( pose_xyz, axis=0)}, mean xyz = {np.mean( pose_xyz, axis=0)}")
            print(f" face min xyz = {np.min( face_xyz, axis=0)}, max xyz = {np.max( face_xyz, axis=0)}, mean xyz = {np.mean( face_xyz, axis=0)}")
            print(f"lhand min xyz = {np.min(lhand_xyz, axis=0)}, max xyz = {np.max(lhand_xyz, axis=0)}, mean xyz = {np.mean(lhand_xyz, axis=0)}")
            print(f"rhand min xyz = {np.min(rhand_xyz, axis=0)}, max xyz = {np.max(rhand_xyz, axis=0)}, mean xyz = {np.mean(rhand_xyz, axis=0)}")
        if False:
            print("")
            print(f" lips min xyz = {np.min( lips_xyz, axis=0)}, max xyz = {np.max( lips_xyz, axis=0)}, mean xyz = {np.mean( lips_xyz, axis=0)}")
            print(f" nose min xyz = {np.min( nose_xyz, axis=0)}, max xyz = {np.max( nose_xyz, axis=0)}, mean xyz = {np.mean( nose_xyz, axis=0)}")
            print(f" leye min xyz = {np.min( leye_xyz, axis=0)}, max xyz = {np.max( leye_xyz, axis=0)}, mean xyz = {np.mean( leye_xyz, axis=0)}")
            print(f" reye min xyz = {np.min( reye_xyz, axis=0)}, max xyz = {np.max( reye_xyz, axis=0)}, mean xyz = {np.mean( reye_xyz, axis=0)}")
            print(f" larm min xyz = {np.min( larm_xyz, axis=0)}, max xyz = {np.max( larm_xyz, axis=0)}, mean xyz = {np.mean( larm_xyz, axis=0)}")
            print(f" rarm min xyz = {np.min( rarm_xyz, axis=0)}, max xyz = {np.max( rarm_xyz, axis=0)}, mean xyz = {np.mean( rarm_xyz, axis=0)}")
            print(f"lhand min xyz = {np.min(lhand_xyz, axis=0)}, max xyz = {np.max(lhand_xyz, axis=0)}, mean xyz = {np.mean(lhand_xyz, axis=0)}")
            print(f"rhand min xyz = {np.min(rhand_xyz, axis=0)}, max xyz = {np.max(rhand_xyz, axis=0)}, mean xyz = {np.mean(rhand_xyz, axis=0)}")
        
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

# Create an Open3D point cloud
pose_landmarks_pcd = o3d.geometry.PointCloud()
face_landmarks_pcd = o3d.geometry.PointCloud()
lhand_landmarks_pcd = o3d.geometry.PointCloud()
rhand_landmarks_pcd = o3d.geometry.PointCloud()

use_zoffset = True
denormalize_pointcloud = True
print("[INFO] use_zoffset = ",use_zoffset)
print("[INFO] denormalize_pointcloud = ",denormalize_pointcloud)

def create_landmark_pointcloud( pose_landmarks, face_landmarks, left_hand_landmarks, right_hand_landmarks,
                                use_zoffset=False, image_width=1.0, image_height=1.0 ):
        pose_3d_points = []
        pose_3d_colors = []
        pose_face_zoffset = 0
        pose_lhand_zoffset = 0
        pose_rhand_zoffset = 0
        try:
            for landmark in pose_landmarks.landmark:
                #pose_3d_points.append((landmark.x * image_width, -landmark.y * image_height, -landmark.z * image_width))
                pose_3d_points.append((-landmark.z * image_width, -landmark.y * image_height, landmark.x * image_width))
                pose_3d_colors.append((80/256.0,22/256.0,10/256.0))
                #pose_3d_colors.append((10/256.0,22/256.0,80/256.0))
            if use_zoffset == True:
                #pose_face_zoffset = (pose_landmarks.landmark[10].z + pose_landmarks.landmark[9].z)/2 # mouth
                pose_face_zoffset = (pose_landmarks.landmark[5].z + pose_landmarks.landmark[2].z)/2 # eyes
                pose_lhand_zoffset = pose_landmarks.landmark[15].z # left wrist
                pose_rhand_zoffset = pose_landmarks.landmark[16].z # right write
        except:
            pass

        face_3d_points = []
        face_3d_colors = []
        try:
            for landmark in face_landmarks.landmark:
                #face_3d_points.append((landmark.x * image_width, -landmark.y * image_height, -(landmark.z+pose_face_zoffset) * image_width))
                face_3d_points.append((-(landmark.z+pose_face_zoffset) * image_width, -landmark.y * image_height, landmark.x * image_width))
                face_3d_colors.append((80/256.0,110/256.0,10/256.0))
                #face_3d_colors.append((10/256.0,110/256.0,80/256.0))
        except:
            pass

        lhand_3d_points = []
        lhand_3d_colors = []
        try:
            for landmark in left_hand_landmarks.landmark:
                #lhand_3d_points.append((landmark.x * image_width, -landmark.y * image_height, -(landmark.z+pose_lhand_zoffset) * image_width))
                lhand_3d_points.append((-(landmark.z+pose_lhand_zoffset) * image_width, -landmark.y * image_height, landmark.x * image_width))
                lhand_3d_colors.append((245/256.0,117/256.0,66/256.0))
                #lhand_3d_colors.append((66/256.0,117/256.0,245/256.0))
        except:
            pass

        rhand_3d_points = []
        rhand_3d_colors = []
        try:
            for landmark in right_hand_landmarks.landmark:
                #rhand_3d_points.append((landmark.x * image_width, -landmark.y * image_height, -(landmark.z+pose_rhand_zoffset) * image_width))
                rhand_3d_points.append((-(landmark.z+pose_rhand_zoffset) * image_width, -landmark.y * image_height, landmark.x * image_width))
                rhand_3d_colors.append((245/256.0,117/256.0,66/256.0))
                #rhand_3d_colors.append((66/256.0,117/256.0,245/256.0))
        except:
            pass
        
        # Create Open3D point cloud
        #pose_landmarks_pcd = o3d.geometry.PointCloud()
        #face_landmarks_pcd = o3d.geometry.PointCloud()
        #lhand_landmarks_pcd = o3d.geometry.PointCloud()
        #rhand_landmarks_pcd = o3d.geometry.PointCloud()
        pose_landmarks_pcd.points = o3d.utility.Vector3dVector(pose_3d_points)
        face_landmarks_pcd.points = o3d.utility.Vector3dVector(face_3d_points)
        lhand_landmarks_pcd.points = o3d.utility.Vector3dVector(lhand_3d_points)
        rhand_landmarks_pcd.points = o3d.utility.Vector3dVector(rhand_3d_points)
        pose_landmarks_pcd.colors = o3d.utility.Vector3dVector(pose_3d_colors)
        face_landmarks_pcd.colors = o3d.utility.Vector3dVector(face_3d_colors)
        lhand_landmarks_pcd.colors = o3d.utility.Vector3dVector(lhand_3d_colors)
        rhand_landmarks_pcd.colors = o3d.utility.Vector3dVector(rhand_3d_colors)

        return pose_landmarks_pcd, face_landmarks_pcd, lhand_landmarks_pcd, rhand_landmarks_pcd

# Create an Open3D visualizer(s)
vis_holistic = o3d.visualization.Visualizer()
vis_holistic.create_window(window_name="MediaPipe Holistic pointcloud",width=640, height=480)
#vis_face = o3d.visualization.Visualizer()
#vis_face.create_window(window_name="MediaPipe Face pointcloud",width=640, height=480)
#vis_hand = o3d.visualization.Visualizer()
#vis_hand.create_window(window_name="MediaPipe Hand pointcloud",width=640, height=480)
ctr_holistic = vis_holistic.get_view_control()
pcp = o3d.io.read_pinhole_camera_parameters("asl_selfie_camera.json")
print("[INFO] Pinhold Camera parameters = ",pcp)
print("[INFO]    Extrinsic = ",pcp.extrinsic)
print("[INFO]    Intrinsic = ",pcp.intrinsic)
print("[INFO]    Intrinsic Matrix = ",pcp.intrinsic.intrinsic_matrix)


vis_holistic.add_geometry(pose_landmarks_pcd)
vis_holistic.add_geometry(face_landmarks_pcd)
vis_holistic.add_geometry(lhand_landmarks_pcd)
vis_holistic.add_geometry(rhand_landmarks_pcd)
#vis_face.add_geometry(face_landmarks_pcd)
#vis_hand.add_geometry(lhand_landmarks_pcd)
#vis_hand.add_geometry(rhand_landmarks_pcd)
ctr_holistic.convert_from_pinhole_camera_parameters(pcp)
#vis_holistic.run()


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


        pose_landmarks_pcd, face_landmarks_pcd, lhand_landmarks_pcd, rhand_landmarks_pcd = create_landmark_pointcloud( 
            pose_landmarks, face_landmarks, 
            left_hand_landmarks, right_hand_landmarks,
            use_zoffset=use_zoffset,
            image_width=IMAGE_WIDTH if denormalize_pointcloud==True else 1.0,
            image_height=IMAGE_HEIGHT if denormalize_pointcloud==True else 1.0
            )


        # Open3D visualization
        vis_holistic.add_geometry(pose_landmarks_pcd)
        vis_holistic.add_geometry(face_landmarks_pcd)
        vis_holistic.add_geometry(lhand_landmarks_pcd)
        vis_holistic.add_geometry(rhand_landmarks_pcd)
        #vis_face.add_geometry(face_landmarks_pcd)
        #vis_hand.add_geometry(lhand_landmarks_pcd)
        #vis_hand.add_geometry(rhand_landmarks_pcd)
        ctr_holistic.convert_from_pinhole_camera_parameters(pcp)
        vis_holistic.run()
        #vis_face.run()
        #vis_hand.run()

        #c = cv2.waitKey(100) # x1 time scale
        c = cv2.waitKey(50) # x2 times faster
        if c == ord('c'):
            break
        if c == ord('n'):
            break
        if c == ord('q'):
            break
        if c == ord('z'):
            use_zoffset = not use_zoffset
            print("[INFO] use_zoffset = ",use_zoffset)
        if c == ord('d'):
            denormalize_pointcloud = not denormalize_pointcloud
            print("[INFO] denormalize_pointcloud = ",denormalize_pointcloud)

    if c == ord('c'):
        continue
    if c == ord('n'):
        continue
    if c == ord('q'):
        break

# Cleanup windows
cv2.destroyAllWindows()

# Cleanup Open3d visualizer(s)
vis_holistic.destroy_window()
#vis_face.destroy_window()
#vis_hand.destroy_window()
