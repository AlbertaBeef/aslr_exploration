# References:
#   https://github.com/nicknochnack/Full-Body-Estimation-using-Media-Pipe-Holistic/blob/main/Media%20Pipe%20Holistic%20Tutorial.ipynb
#   https://github.com/nicknochnack/Body-Language-Decoder/blob/main/Body%20Language%20Decoder%20Tutorial.ipynb

import mediapipe
import cv2
import numpy as np

print("Mediapipe v" + mediapipe.__version__)

mp_pose = mediapipe.solutions.pose
mp_hands = mediapipe.solutions.hands
mp_face = mediapipe.solutions.face_mesh
mp_drawing = mediapipe.solutions.drawing_utils 
mp_drawing_styles = mediapipe.solutions.drawing_styles

mp_holistic = mediapipe.solutions.holistic

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
# Capture live feed to view landmarks
#

input_video=0
cap = cv2.VideoCapture(input_video)
frame_width = 640
frame_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)
#frame_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
#frame_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("[INFO] input : camera",input_video," (",frame_width,",",frame_height,")")

# 1:1 aspect ratio
#IMAGE_WIDTH  = NLANDMARKS*4
#IMAGE_HEIGHT = NLANDMARKS*4

# 4:3 aspect ratio
IMAGE_WIDTH  = NLANDMARKS*3
IMAGE_HEIGHT = NLANDMARKS*4

# 16:9 aspect ratio
#IMAGE_WIDTH  = 290
#IMAGE_HEIGHT = NLANDMARKS*4

TFRAME_WIDTH  = NLANDMARKS*4
TFRAME_HEIGHT = NLANDMARKS*4

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        # aspect ratio 1:1 - crop middle 480x480 from 640x480
        #frame = frame[:,80:560,:]
        # aspect ratio 4:3 - crop middle 360x480 from 640x480
        frame = frame[:,140:500,:]
        # aspect ratio 16:9 - crop middle 228480 from 640x480
        #frame = frame[:,126:354,:]
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Make Detections
        results = holistic.process(image)
        # print(results.face_landmarks)
        
        # Recolor image back to BGR for rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 1. Draw pose landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        # 2. Draw face landmarks
        #mp_drawing.draw_landmarks(image, results.face_landmarks, FACEMESH_TESSELATION,
        mp_drawing.draw_landmarks(image, results.face_landmarks, FACEMESH_LIPS,
                                 landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,256,121), thickness=2, circle_radius=2)
                                 )
        mp_drawing.draw_landmarks(image, results.face_landmarks, FACEMESH_NOSE,
                                 landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,256,121), thickness=2, circle_radius=2)
                                 )
        mp_drawing.draw_landmarks(image, results.face_landmarks, FACEMESH_LEFT_EYE,
                                 landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,256,121), thickness=2, circle_radius=2)
                                 )
        mp_drawing.draw_landmarks(image, results.face_landmarks, FACEMESH_RIGHT_EYE,
                                 landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,256,121), thickness=2, circle_radius=2)
                                 )
        
        # 3. Draw Left Hand landmarks
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
        # 4. Draw Right hand landmarks
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )

        
        cv2.imshow('Raw Webcam Feed', image)
        
        # Generate landmarks-only image


        image_landmarks = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3),np.uint8)
	
        # 1. Draw pose landmarks
        pose_landmarks = results.pose_landmarks
        mp_drawing.draw_landmarks(image_landmarks, pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                 landmark_drawing_spec=mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 connection_drawing_spec=mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        # 2. Draw face landmarks
        face_landmarks = results.face_landmarks
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
                
        # 3. Draw Left Hand landmarks
        left_hand_landmarks = results.left_hand_landmarks
        mp_drawing.draw_landmarks(image_landmarks, left_hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                 landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )                
        # 4. Draw Right hand landmarks
        right_hand_landmarks = results.right_hand_landmarks
        mp_drawing.draw_landmarks(image_landmarks, right_hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                 landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )                
	
        image_landmarks = cv2.putText(image_landmarks, 'Live Feed', (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)                 
        image_landmarks = image_landmarks.astype(np.float64)/256.0
        	
        tframe = create_landmark_tframe( pose_landmarks, face_landmarks, left_hand_landmarks, right_hand_landmarks )
        tframe = cv2.resize(tframe,dsize=[TFRAME_WIDTH, TFRAME_HEIGHT])

        #print(image_landmarks.shape,image_landmarks.dtype)
        #print(tframe.shape,tframe.dtype)
        
        #cv2.imshow(sign,image_landmarks)
        #cv2.imshow('asl_signs_viewer',image_landmarks)
        #cv2.imshow('130 top landmarks (lips, hands, nose, eyes, arms)',tframe)
        
        output = cv2.hconcat([image_landmarks,tframe])
        cv2.imshow('mediapipe_holistic_viewer',output)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

