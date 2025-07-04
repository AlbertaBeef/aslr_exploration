# Overview

Python code for exploration of datasets for ASL recognition.

![](images/mediapipe_holistic_viewer_animation_02.gif)

## Kaggle Datasets

The following datasets are used for this exploration:

| DataSet            | Archive Size | Extracted Size | Kaggle reference                                             | 
| :----------------: | :----------: | :------------: | :----------------------------------------------------------: | 
| asl-signs          |         40GB |           57GB | https://www.kaggle.com/competitions/asl-signs/data           |
| asl-fingerspelling |        170GB |          190GB | https://www.kaggle.com/competitions/asl-fingerspelling/data  |


## Instructions

Perform the following steps to install the viewers and datasets on your platform.

1. Clone repository

'

    $ git clone https://github.com/AlbertaBeef/aslr_exploration
    $ cd alsr_exploration
   

3. Download Kaggle datasets (using Kaggle API, or directory from above URLs)

'

    $ kaggle competitions download -c asl-signs
    $ kaggle competitions download -c asl-fingerspelling
  

5. Extract Kaggle datasets

'
    For asl-signs dataset :

    $ mkdir asl-signs
    $ cd asl-signs
    $ unzip ../asl-signs.zip
    $ cd ..

'
    For asl-fingerspelling dataset :

    $ mkdir asl-fingerspelling
    $ cd asl-fingerspelling
    $ unzip ../asl-fingerspelling.zip
    $ cd ..
   

6. Launch viewer scripts

'
    For asl-signs dataset :

    $ python3 asl_signs_viewer.py

![](images/asl_signs_viewer_animation.gif)

'
    For asl-fingerspelling dataset :

    $ python3 asl_fingerspelling_viewer.py

![](images/asl_fingerspelling_viewer_animation.gif)

'
    For live video :

    $ python3 mediapipe_holistic_viewer.py

![](images/mediapipe_holistic_viewer_animation_01.gif)



