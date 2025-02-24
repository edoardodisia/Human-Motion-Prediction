# Human-Motion-Prediction
This repo contains the code related to my master thesis [Human robot collaboration: integration of human motion prediction in a safety-aware kinodynamic architecture](https://drive.google.com/file/d/1cHVJ08ti55-kDEPAihAmyuEOgTdZMH5S/view?usp=drive_link). 

## Acknowledgments
This code is working thanks to two different frameworks working together:
- [MMPose](https://mmpose.readthedocs.io/en/latest/) : manages and collect the webcam frames as input to compute 3D poses as output;
- [TIM](https://github.com/tileb1/motion-prediction-tim?tab=readme-ov-file) : takes a buffer of human 3D poses as input and generates as output a buffer of predicted future poses.

## Dependecies
- Clone the GitHub Repository:
```bash
    git clone https://github.com/edoardodisia/Human-Motion-Prediction
```

- Install all the required dependecies:
```bash
    pip install -r requirements.txt
```

## Running the code
To start the program run:
```bash
    python main.py webcam
```
As default the application manages a webcam to collect frames of the human who is interacting with the robot in order to extract a 3D pose for each frame captured. Using these information the robot is driven to avoid collision with the operator while working in a shared environment.
If no webcam is available it is possible to emulate the operator by using a previously saved file.
In the same way the robot can be simulate by using an existent file with its joint position data.

The application has two running options to get data as input:
- *live* : data are collected using external hardware devices. A webcam is used to capture frames of the operator, while an *OptiTrack* tracking system is required to collect position data of the robot;
- *file* : previously stored data, saved as json files, can be used to emulate both the human and the robot while running the application.

Inside the root folder different subfolders can be found:
- *TIM* : contains the TIM framework vanilla source code; 
- *ROS* : contains the source file used to establish ethernet communication using *ROS (Robot Operating System)*;
- *Output_files* : contains data computed by the application. Those files are created only if the *--savetofile* parsing option is active, as json or txt files, and are stored in different folders:
    - *robot_data* : contains files with robot movement data;
    - *pose_data* : each file contains, frame by frame, the *(x,y,z)* position of each joint related to the human 3D pose;
    - *videos* : raw footage recorded by the webcam (if used as input source).
- *MMPose* : contains the MMPose framework vanilla source code; 

While running the demo it is possible to visualize some of the information processed by the application. Here you can find an example of what it will be displayed:

![Image](https://github.com/user-attachments/assets/00a5106d-0896-44ce-b655-4027eb3ba660)

In particalur the orange link represents the last link of the robot arm, which holds the end effector (EE). In yellow it is reported the minimum distance between the EE and the real position of the operator (Ground of Truth, GT), while in light-blue the distance between the predicted position of the operator and the EE. 

**Note**: this demo is intended to pilot a **UR5e robot** model produced by the *Universal Robots*. If this is not available it is recommended to emulate the robot to avoid misbehaviour.

## Results
Here it is reported one of the scenarios used to test the demo. In this case the robot is executing a pick and place operation, while the operator plays the role of a dynamic obstacle:

![Image](https://github.com/user-attachments/assets/7767d9fe-0cfb-4df0-86d3-76079e71d007)

This are the data related to the robot performance:

![Image](https://github.com/user-attachments/assets/09f24952-7bbf-4a9e-9e0f-f8b0d219571a)

## Citing
If you use my code, please cite my work
```bash
@mastersthesis{hmp,
    title   = {Human robot collaboration: integration of human motion prediction in a safety-aware kinodynamic architecture},
    author  = {Edoardo Di Sia},
    year    = {2024},
    school  = {Università degli Studi di Modena e Reggio Emilia},
}
```
