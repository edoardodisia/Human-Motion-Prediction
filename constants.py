import numpy as np


SCALE_M_TO_MM = 1000
SCALE_MM_TO_M = 0.001
NN_TIM_INPUT_JOINTS = 22
NN_TIM_POSE25_JOINTS = 25
NN_TIM_POSE17_JOINTS = 17
H36_TOT_JOINTS = 32
KEYBOARD_BTN_ESC = 27
HUMAN_POSE17J_INDEXS = np.array([0,1,2,3,6,7,8,11,12,13,14,15,16,17,20,21,22])
HUMAN_POSE22J_INDEXS = np.array([2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
HUMAN_POSE25J_INDEXS = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
HUMAN_POSE17J_LINK_JOINT_INDEXS = [[0,1], [1,2], [2,3], 
                             [0,4], [4,5], [5,6], 
                             [0,7], [7,8], [8,9], [9,10], 
                             [8,11], [11,12], [12,13], 
                             [8,14], [14,15], [15,16]]
HUMAN_POSE25J_REDUCED_LINK_JOINT_INDEXS = [[0,1], [1,2], [2,3], 
                                     [0,6], [6,7], [7,8], 
                                     [0,11], [11,12], [12,13], [13,14], 
                                     [12,15], [15,16], [16,17], 
                                     [12,20], [20,21], [21,22]]
HUMAN_POSE25J_LINK_JOINT_INDEXS = [[0,1], [1,2], [2,3], [3,4], [4,5],
                             [0,6], [6,7], [7,8], [8,9], [9,10],
                             [0,11], [11,12], [12,13], [13,14],
                             [12,15], [15,16], [16,17], [17,18], [17,19],
                             [12,20], [20,21], [21,22], [22,23], [22,24]]
HUMAN_POSE25J_JOINT_NAMES = {
            0: "master",
            1: "right hip",
            2: "right thigh",
            3: "right leg",
            4: "right ankle",
            5: "right foot",
            6: "left hip",
            7: "left thigh",
            8: "right leg",
            9: "right ankle",
            10: "right foot",
            11: "spine",
            12: "torax",
            13: "neck",
            14: "head",
            15: "left shoulder",
            16: "left elbow",
            17: "left wrist",
            18: "left finger1",
            19: "left finger2",
            20: "right shoulder",
            21: "right elbow",
            22: "right wrist",
            23: "right finger1",
            24: "right finger2"
        }
HUMAN_POSE_MASK_NO_LEGS = np.array([[0, 0, 0],
                                    [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                                    [0, 0 ,0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
                                    [1, 1, 1], [1, 1, 1],
                                    [1, 1, 1], [1, 1, 1],
                                    [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1],
                                    [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
ROBOT_LINK_JOINT_INDEXS = [[0,1]]
ROBOT_DUMMY_JOINT_POS1 = [[[0,-1000,0], [100,-1100,100]], [[100,-1100,100], [200,-1000,200]]]
ROBOT_DUMMY_JOINT_POS2 = [[[0,-100,0], [100,-110,100]], [[100,-110,100], [200,-100,200]]]
ROBOT_DUMMY_JOINT_POS3 = [[[0,-600,500], [100,-700,600]], [[100,-700,600], [200,-700,700]]]
ROBOT_DUMMY_JOINT_POS4 = [[[0,-350,500], [100,-450,600]], [[100,-450,600], [200,-450,700]]]
ROBOT_SAMPLE_PERIOD = 0.002
ROBOT_TOT_JOINTS = 6

SSM_MIN_DISTANCE = 0.120