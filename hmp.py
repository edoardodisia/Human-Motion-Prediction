# nn MMPose (pose 3D calculation from image)
# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict
from mmpose.apis.inferencers import MMPoseInferencer, get_model_aliases

# nn TIM (pose 3D prediction)
import torch
import torch.optim
from torch.utils.data import Dataset
from TIM.utils.model import IdentityAutoencoder
import TIM.utils.model as nnmodel

# extra
from parsers import Options
import constants
import cv2
import numpy as np
import time
import gc
from space_geometry import *
from tools import * 
from ROS.ros_modules import *
from visualization_msgs.msg import *
from geometry_msgs.msg import *
from std_msgs.msg import *
from sensor_msgs.msg import *
from trajectory_msgs.msg import *
import scipy.stats as stats

from plots import *
from pose import *
from controls import *
from robots import *


class DatasetH36M(Dataset):
    def __init__(self):
        self.indexXYZFullPose = []
        self.indexXYZPred = []
        self.indexXYZVisual = []      

        # creating index matrices to extract (x,y,z) coordinates from the raw data output of nn TIM
        for i in range(len(constants.HUMAN_POSE25J_INDEXS)):
            if (i < len(constants.HUMAN_POSE17J_INDEXS)):
                self.indexXYZVisual.append(constants.HUMAN_POSE17J_INDEXS[i] * 3)
                self.indexXYZVisual.append(constants.HUMAN_POSE17J_INDEXS[i] * 3 + 1)
                self.indexXYZVisual.append(constants.HUMAN_POSE17J_INDEXS[i] * 3 + 2)
            
            if (i < len(constants.HUMAN_POSE22J_INDEXS)):
                self.indexXYZPred.append(constants.HUMAN_POSE22J_INDEXS[i] * 3)
                self.indexXYZPred.append(constants.HUMAN_POSE22J_INDEXS[i] * 3 + 1)
                self.indexXYZPred.append(constants.HUMAN_POSE22J_INDEXS[i] * 3 + 2)

            self.indexXYZFullPose.append(constants.HUMAN_POSE25J_INDEXS[i] * 3)
            self.indexXYZFullPose.append(constants.HUMAN_POSE25J_INDEXS[i] * 3 + 1)
            self.indexXYZFullPose.append(constants.HUMAN_POSE25J_INDEXS[i] * 3 + 2)

        self.indexXYZFullPose = np.array(self.indexXYZFullPose)
        self.indexXYZPred = np.array(self.indexXYZPred)
        self.indexXYZVisual = np.array(self.indexXYZVisual)

    def __len__(self):
        return self.encodedFullSeq.shape[0]

    def __getitem__(self, item):
        return self.encodedFullSeq[item], self.xyz25J[item]
    
    def Update(self, dataBuffer, dataSegments, autoencoder = IdentityAutoencoder()):
        # conversion [m] -> [mm] is needed for a correct prediction from the nn TIM!
        xyzJoints_mm = []

        for frame in dataBuffer:
            tempList = []
            firstJoint = True
            
            # single i-th joint (x,y,z) coordinates extraction
            for joint in frame:
                # N.B: the "-" for axes X is needed to align with the default reference system used by RVIZ (ROS module).
                #      The axes Y,Z are instead the same!
                x = - joint[0] * constants.SCALE_M_TO_MM
                y = joint[1] * constants.SCALE_M_TO_MM
                z = joint[2] * constants.SCALE_M_TO_MM
                if (firstJoint):
                    firstJoint = False
                    xOffset = x
                    yOffset = y
                    zOffset = z
                
                tempList.append([x - xOffset, y - yOffset, z - zOffset])

            xyzJoints_mm.append(tempList)

        # change type list -> array
        xyzJoints_mm = np.array(xyzJoints_mm)

        # compute how many frames are related to a single sequence
        framePerSequence = int(len(xyzJoints_mm) / dataSegments)

        # check how many joints the 3D pose has
        if (dataBuffer.shape[1] == constants.NN_TIM_POSE25_JOINTS):
            xyzJoints_mm = xyzJoints_mm[:, constants.HUMAN_POSE17J_INDEXS, :]
        elif (dataBuffer.shape[1] == constants.NN_TIM_POSE17_JOINTS):
            pass
        else:
            raise NotImplementedError    

        # tensor pose 17J (17x3, N * frames) => x,y,z of 17 joint per N * captured frames)
        self.xyz17J = torch.FloatTensor(xyzJoints_mm).reshape(dataSegments, framePerSequence, -1).transpose(1, 2)

        # tensor pose 25J (complete pose)
        self.xyz25J = torch.zeros(dataSegments, constants.NN_TIM_POSE25_JOINTS * 3, framePerSequence)
        self.xyz25J[:, self.indexXYZVisual, :] = self.xyz17J

        # tensor predicted pose 22J (only movable joints, 3 are fixed => no prediction required!)
        self.xyz22J = self.xyz25J[:, self.indexXYZPred, :]

        # N.B: this is the input used by the nn TIM as input to output its motion prediction!
        self.encodedFullSeq = autoencoder(self.xyz22J)[1]

class CaptureOptions(Enum):
    LIVE_CAPTURE = 1
    DATA_FROM_FILE = 2

class HumanMotionPrediction:
    def __init__(self, opt) -> None:
        # if "True" => breaks infinite loop of the "Live" function
        self.end = False

        # saving parsing options 
        self.opt = opt

        # check if any Link Of Interested (LOI) is specified
        if (not self.opt["loi"] is None):
            self.linksOfInterest = self.opt["loi"]  
        else:
            # if not specified all joint will be managed (only J0 is cutted, which is the node master fixed in the origin)
            self.linksOfInterest = np.array([1,2,3,6,7,8,11,12,13,14,15,16])

        if (self.opt["humancap"] == "live"):
            self.human_capture = CaptureOptions.LIVE_CAPTURE
        elif (self.opt["humancap"] == "file"):
            self.human_capture = CaptureOptions.DATA_FROM_FILE
        else:
            raise NotImplementedError

        if (self.opt["robotcap"] == "live"):
            self.robot_capture = CaptureOptions.LIVE_CAPTURE
        elif (self.opt["robotcap"] == "file"):
            self.robot_capture = CaptureOptions.DATA_FROM_FILE
        else:
            raise NotImplementedError

        # list containing the real frames, either capture with a camera or read from a file
        self.realFrames_human = []
        
        # list containing the real frames related to the robot EE position
        self.realFrames_robot = []

        # data used for the pose prediction (containing also the virtual frames)
        self.dataBuffer = None

        # frame captured counter
        self.framesCaptured = 0         # DA ELIMINARE, INCAPSULATO ALL'INTERNO DELL'OGGETTO POSE3D

        # nn MMPose (estrazione posa 3D da immagine)
        self.nnMMPose = MMPoseInferencer(**Options.init_args)
        self.cameraGenerator = None

        # nn TIM (human 3d pose predictor)
        # N.B: it is necessary to use the default naming logic used by the vanilla nn TIM, in order to retrieve
        #      automatically the "input_n" and "output_n" parameters used by the neural network.
        self.ckpt_nnTIM = self.opt['ckpt_dir'] + self.opt['ckpt_file']
        self.opt['input_n'] = int(self.opt['ckpt_file'][18:20])
        self.opt['output_n'] = int(self.opt['ckpt_file'][24:26])
        
        # check if GPU is available
        if (torch.cuda.is_available()):
            ckpt = torch.load(self.ckpt_nnTIM)
        else:
            ckpt = torch.load(self.ckpt_nnTIM, map_location='cpu')

        self.nnTIM = nnmodel.InceptionGCN(opt['linear_size'], opt['dropout'], num_stage=opt['num_stage'], node_n=66, opt=opt)

        self.nnTIM.load_state_dict(ckpt['state_dict'])
        self.nnTIM.eval()

        # definition of the dataset used by the nn TIM
        self.dataset = DatasetH36M()
        
        self.startup = False

        # creation of 3d poses
        self.humanPose = Pose3D(name = "human", 
                                totLinks = int(constants.NN_TIM_POSE17_JOINTS - 1), 
                                maxFrames = int(5 * (self.opt['input_n'] + self.opt['output_n'])),
                                jointIds = constants.HUMAN_POSE25J_REDUCED_LINK_JOINT_INDEXS, 
                                k0 = self.opt["k0"])
        self.robotPose = Pose3D(name = "robot", 
                                totLinks = 1, 
                                maxFrames = int(5 * (self.opt['input_n'] + self.opt['output_n'])), 
                                poseIsHuman = False)

        # The modular structure related to this application is the following:
        # Optitrack => HMP => SSM => UR (drivers + MPC + PD)
        # where Optitrack => the camera tracks both the human and the robot
        #       HMP = Human Motion Prediction (this application!) => output = min distance human-robot
        #       SSM = Speed Separation Monitoring => output = Vmax constraint by the safety distance
        #       MPC = Model Predictive Control => output = optimal "alpha" to apply at current time "k" (from time horizon "alphas")
        #       UR driver => output joint positions and velocities passed to the robot
        
        # creation of ROS node for data communication.
        self.rosNode = ROSNode("hmp_manager")

        # Optitrack
        self.rosNode.AddTopic("vrpn_client_node/UR5_filippo/pose", ROSCommOptions.SUBSCRIBING, PoseStamped)
        self.rosNode.AddTopic("vrpn_client_node/polsino_ed/pose", ROSCommOptions.SUBSCRIBING, PoseStamped)

        # RVIZ topic for 3d visualization of human pose GT and prediction
        self.rosNode.AddTopic("hmp_visualization", ROSCommOptions.PUBLISHING, MarkerArray)

        # HMP topic
        self.rosNode.AddTopic("multiple_minimum_distance", ROSCommOptions.PUBLISHING, Float32MultiArray)

        # SSM topic
        self.rosNode.AddTopic("multiple_velocity_limit", ROSCommOptions.SUBSCRIBING, Float32MultiArray)

        # robot manager topics
        self.rosNode.AddTopic("robot_manager/command", ROSCommOptions.PUBLISHING, String)
        self.rosNode.AddTopic("robot_manager/task", ROSCommOptions.PUBLISHING, String) 
        self.rosNode.AddTopic("robot_manager/trajectory", ROSCommOptions.PUBLISHING, Float64MultiArray) 
        self.rosNode.AddTopic("robot_manager/min_distance_versor", ROSCommOptions.PUBLISHING, Float64MultiArray) 
        self.rosNode.AddTopic("robot_manager/position_reached", ROSCommOptions.SUBSCRIBING, Bool) 
        self.rosNode.AddTopic("robot_manager/status/joints", ROSCommOptions.SUBSCRIBING, Float64MultiArray) 

        # callback function called when the script ends
        rospy.on_shutdown(self.OnShutdown)

        # object used for displaying data and results, using Matplotlib
        self.matlabDrawer = MatlabDrawer()   

        self.CalculateFowardKinematic = dill.load(open("ur5e_fk", "rb")) 

    def OnShutdown(self):
        # closing ROS communication
        self.rosNode.CloseCommunication()

    def Live(self):
        # N.B: this is essential to avoid exeption duo to the fact that the GC may be unable to destroy
        #      unused variables while the infinite loop is running
        gc.disable()

        if (not self.human_capture == CaptureOptions.DATA_FROM_FILE):
            # generator function used to ask a new frame to the camera
            self.cameraGenerator = self.nnMMPose(**self.opt)

        # waiting ROS node to be ready
        time.sleep(2)

        # setting robot task
        self.rosNode.SendDataOnTopic("robot_manager/task", self.opt["robottask"])

        # infinite loop
        while (not self.end and not rospy.is_shutdown()):
            startTime = time.perf_counter()

            # starting task if robot is in position
            if (self.rosNode.dictTopics["robot_manager/position_reached"].dataReceived):
                self.rosNode.dictTopics["robot_manager/position_reached"].dataReceived = False
                self.rosNode.SendDataOnTopic("robot_manager/command", "start")

            if (self.ReadDataInput()):
                self.ProcessData()
                self.WriteDataOutput()

            # manual call of the GC
            gc.collect()

            # delay to avoid stressing the CPU
            time.sleep(0.001)

            print(f"Time passed = {time.perf_counter() - startTime}")

    def ReadDataInput(self):
        dataInputReady = False

        if (not self.human_capture == CaptureOptions.DATA_FROM_FILE): 
            newFrames = self.GetDataFromCamera()

            if (len(self.realFrames_human) <= 1):
                for i in range(0, len(newFrames)):
                    self.realFrames_human.append(newFrames[i]) 
            else:
                self.realFrames_human[0] = self.realFrames_human[1]
                self.realFrames_human[1] = newFrames[0]        
        else:
            if (len(self.realFrames_human) == 0):
                self.realFrames_human = FileManager.DeserializeJson(self.opt["humansrc"])

        if (not self.robot_capture == CaptureOptions.DATA_FROM_FILE): 
            pass
        else:
            if (len(self.realFrames_robot) == 0):
                self.realFrames_robot = FileManager.DeserializeJson(self.opt["robotsrc"])

        self.framesCaptured = self.framesCaptured + 1

        print("Frames captured: " + str(self.framesCaptured))

        if (self.framesCaptured >= 2):
            self.dataBuffer = self.CreateVirtualFrames()
            dataInputReady = True

        return dataInputReady

    def GetDataFromCamera(self):
        poses = []
   
        # temp frame counter
        cntFrames = 0
        while (True):
            # new frame request to the generator
            pose3DInFrame = next(self.cameraGenerator)

            # 3d pose estraction
            actualPose = pose3DInFrame['predictions'][0][0]['keypoints']

            # adding new pose
            poses.append(actualPose)
            cntFrames = cntFrames + 1 

            # check how many frames has to be taken
            targetFrames = 1

            # check if all target frames were captured
            if (cntFrames >= targetFrames):
                break

        return np.array(poses)

    def CreateVirtualFrames(self):
        # casting to array to be sure about the data type
        if (self.human_capture == CaptureOptions.LIVE_CAPTURE):
            previousFrame = np.array(self.realFrames_human[0])
            actualFrame = np.array(self.realFrames_human[1])
        elif (self.human_capture == CaptureOptions.DATA_FROM_FILE):
            previousFrame = np.array(self.realFrames_human[self.framesCaptured - 2])
            actualFrame = np.array(self.realFrames_human[self.framesCaptured - 1])
        else:
            raise NotImplementedError

        # the difference matrix is splitted on the total input frames managed by the nnTIM
        deltaPose = (actualFrame - previousFrame) * self.opt['k0'] * 2
        stepMatrix = deltaPose / (self.opt['input_n'] + self.opt['output_n'])

        # creation of the virtual frames
        virtualFrames = []
        for i in range(0, self.opt['input_n'] + self.opt['output_n']):
            virtualFrames.append(previousFrame + i * stepMatrix)
            
        # the last virtual frame is overwritten with the value of the actual real one (i.e the known actual pose)
        virtualFrames[len(virtualFrames) - 1] = actualFrame 

        return np.array(virtualFrames)

    def ProcessData(self):
        self.GetFuturePoses()

    def GetFuturePoses(self) -> None:         
        # N.B: the vanilla nnTIM uses the DataLoader() Pytorch function to get the input data. Since the live mode
        #      uses only 1 frame per time the data dimensions are small and the pipeline time needed to be reduced
        #      this was omitted here

        # updating data with the last capturing
        self.dataset.Update(self.dataBuffer, self.opt['segments'])

        # pose prediction (22 joint of the total 25 of the pose used)      
        preds22J = self.nnTIM(self.dataset.encodedFullSeq)
             
        # clone GT to add the unmovable joints (j0, j1, j6) and overwrite predicted values of the movable joints
        preds25J = self.dataset.xyz25J.detach().clone()
        preds25J[:, self.dataset.indexXYZPred, :] = preds22J

        # N.B: only the first and last frame of the GT are real!
        gtVirtualFrames_human = self.dataset.xyz25J[-1, ...].detach().clone().transpose(0, 1).reshape(-1, constants.NN_TIM_POSE25_JOINTS, 3).numpy()
        predVirtualFrames_human = preds25J[-1, ...].detach().clone().transpose(0, 1).reshape(-1, constants.NN_TIM_POSE25_JOINTS, 3).numpy()

        #region human frames processing

        gtFrames_human = []
        predFrames_human = []
        if (not self.human_capture == CaptureOptions.DATA_FROM_FILE):
            # updating robot position only when data from Optitrack are available
            if (self.rosNode.dictTopics["vrpn_client_node/polsino_ed/pose"].dataReceived):
                self.rosNode.dictTopics["vrpn_client_node/polsino_ed/pose"].dataReceived = False

                msgFromOptitrack : PoseStamped = self.rosNode.dictTopics["vrpn_client_node/polsino_ed/pose"].buffer[0]

                # calculating distance vector
                d = np.array([msgFromOptitrack.pose.position.x, msgFromOptitrack.pose.position.y, msgFromOptitrack.pose.position.z])

                Tf = tf.transformations.quaternion_matrix([msgFromOptitrack.pose.orientation.x, 
                                                            msgFromOptitrack.pose.orientation.y, 
                                                            msgFromOptitrack.pose.orientation.z, 
                                                            msgFromOptitrack.pose.orientation.w])

                # updating distance vector
                Tf[0:3, 3] = d * constants.SCALE_M_TO_MM

                for ptrFrame in range(gtVirtualFrames_human.shape[0]):
                    gt_frame = []
                    pred_frame = []
                    for ptrJoint in range(gtVirtualFrames_human.shape[1]):
                        # switching to homogeneous space and then compute new position
                        gt_x0 = np.array([gtVirtualFrames_human[ptrFrame, ptrJoint][0], 
                                          gtVirtualFrames_human[ptrFrame, ptrJoint][1], 
                                          gtVirtualFrames_human[ptrFrame, ptrJoint][2], 
                                          1])
                        gt_x1 = np.dot(Tf, gt_x0)
                        gt_frame.append(deepcopy(gt_x1[0:3]))

                        # switching to homogeneous space and then compute new position
                        pred_x0 = np.array([predVirtualFrames_human[ptrFrame, ptrJoint][0], 
                                            predVirtualFrames_human[ptrFrame, ptrJoint][1], 
                                            predVirtualFrames_human[ptrFrame, ptrJoint][2], 
                                            1])
                        pred_x1 = np.dot(Tf, pred_x0)
                        pred_frame.append(deepcopy(pred_x1[0:3]))

                    gtFrames_human.append(deepcopy(gt_frame))
                    predFrames_human.append(deepcopy(pred_frame))

                gtFrames_human = np.array(gtFrames_human)
                predFrames_human = np.array(predFrames_human)
        else:
            gtFrames_human = gtVirtualFrames_human
            predFrames_human = predVirtualFrames_human

        if (self.opt["ignorelegs"]):
            gtFrames_human = gtFrames_human[:] * constants.HUMAN_POSE_MASK_NO_LEGS
            predFrames_human = predFrames_human[:] * constants.HUMAN_POSE_MASK_NO_LEGS  

        gtFrames_human = np.array(gtFrames_human)
        predFrames_human = np.array(predFrames_human)

        #endregion

        #region robot frames processing

        gtFrames_robot = []
        if (not self.robot_capture == CaptureOptions.DATA_FROM_FILE):
            # updating robot position only when data from Optitrack are available
            if (self.rosNode.dictTopics["vrpn_client_node/UR5_filippo/pose"].dataReceived
                and self.rosNode.dictTopics["robot_manager/status/joints"].dataReceived):
                self.rosNode.dictTopics["vrpn_client_node/UR5_filippo/pose"].dataReceived = False
                self.rosNode.dictTopics["robot_manager/status/joints"].dataReceived = False

                msgFromOptitrack : PoseStamped = self.rosNode.dictTopics["vrpn_client_node/UR5_filippo/pose"].buffer[0]
                msgFromRobot : Float64MultiArray = self.rosNode.dictTopics["robot_manager/status/joints"].buffer[0]

                # calculating distance vector
                d = np.array([msgFromOptitrack.pose.position.x, msgFromOptitrack.pose.position.y, msgFromOptitrack.pose.position.z])

                # calculating transform matrix (Optitrack => robot reference system) with d = 0
                Tf = tf.transformations.quaternion_matrix([msgFromOptitrack.pose.orientation.x, 
                                                           msgFromOptitrack.pose.orientation.y, 
                                                           msgFromOptitrack.pose.orientation.z, 
                                                           msgFromOptitrack.pose.orientation.w])
                # updating distance vector
                Tf[0:3, 3] = d

                # foward cinematic
                poseEE = np.array(self.CalculateFowardKinematic(msgFromRobot.data))

                # moving end-effector
                gt_x0 = np.array([poseEE[0], poseEE[1], poseEE[2], 1])
                gt_x1 = np.dot(Tf, gt_x0)

                endEffectorXYZ = gt_x1[0:3]

                # updating robot frame position (only EE is present!)
                gtFrames_robot = np.array([endEffectorXYZ, endEffectorXYZ - np.array([0.10, 0.10, 0.10])])
        else:
            gtFrames_robot = self.realFrames_robot[self.framesCaptured - 1]

        gtFrames_robot = np.array(gtFrames_robot)

        #endregion

        # updating human pose
        for i in range(gtFrames_human.shape[0]):
            # N.B: saving only real frames as data (first and last one), while throwing away the virtual ones
            #      (i.e "saveFrames = True"). 
            if (not self.startup):
                self.startup = True
                saveFrames = True
            else:
                if (i == int(gtFrames_human.shape[0] - 1)):
                    saveFrames = True
                else:
                    saveFrames = False

            self.humanPose.Update(gtFrames_human[i, constants.HUMAN_POSE17J_INDEXS, :], 
                                  FrameTypes.GT,
                                  "gt",
                                  saveFrames)

            self.humanPose.Update(predFrames_human[i, constants.HUMAN_POSE17J_INDEXS, :],
                                  FrameTypes.PRED,
                                  f"pred{i}",
                                  saveFrames)

        # updating robot pose
        if (len(gtFrames_robot) > 0):
            self.robotPose.Update(gtFrames_robot, FrameTypes.GT, "gt")

        # sending data using ROS
        msgToSSM = Float32MultiArray()
        msgToRobot = Float64MultiArray()

        # adding PRED data to both messages, based on the horizon length
        for i in range(self.opt["horizon"]):
            # check if data is available
            if (f"pred{i}" in self.humanPose.linkInfos[0].boundings):
                CalculateDistancePoses(self.humanPose, self.robotPose, f"pred{i}", "gt")

                # forcing minimum value to avoid exception
                if (f"pred{i}" in self.humanPose.minimumDistances):
                    if (self.humanPose.minimumDistances[f"pred{i}"].norm >= constants.SSM_MIN_DISTANCE):
                        msgToSSM.data.append(deepcopy(self.humanPose.minimumDistances[f"pred{i}"].norm))
                    else:
                        msgToSSM.data.append(constants.SSM_MIN_DISTANCE)

                    # N.B: x,y,z,rotX,rotY,rotZ => 6 dimension for 1 vector!
                    msgToRobot.data.append(deepcopy(self.humanPose.minimumDistances[f"pred{i}"].versor[0]))
                    msgToRobot.data.append(deepcopy(self.humanPose.minimumDistances[f"pred{i}"].versor[1]))
                    msgToRobot.data.append(deepcopy(self.humanPose.minimumDistances[f"pred{i}"].versor[2]))
                    msgToRobot.data.append(deepcopy(0))
                    msgToRobot.data.append(deepcopy(0))
                    msgToRobot.data.append(deepcopy(0))

        # overwriting GT data and removing first prediction frame
        CalculateDistancePoses(self.humanPose, self.robotPose, "gt", "gt")

        if ("gt" in self.humanPose.minimumDistances):
            # forcing minimum value to avoid exception
            if (self.humanPose.minimumDistances["gt"].norm >= constants.SSM_MIN_DISTANCE):
                min_dist = self.humanPose.minimumDistances["gt"].norm
            else:
                min_dist = constants.SSM_MIN_DISTANCE

            if (len(msgToSSM.data) > 0):
                msgToSSM.data[0] = min_dist
            else:
                msgToSSM.data.append(deepcopy(min_dist))

            # overwriting "pred0" versor data with "GT" versor data
            if (len(msgToRobot.data) > 0):
                # N.B: x,y,z,rotX,rotY,rotZ => 6 dimension for 1 vector!
                msgToRobot.data[0] = self.humanPose.minimumDistances["gt"].versor[0]
                msgToRobot.data[1] = self.humanPose.minimumDistances["gt"].versor[1]
                msgToRobot.data[2] = self.humanPose.minimumDistances["gt"].versor[2]
                msgToRobot.data[3] = 0
                msgToRobot.data[4] = 0
                msgToRobot.data[5] = 0
            else:
                # N.B: x,y,z,rotX,rotY,rotZ => 6 dimension for 1 vector!
                msgToRobot.data.append(deepcopy(self.humanPose.minimumDistances["gt"].versor[0]))
                msgToRobot.data.append(deepcopy(self.humanPose.minimumDistances["gt"].versor[1]))
                msgToRobot.data.append(deepcopy(self.humanPose.minimumDistances["gt"].versor[2]))
                msgToRobot.data.append(deepcopy(0))
                msgToRobot.data.append(deepcopy(0))
                msgToRobot.data.append(deepcopy(0))

        self.rosNode.SendDataOnTopic("multiple_minimum_distance", msgToSSM)
        self.rosNode.SendDataOnTopic("robot_manager/min_distance_versor", msgToRobot)

    def WriteDataOutput(self):   
        self.PlotWithRviz()      

        if (self.human_capture == CaptureOptions.DATA_FROM_FILE):
            if (self.framesCaptured >= len(self.realFrames_human)):
                # calculate how many data must be displayed. If they are too many the axis X ticks
                # will not be shown as full scale
                totBatches = len(self.realFrames_human) / (self.opt["input_n"] + self.opt["output_n"])
                if (totBatches <= 2):
                    showFullScaleX = True
                else:
                    showFullScaleX = False

                for linkNbr in self.linksOfInterest:
                    linkInfo : LinkInfo = self.humanPose.linkInfos[linkNbr - 1]

                    self.CreateGraphs(showFullScaleX, linkInfo.name, linkInfo.startJoint)
                    self.CreateGraphs(showFullScaleX, linkInfo.name, linkInfo.endJoint)

                # showing plot and figure generated   
                self.matlabDrawer.ShowPlots()

                self.end = True  
        elif (self.human_capture == CaptureOptions.LIVE_CAPTURE):
            if (self.opt["savetofile"] == True):
                # region file saving via serialization

                if (self.framesCaptured >= (self.opt["input_n"] + self.opt["output_n"]) * self.opt["targetbatches"]):
                    self.SaveToFile(self.humanPose, "human")
                    self.SaveToFile(self.robotPose, "robot")
                    self.end = True   

                # endregion

    def PlotWithRviz(self):
        markerArray = MarkerArray()

        # adding human 3d pose markers
        for linkInfo in self.humanPose.linkInfos:
            for key in linkInfo.markers:
                markerArray.markers.append(linkInfo.markers[key])

        # adding GT minimum distance marker
        if ("link_gt_min_distance" in self.humanPose.gt_linkMinDistance.markers):
            markerArray.markers.append(self.humanPose.gt_linkMinDistance.markers["link_gt_min_distance"])

        # adding PRED minimum distance marker
        if ("link_pred_min_distance" in self.humanPose.pred_linkMinDistance.markers):
            markerArray.markers.append(self.humanPose.pred_linkMinDistance.markers["link_pred_min_distance"])

        # adding robot 3d pose markers
        for linkInfo in self.robotPose.linkInfos:
            for key in linkInfo.markers:
                markerArray.markers.append(linkInfo.markers[key])

        self.rosNode.SendDataOnTopic("hmp_visualization", markerArray)

    def SaveToFile(self, pose : Pose3D, prefixFileName : str):
        gtFrames = {}
        predFrames = {}

        # coverting data into a dictionary before serializing
        for linkInfo in pose.linkInfos:
            id1, id2 = linkInfo.GetJoints()
            for i in range(0, len(linkInfo.startJoint.gts)):
                # check if key is already present
                if (not gtFrames.get(f"frame{i}")):
                    gtFrames[f"frame{i}"] = {}
                    predFrames[f"frame{i}"] = {}

                if (len(linkInfo.startJoint.gts) > 0):
                    gt_startJoint_xyz = {}
                    gt_startJoint_xyz["x"] = str(linkInfo.startJoint.gts[i].quotes[0])
                    gt_startJoint_xyz["y"] = str(linkInfo.startJoint.gts[i].quotes[1])
                    gt_startJoint_xyz["z"] = str(linkInfo.startJoint.gts[i].quotes[2])

                    gtFrames[f"frame{i}"][f"joint{id1}"] = gt_startJoint_xyz

                if (len(linkInfo.endJoint.gts) > 0):
                    gt_endJoint_xyz = {}
                    gt_endJoint_xyz["x"] = str(linkInfo.endJoint.gts[i].quotes[0])
                    gt_endJoint_xyz["y"] = str(linkInfo.endJoint.gts[i].quotes[1])
                    gt_endJoint_xyz["z"] = str(linkInfo.endJoint.gts[i].quotes[2])

                    gtFrames[f"frame{i}"][f"joint{id2}"] = gt_endJoint_xyz

                if (len(linkInfo.startJoint.preds) > 0):
                    pred_startJoint_xyz = {}
                    pred_startJoint_xyz["x"] = str(linkInfo.startJoint.preds[i].quotes[0])
                    pred_startJoint_xyz["y"] = str(linkInfo.startJoint.preds[i].quotes[1])
                    pred_startJoint_xyz["z"] = str(linkInfo.startJoint.preds[i].quotes[2])

                    predFrames[f"frame{i}"][f"joint{id1}"] = pred_startJoint_xyz

                if (len(linkInfo.endJoint.preds) > 0):
                    pred_endJoint_xyz = {}
                    pred_endJoint_xyz["x"] = str(linkInfo.endJoint.preds[i].quotes[0])
                    pred_endJoint_xyz["y"] = str(linkInfo.endJoint.preds[i].quotes[1])
                    pred_endJoint_xyz["z"] = str(linkInfo.endJoint.preds[i].quotes[2])

                    predFrames[f"frame{i}"][f"joint{id2}"] = pred_endJoint_xyz

        if (not prefixFileName == "robot"):
            name = f"_in{self.opt['input_n']}_" + f"out{self.opt['output_n']}_" + f"k0_{self.opt['k0']}"
        else:
            name = "_"

        if (len(gtFrames) > 0):
            FileManager.SerializeJson(gtFrames, prefixFileName + "_gts" + name, FileManager.SINGLE_SEGMENT_LABLES)
        
        if (len(predFrames) > 0):
            FileManager.SerializeJson(predFrames, prefixFileName + "_preds" + name, FileManager.SINGLE_SEGMENT_LABLES)

    def CreateGraphs(self, showFullScaleX : bool, linkName : str, jointInfo : JointInfo):
        figureTitle = f"{linkName}, J{jointInfo.id} = {jointInfo.name}. k0 = " + str(self.opt["k0"])
        self.matlabDrawer.AddFigure(figureTitle)

        # region GT and pred

        gts = []
        preds = []
        for i in range(0, len(jointInfo.gts)):
            gts.append(jointInfo.gts[i].norm)
            preds.append(jointInfo.preds[i].norm)

        axTitle = "Norm GT and PRED"
        self.matlabDrawer.AddAx(figureTitle, f"J{jointInfo.id}", True, GraphTypes.GRAPH_2D)       

        dataY = np.array(gts)
        dataX = range(1, dataY.shape[0] + 1)
        legend = "gt, " + "k0 = " + str(self.opt["k0"])

        xInfo = (AxisInfo("x", showFullScaleX, dataX, MeasureUnits.FRAMES))
        yInfo = AxisInfo("y", False, dataY, MeasureUnits.METERS)
        graph = Graph2D(xInfo, yInfo)
        plotInfo = PlotInfo(graph, axTitle, legend)
                        
        self.matlabDrawer.AddPlot(f"J{jointInfo.id}", deepcopy(plotInfo))    

        dataY = np.array(preds)
        dataX = range(1, dataY.shape[0] + 1)
        legend = "pred "  + "k0 = " + str(self.opt["k0"])

        xInfo = AxisInfo("x", showFullScaleX, dataX, MeasureUnits.FRAMES)
        yInfo = AxisInfo("y", False, dataY, MeasureUnits.METERS)
        graph = Graph2D(xInfo, yInfo)
        plotInfo = PlotInfo(graph, axTitle, legend)
                        
        self.matlabDrawer.AddPlot(f"J{jointInfo.id}", deepcopy(plotInfo)) 

        # endregion

        # region Joint Position Error (JPE)

        axTitle = "Joint Position Error (JPE)"
        self.matlabDrawer.AddAx(figureTitle, f"J{jointInfo.id}", True, GraphTypes.GRAPH_2D)

        dataY = np.array(jointInfo.jpes)
        dataX = range(int(self.humanPose.k0 + 1), dataY.shape[0] + int(self.humanPose.k0 + 1))
        legend = "jpe"

        xInfo = AxisInfo("x", showFullScaleX, dataX, MeasureUnits.FRAMES)
        yInfo = AxisInfo("y", False, dataY, MeasureUnits.METERS)
        graph = Graph2D(xInfo, yInfo)
        plotInfo = PlotInfo(graph, axTitle, legend)

        self.matlabDrawer.AddPlot(f"J{jointInfo.id}", deepcopy(plotInfo))  

        # endregion
                        
        # region statistics JPE (mean and standard deviation)

        mu, sigma2, sigma = DataAnalyzer.CalculateStatistics(dataY) 

        xInfo = AxisInfo("x", showFullScaleX, dataX, MeasureUnits.FRAMES)
        yInfo = AxisInfo("y", False, np.full(dataY.shape, mu), MeasureUnits.METERS)
        graph = Graph2D(xInfo, yInfo)
        plotInfo = PlotInfo(graph, axTitle, r'$\mu$' + f"={mu:.2f}")
               
        self.matlabDrawer.AddPlot(f"J{jointInfo.id}", plotInfo, "red", "-")     

        xInfo = AxisInfo("x", showFullScaleX, dataX, MeasureUnits.FRAMES)
        yInfo = AxisInfo("y", False, np.full(dataY.shape, 3 * sigma), MeasureUnits.METERS)
        graph = Graph2D(xInfo, yInfo)
        plotInfo = PlotInfo(graph, axTitle, r'3$\sigma$' + f"={3 * sigma:.2f}")
                        
        self.matlabDrawer.AddPlot(f"J{jointInfo.id}", plotInfo, "red", "--")  

        axTitle = "Normal distribution (JPE)"
        self.matlabDrawer.AddAx(figureTitle, f"J{jointInfo.id}", True, GraphTypes.GRAPH_2D)

        dataX = np.linspace(mu - 3 * sigma, mu + 3 * sigma)

        xInfo = AxisInfo("x", False, dataX, MeasureUnits.NONE)
        yInfo = AxisInfo("y", False, stats.norm.pdf(dataX, mu, sigma), MeasureUnits.NONE)
        graph = Graph2D(xInfo, yInfo)
        plotInfo = PlotInfo(graph, axTitle, "")

        self.matlabDrawer.AddPlot(f"J{jointInfo.id}", plotInfo, "purple")

        # endregion

