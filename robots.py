from enum import Enum
import math
import dill
import numpy as np
from copy import deepcopy

from controls import *
from tools import *

from ROS.ros_modules import *
from visualization_msgs.msg import *
from geometry_msgs.msg import *
from std_msgs.msg import *
from sensor_msgs.msg import *
from trajectory_msgs.msg import *

from ur_rtde_controller.srv import *

class PolinomialTrajectory:
    def __init__(self, q0: float, qf : float, dq0 : float, dqf: float, ddq0 : float, ddqf : float, 
                 t0 : float, tf : float, n : int) -> None:
        # calculating position coefficients
        self.coeff_as = self.GetPolinomialCoefficients(q0, qf, dq0, dqf, ddq0, ddqf, t0, tf, n)

        # calculating velocity coefficients
        self.coeff_das = self.DerivePolinomialCoeff(self.coeff_as)

        # variables used during the movement over the trajectory
        self.q = 0
        self.q_dot = 0

    def GetPolinomialCoefficients(self, q0: float, qf : float, dq0 : float, dqf: float, ddq0 : float, ddqf : float, 
                                  t0 : float, tf : float, n : int):        
        if (n < 5):
            raise NotImplementedError
        else:
            # boundary conditions
            self.q0 = q0        # start position
            self.qf = qf        # end position
            self.dq0 = dq0      # start velocity
            self.dqf = dqf      # end velocity
            self.ddq0 = ddq0    # start acceleration
            self.ddqf = ddqf    # end acceleration

            self.t0 = t0    # start time (always 0)
            self.tf = tf    # end time. Implicitly defines the velocity requested while following the trajectory

            A = np.zeros((6, n + 1))
        
            for j in range(1, n + 2):
                A[0, j - 1] = math.pow(t0, (n + 1 - j))
                A[1, j - 1] = math.pow(tf, (n + 1 - j))

                if n - j >= 0:
                    A[2, j - 1] = (n + 1 - j) * math.pow(t0, (n - j))
                    A[3, j - 1] = (n + 1 - j) * math.pow(tf, (n - j))
                    
                if n - j - 1 >= 0:
                    A[4, j - 1] = (n + 1 - j) * (n - j) * math.pow(t0, (n - j - 1))
                    A[5, j - 1] = (n + 1 - j) * (n - j) * math.pow(tf, (n - j - 1))    
            
        b = [self.q0, self.qf, self.dq0, self.dqf, self.ddq0, self.ddqf]
        a = np.dot(np.linalg.pinv(A), b)

        return a

    def DerivePolinomialCoeff(self, a : np.array):
        n = a.shape[0]
        da = deepcopy(a[0 : n - 1])

        for i in range(n - 1):
            da[i] = a[i] * (n - (i + 1))
        
        return da

    def CalculatePolinomial(self, a : np.array, t : float):
        n = a.shape[0]
        p = 0

        for i in range(n):
            p = p + math.pow(t, n - (i + 1)) * a[i]

        return p

class Steps(Enum):
    IDLE = 0
    START_MOVING = 1
    MOVING = 5
    STOP_MOVING = 10
    CMD_GRIPPER = 13
    SAVE_DATA = 15

class RobotTypes(Enum):
    UR5E = 0

class GripperCommands(Enum):
    PASS = 0
    CLOSE = 1
    OPEN = 2

class RobotPose:
    # pi = 3.14159 / 180
    DEG_TO_RAD_FACTOR = 0.01745

    def __init__(self, tf : float, qfs : list, gripperCommand : GripperCommands, um : str) -> None:
        self.tf = tf
        
        if (um == "rad"):
            self.qfs = qfs
        elif(um == "deg"):
            temp = []
            for qf in qfs:
                temp.append(deepcopy(qf * RobotPose.DEG_TO_RAD_FACTOR))

            self.qfs = deepcopy(temp) 
        else:
            raise NotImplementedError
        
        self.gripperCommand = gripperCommand
        self.um = um

class RobotHorizon:
    def __init__(self, horizonLength : int) -> None:
        self.length = horizonLength

        self.qs = []

        # Vmaxs - n*J(q)*q_dot >= 0 (vector equation, n = minimum distance versor)
        self.Vmaxs = []
        self.versor_distances = []
        self.Jqs = []
        self.q_dots = []
    
    def AddSpeedLimit(self, Vmax):
        if (len(self.Vmaxs) < self.length):
            self.Vmaxs.append(deepcopy(np.array(Vmax)))
        else:
            # removing oldest element
            self.Vmaxs = self.Vmaxs[1:len(self.Vmaxs)]

            # adding new element
            self.Vmaxs.append(deepcopy(np.array(Vmax)))

    def AddVersorDistance(self, versor_distance):
        if (len(self.versor_distances) < self.length):
            self.versor_distances.append(deepcopy(np.array(versor_distance)))
        else:
            # removing oldest element
            self.versor_distances = self.versor_distances[1:len(self.versor_distances)]

            # adding new element
            self.versor_distances.append(deepcopy(np.array(versor_distance)))

    def AddPositions(self, q):
        if (len(self.qs) < self.length):
            self.qs.append(deepcopy(np.array(q)))
        else:
            # removing oldest element
            self.qs = self.qs[1:len(self.qs)]

            # adding new element
            self.qs.append(deepcopy(np.array(q)))

    def AddJacobianMatrix(self, Jq):
        if (len(self.Jqs) < self.length):
            self.Jqs.append(deepcopy(np.array(Jq)))
        else:
            # removing oldest element
            self.Jqs = self.Jqs[1:len(self.Jqs)]

            # adding new element
            self.Jqs.append(deepcopy(np.array(Jq)))

    def AddVelocities(self, q_dot):
        if (len(self.q_dots) < self.length):
            self.q_dots.append(deepcopy(np.array(q_dot)))
        else:
            # removing oldest element
            self.q_dots = self.q_dots[1:len(self.q_dots)]

            # adding new element
            self.q_dots.append(deepcopy(np.array(q_dot)))

class Robot:
    CMD_START_MOVING : str = "start"
    CMD_STOP_MOVING : str = "stop"
    CMD_KILL_MANAGER : str = "kill"
    MAX_DATA_FRAMES : int = 10000

    def __init__(self, robotType : RobotTypes) -> None:
        # "True" => end the infinite loop script
        self.end = False

        # "True" => requested movement task was completed
        self.targetReached = False

        # "True" => robot will use the task selected from the lookup table to move
        self.taskIsActive = False
        self.robotPoses : list[RobotPose] = None
        self.ptrTask = 0

        # time variables: counter, trajectory end time
        self.t = 0
        self.tf = 0

        # horizon variables (used in the optimization problem => optimal "alpha")
        self.horizon = None

        # Model Predictive Control (MPC). Used to calculate the velocity scaling factor "alpha"
        # accordingly to the human-robot distance and the safety constraints
        self.mpc = MPC()

        # PD controller (Kd = 1) => q_dot_traj = q_dot_actual + Kp * (q_actual - q_traj)
        self.Kp = 5

        # velocity scaling factor
        self.alpha = 1

        # robot target trajectories, one for each joint (must be passed externally)
        self.JointTrajectories : list[PolinomialTrajectory] = []

        # dictionary containing the data to save after the task has been completed
        self.dictSavedData : dict[str, list]= {}
        self.dictSavedData["speed_limit"] = []
        self.dictSavedData["vmax_human"] = [] 
        self.dictSavedData["alpha"] = [] 
        
        # creating ROS node for communication
        self.rosNode = ROSNode("robot_manager")

        # Speed Separation Monitoring (SSM) topic. Return the allowed maximum speed
        self.rosNode.AddTopic("multiple_velocity_limit", ROSCommOptions.SUBSCRIBING, Float32MultiArray)

        # communication with this node. The main application will use this topic to communicate
        self.rosNode.AddTopic("robot_manager/command", ROSCommOptions.SUBSCRIBING, String) 

        # communication with this node. The main application will use this topic to communicate
        self.rosNode.AddTopic("robot_manager/trajectory", ROSCommOptions.SUBSCRIBING, Float64MultiArray)

        # communication with this node. The main application will use this topic to communicate
        self.rosNode.AddTopic("robot_manager/task", ROSCommOptions.SUBSCRIBING, String)

        # communication with this node. The main application will use this topic to communicate
        self.rosNode.AddTopic("robot_manager/min_distance_versor", ROSCommOptions.SUBSCRIBING, Float64MultiArray)

        # set "True" when robot is in position
        self.rosNode.AddTopic("robot_manager/position_reached", ROSCommOptions.PUBLISHING, Bool)

        # communication with this node. The main application will use this topic to communicate
        self.rosNode.AddTopic("robot_manager/status/joints", ROSCommOptions.PUBLISHING, Float64MultiArray)

        # gripper command
        self.rosNode.AddService("/ur_rtde/robotiq_gripper/command", RobotiQGripperControl)

        # callback function called when the script ends
        rospy.on_shutdown(self.OnShutdown)

        if (robotType == RobotTypes.UR5E):
            self.totJoints : int = 6
            
            # loading dll external functions
            self.CalculateJacobian = dill.load(open("ur5e_jacobian", "rb"))
            self.CalculateFowardKinematic = dill.load(open("ur5e_fk", "rb"))
            
            # minimum time at which the robot can work properly
            self.sampleTime = 0.015

            # lookup table of predefined task. Specific for each robot type (related to its total number of joints!)
            self.lookupTable : dict[str, list[RobotPose]] = {
                "home" : [RobotPose(5, [-98.50,-35.85,-100.42,-133.01,90.03, 86.22], GripperCommands.CLOSE, "deg")],
                "ping_pong" : [RobotPose(5, [-98.50,-35.85,-100.42,-133.01,90.03, 86.22], GripperCommands.PASS, "deg"),
                                RobotPose(5, [-87.16,-167.25,-15.18,-51.32,94.65,95.34], GripperCommands.PASS, "deg"),
                                RobotPose(5, [-98.50,-35.85,-100.42,-133.01,90.03, 86.22], GripperCommands.PASS, "deg")],
                "pick_place" : [RobotPose(2.5, [-98.50,-35.85,-100.42,-133.01,90.03, 86.22], GripperCommands.OPEN, "deg"), # home
                                RobotPose(5, [-65.92,-129.45,-78.46,-63.73,94.08, 86.80], GripperCommands.CLOSE, "deg"), # pick1
                                RobotPose(2.5, [-80.64,-96.36,-100.29,-74.62,89.37, 101.15], GripperCommands.PASS, "deg"), # midair
                                RobotPose(2.5, [-73.66,-82.34,-141.59,-49.46,90.53, 104.87], GripperCommands.OPEN, "deg"), # place
                                RobotPose(2.5, [-80.64,-96.36,-100.29,-74.62,89.37, 101.15], GripperCommands.PASS, "deg"), # midair
                                RobotPose(5, [-74.99,-135.56,-68.13,-63.85,91.50, 86.79], GripperCommands.CLOSE, "deg"), # pick2
                                RobotPose(2.5, [-80.64,-96.36,-100.29,-74.62,89.37, 101.15], GripperCommands.PASS, "deg"), # midair
                                RobotPose(2.5, [-73.66,-82.34,-141.59,-49.46,90.53, 104.87], GripperCommands.OPEN, "deg"), # place
                                RobotPose(5, [-80.64,-96.36,-100.29,-74.62,89.37, 101.15], GripperCommands.PASS, "deg"), # midair
                                RobotPose(2.5, [-97.25,-135.95,-68.00,-66.26,91.43, 149.13], GripperCommands.CLOSE, "deg"), # pick3
                                RobotPose(2.5, [-80.64,-96.36,-100.29,-74.62,89.37, 101.15], GripperCommands.PASS, "deg"), # midair
                                RobotPose(2.5, [-73.66,-82.34,-141.59,-49.46,90.53, 104.87], GripperCommands.OPEN, "deg"), # place
                                RobotPose(2.5, [-73.62,-70.84,-105.42,-99.47,84.38, 104.83], GripperCommands.OPEN, "deg")] # end
            }

            # # # ROS UR drivers # # #

            # sets trajectory s(t) as a space-time law. Velocities are therefore automatically calculated
            self.rosNode.AddTopic("ur_rtde/controllers/joint_space_controller/command", 
                          ROSCommOptions.PUBLISHING, JointTrajectoryPoint) 

            # sets target q_dot
            self.rosNode.AddTopic("ur_rtde/controllers/joint_velocity_controller/command",
                                  ROSCommOptions.PUBLISHING, Float64MultiArray) 

            # sets target x_dot
            self.rosNode.AddTopic("ur_rtde/controllers/cartesian_velocity_controller/command", 
                          ROSCommOptions.PUBLISHING, Twist) 

            # reads actual q, q_dot
            self.rosNode.AddTopic("joint_states", ROSCommOptions.SUBSCRIBING, JointState) 

            # # # # # # # # # # # #
        else:
            raise NotImplementedError
        
        self.step : Steps = Steps.STOP_MOVING

    def UpdateJointTrajectories(self, tf : float = None, qfs : list = None):
        # default boundary conditions (null speed and acceleration at start and end point)
        dq0 = 0
        dqf = 0
        ddq0 = 0
        ddqf = 0
            
        # default trajectory parameters
        t0 = 0
        n = 5

        # N.B: tf is the same for every joint! If the value is incosistent the trajectory will
        #      not be updated
        # updating final time
        if (not tf is None):
            if (tf > 0):
                self.tf = tf
            else:
                print("tf must be greater than 0!")
        
        # check if must be created or updated
        if (len(self.JointTrajectories) == 0):
            listIsEmpty = True
        else:
            listIsEmpty = False

        # selecting the most recent robot pose frome the horizon
        q_actuals = self.horizon.qs[int(len(self.horizon.qs) - 1)]

        for i in range(self.totJoints):
            # using current joint position as starting point
            q0 = q_actuals[i]

            # setting trajectory end position
            if (not qfs is None):
                qf = qfs[i]
            else:
                qf = q0

            if (listIsEmpty):
                self.JointTrajectories.append(PolinomialTrajectory(deepcopy(q0), 
                                                              deepcopy(qf), 
                                                              dq0, 
                                                              dqf, 
                                                              ddq0, 
                                                              ddqf, 
                                                              t0, 
                                                              deepcopy(self.tf), 
                                                              n))
            else:
                self.JointTrajectories[i] = PolinomialTrajectory(deepcopy(q0), 
                                                              deepcopy(qf), 
                                                              dq0, 
                                                              dqf, 
                                                              ddq0, 
                                                              ddqf, 
                                                              t0, 
                                                              deepcopy(self.tf), 
                                                              n)
                    
            print(f"        J{i} => trajectory updated. q0 = {q0}, qf = {qf}, tf = {self.tf}")                     

    def OnShutdown(self):
        msgToRobot = Float64MultiArray()
        msgToRobot.data = [0, 0, 0, 0, 0, 0]
        self.rosNode.SendDataOnTopic("ur_rtde/controllers/joint_velocity_controller/command", msgToRobot)
        import pdb; pdb.set_trace()

    def Live(self):
        print("Robot manager => started")

        while (not self.end and not rospy.is_shutdown()):
            startTime = time.perf_counter()
            
            #region ROS subscribed messages

            # updating safety speed limit
            if (self.rosNode.dictTopics["multiple_velocity_limit"].dataReceived):
                self.rosNode.dictTopics["multiple_velocity_limit"].dataReceived = False

                msgVmaxs : Float32MultiArray = self.rosNode.dictTopics["multiple_velocity_limit"].buffer[0]

                # creating horizon based on the available velocities limits
                if (self.horizon is None or len(msgVmaxs.data) != self.horizon.length):
                    self.horizon = RobotHorizon(len(msgVmaxs.data))
                
                if (not self.horizon is None):
                    for vmax in msgVmaxs.data:
                        self.horizon.AddSpeedLimit(vmax)
            else:
                # if no message from SSM is received a default horizon of 1 frame lenght is created (first run only!)
                if (self.horizon is None):
                    self.horizon = RobotHorizon(1)

            # updating distance versor
            if (self.rosNode.dictTopics["robot_manager/min_distance_versor"].dataReceived):
                self.rosNode.dictTopics["robot_manager/min_distance_versor"].dataReceived = False

                msgVersorDistances : Float64MultiArray = self.rosNode.dictTopics["robot_manager/min_distance_versor"].buffer[0]

                if (not self.horizon is None):
                    # N.B: x,y,z,rotX,rotY,rotZ => 6 dimension for 1 vector!
                    for i in range(0, len(msgVersorDistances.data), 6):
                        self.horizon.AddVersorDistance(deepcopy(msgVersorDistances.data[i : i + 6]))

            # updating current robot state
            if (self.rosNode.dictTopics["joint_states"].dataReceived):
                self.rosNode.dictTopics["joint_states"].dataReceived = False

                msgJointStates : JointState = self.rosNode.dictTopics["joint_states"].buffer[0]

                if (not self.horizon is None):
                    self.horizon.AddPositions(msgJointStates.position)

                # updating trajectory
                if (self.rosNode.dictTopics["robot_manager/trajectory"].dataReceived):      
                    self.rosNode.dictTopics["robot_manager/trajectory"].dataReceived = False
                    # reading new trajectory data received
                    msgTraj : Float64MultiArray = self.rosNode.dictTopics["robot_manager/trajectory"].buffer[0]

                    self.UpdateJointTrajectories(msgTraj.data[0], msgTraj.data[1:len(msgTraj.data)])
                    self.taskIsActive = False

            # updating task
            if (self.rosNode.dictTopics["robot_manager/task"].dataReceived):      
                self.rosNode.dictTopics["robot_manager/task"].dataReceived = False    

                msgTask : String = self.rosNode.dictTopics["robot_manager/task"].buffer[0]

                if (msgTask.data in self.lookupTable):
                    self.robotPoses : list[RobotPose] = self.lookupTable[msgTask.data]

                    # check data consistency
                    taskOk = True
                    for robotPose in self.robotPoses:
                        if (len(robotPose.qfs) != self.totJoints):
                            taskOk = False

                    if (taskOk):
                        print(f"Robot => task {msgTask.data} is now active")
                        
                        for i in range(len(self.robotPoses)):
                            print(f"        P{i} => {self.robotPoses[i].qfs}")
                            print(f"        tf = {self.robotPoses[i].tf}")
                            print(f"        Gripper command = {self.robotPoses[i].gripperCommand}")
                        
                        self.taskIsActive = True   
                    else:
                        print(f"Robot => tot joints = {self.totJoints} => {robotPose} inconsistence!")    
                else:
                    print("Task not found") 

            # check movement commands
            if (self.rosNode.dictTopics["robot_manager/command"].dataReceived):
                self.rosNode.dictTopics["robot_manager/command"].dataReceived = False

                msgMovement : String = self.rosNode.dictTopics["robot_manager/command"].buffer[0]

                if (msgMovement.data == Robot.CMD_START_MOVING):
                    if (self.step == Steps.IDLE):
                        self.step = Steps.START_MOVING
                elif (msgMovement.data == Robot.CMD_STOP_MOVING):
                    self.step = Steps.STOP_MOVING
                elif (msgMovement.data == Robot.CMD_KILL_MANAGER):
                    self.end = True
                    self.step = Steps.STOP_MOVING
                else:
                    print("Unknown command")
            
            #endregion

            #region state machine
                    
            # state machine
            if (self.step == Steps.IDLE):
                self.ptrTask = 0
                pass
            elif (self.step == Steps.START_MOVING):  
                if (self.taskIsActive):
                    self.UpdateJointTrajectories(self.robotPoses[self.ptrTask].tf, self.robotPoses[self.ptrTask].qfs)
                else:  
                    self.UpdateJointTrajectories()

                # check if trajectories are set before start moving
                if (len(self.JointTrajectories) > 0):
                    # reset variable
                    self.targetReached = False
                    self.t = 0
                    
                    print("Robot => starts moving")
                    self.step = Steps.MOVING
                else:
                    print("Robot => unable to define trajectories")
                    self.step = Steps.IDLE
            elif (self.step == Steps.MOVING):
                # calculate Jacobian matrixes. Required to solve the horizon optimization problem.
                # N.B: for 1 iteration it takes aboout 4ms. Must be taken into account when
                #      the robot sample time is defined!!!
                if (len(self.horizon.qs) == self.horizon.length):
                    Jqi = None
                    for i in range(self.horizon.length):
                        if (i == 0):
                            Jqi = np.array(self.CalculateJacobian(self.horizon.qs[i]))
                        else:
                            # calculating Jq based on the target trajectory
                            Jq_traj_prev = np.array(self.CalculateJacobian(self.horizon.qs[i - 1]))
                            Jq_traj_actual = np.array(self.CalculateJacobian(self.horizon.qs[i]))

                            Jqi = Jq_traj_prev + (Jq_traj_prev - Jq_traj_actual) * self.alpha

                        self.horizon.AddJacobianMatrix(deepcopy(Jqi))

                # selecting the joint positions of the last frame of the horizon
                ptrLast = int(len(self.horizon.qs) - 1)
                q_actuals = self.horizon.qs[ptrLast]
                q_dot_maxs = []

                # setting velocities
                msgToRobot = Float64MultiArray()
                for i in range(self.totJoints):
                    # calculate trajectory position and velocity
                    q_traj = self.JointTrajectories[i].CalculatePolinomial(self.JointTrajectories[i].coeff_as, self.t)
                    q_dot_traj = self.JointTrajectories[i].CalculatePolinomial(self.JointTrajectories[i].coeff_das, self.t)

                    # PD controller (Kd = 1) => q_dot_traj = q_dot_actual + Kp * (q_actual - q_traj)
                    q_dot_max = q_dot_traj + self.Kp * (q_traj - q_actuals[i])
                    q_dot_maxs.append(deepcopy(q_dot_max))
                    msgToRobot.data.append(deepcopy(q_dot_max * self.alpha))

                # adding max velocities to the horizon
                self.horizon.AddVelocities(deepcopy(q_dot_maxs))

                # check all data are available
                if ((len(self.horizon.qs) == len(self.horizon.Vmaxs) == len(self.horizon.versor_distances) == len(self.horizon.Jqs) == len(self.horizon.q_dots))
                    and (len(self.horizon.qs) > 0 and len(self.horizon.Vmaxs) > 0 and len(self.horizon.versor_distances) > 0 and len(self.horizon.Jqs) > 0
                    and len(self.horizon.q_dots) > 0)):
                    # solving the optimization problem
                    alpha_current = deepcopy(self.alpha)
                    alpha_next, vmax_human = self.mpc.GetOptimalSolution(self.horizon.Vmaxs, 
                                                             self.horizon.versor_distances, 
                                                             self.horizon.Jqs,
                                                             self.horizon.q_dots,
                                                             alpha_current)

                    print(f"Optimal alpha: {alpha_next}")

                    # updating alpha value
                    self.alpha = deepcopy(alpha_next)

                    ptrLast = int(len(self.horizon.Vmaxs) - 1)

                    # circling data to reduce memory usage
                    if (len(self.dictSavedData["speed_limit"]) > Robot.MAX_DATA_FRAMES):
                        self.dictSavedData["speed_limit"] = self.dictSavedData["speed_limit"][1:Robot.MAX_DATA_FRAMES]
                        self.dictSavedData["alpha"] = self.dictSavedData["alpha"][1:Robot.MAX_DATA_FRAMES]

                    self.dictSavedData["speed_limit"].append(deepcopy(float(self.horizon.Vmaxs[ptrLast])))
                    self.dictSavedData["alpha"].append(deepcopy(float(self.alpha)))

                    if (len(vmax_human) < 1):
                        # circling data to reduce memory usage
                        if (len(self.dictSavedData["vmax_human"]) > Robot.MAX_DATA_FRAMES):
                            self.dictSavedData["vmax_human"] = self.dictSavedData["vmax_human"][1:Robot.MAX_DATA_FRAMES]

                        self.dictSavedData["vmax_human"].append(deepcopy(float(vmax_human * self.alpha)))
                    else:
                        # circling data to reduce memory usage
                        if (len(self.dictSavedData["vmax_human"]) > Robot.MAX_DATA_FRAMES):
                            self.dictSavedData["vmax_human"] = self.dictSavedData["vmax_human"][1:Robot.MAX_DATA_FRAMES]

                        ptrLast = int(len(vmax_human) - 1)
                        self.dictSavedData["vmax_human"].append(deepcopy(float(vmax_human[ptrLast] * self.alpha)))

                # sending new target velocities to the robot
                self.rosNode.SendDataOnTopic("ur_rtde/controllers/joint_velocity_controller/command", msgToRobot)

                # selecting the joint positions of the last frame of the horizon
                ptrLast = int(len(self.horizon.qs) - 1)
                q_actuals = self.horizon.qs[ptrLast]

                q_actual_norm = np.linalg.norm(q_actuals)

                qfs = []
                for i in range(self.totJoints):
                    qfs.append(self.JointTrajectories[i].qf)

                qf_norm = np.linalg.norm(qfs)
                qf_norm_min = qf_norm - qf_norm * 0.001
                qf_norm_max = qf_norm + qf_norm * 0.001

                # check if robot is in position and parameter "t" is at its final value
                if (q_actual_norm >= qf_norm_min and q_actual_norm <= qf_norm_max and self.t == self.JointTrajectories[0].tf):
                    self.step = Steps.STOP_MOVING      
            elif (self.step == Steps.STOP_MOVING):
                msgToRobot = Float64MultiArray()
                msgToRobot.data = [0, 0, 0, 0, 0, 0]
                self.rosNode.SendDataOnTopic("ur_rtde/controllers/joint_velocity_controller/command", msgToRobot)

                print("Robot => stops moving")

                self.step = Steps.CMD_GRIPPER
            elif (self.step == Steps.CMD_GRIPPER):
                # check gripper command
                if (not self.robotPoses is None):
                    if (self.robotPoses[self.ptrTask].gripperCommand == GripperCommands.OPEN):
                        self.rosNode.services["/ur_rtde/robotiq_gripper/command"](100, 100, 100)
                    elif (self.robotPoses[self.ptrTask].gripperCommand == GripperCommands.CLOSE):
                        self.rosNode.services["/ur_rtde/robotiq_gripper/command"](0, 50, 50)

                if (self.taskIsActive):
                    if (self.ptrTask >= int(len(self.robotPoses) - 1)):
                        self.targetReached = True
                    else:
                        self.ptrTask = self.ptrTask + 1
                        self.step = Steps.START_MOVING
                else:
                    self.targetReached = True

                if (self.targetReached):
                    print("\n")
                    print("Robot => target reached")
                    print("\n")

                    self.step = Steps.SAVE_DATA              
            elif (self.step == Steps.SAVE_DATA):
                # check if there is any data to save
                dataIsPresent = False
                for key in self.dictSavedData:
                    print(f"    {key} => tot samples = {len(self.dictSavedData[key])}")
                    
                    if (len(self.dictSavedData[key]) > 0):
                        dataIsPresent = True
                
                if (dataIsPresent):
                    fileName = f"experiment_horiz{self.horizon.length}_"
                    FileManager.SerializeJson(data = self.dictSavedData, fileName = fileName, additionalPath = "robot_data")
                    print("Robot => data saved to file")

                    # clearing data in memory after saving
                    for key in self.dictSavedData:
                        self.dictSavedData[key] = []

                self.step = Steps.IDLE

            #endregion

            #region ROS published messages

            # position reached message
            if (self.targetReached):
                msgFromRobot = Bool()
                msgFromRobot.data = True
                self.rosNode.SendDataOnTopic("robot_manager/position_reached", msgFromRobot)

            # joints actual positions
            if (len(self.horizon.qs) > 0):
                ptrLast = int(len(self.horizon.qs) - 1)

                msgFromRobot = Float64MultiArray()
                msgFromRobot.data.append(self.horizon.qs[ptrLast][0])
                msgFromRobot.data.append(self.horizon.qs[ptrLast][1])
                msgFromRobot.data.append(self.horizon.qs[ptrLast][2])
                msgFromRobot.data.append(self.horizon.qs[ptrLast][3])
                msgFromRobot.data.append(self.horizon.qs[ptrLast][4])
                msgFromRobot.data.append(self.horizon.qs[ptrLast][5])

                self.rosNode.SendDataOnTopic("robot_manager/status/joints", msgFromRobot)

            #endregion

            # loop execution time
            timePassed = time.perf_counter() - startTime

            deltaT = self.sampleTime - timePassed
            if (deltaT > 0):
                time.sleep(deltaT)
            else:
                deltaT = 0

            # updating trajectory time parameter
            if (self.step == Steps.MOVING): 
                if (self.t < self.tf):
                    self.t = self.t + self.alpha * (timePassed + deltaT)
                else:
                    self.t = self.tf


if __name__ == '__main__':
    # process to launch and execute in parallel
    dictProcesses = {
        "/ssm" : "roslaunch speed_limitation ur10_ssm.launch",
        "/ur_rtde_controller" : "roslaunch ur_rtde_controller rtde_controller.launch",
    }

    # command to open a new terminal window with no timeout (it will always stay open)
    newTerminalCmd = "gnome-terminal -e 'bash -c \"_\" '"

    # check if ROS master is active
    if (not rosgraph.is_master_online()):
        os.system(newTerminalCmd.replace("_", "roscore"))

        # waiting activation
        time.sleep(1)

    # checking if needed ROS nodes are already active
    activeNodes = rosnode.get_node_names()
    for key in dictProcesses:
        nodeIsActive = False

        for activeNode in activeNodes:
            if (key in activeNode):
                nodeIsActive = True
                break

        if (not nodeIsActive or key == ""):
            print("\n")
            print("Started ROS node => " + newTerminalCmd.replace("_", dictProcesses[key]))
            print("\n")
            os.system(newTerminalCmd.replace("_", dictProcesses[key]))

    robot = Robot(RobotTypes.UR5E)
    robot.Live()
