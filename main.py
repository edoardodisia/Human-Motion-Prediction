from hmp import HumanMotionPrediction
from parsers import Options

import os
import time
import rosgraph
import rosnode

if __name__ == '__main__':
    # process to launch and execute in parallel
    dictProcesses = {
        "/rviz" : "rosrun rviz rviz",
        "/ssm" : "roslaunch speed_limitation ur10_ssm.launch",
        "/ur_rtde_controller" : "roslaunch ur_rtde_controller rtde_controller.launch",
        "/vrpn_client_node" : "roslaunch vrpn_client_ros sample.launch",
        "robot_manager" : "python robots.py"
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

        if (not nodeIsActive):
            print("\n")
            print("Started ROS node => " + newTerminalCmd.replace("_", dictProcesses[key]))
            print("\n")
            os.system(newTerminalCmd.replace("_", dictProcesses[key]))

    # waiting connection to the robot
    time.sleep(5)

    # reading options from command line
    opt = Options.ReadArgs()

    hmp = HumanMotionPrediction(opt)
    
    # starting human motion prediction (infinite loop)
    hmp.Live()