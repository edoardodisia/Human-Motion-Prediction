#!7/usr/bin/env python3

import time
from enum import Enum
import rospy
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion
import tf
import math
from space_geometry import *
import os
import rosgraph
import rosnode

class ROSCommOptions(Enum):
    PUBLISHING = 0
    SUBSCRIBING = 1

class ROSTopic:
    def __init__(self) -> None:
        self.dataReceived : bool = False
        self.messageType = None
        self.bufferSize = int
        self.buffer : list = []

class ROSNode:
    def __init__(self, nodeName : str):
        self.rosNodeName :str = nodeName

        # check if ROS master is active
        if (not rosgraph.is_master_online()):
            os.system("gnome-terminal -e 'bash -c \"roscore\" '")

            # waiting activation
            time.sleep(1)

        # initialize ROS node
        rospy.init_node(self.rosNodeName, anonymous = True)
        
        # creazione dizionari per gestioni comunicazioni multiple 
        # (chiave = topic, valore = publisher o subscriber)
        self.publishers : dict[str, rospy.Publisher] = {}
        self.subscribers : dict[str, rospy.Subscriber] = {}

        # as client communication to the service (server)
        self.services : dict[str, rospy.Service] = {}

        # creazione dizionario dati gestiti 
        # (chiave = topic, valore = dati pubblicati o ricevuti)
        self.dictTopics : dict[str, ROSTopic] = {}

    def AddTopic(self, topicName : str, commOption : ROSCommOptions, messageType, bufferSize : int = 1):
        # check if topic name syntax is correct
        topicFullName = ''     
        if (not topicName[0] == '/'):
            topicFullName = '/' + topicName
        else:
            topicFullName = topicName

        if (commOption == ROSCommOptions.PUBLISHING):
            self.publishers[topicName] = rospy.Publisher(topicFullName, messageType, queue_size = 10)
        elif (commOption == ROSCommOptions.SUBSCRIBING):
            self.subscribers[topicName] = rospy.Subscriber(topicFullName, messageType, self.OnTopicDataReceived, topicName)
        else:
            raise NotImplementedError

        # creating new topic
        rosTopic = ROSTopic()
        rosTopic.messageType = messageType
        rosTopic.bufferSize = bufferSize

        # creating buffer for storing data
        for _ in range(bufferSize):
            rosTopic.buffer.append(messageType)

        # adding topic object to the managed topic dictionary
        self.dictTopics[topicName] = rosTopic

    def AddService(self, serviceName : str, serviceClass):
        self.services[serviceName] = rospy.ServiceProxy(serviceName, serviceClass)

    def SendDataOnTopic(self, topicName : str, dataToSend = None):
        if (topicName != ""):
            if (not dataToSend is None):
                # sending new data passed as parameter
                self.publishers[topicName].publish(dataToSend)     
            else:  
                # sending current data stored in the buffer
                self.publishers[topicName].publish(self.dictTopics[topicName].buffer) 

    def OnTopicDataReceived(self, dataReceived, topicName):
        if (not self.dictTopics[topicName].dataReceived):
            self.dictTopics[topicName].dataReceived = True

            totDataStored = len(self.dictTopics[topicName].buffer)

            # verifica se il buffer è pieno
            if (totDataStored < self.dictTopics[topicName].bufferSize):
                # caso buffer non ancora pieno
                self.dictTopics[topicName].buffer.append(dataReceived)
            else:
                # ricircolo memoria
                self.dictTopics[topicName].buffer = self.dictTopics[topicName].buffer[1:(totDataStored)]
                self.dictTopics[topicName].buffer.append(dataReceived)

    def CloseCommunication(self):
        pass

    def GetActiveTopics(self):
        topics = {}

        topics["published"] = self.publishers.keys()
        topics["subscribed"] = self.subscribers.keys()

        return topics


if __name__ == '__main__':
    rosNode = ROSNode("ros_node")
    rosNode.AddTopic("test_rviz", ROSCommOptions.PUBLISHING, MarkerArray)

    m1 = Marker()
    m1.type = Marker.ARROW
    m1.header.frame_id = "map"
    m1.id = 1
    m1.scale.x = 0.1
    m1.scale.y = 0.1
    m1.scale.z = 0.1

    m1.color.a = 1
    m1.color.r = 1.0
    m1.color.g = 0.0
    m1.color.b = 0.0

    p1 = Point(0, 0, 0)
    p2 = Point(1, 2, 3)
    m1.points.append(p1)
    m1.points.append(p2)

    m1.pose.position.x = 0
    m1.pose.position.y = 0
    m1.pose.position.z = 0

    m2 = Marker()
    m2.type = Marker.CYLINDER
    m2.header.frame_id = "map"
    m2.id = 2
        
    # dimensione rispetto gli assi, espressa in [m]
    # N.B: i marker LINE utilizzano solo "scale.x"!
    m2.scale.x = 0.25
    m2.scale.y = 0.25
    m2.scale.z = 0.25

    m2.color.a = 1
    m2.color.r = 0.0
    m2.color.g = 1.0
    m2.color.b = 0.0

    # impostazione sistema di riferimento
    m2.pose.position.x = 0
    m2.pose.position.y = 0
    m2.pose.position.z = 0

    deltaX = p2.x - p1.x
    deltaY = p2.y - p1.y
    deltaZ = p2.z - p1.z

    rotX = math.atan2(deltaZ, deltaY)
    rotY = math.atan2(deltaZ, deltaX)
    rotZ = math.atan2(deltaY, deltaX)

    q = tf.transformations.quaternion_from_euler(rotX, rotY, rotZ)

    m2.pose.orientation = Quaternion(*q)

    m3 = Marker()
    m3.type = Marker.ARROW
    m3.header.frame_id = "map"
    m3.id = 3
    m3.scale.x = 0.1
    m3.scale.y = 0.1
    m3.scale.z = 0.1

    m3.color.a = 1
    m3.color.r = 0.0
    m3.color.g = 0.0
    m3.color.b = 1.0

    m3.pose.position.x = 0
    m3.pose.position.y = 0
    m3.pose.position.z = 0

    q = tf.transformations.quaternion_from_euler(rotX + 1.57, rotY, rotZ)

    m3.pose.orientation = Quaternion(*q)

    markerArray = MarkerArray()
    markerArray.markers.append(m1)
    markerArray.markers.append(m2)
    markerArray.markers.append(m3)

    i = 0
    while(i < 50):
        rosNode.SendDataOnTopic("test_rviz", markerArray)
        i = i + 1

        # delay to avoid stressing the CPU
        time.sleep(0.025)
        
       