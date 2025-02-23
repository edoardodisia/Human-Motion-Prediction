import math
import numpy as np
import tf
from geometry_msgs.msg import Point
import tf.transformations
from copy import deepcopy
import dill

class Space3D:
    def GetAnglesFromPoints(startPoint : Point, endPoint : Point):
        roll = 0
        pitch = - (math.pi/2 + math.asin((endPoint.z - startPoint.z) / math.sqrt(pow(endPoint.x - startPoint.x, 2) 
                + pow(endPoint.y - startPoint.y, 2) + pow(endPoint.z - startPoint.z, 2))))
        yaw = - math.atan2((startPoint.y - endPoint.y), (endPoint.x - startPoint.x))

        return roll, pitch, yaw
    
    def GetAnglesFromArrays(startPoint : np.array, endPoint : np.array):
        if (not (len(startPoint) == 3 and len(endPoint) == 3)):
            raise NotImplementedError
        
        roll = 0
        pitch = - (math.pi/2 + math.asin((endPoint[2] - startPoint[2]) / math.sqrt(pow(endPoint[0] - startPoint[0], 2) 
                + pow(endPoint[1] - startPoint[1], 2) + pow(endPoint[2] - startPoint[2], 2))))
        yaw = - math.atan2((startPoint[1] - endPoint[1]), (endPoint[0] - startPoint[0]))

        return roll, pitch, yaw
    
    def GetQuaternionFromPointsXYZ(startPoint : Point, endPoint : Point): 
        roll = 0
        pitch = - (math.pi/2 + math.asin((endPoint.z - startPoint.z) / math.sqrt(pow(endPoint.x - startPoint.x, 2) 
                + pow(endPoint.y - startPoint.y, 2) + pow(endPoint.z - startPoint.z, 2))))
        yaw = - math.atan2((startPoint.y - endPoint.y), (endPoint.x - startPoint.x))

        q = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        q_normalized = tf.transformations.unit_vector(q)

        return q_normalized
    
    def GetQuaternionFromPointsArray(startPoint : np.array, endPoint : np.array):
        if (not (len(startPoint) == 3 and len(endPoint) == 3)):
            raise NotImplementedError
        
        roll = 0
        pitch = - (math.pi/2 + math.asin((endPoint[2] - startPoint[2]) / math.sqrt(pow(endPoint[0] - startPoint[0], 2) 
                + pow(endPoint[1] - startPoint[1], 2) + pow(endPoint[2] - startPoint[2], 2))))
        yaw = - math.atan2((startPoint[1] - endPoint[1]), (endPoint[0] - startPoint[0]))

        q = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        q_normalized = tf.transformations.unit_vector(q)

        return q_normalized
    
    def GetRotationMatrixFromPointsArray(startPoint : np.array, endPoint : np.array):
        if (not (len(startPoint) == 3 and len(endPoint) == 3)):
            raise NotImplementedError
        
        roll = 0
        pitch = - (math.pi/2 + math.asin((endPoint[2] - startPoint[2]) / math.sqrt(pow(endPoint[0] - startPoint[0], 2) 
                + pow(endPoint[1] - startPoint[1], 2) + pow(endPoint[2] - startPoint[2], 2))))
        yaw = - math.atan2((startPoint[1] - endPoint[1]), (endPoint[0] - startPoint[0]))

        q = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        q_normalized = tf.transformations.unit_vector(q)

        # N.B: "quaternion_matrix" is a matrix 4x4 in the homogeneous space, so 
        #       the 3x3 submatrix will be returned (Cartesian space)
        R = tf.transformations.quaternion_matrix(q_normalized)[0:3, 0:3]

        return R
    
    