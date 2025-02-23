import math
import fcl
import torch
from enum import Enum

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion

import constants
from tools import *
from space_geometry import *


RVIZ_DEFAULT_HEADER_NAME = "map" # change to "world" while using Optitrack
RVIZ_DEFAULT_LINK_SCALE = 0.01
COLOR_RED = [255, 255, 0, 0]
COLOR_RED_SHADED = [123, 255, 0, 0]
COLOR_GREEN = [255, 0, 255, 0]
COLOR_GREEN_SHADED = [123, 0, 255, 0]
COLOR_BLUE = [255, 0, 0, 255]
COLOR_YELLOW = [255, 255, 255, 0]
COLOR_PURPLE = [255, 255, 0, 255]
COLOR_PURPLE_SHADED = [123, 255, 0, 255]
COLOR_GRAY = [255, 123, 123, 123]
COLOR_GRAY_SHADED = [123, 123, 123, 123]
COLOR_ORANGE = [255, 255, 123, 0]
COLOR_ORANGE_SHADED = [123, 255, 123, 0]
COLOR_CYAN = [255, 0, 183, 235]
MARKER_ARROW = 0
MARKER_CUBE = 1
MARKER_SPHERE = 2
MARKER_CYLINDER = 3
MARKER_LINE_STRIP = 4
MARKER_LINE_LIST = 5
MARKER_CUBE_LIST = 6
MARKER_SPHERE_LIST = 7
MARKER_POINTS = 8
MARKER_TEXT_VIEW_FACING = 9
MARKER_MESH_RESOURCE = 10
MARKER_TRIANGLE_LIST = 11

class CollisionGeomOptions(Enum):
    TRIANGLE = 0
    BOX = 1
    SPHERE = 2
    ELLIPSOID = 3
    CAPSULE = 4
    CONE = 5
    CYLINDER = 6
    HALFSPACE = 7
    PLANE = 8

class LinkInfo:
    markerId = 1

    def __init__(self, linkName : str, linkId : int, jointIds: list = None) -> None:
        self.name = linkName
        self.id = linkId  

        if (jointIds is None):
            self.startJoint : JointInfo = JointInfo(f"link{linkId}", linkId)
            self.endJoint : JointInfo = JointInfo(f"link{linkId + 1}", linkId + 1)
        else:       
            self.startJoint : JointInfo = JointInfo(constants.HUMAN_POSE25J_JOINT_NAMES[jointIds[0]], jointIds[0])
            self.endJoint : JointInfo = JointInfo(constants.HUMAN_POSE25J_JOINT_NAMES[jointIds[1]], jointIds[1])

        self.boundings : dict[str, LinkBounding] = {}

        self.markers = {}

    def CreateBounding(self, startPoint : np.array, endPoint : np.array, collGeomType : CollisionGeomOptions, collGeomData,
                       key : str):
        if (collGeomType == CollisionGeomOptions.CAPSULE):
            bounding = LinkBounding()
            
            bounding.radius = collGeomData
            bounding.lz = math.sqrt(pow(endPoint[0] - startPoint[0], 2) 
                                   + pow(endPoint[1] - startPoint[1], 2) 
                                   + pow(endPoint[2] - startPoint[2], 2))
            
            # setting radius (plane XY) and height (axis Z) of the capsule
            bounding.collisionGeometry = fcl.Capsule(bounding.radius, bounding.lz)
        else:
            raise NotImplementedError

        # calculating distance vector (from absolute RS to the local RS of the link)
        deltaX = (endPoint[0] - startPoint[0]) / 2
        deltaY = (endPoint[1] - startPoint[1]) / 2
        deltaZ = (endPoint[2] - startPoint[2]) / 2
        bounding.d = np.array([startPoint[0] + deltaX, startPoint[1] + deltaY, startPoint[2] + deltaZ])

        # calculate rotation data
        bounding.q = np.array(Space3D.GetQuaternionFromPointsArray(startPoint, endPoint))
        bounding.R = Space3D.GetRotationMatrixFromPointsArray(startPoint, endPoint)

        # get transformation matrix from quaternion (i.e rotation matrix) and distance vector
        tf_matrix = fcl.Transform(bounding.R, bounding.d)

        # creating collision object
        bounding.collisionObject = fcl.CollisionObject(bounding.collisionGeometry, tf_matrix)

        # saving new bounding in the dictionary
        self.boundings[key] = bounding

    def CreateMarkerFromPoints(self, startPoint : np.array, endPoint : np.array, markerName : str, markerType : int, 
                  markerGeomData : np.array, color : np.array):
        if (markerType >= MARKER_ARROW and markerType <= MARKER_TRIANGLE_LIST):
            marker = Marker()

            marker.type = markerType
            marker.header.frame_id = RVIZ_DEFAULT_HEADER_NAME

            geomData = np.array(markerGeomData)
            if (geomData.shape[0] == 1):
                marker.scale.x = geomData[0]
            elif (geomData.shape[0] == 2):
                marker.scale.x = geomData[0]
                marker.scale.y = geomData[1]
            elif (geomData.shape[0] == 3):
                marker.scale.x = geomData[0]
                marker.scale.y = geomData[1]
                marker.scale.z = geomData[2]
            else:
                raise NotImplementedError

            marker.color.a = color[0] / 255
            marker.color.r = color[1] / 255
            marker.color.g = color[2] / 255
            marker.color.b = color[3] / 255

            if (not self.name + "_" + markerName in self.markers):
                marker.id = LinkInfo.markerId
                LinkInfo.markerId = LinkInfo.markerId + 1
            else:
                marker.id = self.markers[self.name + "_" + markerName].id

            if (markerType == MARKER_ARROW or markerType == MARKER_LINE_LIST or markerType == MARKER_LINE_STRIP):
                if (len(marker.points) == 0):
                    marker.points.append(Point(startPoint[0], startPoint[1], startPoint[2]))
                    marker.points.append(Point(endPoint[0], endPoint[1], endPoint[2]))
                else:
                    marker.points[0] = Point(startPoint[0], startPoint[1], startPoint[2])
                    marker.points[1] = Point(endPoint[0], endPoint[1], endPoint[2])

                # defining default position and orientation
                marker.pose.position.x = 0
                marker.pose.position.y = 0
                marker.pose.position.z = 0
                marker.pose.orientation.x = 0
                marker.pose.orientation.y = 0
                marker.pose.orientation.z = 0
                marker.pose.orientation.w = 1
            else:
                # check if the start point and end point are the same
                if (not np.all(startPoint == endPoint)):
                    pivot = startPoint + (endPoint - startPoint) / 2

                    marker.pose.position.x = pivot[0]
                    marker.pose.position.y = pivot[1]
                    marker.pose.position.z = pivot[2]

                    q = Space3D.GetQuaternionFromPointsArray(startPoint, endPoint)
                    marker.pose.orientation = Quaternion(*q)
                else:
                    pivot = startPoint

                    marker.pose.position.x = pivot[0]
                    marker.pose.position.y = pivot[1]
                    marker.pose.position.z = pivot[2]

                    # defining default orientation
                    marker.pose.orientation.x = 0
                    marker.pose.orientation.y = 0
                    marker.pose.orientation.z = 0
                    marker.pose.orientation.w = 1

            self.markers[self.name + "_" + markerName] = marker
        else:
            raise NotImplementedError
        
    def CreateMarkerFromMatrix(self, d : np.array, q : np.array, markerName : str, markerType : int, 
                  markerGeomData : np.array, color : np.array):
        if (markerType >= MARKER_ARROW and markerType <= MARKER_TRIANGLE_LIST):
            marker = Marker()

            marker.type = markerType
            marker.header.frame_id = RVIZ_DEFAULT_HEADER_NAME

            geomData = np.array(markerGeomData)
            if (geomData.shape[0] == 1):
                marker.scale.x = geomData[0]
            elif (geomData.shape[0] == 2):
                marker.scale.x = geomData[0]
                marker.scale.y = geomData[1]
            elif (geomData.shape[0] == 3):
                marker.scale.x = geomData[0]
                marker.scale.y = geomData[1]
                marker.scale.z = geomData[2]
            else:
                raise NotImplementedError

            marker.color.a = color[0] / 255
            marker.color.r = color[1] / 255
            marker.color.g = color[2] / 255
            marker.color.b = color[3] / 255

            if (not self.name + "_" + markerName in self.markers):
                marker.id = LinkInfo.markerId
                LinkInfo.markerId = LinkInfo.markerId + 1
            else:
                marker.id = self.markers[self.name + "_" + markerName].id

            marker.pose.position.x = d[0]
            marker.pose.position.y = d[1]
            marker.pose.position.z = d[2]

            marker.pose.orientation = Quaternion(*q)

            self.markers[self.name + "_" + markerName] = marker
        else:
            raise NotImplementedError

    def GetJoints(self):
        return self.startJoint.id, self.endJoint.id
    
    def GetNormGts(self):
        sj_norms = []
        ej_norms = []

        for i in range(len(self.startJoint.gts)):
            sj_norms.append(deepcopy(self.startJoint.gts[i].norm))
            ej_norms.append(deepcopy(self.endJoint.gts[i].norm))
        
        return sj_norms, ej_norms

    def GetQuotesGts(self):
        sj_quotes = []
        ej_quotes = []

        for i in range(len(self.startJoint.gts)):
            sj_quotes.append(deepcopy(self.startJoint.gts[i].quotes))
            ej_quotes.append(deepcopy(self.endJoint.gts[i].quotes))
        
        return sj_quotes, ej_quotes

    def GetNormPreds(self):
        sj_norms = []
        ej_norms = []

        for i in range(len(self.startJoint.preds)):
            sj_norms.append(deepcopy(self.startJoint.preds[i].norm))
            ej_norms.append(deepcopy(self.endJoint.preds[i].norm))
        
        return sj_norms, ej_norms
    
    def GetQuotesGts(self):
        sj_quotes = []
        ej_quotes = []

        for i in range(len(self.startJoint.preds)):
            sj_quotes.append(deepcopy(self.startJoint.preds[i].quotes))
            ej_quotes.append(deepcopy(self.endJoint.preds[i].quotes))
        
        return sj_quotes, ej_quotes

    def GetJpes(self):
        sj_jpes = []
        ej_jpes = []

        for i in range(len(self.startJoint.jpes)):
            sj_jpes.append(deepcopy(self.startJoint.jpes[i]))
            sj_jpes.append(deepcopy(self.endJoint.jpes[i]))

        return sj_jpes, ej_jpes

class LinkBounding:
    def __init__(self) -> None:
        # distance vector, rotation matrix and its quaternion representation for orientation
        self.d = [0, 0, 0]
        self.q = [0, 0, 0, 1]
        self.R = np.eye(3, 3)
        self.radius = 0
        self.lz = 0

        # defining fcl library objects
        self.collisionGeometry = None
        self.collisionObject = None

class JointInfo:
    def __init__(self, name : str, id : int) -> None:
        self.name = name
        self.id = id
        self.gts : list[JointPosition] = []
        self.preds : list[JointPosition] = []
        self.jpes = []            # Joint Position Error = abs(gts[i].norm - preds[i].norm)
        self.sigmas = []          # standard deviation samples of JPE
        self.mus = []             # mean value samples of JPE
        self.actualMu = 0         # last calculated mean value of JPEs samples
        self.actualSigma = 0      # last calculated standard deviation of JPE samples

class JointPosition:
    def __init__(self, quotes : np.array) -> None:
        # maximum 3 coordinates must be present (i.e x,y,z)
        if (quotes.shape[0] <= 3 and len(quotes.shape) == 1):
            self.quotes = np.array(quotes)
        else:
            raise NotImplementedError
        
        self.norm = np.linalg.norm(self.quotes)

class MinimumDistance:
    def __init__(self, p1 : np.array, p2 : np.array) -> None:
        if (p1.shape[0] == 3 and p1.shape[0] == 3):
            self.p1 = p1
            self.p2 = p2
            self.norm = np.linalg.norm(p1 - p2)
            self.versor = (p1 - p2) / self.norm
        else:
            raise NotImplementedError

class FrameTypes(Enum):
    GT = 0
    PRED = 1

class Pose3D:
    def __init__(self, name : str, totLinks : int, maxFrames : int, jointIds : list = None, k0 : int = None, poseIsHuman = True) -> None:
        self.name = name
        self.k0 = k0
        self.poseIsHuman = poseIsHuman

        self.maxFrames = maxFrames # maximum number of frames saved before cycling the data buffer

        self.linkInfos : list[LinkInfo] = []
        
        for i in range(0, totLinks):
            if (jointIds is None):
                linkInfo = LinkInfo(f"link{i}", i, constants.HUMAN_POSE25J_REDUCED_LINK_JOINT_INDEXS[i])
            else:
                linkInfo = LinkInfo(f"link{i}", i, jointIds[i])
                
            self.linkInfos.append(linkInfo)

        self.minimumDistances : dict[str, MinimumDistance] = {}

        self.gt_linkMinDistance : LinkInfo = LinkInfo("link", totLinks)
        self.pred_linkMinDistance : LinkInfo = LinkInfo("link", totLinks)

        self.mpjpes : list[float] = []

    def Update(self, newFrame : np.array, frameType : FrameTypes, boundingKey : str, saveFrames : bool = True):
        startPoint = None
        endPoint = None
        ptrLink = 0

        if (self.poseIsHuman):
            # scorro lista indici joint associati ai singoli link della posa
            for startEndIndexs in constants.HUMAN_POSE17J_LINK_JOINT_INDEXS:
                startPoint = JointPosition(newFrame[startEndIndexs[0]] * constants.SCALE_MM_TO_M)
                endPoint = JointPosition(newFrame[startEndIndexs[1]] * constants.SCALE_MM_TO_M)
            
                if (frameType == FrameTypes.GT):
                    # check if buffer is full
                    if (len(self.linkInfos[ptrLink].startJoint.gts) > self.maxFrames):
                        totFrames = len(self.linkInfos[ptrLink].startJoint.gts)

                        # removing first frame saved, which is the oldest one (FIFO logic)
                        self.linkInfos[ptrLink].startJoint.gts = self.linkInfos[ptrLink].startJoint.gts[1:totFrames]
                        self.linkInfos[ptrLink].endJoint.gts = self.linkInfos[ptrLink].endJoint.gts[1:totFrames]

                    if (saveFrames):
                        self.linkInfos[ptrLink].startJoint.gts.append(deepcopy(startPoint))
                        self.linkInfos[ptrLink].endJoint.gts.append(deepcopy(endPoint))

                    self.linkInfos[ptrLink].CreateMarkerFromPoints(startPoint.quotes, 
                                                         endPoint.quotes, 
                                                         f"gt_link{ptrLink}", 
                                                         MARKER_LINE_LIST, 
                                                         [RVIZ_DEFAULT_LINK_SCALE], 
                                                         COLOR_GREEN_SHADED)

                    # N.B: using a default radius value for the GT capsules, since the 3d pose does not take into
                    #      account the "physical" dimensions of the real body
                    self.linkInfos[ptrLink].CreateBounding(startPoint.quotes, 
                                                           endPoint.quotes, 
                                                           CollisionGeomOptions.CAPSULE, 
                                                           RVIZ_DEFAULT_LINK_SCALE,
                                                           boundingKey)
                elif (frameType == FrameTypes.PRED):
                    # check if buffer is full
                    if (len(self.linkInfos[ptrLink].startJoint.preds) > self.maxFrames):
                        totFrames = len(self.linkInfos[ptrLink].startJoint.preds)

                        # removing first frame saved, which is the oldest one (FIFO logic)
                        self.linkInfos[ptrLink].startJoint.preds = self.linkInfos[ptrLink].startJoint.preds[1:totFrames]
                        self.linkInfos[ptrLink].endJoint.preds = self.linkInfos[ptrLink].endJoint.preds[1:totFrames]

                    if (saveFrames):
                        self.linkInfos[ptrLink].startJoint.preds.append(deepcopy(startPoint))
                        self.linkInfos[ptrLink].endJoint.preds.append(deepcopy(endPoint))

                    self.linkInfos[ptrLink].CreateMarkerFromPoints(startPoint.quotes, 
                                                         endPoint.quotes, 
                                                         f"pred_link{ptrLink}",
                                                         MARKER_LINE_LIST, 
                                                         [RVIZ_DEFAULT_LINK_SCALE], 
                                                         COLOR_BLUE)

                    if (self.CalculateErrors(ptrLink, saveFrames)):
                        if (self.linkInfos[ptrLink].startJoint.actualSigma >= self.linkInfos[ptrLink].endJoint.actualSigma):
                            rad = 3 * self.linkInfos[ptrLink].startJoint.actualSigma
                        else:
                            rad = 3 * self.linkInfos[ptrLink].endJoint.actualSigma

                        self.linkInfos[ptrLink].CreateBounding(startPoint.quotes, 
                                                                   endPoint.quotes, 
                                                                   CollisionGeomOptions.CAPSULE, 
                                                                   rad,
                                                                   boundingKey)

                        self.linkInfos[ptrLink].CreateMarkerFromPoints(startPoint.quotes, 
                                                                 endPoint.quotes, 
                                                                 f"pred_link_variance{ptrLink}", 
                                                                 MARKER_CYLINDER, 
                                                                 [2 * self.linkInfos[ptrLink].boundings[boundingKey].radius, 
                                                                  2 * self.linkInfos[ptrLink].boundings[boundingKey].radius, 
                                                                  self.linkInfos[ptrLink].boundings[boundingKey].lz], 
                                                                 COLOR_PURPLE_SHADED)  
                        self.linkInfos[ptrLink].CreateMarkerFromPoints(startPoint.quotes, 
                                                                 startPoint.quotes, 
                                                                 f"start_joint_variance{ptrLink}", 
                                                                 MARKER_SPHERE, 
                                                                 [2 * self.linkInfos[ptrLink].boundings[boundingKey].radius, 
                                                                  2 * self.linkInfos[ptrLink].boundings[boundingKey].radius, 
                                                                  2 * self.linkInfos[ptrLink].boundings[boundingKey].radius], 
                                                                 COLOR_RED_SHADED)
                        
                        # N.B: the marker related to each joint is a sphere in a single point! That's why the same coordinates are given
                        #      for its creation
                        self.linkInfos[ptrLink].CreateMarkerFromPoints(endPoint.quotes, 
                                                                 endPoint.quotes, 
                                                                 f"end_joint_variance{ptrLink}", 
                                                                 MARKER_SPHERE, 
                                                                 [2 * self.linkInfos[ptrLink].boundings[boundingKey].radius, 
                                                                  2 * self.linkInfos[ptrLink].boundings[boundingKey].radius, 
                                                                  2 * self.linkInfos[ptrLink].boundings[boundingKey].radius], 
                                                                 COLOR_RED_SHADED)

                ptrLink = ptrLink + 1
        else:
            for startEndXYZs in constants.ROBOT_LINK_JOINT_INDEXS:
                startPoint = JointPosition(newFrame[startEndXYZs[0]])
                endPoint = JointPosition(newFrame[startEndXYZs[1]])

                # check if buffer is full
                if (len(self.linkInfos[ptrLink].startJoint.gts) > self.maxFrames):
                    totFrames = len(self.linkInfos[ptrLink].startJoint.gts)

                    # removing first frame saved, which is the oldest one (FIFO logic)
                    self.linkInfos[ptrLink].startJoint.gts = self.linkInfos[ptrLink].startJoint.gts[1:totFrames]
                    self.linkInfos[ptrLink].endJoint.gts = self.linkInfos[ptrLink].endJoint.gts[1:totFrames]

                if (saveFrames):
                    self.linkInfos[ptrLink].startJoint.gts.append(startPoint)
                    self.linkInfos[ptrLink].endJoint.gts.append(endPoint)

                self.linkInfos[ptrLink].CreateMarkerFromPoints(startPoint.quotes, 
                                                     endPoint.quotes, 
                                                     f"gt_link{ptrLink}", 
                                                     MARKER_ARROW, 
                                                     [RVIZ_DEFAULT_LINK_SCALE * 2], 
                                                     COLOR_ORANGE)
                
                self.linkInfos[ptrLink].CreateBounding(startPoint.quotes, 
                                       endPoint.quotes, 
                                       CollisionGeomOptions.CAPSULE, 
                                       RVIZ_DEFAULT_LINK_SCALE,
                                       "gt")
                
                ptrLink = ptrLink + 1

    def ClearData(self):
        for linkInfo in self.linkInfos:
            linkInfo.startJoint = JointInfo("J" + str(linkInfo.id), linkInfo.id)
            linkInfo.endJoint = JointInfo("J" + str(linkInfo.id + 1), linkInfo.id + 1)

    def CalculateErrors(self, linkNbr : int, saveFrames : bool = True):
        ptr_gtFrame = len(self.linkInfos[linkNbr]. startJoint.gts) - 1
        ptr_predFrame = (len(self.linkInfos[linkNbr].startJoint.preds) - 1) - self.k0

        if (ptr_predFrame >= 0):
            jpe_startJoint = abs(self.linkInfos[linkNbr].startJoint.gts[ptr_gtFrame].norm 
                                 - self.linkInfos[linkNbr].startJoint.preds[ptr_predFrame].norm)
            jpe_endJoint = abs(self.linkInfos[linkNbr].endJoint.gts[ptr_gtFrame].norm 
                               - self.linkInfos[linkNbr].endJoint.preds[ptr_predFrame].norm)

            # check if buffer is full
            if (len(self.linkInfos[linkNbr].startJoint.jpes) > self.maxFrames):
                totFrames = len(self.linkInfos[linkNbr].startJoint.jpes)

                # removing first frame saved, which is the oldest one (FIFO logic)
                self.linkInfos[linkNbr].startJoint.jpes = self.linkInfos[linkNbr].startJoint.jpes[1:totFrames]
                self.linkInfos[linkNbr].endJoint.jpes = self.linkInfos[linkNbr].endJoint.jpes[1:totFrames]

            if (saveFrames):
                self.linkInfos[linkNbr].startJoint.jpes.append(jpe_startJoint)
                self.linkInfos[linkNbr].endJoint.jpes.append(jpe_endJoint)

                mu, sigma2, sigma = DataAnalyzer.CalculateStatistics(self.linkInfos[linkNbr].startJoint.jpes)
                self.linkInfos[linkNbr].startJoint.mus.append(mu)
                self.linkInfos[linkNbr].startJoint.actualMu = mu
                self.linkInfos[linkNbr].startJoint.sigmas.append(sigma)
                self.linkInfos[linkNbr].startJoint.actualSigma = sigma

                mu, sigma2, sigma = DataAnalyzer.CalculateStatistics(self.linkInfos[linkNbr].endJoint.jpes)
                self.linkInfos[linkNbr].endJoint.mus.append(mu)
                self.linkInfos[linkNbr].endJoint.actualMu = mu
                self.linkInfos[linkNbr].endJoint.sigmas.append(sigma)
                self.linkInfos[linkNbr].endJoint.actualSigma = sigma

            return True
        else:
            return False

def CalculateDistanceLinks(link1 : LinkInfo, link2 : LinkInfo, boundingKey1: str, boundingKey2: str):
    if (boundingKey1 in link1.boundings and boundingKey2 in link2.boundings):
        if (not link1.boundings[boundingKey1].collisionObject is None and not link2.boundings[boundingKey2].collisionObject is None):
            request = fcl.DistanceRequest()
            result = fcl.DistanceResult()

            distance = fcl.distance(link1.boundings[boundingKey1].collisionObject, 
                                    link2.boundings[boundingKey2].collisionObject, 
                                    request, 
                                    result)

            p1 = result.nearest_points[0]
            p2 = result.nearest_points[1]

            return distance, p1, p2
        else:
            return 0, 0, 0
    else:
        return 0, 0, 0
    
def CalculateDistancePoses(pose1 : Pose3D, pose2 : Pose3D, boundingKey1 : str, boundingKey2 : str, drawDistance = True):
    distance = 0
    p1 = np.array([0, 0, 0])
    p2 = np.array([0, 0, 0])
    flagStartup = False

    for linkInfo1 in pose1.linkInfos:
        for linkInfo2 in pose2.linkInfos:
            current_dist, current_p1, current_p2 = CalculateDistanceLinks(linkInfo1, linkInfo2, boundingKey1, boundingKey2)

            if ((current_dist > 0 and current_dist < distance) or not flagStartup):
                distance = current_dist
                p1 = current_p1
                p2 = current_p2

            flagStartup = True

    if (distance > 0 and drawDistance):
        # saving information
        pose1.minimumDistances[boundingKey1] = MinimumDistance(p1, p2)

        # drawing minimum distance
        if ("gt" in boundingKey1):
            pose1.gt_linkMinDistance.CreateMarkerFromPoints(pose1.minimumDistances[boundingKey1].p1, 
                                     pose1.minimumDistances[boundingKey1].p2, 
                                     "gt_min_distance",
                                     MARKER_CYLINDER, 
                                     [0.01, 0.01, distance], 
                                     COLOR_YELLOW)
        elif ("pred" in boundingKey1):
            pose1.pred_linkMinDistance.CreateMarkerFromPoints(pose1.minimumDistances[boundingKey1].p1, 
                                     pose1.minimumDistances[boundingKey1].p2, 
                                     "pred_min_distance",
                                     MARKER_CYLINDER, 
                                     [0.01, 0.01, distance], 
                                     COLOR_CYAN)
        else:
            raise NotImplementedError

