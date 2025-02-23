from matplotlib import pyplot as plt
from enum import Enum
import numpy as np
from tools import *


class MeasureUnits(Enum):
    NONE = ""
    METERS = "[m]"
    MILLIMETERS = "[mm]"
    FRAMES = "[frame]"
    METERS_PER_SECONDS = "[m/s]"
    DISCRE_TIME = "[k]"

class GraphTypes(Enum):
    GRAPH_2D = 0
    GRAPH_3D = 1

class AxisInfo:
    def __init__(self, label : str, showFullScale : bool, data, measureUnit : MeasureUnits = MeasureUnits.NONE, limits = None):
        self.label = label
        self.showFullScale = showFullScale

        # check if data has only 1 dimension, since it will be relate to a single ax of a plot
        if (len(np.array(data).shape) == 1):
            self.data = np.array(data)
        else:
            raise NotImplementedError

        self.limits = limits
        self.measureUnit = " " + measureUnit.value

class Graph2D:
    def __init__(self, x : AxisInfo, y : AxisInfo):
        self.x = x
        self.y = y

class Graph3D:
    def __init__(self, x : AxisInfo, y : AxisInfo, z : AxisInfo) -> None:
        self.x = x
        self.y = y
        self.z = z

class PlotInfo:
    def __init__(self, graph, title : str = "", legend : str = ""):
        self.title = title
        self.legend = legend

        if (isinstance(graph, Graph2D) or isinstance(graph, Graph3D)):
            self.graph = graph
        else:
            raise NotImplementedError

class MatlabFigureInfo:
    def __init__(self):
        self.cntAxes = 1
        self.figure = None

class MatlabAxesInfo: 
    def __init__(self):
        self.activeColor = 0
        self.activeStyle = 0
        self.ax = None

class MatlabDrawer:
    defaultColors =  {0 : "green",
                      1 : "blue",
                      2: "magenta",
                      3 : "red",
                      4 : "purple",
                      5 : "brown",
                      6: "pink",
                      7: "gray",
                      8: "olive",
                      9: "black",
                      10 : "orange"
                      }    
    lineStyles = {0 : "-",
                  1 : "--",
                  2 : "-.",
                  3 : ":"}

    cntFigures = 1

    def __init__(self):
        self.figureInfos = {}
        self.matlabAxInfos = {}

    def AddFigure(self, figureTitle : str):
        self.figureInfos[figureTitle] = MatlabFigureInfo()
        self.figureInfos[figureTitle].cntAxes = self.figureInfos[figureTitle].cntAxes
        self.figureInfos[figureTitle].figure = plt.figure(MatlabDrawer.cntFigures)
        self.figureInfos[figureTitle].figure.suptitle(figureTitle)

        MatlabDrawer.cntFigures = MatlabDrawer.cntFigures + 1

    def AddAx(self, figureTitle : str, axTitle : str, showGrid : bool, graphType : GraphTypes):
        if (graphType == GraphTypes.GRAPH_2D):
            self.matlabAxInfos[axTitle] = MatlabAxesInfo()
            self.matlabAxInfos[axTitle].ax = self.figureInfos[figureTitle].figure.add_subplot(
                int(220 + self.figureInfos[figureTitle].cntAxes))

        elif (graphType == GraphTypes.GRAPH_3D):
            self.matlabAxInfos[axTitle] = MatlabAxesInfo()
            self.matlabAxInfos[axTitle].ax = self.figureInfos[figureTitle].figure.add_subplot(
                int(220 + self.figureInfos[figureTitle].cntAxes), projection='3d')

        self.matlabAxInfos[axTitle].ax.grid(showGrid)
        
        self.figureInfos[figureTitle].cntAxes = self.figureInfos[figureTitle].cntAxes + 1

    def AddPlot(self, axTitle : str, plotInfo : PlotInfo, graphColor : str = "", graphStyle : str = ""):
        # check if there are specific color to use
        if (graphColor != ""):
            color = graphColor
        else:
            color = MatlabDrawer.defaultColors[self.matlabAxInfos[axTitle].activeColor]

        # check if there are specific line style to use
        if (graphStyle != ""):
            linestyle = graphStyle
        else:
            linestyle = MatlabDrawer.lineStyles[self.matlabAxInfos[axTitle].activeStyle]

        if (isinstance(plotInfo.graph, Graph2D)):
            self.matlabAxInfos[axTitle].ax.set_xlabel(plotInfo.graph.x.label + plotInfo.graph.x.measureUnit, rotation = 0)
            self.matlabAxInfos[axTitle].ax.set_ylabel(plotInfo.graph.y.label + plotInfo.graph.y.measureUnit, rotation = 90)

            if (not plotInfo.graph.x.limits == None):
                self.matlabAxInfos[axTitle].ax.set_xlim(plotInfo.graph.x.limits)
            if (not plotInfo.graph.y.limits == None):
                self.matlabAxInfos[axTitle].ax.set_ylim(plotInfo.graph.y.limits)

            if (plotInfo.graph.x.showFullScale):
                self.matlabAxInfos[axTitle].ax.set_xticks(plotInfo.graph.x.data)
            if (plotInfo.graph.y.showFullScale):
                self.matlabAxInfos[axTitle].ax.set_yticks(plotInfo.graph.y.data)

            self.matlabAxInfos[axTitle].ax.plot(plotInfo.graph.x.data, plotInfo.graph.y.data, 
                                    color = color,
                                    linestyle = linestyle,
                                    label = plotInfo.legend)
        elif (isinstance(plotInfo.graph, Graph3D)):
            self.matlabAxInfos[axTitle].ax.set_xlabel(plotInfo.graph.x.label + plotInfo.graph.x.measureUnit, rotation = 0)
            self.matlabAxInfos[axTitle].ax.set_ylabel(plotInfo.graph.y.label + plotInfo.graph.y.measureUnit, rotation = 0)
            self.matlabAxInfos[axTitle].ax.set_zlabel(plotInfo.graph.z.label + plotInfo.graph.z.measureUnit, rotation = 0)

            if (not plotInfo.graph.x.limits == None):
                self.matlabAxInfos[axTitle].ax.set_xlim3d(plotInfo.graph.x.limits)
            if (not plotInfo.graph.y.limits == None):
                self.matlabAxInfos[axTitle].ax.set_ylim3d(plotInfo.graph.y.limits)
            if (not plotInfo.graph.z.limits == None):
                self.matlabAxInfos[axTitle].ax.set_zlim3d(plotInfo.graph.z.limits)

            if (plotInfo.graph.x.showFullScale):
                self.matlabAxInfos[axTitle].ax.set_xticks(plotInfo.graph.x.data)
            if (plotInfo.graph.y.showFullScale):
                self.matlabAxInfos[axTitle].ax.set_yticks(plotInfo.graph.y.data)
            if (plotInfo.graph.z.showFullScale):
                self.matlabAxInfos[axTitle].ax.set_zticks(plotInfo.graph.z.data)

            self.matlabAxInfos[axTitle].ax.plot(plotInfo.graph.x.data, plotInfo.graph.y.data, plotInfo.graph.z.data, 
                                    color = color,
                                    linestyle = linestyle,
                                    label = plotInfo.legend)
        else:
            raise NotImplementedError

        self.matlabAxInfos[axTitle].ax.set_title(plotInfo.title)

        # command required in order to show the legend
        self.matlabAxInfos[axTitle].ax.legend()

        # changing active color and style if one of the default was used (i.e none where specified as parameters)
        if (graphColor == "" and graphStyle == ""):
            if (self.matlabAxInfos[axTitle].activeColor < len(MatlabDrawer.defaultColors) - 1):
                self.matlabAxInfos[axTitle].activeColor = self.matlabAxInfos[axTitle].activeColor + 1
            else:
                self.matlabAxInfos[axTitle].activeColor = 0

                if (self.matlabAxInfos[axTitle].activeStyle < len(MatlabDrawer.lineStyles) - 1):
                    self.matlabAxInfos[axTitle].activeStyle = self.matlabAxInfos[axTitle].activeStyle + 1
                else:
                    self.matlabAxInfos[axTitle].activeStyle = 0        

    def AddText(self, axTitle, text : str):
        self.matlabAxInfos[axTitle].ax.text(8, 13, text)

    def ShowPlots(self):
        for key in self.figureInfos.keys():
            self.figureInfos[key].figure.tight_layout()
            self.figureInfos[key].figure.show()
        
        # waiting user input to avoid automatic closing of the plots displayed
        input()

if __name__ == '__main__':
    ptrColor = 0

    files = ["experiment_h1_ test0",
             "experiment_h2_ test0",
             "experiment_h3_ test0"]
    
    axLabels = [[],
                [],
                []]

    dataFiles = {}
    for file in files:
        dataFiles[file] = FileManager.DeserializeJson(fileName = file, subFolder = "robot_data", dataAsDict = True)

    figureTitle = "Robot experiment data"
    matlabDrawer = MatlabDrawer()
    matlabDrawer.AddFigure(figureTitle)

    matlabDrawer.AddAx(figureTitle, "Velocity speed limit", False, GraphTypes.GRAPH_2D) 
    matlabDrawer.AddAx(figureTitle, "Velocity robot versus human", False, GraphTypes.GRAPH_2D) 
    matlabDrawer.AddAx(figureTitle, "Speed scaling factor", False, GraphTypes.GRAPH_2D) 

    ptrHorizon = 1
    for keyFile in dataFiles:
        data = dataFiles[keyFile]
        for keyData in data:
            if (keyData == "speed_limit"):
                unitMeasure = MeasureUnits.METERS_PER_SECONDS
                axTitle = "Velocity speed limit"
                labelY = r'$V_{safety}$'
            elif (keyData == "vmax_human"):
                unitMeasure = MeasureUnits.METERS_PER_SECONDS
                axTitle = "Velocity robot versus human"
                labelY = r'$V_{rh}$'
            elif (keyData == "alpha"):
                unitMeasure = MeasureUnits.NONE
                axTitle = "Speed scaling factor"
                labelY = r'$\alpha$'

            dataX = range(len(data[keyData]))
            dataY = data[keyData]

            xInfo = AxisInfo("x", False, dataX, MeasureUnits.DISCRE_TIME)
            yInfo = AxisInfo(labelY, False, dataY, unitMeasure)
            graph = Graph2D(xInfo, yInfo)
            plotInfo = PlotInfo(graph, axTitle, "")

            matlabDrawer.AddPlot(axTitle = axTitle, plotInfo = plotInfo, graphColor = MatlabDrawer.defaultColors[ptrColor]) 

            mu, sigma2, sigma = DataAnalyzer.CalculateStatistics(dataY) 

            mu_xInfo = AxisInfo("x", False, dataX, MeasureUnits.DISCRE_TIME)
            mu_yInfo = AxisInfo(labelY, False, np.full(len(dataY), mu), unitMeasure)
            mu_graph = Graph2D(mu_xInfo, mu_yInfo)
            mu_plotInfo = PlotInfo(mu_graph, axTitle, f"W = {ptrHorizon}" + ", " + r'$\mu$' + f"={mu:.2f}")

            matlabDrawer.AddPlot(axTitle, mu_plotInfo, graphStyle = "--", graphColor = MatlabDrawer.defaultColors[ptrColor]) 

        ptrColor = ptrColor + 1
        ptrHorizon = ptrHorizon + 1
                    
    matlabDrawer.ShowPlots()