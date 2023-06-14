import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist


class SelectionWindow:
    def __init__(self, title, frame):
        self.title = title
        self.frame = frame.copy()

        self.minPointsLeft = 0
        self.func = None

    def displayWindow(self):
        cv2.namedWindow(self.title)
        if self.func != None:
            cv2.setMouseCallback(self.title, self.func)
        cv2.imshow(self.title, self.frame)

        while True:
            key = cv2.waitKey(0)

            if self.minPointsLeft <= 0 and key == 27:
                cv2.destroyWindow(self.title)
                break


class PickColorWindow(SelectionWindow):
    def __init__(self, title, frame):
        super().__init__(title, frame)
        self.pickedColors = []
        self.minPointsLeft = 1
        self.func = self.pick_color

    def pick_color(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
            self.minPointsLeft -= 1
            self.pickedColors.append(self.frame[y, x])
            cv2.circle(self.frame, (x, y), 2, (255, 255, 255), thickness=1)
            cv2.imshow(self.title, self.frame)


def display(frame, title="Display"):
    cv2.imshow(title, frame.copy())
    key = cv2.waitKey(0)
    if key == 27 or key == ord("q"):
        cv2.destroyAllWindows()


def matchColor(boundary, frame):
    lower, upper = np.array(boundary, dtype="uint8")
    mask = cv2.inRange(frame, lower, upper)
    output = cv2.bitwise_and(frame, frame, mask=mask)
    return mask, output


def displayContours(cnts, frameShape):
    blank = np.zeros(frameShape)
    cv2.drawContours(blank, cnts, -1, (0, 0, 255), 1)
    display(blank)


def checkContours(cnts, frame):
    frameCopy = frame.copy()
    cv2.drawContours(frameCopy, cnts, -1, (0, 255, 0), 1)
    display(frameCopy)


class Blob:
    def __init__(
        self,
        bound,
        frame,
        dilate_iter=0,
        erode_iter=0,
        filter_by_size=None,
        heightMin=0,
        widthMin=0,
    ):
        self.frame = frame
        self.bound = bound
        self.mask, self.output = matchColor(self.bound, frame)

        # dilate
        kernel = np.ones((4, 4), np.uint8)
        self.mask = cv2.dilate(self.mask, kernel, iterations=dilate_iter)
        self.output = cv2.dilate(self.output, kernel, iterations=dilate_iter)

        # erode
        self.mask = cv2.erode(self.mask, kernel, iterations=erode_iter)
        self.output = cv2.erode(self.output, kernel, iterations=erode_iter)

        # cnts
        cnts, _ = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # filter out contours less than a certain size
        self.cnts = []
        for cnt in cnts:
            rect = cv2.minAreaRect(cnt)
            width = rect[1][1]
            height = rect[1][0]
            if (width >= widthMin) and (height >= heightMin):
                self.cnts.append(cnt)

        if filter_by_size:
            sortedCnts = sorted(self.cnts, key=lambda c: cv2.contourArea(c))
            self.cnts = sortedCnts[:filter_by_size]
        # cv2.drawContours(self.output, self.cnts, -1, (0,0,255), 1)

    def displayContours(self):
        displayContours(self.cnts, self.output.shape)

    def checkContours(self):
        checkContours(self.cnts, self.frame)

    def getCntExtremePts(self):
        c = max(self.cnts, key=cv2.contourArea)  # get largest cnt by area
        x, y, w, h = cv2.boundingRect(c)
        return x, y, w, h

    def getAllCntExtremePts(self):
        pts = []
        for c in self.cnts:
            x, y, w, h = cv2.boundingRect(c)
            x2 = x + w / 2
            y2 = y + h / 2
            pts.append([x, x2, y, y2])
        return pts

    def getAllCntCentres(self):
        pts = []
        for c in self.cnts:
            x, y, w, h = cv2.boundingRect(c)
            xC = x + w * 0.5
            yC = y + h * 0.5
            pts.append((xC, yC))
        return pts

    def displayExtremePoints(self):
        h, w, c = self.output.shape
        blank = np.zeros([h, w, c])
        cv2.drawContours(blank, self.cnts, -1, (0, 0, 255), 1)
        x, y, w, h = self.getCntExtremePts()

        for i, j in ((0, 0), (1, 0), (0, 1), (1, 1)):
            blank = cv2.circle(blank, (x + w * i, y + h * j), 1, (0, 255, 0), 1)

        display(blank)

    def mapCntToRect(self):
        c = self.cnts[0]
        epsilon = 0.1 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        self.cnts = [approx]

    def mapAllCntsToRect(self, epsilonCoeff):
        cnts = []
        for c in self.cnts:
            epsilon = epsilonCoeff * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            cnts.append(approx)
        self.cnts = cnts


def getDistBetweenCoords(coord1, coord2):
    # print(coord1, coord2)
    return ((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2) ** 0.5


def drawGrids(image, xN, yN):
    try:
        h, w, _ = image.shape
    except:
        h, w = image.shape

    imageCopy = image.copy()
    gridW, gridH = w / xN, h / yN

    for x in range(1, xN):
        xCoord = round(gridW * x)
        cv2.line(imageCopy, (xCoord, 0), (xCoord, h), 255, thickness=1)
    for y in range(1, yN):
        yCoord = round(gridH * y)
        cv2.line(imageCopy, (0, yCoord), (w, yCoord), 255, thickness=1)
    return imageCopy


class ParticleFilter:
    def __init__(self, frame, cnts, P, N):
        """
        P: no. of points in a grid in each direction.
        N: no. of grids in each direction.
        """

        self.frame = frame
        self.cnts = sorted(
            cnts, key=lambda x: cv2.contourArea(x)
        )  # contours, sorted by size
        self.P = np.array(P)  # np array, no. of points in a grid in each direction.
        self.N = np.array(N)  # np array, no. of grids in each direction.
        self.pixelDistDebug = False

        self.blank = np.zeros(self.frame.shape[:2]).astype(np.uint8)

        # Generate frame where all pixels inside obstacles are 1.
        self.walls = self.blank.copy()
        for cnt in self.cnts:
            cv2.fillPoly(self.walls, pts=[cnt], color=255)
        display(self.walls)

        h, w, _ = self.frame.shape
        self.boundaryCnt = [
            np.array([[[0, 0]], [[w - 1, 0]], [[w - 1, h - 1]], [[0, h - 1]]])
        ]

        # Generate frame where all pixels inside boundary and obstacles are 1.
        self.wallsAndBoundary = cv2.drawContours(
            self.walls.copy(), self.boundaryCnt, -1, 255, 1
        )

    def pixelDistFromWall(self, coord, angle):
        h, w, _ = self.frame.shape

        xCoord, yCoord = coord
        rad = angle * np.pi / 180

        if angle == 0 or angle == 180:
            gradient = 1e5
        else:
            gradient = np.cos(rad) / np.sin(rad)

        if angle < 180:
            travel = -w
        else:
            travel = w

        # Check if coord is inside an obstacle
        xCoord = min(xCoord, w - 1)
        yCoord = min(yCoord, h - 1)
        if self.wallsAndBoundary[yCoord][xCoord] == 255:
            return np.nan

        # Get point on wall closest to current coord
        xEnd, yEnd = round(xCoord + travel), round(yCoord + travel * gradient)
        lineFrame = cv2.line(self.blank.copy(), (xCoord, yCoord), (xEnd, yEnd), 255, 1)
        img = cv2.bitwise_and(lineFrame, self.wallsAndBoundary)
        coords = cv2.findNonZero(img)

        if self.pixelDistDebug:
            debug = cv2.bitwise_or(lineFrame, self.wallsAndBoundary)
            debug = cv2.circle(debug, (xCoord, yCoord), 2, 255, -1)
            debug = drawGrids(debug, *self.N)
            display(debug)

        # Smallest euclidean distance
        return cdist(coords[:, 0], [np.array(coord)]).min()

    # TODO: Account for case where position is inside wall
    def getDistGrid(self, gridCoord, angle):
        xP, yP = self.P
        xN, yN = self.N
        x, y = gridCoord
        assert x < xN and x >= 0
        assert y < yN and y >= 0

        h, w, _ = self.frame.shape
        gridH, gridW = h / yN, w / xN

        xMin, yMin = round(x * gridW), round((yN - y) * gridH)
        xCoords, yCoords = [], []
        distsGrid = np.zeros(self.P)

        for xi in range(xP):
            for yi in range(yP):
                xCoord, yCoord = round(xMin + xi / xP * gridW), round(
                    yMin - yi / yP * gridH
                )
                # print(xCoord, yCoord)
                dist = self.pixelDistFromWall((xCoord, yCoord), angle)
                xCoords.append(xCoord)
                yCoords.append(yCoord)
                distsGrid[xi][yi] = dist

        return xCoords, yCoords, distsGrid

    def plotDistMap(self, angle):
        xP, yP = self.P
        xN, yN = self.N

        h, w, _ = self.frame.shape
        gridH, gridW = h / yN, w / xN

        xMin, yMin = 0, round(yN * gridH)
        distsGrid = np.zeros(self.P * self.N)

        for xi in range(xP * xN):
            for yi in range(yP * yN):
                xCoord, yCoord = round(xMin + xi / xP * gridW), round(
                    yMin - yi / yP * gridH
                )
                # print(xCoord, yCoord)
                dist = self.pixelDistFromWall((xCoord, yCoord), angle)
                distsGrid[xi][yi] = dist

        return distsGrid


def plotDistGrid(distsGrids, deg, export=True):
    plt.imshow(distsGrids[deg].T, origin="lower", interpolation=None)
    plt.xlabel("xCoord")
    plt.ylabel("yCoord")
    plt.axes().set_aspect(aspect=270 / 360)
    plt.colorbar().set_label("Distance")
    if export:
        plt.savefig(
            f"Presentation Images/Distance Grids/{deg}.svg",
            transparent=True,
            bbox="tight",
        )
    plt.show()


def addCompass(deg, change):
    deg += change
    if deg < 0:
        deg += 360
    elif deg >= 360:
        deg -= 360
    return deg


def sensorFuseTest(deg, distsAll, sensors, alpha):
    leftDeg = addCompass(deg, 45)
    rightDeg = addCompass(deg, -45)

    frontCond = np.abs(distsAll[deg] - sensors[0]) < 5
    leftCond = np.abs(distsAll[leftDeg] - sensors[1]) < 5
    rightCond = np.abs(distsAll[rightDeg] - sensors[2]) < 5

    frontCoords = np.where(frontCond)
    leftCoords = np.where(leftCond)
    rightCoords = np.where(rightCond)

    intersectCoords = np.where(
        np.logical_and(np.logical_and(frontCond, leftCond), rightCond)
    )

    # print("Coords:", intersectCoords)

    plt.scatter(*frontCoords, marker=".", alpha=alpha, label="Front US")
    plt.scatter(*leftCoords, marker=".", alpha=alpha, label="Left US")
    plt.scatter(*rightCoords, marker=".", alpha=alpha, label="Right US")
    plt.scatter(*intersectCoords, marker="x", label="Common")
    plt.legend()

    plt.xlim(0, 90)
    plt.ylim(0, 90)
    plt.axes().set_aspect(aspect=270 / 360)
    plt.xlabel("xCoord")
    plt.ylabel("yCoord")
    plt.savefig("Presentation Images/Sensor Fusion.svg", transparent=True)
    plt.show()


def convertToInt(arr):
    arr = np.array(arr)
    arr[np.isnan(arr)] = -1
    return np.round(arr).astype("int")


def numPyToCArrayStr(arrStr):
    arrStr = arrStr.replace(" ", "")
    arrStr = arrStr.replace("[", "{")
    arrStr = arrStr.replace("]", "}")
    arrStr = arrStr.replace("\n", "")
    return arrStr


def exportNumPyToCArr(fileName, openType, arr, arrName):
    np.set_printoptions(threshold=np.prod(arr.shape))
    dims = len(arr.shape)
    arrStr = np.array2string(arr, separator=",")
    arrStr = numPyToCArrayStr(arrStr)

    with open(fileName, openType) as f:
        if openType == "w":
            f.write("#include <distsAll.h>\n")
        f.write(f"int {arrName}")

        for i in range(dims):
            f.write(f"[{arr.shape[i]}]")

        f.write(" = ")
        f.write(arrStr)
        f.write(";\n")

    # with open("run/include/distsAll.h", "w") as f:
    #     f.write(f"extern distsAll[{arr.shape[0]}][{arr.shape[1]}][{arr.shape[2]}];")

    np.set_printoptions(threshold=1000)  # Set back to default


def seeMapValue(arr, xi, yi):
    plt.scatter(xi, yi, s=1, color="r")
    plt.imshow(arr.T, origin="lower")
    plt.colorbar()
    print(arr[xi][yi])
