'''
__author__ = Hermann Ndeh
__author__ = Sharon Gilman
__author__ = Virginia Jackson

Make sure all dependencies imported below are installed before running this script, you will need
a google API key with the googlemaps API configured to it. If you are grading this, the key is in
the 'apiKey.txt' file. Wait for the prompt to input they key after you run the code.
'''
import csv
import googlemaps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from mpl_toolkits.basemap import Basemap
from scipy.sparse import csr_matrix, save_npz
from math import pi, atan2, cos, radians, sin, sqrt
from pulp import LpProblem, LpVariable, LpMinimize, LpBinary, lpSum

# Constants
EDGE_OF_MAP_FROM_LOCATION = 2.5
EARTH_RADIUS_IN_MILES = 3959.0

# Global Variables
apiKey = input('Enter your google API key: ')
df = pd.read_csv('SeattleAddressess.csv', delimiter = ',', quotechar = "'")
googleMapsAPIKey = googlemaps.Client(key = apiKey)
mapOfSeattle = None
fileNameForTable = 'Locations.txt'
fileNameForMatrix = 'Distances.csv'
fileNameForTask2 = 'Solution'
fileNameForOD = 'OD.txt'
numberOfCloudKitchens = 25
numberOfServiceStations = 50
cloudKitchenCoordinates = []
cloudKitchenLatitudes = []
cloudKitchenLongitudes = []
serviceStationLatitudes = []
serviceStationLongitudes = []
edgeOfMapCoordinates = []
serviceStationCoordinates = []
cloudKitchenZipCodes = []
cloudKitchenAddresses = []
serviceStations = []
cloudKitchens = []
kitchenAndStationData = []
coordinatePointN = ()
coordinatePointW = ()
coordinatePointE = ()
coordinatePointS = ()

class createEdgePoints:
    def createWestPoint(coordinates, distance):
        """
        The function "createWestPoint" takes in a set of coordinates and a distance, and returns the
        longitude of a point that is west of the given coordinates by the specified distance.
        
        :param coordinates: The `coordinates` parameter is a tuple containing the latitude and longitude of
        a location
        :param distance: The distance parameter represents the distance in miles from the given coordinates
        to the desired location, in this case, West Point
        :return: the longitude of the edge of the map, calculated by subtracting the distance divided by
        69.1 from the given longitude.
        """
        latitude, longitude = coordinates
        return edgeOfMapCoordinates.append(longitude - distance / 69.1)

    def createNorthPoint(coordinates, distance):
        """
        The function `createNorthPoint` calculates the latitude of a point that is a certain distance north
        of a given coordinate.
        
        :param coordinates: The coordinates parameter is a tuple containing the latitude and longitude
        values of a location
        :param distance: The distance parameter represents the distance in miles from the given coordinates
        to the desired point
        :return: the updated list `edgeOfMapCoordinates` after appending the calculated latitude value.
        """
        latitude, longitude = coordinates
        return edgeOfMapCoordinates.append(latitude + (distance / EARTH_RADIUS_IN_MILES * (180 / pi)))

    def createSouthPoint(coordinates, distance):
        """
        The function `createSouthPoint` calculates the latitude of a point that is a given distance south of
        a given coordinate.
        
        :param coordinates: The coordinates parameter is a tuple containing the latitude and longitude
        values of a location
        :param distance: The distance parameter represents the distance in miles from the given coordinates
        to the southernmost point on the map
        :return: the updated list `edgeOfMapCoordinates` after appending a new latitude value.
        """
        latitude, longitude = coordinates
        return edgeOfMapCoordinates.append(latitude - (distance / EARTH_RADIUS_IN_MILES * (180 / pi)))

    def createEastPoint(coordinates, distance):
        """
        The function `createEastPoint` takes in a set of coordinates and a distance, and returns the
        longitude of a point that is `distance` miles east of the given coordinates.
        
        :param coordinates: The `coordinates` parameter is a tuple containing the latitude and longitude
        values of a point on a map
        :param distance: The distance parameter represents the distance in miles from the given coordinates
        :return: the updated list `edgeOfMapCoordinates` after appending the calculated longitude value.
        """
        latitude, longitude = coordinates
        return edgeOfMapCoordinates.append(longitude + distance / 69.1)
    
class findFurthestPlotPoints:
    def findEastLocation(coordinates):
        """
        The function finds the easternmost location from a list of coordinates.
        
        :param coordinates: A list of tuples representing latitude and longitude coordinates
        :return: the coordinates of the easternmost location.
        """
        if not coordinates:
            return None
        eastLoc = None
        maxLongitude = float('-inf')
        for lat, lon in coordinates:
            if lon > maxLongitude:
                maxLongitude = lon
                eastLoc = (lat, lon)
        return eastLoc

    def findNorthLocation(coordinates):
        """
        The function findNorthLocation takes a list of coordinates and returns the coordinate with the
        highest latitude, representing the northernmost location.
        
        :param coordinates: A list of tuples representing coordinates. Each tuple contains a latitude and
        longitude value
        :return: the coordinates of the northernmost location from the given list of coordinates.
        """
        if not coordinates:
            return None
        northLoc = None
        maxLatitude = float('-inf')
        for lat, lon in coordinates:
            if lat > maxLatitude:
                maxLatitude = lat
                northLoc = (lat, lon)
        return northLoc

    def findSouthLocation(coordinates):
        """
        The function `findSouthLocation` takes a list of coordinates and returns the southernmost location.
        
        :param coordinates: A list of tuples representing latitude and longitude coordinates. Each tuple
        contains two elements: the latitude and longitude values
        :return: the southernmost location from the given coordinates.
        """
        if not coordinates:
            return None
        southLoc = None
        minLatitude = float('inf')
        for lat, lon in coordinates:
            if lat < minLatitude:
                minLatitude = lat
                southLoc = (lat, lon)
        return southLoc

    def findWestLocation(coordinates):
        """
        The function `findWestLocation` takes a list of coordinates and returns the coordinate with the
        smallest longitude, representing the westernmost location.
        
        :param coordinates: A list of tuples representing latitude and longitude coordinates. Each tuple
        contains two elements: the latitude and the longitude
        :return: the coordinates of the westernmost location.
        """
        if not coordinates:
            return None
        westLoc = None
        minLongitude = float('inf')
        for lat, lon in coordinates:
            if lon < minLongitude:
                minLongitude = lon
                westLoc = (lat, lon)
        return westLoc

class mapCreation:
    def createBaseMap(extremeLocations):
        """
        The function creates a basemap object for a given set of extreme locations.
        
        :param extremeLocations: The `extremeLocations` parameter is a list containing the latitude and
        longitude values of the extreme points of the map. The list should have the following format:
        `[min_latitude, max_latitude, min_longitude, max_longitude]`. These values define the boundaries of
        the map
        :return: a Basemap object.
        """
        cityMap = Basemap(
            projection = 'merc',
            llcrnrlat = extremeLocations[0],
            urcrnrlat = extremeLocations[1],
            llcrnrlon = extremeLocations[2],
            urcrnrlon = extremeLocations[3],
            resolution = 'h'
        )
        return cityMap

    def drawAndPlotMap(mapOfSeattle):
        """
        The function "drawAndPlotMap" takes a map object of Seattle and plots the locations of cloud
        kitchens and service stations on the map.
        
        :param mapOfSeattle: The parameter `mapOfSeattle` is an instance of the Basemap class, which
        represents a map of the city of Seattle. It is used to draw the coastlines, countries, and states on
        the map
        """
        mapOfSeattle.drawcoastlines()
        mapOfSeattle.drawcountries()
        mapOfSeattle.drawstates()
        x, y = mapOfSeattle(cloudKitchenLongitudes, cloudKitchenLatitudes)
        a, b = mapOfSeattle(serviceStationLongitudes, serviceStationLatitudes)
        mapOfSeattle.scatter(x, y, marker = '.', color = '#41BE1A', label = 'Cloud Kitchens')
        mapOfSeattle.scatter(a, b, marker = '.', color = 'blue', label = 'Service Locations')

def generateServiceStations():
    """
    The function generates random coordinates for service stations within specified boundaries.
    """
    for _ in range(numberOfServiceStations):
        latitude = np.random.uniform(coordinatePointS[0], coordinatePointN[0])
        longitude = np.random.uniform(coordinatePointW[1], coordinatePointE[1])
        serviceStationCoordinates.append((latitude, longitude))

def addressesToLocations():
    """
    The function "addressesToLocations" takes a list of addresses, uses the Google Maps API to geocode
    each address, and extracts the latitude and longitude coordinates for each address.
    """
    for i in range(len(df['Addresses'])):
        cloudKitchenAddress = df['Addresses'][i]
        googleMap = googleMapsAPIKey.geocode(cloudKitchenAddress)
        latitudeCK = googleMap[0]['geometry']['location']['lat']
        longitudeCK = googleMap[0]['geometry']['location']['lng']
        cloudKitchenCoordinates.append((latitudeCK, longitudeCK))
        cloudKitchenLatitudes.append(latitudeCK)
        cloudKitchenLongitudes.append(longitudeCK)

def getAddresses():
    """
    The function `getAddresses` extracts the street addresses from a dataframe column called 'Addresses'
    and stores them in a global variable called `cloudKitchenAddresses`.
    """
    global cloudKitchenAddresses
    df['StreetAddress'] = df['Addresses'].str.extract(r'^(.*?, Seattle, WA)')
    cloudKitchenAddresses = df['StreetAddress'].values

def getZipCodes():
    """
    The function extracts zip codes from addresses in a dataframe and assigns them to a global variable.
    """
    global cloudKitchenZipCodes
    df['ZipCodes'] = df['Addresses'].str.extract(r'WA (\d+),')
    cloudKitchenZipCodes = df['ZipCodes'].values

def distance(cloudKitchens, serviceStations):
    """
    The function calculates the distance between each cloud kitchen and service station using the
    Euclidean distance formula.
    
    :param cloudKitchens: The parameter `cloudKitchens` is a list of dictionaries, where each dictionary
    represents a cloud kitchen. Each dictionary should have a key called 'Coordinates' which represents
    the coordinates of the cloud kitchen
    :param serviceStations: The parameter "serviceStations" is a list of dictionaries, where each
    dictionary represents a service station. Each dictionary contains information about the service
    station, including its coordinates
    :return: a distance matrix, which is a 2D array containing the distances between each cloud kitchen
    and service station.
    """
    numkitchens = len(cloudKitchens)
    numStations = len(serviceStations)
    distanceMatrix = np.zeros((numkitchens, numStations))
    for i in range(numkitchens):
        for j in range(numStations):
            distanceMatrix[i, j] = haversine(cloudKitchens[i]['Coordinates'], serviceStations[j]['Coordinates'])
    return distanceMatrix

def haversine(coordinate1, coordinate2):
    """
    The haversine function calculates the distance between two coordinates on Earth using the haversine
    formula.
    
    :param coordinate1: The coordinate1 parameter represents the latitude and longitude of the first
    point. It should be a tuple or list containing two values: the latitude in degrees and the longitude
    in degrees
    :param coordinate2: The `coordinate2` parameter is the second set of coordinates (latitude and
    longitude) that you want to calculate the distance to from `coordinate1`
    :return: the distance between two coordinates in miles.
    """
    latitude1, longitude1 = map(radians, coordinate1)
    latitude2, longitude2 = map(radians, coordinate2)

    deltaLong = longitude2 - longitude1
    deltaLat = latitude2 - latitude1
    squaredHalfChord = sin(deltaLat / 2) ** 2 + cos(latitude1) * cos(latitude2) * sin(deltaLong / 2) ** 2
    angularDist = 2 * atan2(sqrt(squaredHalfChord), sqrt(1 - squaredHalfChord))
    distance = EARTH_RADIUS_IN_MILES * angularDist
    return distance

def taskII(distanceMatrix):
    """
    This function formulates a minimization problem using the PuLP library and stores the solution into a sparse
    data structure which is returned and generated to an .npz file in the main() method.

    :param distanceMatrix: The parameter "distanceMatrix" is a 2D array containing the distances between each
    cloud kitchen and service station.
    :return zij: This variable is used in taskIII() definition.
    :return solutionMatrix: The solution matrix, which is a sparse data structure containing the solution to the
    linear programming problem.
    """

    I = range(numberOfCloudKitchens)
    J = range(numberOfServiceStations)
    dij = distanceMatrix

    task2 = LpProblem("Project 1 - Task 2", LpMinimize)
    zij = LpVariable.dicts('zij', (I,J), 0, 1, LpBinary)

    task2 += (
        lpSum(lpSum(dij[i][j] * zij[i][j] for i in I for j in J)), "Total Distance"
    )

    for j in J:
        task2 += (
            lpSum(zij[i][j] for i in I) == 1, "Deliver to service station {0}".format(j)
        )
    for i in I:
        task2 += (
            lpSum(zij[i][j] for j in J) == 2, "Deliver from cloud kitchen {0} to two service stations".format(i)
        )
    
    task2.solve()
    solution = {(i,j): zij[i][j].varValue for i in I for j in J}
    rows, columns, data = [], [], []
    for (i,j), value in solution.items():
        if value == 1:
            rows.append(i)
            columns.append(j)
            data.append(1)

    solutionMatrix = csr_matrix((data, (rows, columns)), shape = (numberOfCloudKitchens, numberOfServiceStations))
    task2.writeMPS("AP.mps")
    return zij, solutionMatrix

def taskIII(distanceMatrix, zij):
    """
    The function `taskIII` takes a distance matrix and a binary decision variable matrix as inputs, and
    generates an origin-destination table, plots the routes on a map, and creates a frequency graph
    based on the distances.
    
    :param distanceMatrix: The distanceMatrix parameter is a matrix that represents the distances
    between cloud kitchens and service stations. It is a 2D array where each element represents the
    distance between a cloud kitchen and a service station
    :param zij: The parameter "zij" is a binary decision variable matrix that represents the assignment
    of cloud kitchens to service stations. It is a 2D matrix where each element zij[i][j] is a binary
    variable that indicates whether cloud kitchen i is assigned to service station j
    :return: the `odTable`, which is a list of dictionaries containing information about the origin,
    destination, and distance for each pair of cloud kitchens and service stations where `zij` is equal
    to 1.
    """
    odTable = []
    odDictionary = {}
    distanceIndex = 0

    for i in range(len(cloudKitchens)):
        for j in range(len(serviceStations)):
            if zij[i][j].value() == 1:
                odTable.append({cloudKitchens[i]['Index']: i + 1, 
                                 serviceStations[j]['Index']: j + 1,
                                 f"Distance {distanceIndex}": distanceMatrix[i][j]})
                odDictionary[cloudKitchens[i]['Index']] = i + 1
                odDictionary[serviceStations[j]['Index']] = j + 1
                odDictionary[f"Distance {distanceIndex}"] = distanceMatrix[i][j]
                distanceIndex += 1
    
    mapOfSeattle = mapCreation.createBaseMap(edgeOfMapCoordinates)
    mapCreation.drawAndPlotMap(mapOfSeattle)
    for pair in odTable:
        origin = None
        destination = None
        for i in range(len(pair)):
            for j in range(len(cloudKitchens)):
                if cloudKitchens[j]['Index'] == list(pair.keys())[0]:
                    origin = cloudKitchens[j]['Coordinates']
            for k in range(len(serviceStations)):
                if serviceStations[k]['Index'] == list(pair.keys())[1]:
                    destination = serviceStations[k]['Coordinates']

        if origin is not None and destination is not None:
            xOrigin, yOrigin = mapOfSeattle(origin[1], origin[0])
            xDestination, yDestination = mapOfSeattle(destination[1], destination[0])
            mapOfSeattle.plot([xOrigin, xDestination], [yOrigin, yDestination], color='blue', linewidth=1)
    plt.savefig('Solution.jpg', format='jpeg', dpi=300)
    plt.close()
    
    shortRange = (0,3)
    mediumRange = (3,6)
    longRange = (6,float("inf"))

    shortFrequency = 0
    mediumFrequency = 0
    longFrequency = 0
    distanceIndex = 0

    for key, value in odDictionary.items():
      if key == f"Distance {distanceIndex}":
        if shortRange[0] <= value < shortRange[1]:
          shortFrequency += 1
        if mediumRange[0] <= value < mediumRange[1]:
            mediumFrequency += 1
        if longRange[0] <= value < longRange[1]:
            longFrequency += 1
        distanceIndex += 1
    
    distanceRanges = ["< 3 miles", "3-6 miles", "> 6 miles"]
    frequencyValues = [shortFrequency, mediumFrequency, longFrequency]

    plt.bar(distanceRanges,frequencyValues)
    plt.xlabel("Distance Ranges (miles)")
    plt.ylabel("Frequency Values")
    plt.title("Frequency Graph of Origin-Destination Table")
    plt.savefig('Frequency.jpg', format = 'jpeg', dpi = 300)

    return odTable

def main():
    """
    The main function performs various tasks related to cloud kitchen and service station locations,
    including generating coordinates, creating points on a map, generating service stations, creating
    data tables, and plotting the locations on a map.
    """
    global serviceStationLatitudes
    global serviceStationLongitudes
    global coordinatePointS
    global coordinatePointN
    global coordinatePointW
    global coordinatePointE

    addressesToLocations()
    getAddresses()
    getZipCodes()
    coordinatePointS = findFurthestPlotPoints.findSouthLocation(cloudKitchenCoordinates)
    createEdgePoints.createSouthPoint(coordinatePointS, EDGE_OF_MAP_FROM_LOCATION)
    coordinatePointN = findFurthestPlotPoints.findNorthLocation(cloudKitchenCoordinates)
    createEdgePoints.createNorthPoint(coordinatePointN, EDGE_OF_MAP_FROM_LOCATION)
    coordinatePointE = findFurthestPlotPoints.findEastLocation(cloudKitchenCoordinates)
    createEdgePoints.createEastPoint(coordinatePointE, EDGE_OF_MAP_FROM_LOCATION)
    coordinatePointW = findFurthestPlotPoints.findWestLocation(cloudKitchenCoordinates)
    createEdgePoints.createWestPoint(coordinatePointW, EDGE_OF_MAP_FROM_LOCATION)
    generateServiceStations()
    serviceStationLatitudes = [coord[0] for coord in serviceStationCoordinates]
    serviceStationLongitudes = [coord[1] for coord in serviceStationCoordinates]
    mapCreation.drawAndPlotMap(mapCreation.createBaseMap(edgeOfMapCoordinates))

    for i in range(len(cloudKitchenAddresses)):
        cloudKitchens.append({
            'Index': f'Kitchen {i}',
            'Street Address': cloudKitchenAddresses[i],
            'Zip Code': cloudKitchenZipCodes[i],
            'Coordinates': cloudKitchenCoordinates[i]
        })

    for j in range(len(serviceStationCoordinates)):
        serviceStations.append({
            'Index': f'Station {j}',
            'Street Address': '',
            'Zip Code': '',
            'Coordinates': serviceStationCoordinates[j]
        })
    combined_data = cloudKitchens + serviceStations
    headers = ['Index', 'Street Address', 'Zip Code', 'Coordinates']
    for item in combined_data:
        index = item['Index']
        address = item['Street Address']
        zip_code = item['Zip Code']
        coordinates = item['Coordinates']
        kitchenAndStationData.append([index, address, zip_code, coordinates])
    
    with open(fileNameForTable, 'w') as file:
        table = tabulate(kitchenAndStationData, headers, tablefmt = 'simple')    
        file.write(table)
    
    with open(fileNameForMatrix, mode = 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerows(distance(cloudKitchens, serviceStations))

    plt.title('Cloud Kitchen Locations')
    plt.legend(loc = 'best')
    plt.savefig('Locations.jpg', format = 'jpeg', dpi = 300)
    # plt.close()

    distanceMatrix = distance(cloudKitchens, serviceStations)
    zij, solutionMatrix = taskII(distanceMatrix)
    save_npz(fileNameForTask2, solutionMatrix)

    od_table = taskIII(distanceMatrix, zij)
    with open(fileNameForOD, 'w') as file:
        od_table = tabulate(od_table, headers = "keys", tablefmt = "simple")   
        file.write(od_table)

if __name__ == '__main__':
    main()