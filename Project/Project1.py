import csv
import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap
from math import *
import random
import googlemaps
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.spatial.distance import euclidean
from pulp import *
from scipy.sparse import *
from networkx import *

# Constants
EDGE_OF_MAP_FROM_LOCATION = 2.5
EARTH_RADIUS_IN_MILES = 3959.0

# Global Variables
df = pd.read_csv('SeattleAddressess.csv', delimiter = ',', quotechar = "'")
googleMapsAPIKey = googlemaps.Client(key = 'AIzaSyB9U3xQQfCIgZqIlySxt_R07cCVOcnygl0')
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

# Functions
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

def findEastLocation(coordinates):
    """
    The function finds the easternmost location from a list of coordinates.
    
    :param coordinates: A list of tuples representing latitude and longitude coordinates
    :return: the coordinates of the easternmost location.
    """
    if not coordinates:
        return None
    eastLoc = None
    max_longitude = float('-inf')
    for lat, lon in coordinates:
        if lon > max_longitude:
            max_longitude = lon
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
    max_latitude = float('-inf')
    for lat, lon in coordinates:
        if lat > max_latitude:
            max_latitude = lat
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
    min_latitude = float('inf')
    for lat, lon in coordinates:
        if lat < min_latitude:
            min_latitude = lat
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
    min_longitude = float('inf')
    for lat, lon in coordinates:
        if lon < min_longitude:
            min_longitude = lon
            westLoc = (lat, lon)
    return westLoc

def createBaseMap(extremeLocations):
    """
    The function creates a basemap object using the given extreme locations.
    
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

def drawAndPlotMap(map_of_seattle):
    """
    The function "drawAndPlotMap" takes a map of Seattle and plots the locations of cloud kitchens and
    random service stations on the map.
    
    :param map_of_seattle: The parameter "map_of_seattle" is a map object that represents the map of
    Seattle. It is used to draw the coastlines, countries, and states on the map
    """
    map_of_seattle.drawcoastlines()
    map_of_seattle.drawcountries()
    map_of_seattle.drawstates()
    x, y = map_of_seattle(cloudKitchenLongitudes, cloudKitchenLatitudes)
    a, b = map_of_seattle(serviceStationLongitudes, serviceStationLatitudes)
    map_of_seattle.scatter(x, y, marker = '.', color = '#41BE1A', label = 'Cloud Kitchens')
    map_of_seattle.scatter(a, b, marker = '.', color = 'blue', label = 'Service Locations')

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
    This function takes the solution found in taskII and creates an Origin-Destination table to be exported to a
    text file in the main() function. 

    :parameter distanceMatrix: The parameter "distanceMatrix" is a 2D array containing the distances between each
    cloud kitchen and service station.
    :parameter zij: Part of the solution to taskII. Used in this function to obtain the service locations selected
    by the model. 
    :return od_table: Origin-Destination Table with columns "Cloud Kitchen Index", "Service Station Index", and
    "Distance" 
    """
    od_table = []
    od_dict = {}

    for i in range(len(cloudKitchens)):
        for j in range(len(serviceStations)):
            if zij[i][j].value() == 1:
                od_table.append({"Cloud Kitchen Index (Origin)": i + 1, 
                                 "Service Station Index (Destination)": j + 1, 
                                 "Distance (miles)": distanceMatrix[i][j]})
                od_dict["Cloud Kitchen Index (Origin)"] = i + 1
                od_dict["Service Station Index (Destination)"] = j + 1
                od_dict["Distance (miles)"] = distanceMatrix[i][j]
    
    short_range = (0,3)
    medium_range = (3,6)
    long_range = (6,float("inf"))

    short_freq = 0
    medium_freq = 0
    long_freq = 0

    for key, value in od_dict.items():
      if key == "Distance (miles)":
        if short_range[0] <= value < short_range[1]:
          short_freq += 1
        if medium_range[0] <= value < medium_range[1]:
            medium_freq += 1
        if long_range[0] <= value < long_range[1]:
            long_freq += 1
    
    dist_ranges = ["< 3 miles", "3-6 miles", "> 6 miles"]
    freq_values = [short_freq, medium_freq, long_freq]

    plt.bar(dist_ranges,freq_values)
    plt.xlabel("Distance Ranges (miles)")
    plt.ylabel("Frequency Values")
    plt.title("Frequency Graph of Origin-Destination Table")
    plt.savefig('Frequency.jpg', format='jpeg', dpi=300)

    return od_table
    # plt.show()

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
    coordinatePointS = findSouthLocation(cloudKitchenCoordinates)
    createSouthPoint(coordinatePointS, EDGE_OF_MAP_FROM_LOCATION)
    coordinatePointN = findNorthLocation(cloudKitchenCoordinates)
    createNorthPoint(coordinatePointN, EDGE_OF_MAP_FROM_LOCATION)
    coordinatePointE = findEastLocation(cloudKitchenCoordinates)
    createEastPoint(coordinatePointE, EDGE_OF_MAP_FROM_LOCATION)
    coordinatePointW = findWestLocation(cloudKitchenCoordinates)
    createWestPoint(coordinatePointW, EDGE_OF_MAP_FROM_LOCATION)
    generateServiceStations()
    serviceStationLatitudes = [coord[0] for coord in serviceStationCoordinates]
    serviceStationLongitudes = [coord[1] for coord in serviceStationCoordinates]
    drawAndPlotMap(createBaseMap(edgeOfMapCoordinates))

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

    # print(tabulate(kitchenAndStationData, headers, tablefmt = 'simple'))
    # print(distance(cloudKitchens, serviceStations))
    plt.title('Cloud Kitchen Locations')
    plt.legend(loc = 'best')
    plt.savefig('Locations.jpg', format = 'jpeg', dpi = 300)
    plt.show()

    distanceMatrix = distance(cloudKitchens, serviceStations)
    zij, solutionMatrix = taskII(distanceMatrix)
    save_npz(fileNameForTask2, solutionMatrix)

    od_table = taskIII(distanceMatrix, zij)
    with open(fileNameForOD, 'w') as file:
        od_table = tabulate(od_table, headers="keys", tablefmt="simple")   
        file.write(od_table)

if __name__ == '__main__':
    main()
