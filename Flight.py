#Oliver Harrison
#Final Project
#Takes inputs of flight paramiters and outputs route
import math
from geopy.distance import geodesic
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import great_circle_calculator.great_circle_calculator as gcc
import heapq
import plotly.graph_objects as go

mach = 0.78

#Declaration of Navaid Class
class navaid:

  def __init__(self, name, coordinates, type, association, country):
    self.coordinates = coordinates
    self.type = type
    self.association = association
    self.country = country
    self.name = name

  def printNav(self):
    return self.coordinates

#Declaration of Airport Class
class airport:

  def __init__(self, identifier, name, lat, lon):
    self.identifier = identifier
    self.name = name
    self.lat = float(lat)
    self.lon = float(lon)
  #print ICAO code
  def printPort(self):
    return self.identifier
    
  def printLat(self):
    return self.lat

  def printLon(self):
    return self.lon

#Importing Data from Files 
class map:

  def __init__(self):
    self.navaids = []
    with open('Navaids.vb', 'r') as f:
      for line in f:
        lat = line[1:13]
        line = line[14:]
        x = line.split('  ')
        lon = x[0]
        lon.replace(" ", "")
        lon.replace("  ", "")
        y = x[1].split(" ")
        if y[0] == "":
          y.remove("")
        name = y[0]
        assc = y[1]
        type = y[2]
        num = y[3]
        self.navaids.append(navaid(name, [lat, lon], 'RNAV', assc, type))

    self.airports = []
    with open('Airports.txt', 'r') as x:
      for line in x:
        if line[0] == 'A':
          x = line.split(",")
          self.airports.append(airport(x[1], x[2], x[3], x[4]))
#Define Node's
class node:

  def __init__(self, lon, lat):
    self.lon = lon
    self.lat = lat
    self.neighbours = []
    self.g_score = math.inf
    self.f_score = math.inf

  def printNode(self):
    return (self.lon, self.lat)

#Graph Network
class graph:
  """
  A class representing a graph for flight path planning.

  Attributes:
    startlon (float): The longitude of the starting point.
    startlat (float): The latitude of the starting point.
    goallon (float): The longitude of the goal point.
    goallat (float): The latitude of the goal point.
    nodes (list): A 2D list representing the nodes in the graph.
    start (node): The starting node.
    goal (node): The goal node.
    trial (node): The trial node.

  Methods:
    __init__(self, startlon, startlat, goallon, goallat): Initializes the graph object.
    cost(self, dept, arriv): Calculates the cost between two nodes.
    heuristic(self, a, b): Calculates the heuristic value between two nodes.
  """
  
  def __init__(self, startlon, startlat, goallon, goallat):
    """
    Initializes the graph object.

    Args:
      startlon (float): The longitude of the starting point.
      startlat (float): The latitude of the starting point.
      goallon (float): The longitude of the goal point.
      goallat (float): The latitude of the goal point.
    """
    res = []
    resCalc = resolution(goallat - startlat, goallon - startlon, 25)
    res.append(resCalc[0])
    res.append(resCalc[1])
    # Rest of the code...
  
  def cost(self, dept, arriv):
    """
    Calculates the cost between two nodes.

    Args:
      dept (node): The departure node.
      arriv (node): The arrival node.

    Returns:
      float: The cost between the two nodes.
    """
    # Rest of the code...
  
  def heuristic(self, a, b):
    """
    Calculates the heuristic value between two nodes.

    Args:
      a (node): The first node.
      b (node): The second node.

    Returns:
      float: The heuristic value between the two nodes.
    """
    # Rest of the code...
class graph:

  def __init__(self, startlon, startlat, goallon, goallat):
    res = []
    resCalc = resolution(goallat - startlat, goallon - startlon, 25)
    res.append(resCalc[0])
    res.append(resCalc[1])
    #print(res)
    if goallon >= startlon and res[0] < 0:
      res[0] = -res[0]
    if goallon <= startlon and res[0] > 0:
      res[0] = -res[0]
    x = np.arange(startlon, goallon, res[0])
    x = np.append(x, goallon)
    y = np.arange(-90, 90, abs(res[1]))
    y = np.append(y, startlat)
    y = np.append(y, goallat)
    x = np.sort(x)
    y = np.sort(y)
    #print(x)
    self.nodes = [[node(x[i], y[j]) for i in range(len(x))]
                  for j in range(len(y))]
    lat = y
    #Assigning the neighbouring of each node
    for y in range(len(self.nodes)):
      for x in range(len(self.nodes[y])):
        if self.nodes[y][x].printNode(
        )[0] == startlon and self.nodes[y][x].printNode()[1] == startlat:
          self.start = self.nodes[y][x]
        if self.nodes[y][x].printNode(
        )[0] == goallon and self.nodes[y][x].printNode()[1] == goallat:
          self.goal = self.nodes[y][x]
          self.trial = self.nodes[y][x - 1]

        if y < len(self.nodes):
          for i in range(-4, 5):
            if y - i < len(self.nodes):
              if x + 1 < len(self.nodes[y]):
                self.nodes[y][x].neighbours.append(self.nodes[y - i][x + 1])
              if x != 0:
                self.nodes[y][x].neighbours.append(self.nodes[y - i][x - 1])
          for i in range(-3, 4, 2):
            if y - i < len(self.nodes):
              if x + 2 < len(self.nodes[y]):
                self.nodes[y][x].neighbours.append(self.nodes[y - (i)][x +
                                                                       (2)])
              if x >= 2:
                self.nodes[y][x].neighbours.append(self.nodes[y - (i)][x -
                                                                       (2)])
            if y - i < len(self.nodes):
              if x + 4 < len(self.nodes[y]):
                self.nodes[y][x].neighbours.append(self.nodes[y - (i)][x +
                                                                       (4)])
              if x >= 4:
                self.nodes[y][x].neighbours.append(self.nodes[y - (i)][x -
                                                                       (4)])
          for j in range(-1, 2, 2):
            if y - j < len(self.nodes):
              if x < len(self.nodes[y]) - 3:
                self.nodes[y][x].neighbours.append(self.nodes[y - j][x + (3)])
              if x >= 3:
                self.nodes[y][x].neighbours.append(self.nodes[y - j][x - (3)])
            if y - 2 * j < len(self.nodes):
              if x < len(self.nodes[y]) - 3:
                self.nodes[y][x].neighbours.append(self.nodes[y - j *
                                                              (2)][x + (3)])
              if x >= 3:
                self.nodes[y][x].neighbours.append(self.nodes[y - j *
                                                              (2)][x - (3)])
            if y - 4 * j < len(self.nodes):
              if x < len(self.nodes[y]) - 3:
                self.nodes[y][x].neighbours.append(self.nodes[y - j *
                                                              (4)][x + (3)])
              if x >= 3:
                self.nodes[y][x].neighbours.append(self.nodes[y - j *
                                                              (4)][x - (3)])
          if y != 0:
            self.nodes[y][x].neighbours.append(self.nodes[y - 1][x])
          if y + 2 < len(self.nodes) and x + 1 < len(self.nodes[y]):
            self.nodes[y][x].neighbours.append(self.nodes[y + 1][x])
  #Cost function, with wind calculation
  def cost(self, dept, arriv):
    alpha = gcc.bearing_at_p1((dept.lon, dept.lat), (arriv.lon, arriv.lat))
    latW = ClosestLat(dept.lat)
    lonW = ClosestLon(dept.lon)
    winds = wind[tuple([str(lonW), str(latW)])]
    u = float(winds[0])
    v = float(winds[1])
    theta = math.radians(90 - alpha)
    Hmatrix = [[math.cos(theta), math.sin(theta)],
               [-math.sin(theta), math.cos(theta)]]
    Wmatrix = [[u], [v]]
    NodelWinds = np.matmul(Hmatrix, Wmatrix)
    tail = NodelWinds[0][0]
    dept.tail = tail
    cross = NodelWinds[1][0]
    dept.cross = cross
    Va = mach * 661.478
    Vg = math.sqrt(Va**2 - cross**2) + tail
    return ((geodesic((dept.lat, dept.lon),
                      (arriv.lat, arriv.lon)).meters / Vg))
  #huristic assuming a constant tailwind to provide an optimal solution
  def heuristic(self, a, b):
    return (geodesic((a.lat, a.lon), (b.lat, b.lon)).meters / (0.78 * 661.478))


world = map()
leaving = input("Enter ICAO of Departure:")
arriving = input("Enter ICAO of Arrival:")
zfw = float(input("Enter the ZFW of the Aircraft(42.5t-64.3t):"))
reserves = float(input("Enter the Reserve Fuel Needed in Tones:")) 
mach = float(input("Enter Mach Number (0.7-0.94):"))
departure = None
arrival = None
for i in world.airports:
  if i.printPort() == leaving:
    departure = i
  elif i.printPort() == arriving:
    arrival = i
  if departure != None and arrival != None:
    #print(departure.lat,arrival.lat)
    break

Nleaving = departure
Narriving = arrival

#Resolution of the grid
def resolution(d_lat, d_lon, max):
  if abs(d_lat) > abs(d_lon):
    r_lat = d_lat / max
    r_lon = 270 / ((180 / r_lat) - 1)
  elif abs(d_lat) <= abs(d_lon):
    r_lon = d_lon / max
    r_lat = 180 / ((270 / r_lon) + 1)
  return (abs(r_lon), abs(r_lat))


wind = {}
#Extract wind data
with open('WindData.txt', 'r') as f:
  for line in f:
    data = line.split(' ')
    wind[tuple([data[0], data[1]])] = tuple([data[2], data[3]])

#Closest node function
takeClosest = lambda num, collection: min(collection,
                                          key=lambda x: abs(x - num))


def ClosestLat(lat):
  pos = np.arange(90, -90, -1.5)
  return takeClosest(lat, pos)


def ClosestLon(lon):
  pos = np.arange(-180, 180, 1.5)
  return takeClosest(lon, pos)

# A* algo
def aStar(graph, start, goal):
  frontier = []
  heapq.heapify(frontier)
  heapq.heappush(frontier, (0, start))
  came_from = {start: None}
  cost_so_far = {start: 0}
  while len(frontier) != 0:
    current = heapq.heappop(frontier)[1]
    if current == goal:
      #print("found!")
      break
    for next in current.neighbours:
      new_cost = cost_so_far[current] + graph.cost(current, next)
      if next not in cost_so_far or new_cost < cost_so_far[next]:
        cost_so_far[next] = new_cost
        priority = new_cost + graph.heuristic(goal, next)
        heapq.heappush(frontier, (priority, next))
        came_from[next] = current
  return came_from


def reconstruct_path(came_from, start, goal):
  current = goal
  path = [current]
  while current != start:
    current = came_from[current]
    path.append(current)
  path.reverse()
  return path

# Breguet range equation implementaiton 
def breguet(rangel, zfw, ltod_ratio, tail, cross, reserves):
  Va = mach * 661.478
  Vg = math.sqrt(Va**2 - cross**2) + tail
  mass1 = (zfw + reserves) * math.exp(
    (rangel * 9.81 * 0.0000535) / (Vg * ltod_ratio))
  return mass1 - (reserves + zfw)



path = graph(Nleaving.lon, Nleaving.lat, Narriving.lon, Narriving.lat)
#print(path.trial.printNode())
#print("Neighbours Start:")
#for i in path.trial.neighbours:
#print(i.printNode())
#print("Neighbours End")

#print(path.start, path.goal)
came = aStar(path, path.start, path.goal)

#print((departure.lon+arrival.lon)/2,(departure.lat+arrival.lat)/2)
plt.figure(figsize=(8, 8))
m = Basemap(projection='ortho',
            lon_0=(departure.lon + arrival.lon) / 2,
            lat_0=(departure.lat + arrival.lat) / 2,
            resolution=None)
vertices = []
output = reconstruct_path(came, path.start, path.goal)

total = 0
#Fuel Calculation
for i in range(len(output)):
  if i != len(output) - 1:
    lon1 = output[i].printNode()[0]
    lat1 = output[i].printNode()[1]
    lon2 = output[i + 1].printNode()[0]
    lat2 = output[i + 1].printNode()[1]
    rangel = geodesic((lat1, lon1), (lat2, lon2)).meters
    total += breguet(rangel, zfw, 18, output[i].tail, output[i].cross,
                     reserves)
print("Minimum Takeoff Fuel --", total + reserves)
print("Estimated Fuel Burn --", total)
for i in output:
  lon = i.printNode()[0]
  lat = i.printNode()[1]
  vertices.append((lon, lat))
  x, y = m(lon, lat)
xs, ys = zip(*vertices)
plt.plot(xs, ys, "*--", lw=2, color='black', ms=10)
plt.title('Gnomonic Projection')
m.bluemarble()

fig = go.Figure()

for i in range(len(vertices)):
  if i + 1 < len(vertices):
    fig.add_trace(
      go.Scattergeo(lon=(vertices[i][0], vertices[i + 1][0]),
                    lat=(vertices[i][1], vertices[i + 1][1]),
                    mode='markers+lines',
                    line=dict(width=2, color='black'),
                    marker={
                      'color': "black",
                      'size': 8
                    }))

fig.update_layout(
  title_text=
  f"{leaving}-{arriving}<br>MTOF -- {round(total+reserves,3)}<br>Estimated Fuel Burn -- {round(total,3)}T<br>Mach -- {mach}<br>ZFW -- {zfw}T",
  showlegend=False,
  geo=dict(showland=True,
           showcountries=True,
           showocean=True,
           countrywidth=0.5,
           landcolor='orange',
           lakecolor='skyblue',
           oceancolor='skyblue',
           projection=dict(type='orthographic',
                           rotation=dict(lon=(departure.lon + arrival.lon) / 2,
                                         lat=(departure.lat + arrival.lat) / 2,
                                         roll=0)),
           lonaxis=dict(showgrid=True,
                        gridcolor='grey',
                        gridwidth=0.5),
           lataxis=dict(showgrid=True,
                        gridcolor='grey',
                        gridwidth=0.5)))
#fig.write_html(f"{leaving}-{arriving}.html")
fig.show()
