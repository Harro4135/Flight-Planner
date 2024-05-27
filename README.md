# Basic Flight-Planner
## Introduction
This project allowed me to take my love and passion for aviation and experiment with concepts I was learning about in my CS class. It uses a database of global airports and wind data to output the optimal path between any two airports, additionally, it calculates the minimum takeoff fuel that could be used.
## System Architecture 
Flight.py takes stored data from various files to process user terminal inputs
1. 4-letter ICAO code of departure airport (e.g. CYYZ)
2. 4-letter ICAO code of arrival airport (e.g. EGLL)

The script then generates a grid of appropriate size, proportional to the manhattan distance between the arrival and departure airport. It proceeds to use an implementation of the a* algorithm with a heuristic based on the wind conditions at the given location. Finally it outputs an interactive globe view of the route and the numeric properties of the route, seen here.

![image](https://github.com/Harro4135/Flight-Planner/assets/91696463/2f611924-3fdf-4802-a2e9-3b012f2eec0c)
[Interactive Version](CYYZ-EGLL.html)
