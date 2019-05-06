import math
from astropy import constants as const

# haversine formula found on https://www.geeksforgeeks.org/haversine-formula-to-find-distance-between-two-points-on-a-sphere/
# slightly modified function to stick with our data
# INPUT : positions of satellites in radians, altitude in km
# OUTPUT : Approximate distance between two satellites in kilometers, as a straight line
def haversine(positions1, positions2, altitude):
    lat1 = positions1[0]
    lon1 = positions1[1]
    lat2 = positions2[0]
    lon2 = positions2[1]

    # distance between latitudes
    # and longitudes
    dLat = (lat2 - lat1)
    dLon = (lon2 - lon1)


    def hav(angle):
        return (1 - math.cos(angle))/2

    # apply formulae
    r = (const.R_earth.value / 1000) + altitude
    distance = 2*r*math.asin(math.sqrt(hav(dLat)+(math.cos(lat1)*math.cos(lat2)*hav(dLon))))
    chord = 2*r*math.sin((0.5*distance/r))
    return chord


SECONDS_IN_DAY = 86400
#Input : semi_major_axis_length in km
#Output : mean_motion in revolutions/day (float)
#TODO : test it
def semi_major_to_mean_motion(axis_length):
    return SECONDS_IN_DAY*math.sqrt(const.G.value*(const.M_earth.value)/math.pow(axis_length*1000,3))/(2*math.pi)


#Input : mean motion (in rotations/day) and time to periapsis
#Output : mean anomaly in degrees (string format)
# (Mean Anomaly / 360) = (time to periapsis / duration of orbit)
def time_to_periapsis_to_mean_anomaly(time_to_periapsis, mean_motion):
    return 360*time_to_periapsis/(SECONDS_IN_DAY/mean_motion)


