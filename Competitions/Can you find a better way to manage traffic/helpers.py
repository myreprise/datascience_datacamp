from math import radians, cos, sin, asin, sqrt

# Haversine function
def haversine_distance(lon1, lat1, lon2, lat2):
    R = 6371  # Radius of earth in kilometers
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    return R * c