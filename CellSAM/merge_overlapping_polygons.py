import numpy as np
from copy import deepcopy

"""
File to merge overlapping polygons
"""
def project_polygon(polygon, axis):
    dots = [np.dot(vertex, axis) for vertex in polygon]
    return min(dots), max(dots)

def normalize(vector):
    length = np.sqrt(vector[0] ** 2 + vector[1] ** 2)
    if length == 0:
        return vector
    return vector / length

def polygons_intersect(poly1, poly2):
    polygons = [poly1, poly2]
    
    for polygon in polygons:
        for i in range(len(polygon)):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % len(polygon)]

            edge = np.array([p2[0] - p1[0], p2[1] - p1[1]])
            axis = np.array([-edge[1], edge[0]])  
            axis = normalize(axis)
            
            min_a, max_a = project_polygon(poly1, axis)
            min_b, max_b = project_polygon(poly2, axis)
            if max_a < min_b or max_b < min_a:
                return False 
    
    return True 

def area_of_polygon(polygon):
    x, y = zip(*polygon)
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def ensure_polygon_closed(polygon):
    if polygon[0] != polygon[-1]:
        polygon.append(polygon[0])
    return polygon

def merge_polygons(poly1, poly2):
    points = np.vstack([poly1, poly2])  
    hull = convex_hull(points)
    return ensure_polygon_closed(hull)

def convex_hull(points):
    points = sorted(points.tolist()) 
    if len(points) <= 1:
        return points

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def get_bbox(polygon):
    return [min([p[0] for p in polygon]), min([p[1] for p in polygon]), max([p[0] for p in polygon]), max([p[1] for p in polygon])]

def not_bbox_overlap(bbox1, bbox2):
    return bbox1[2] < bbox2[0] or bbox2[2] < bbox1[0] or bbox1[3] < bbox2[1] or bbox2[3] < bbox1[1]

#
# Main function for merging overlapping polygons
#
def merge_overlapping_polygons(geojson):
    features = geojson['features']
    merged_features = []
    
    while features:
        current = features.pop(0)
        current_poly = current['geometry']['coordinates'][0]
        current_poly_bbox = get_bbox(current_poly)
        current_class = current['properties']['classification']['name']
        current_area = area_of_polygon(current_poly)
        overlap_found = False
        
        for i, other in enumerate(features):
            other_poly = other['geometry']['coordinates'][0]
            other_class = other['properties']['classification']['name']
            other_poly_bbox = get_bbox(other_poly)

            if not_bbox_overlap(current_poly_bbox, other_poly_bbox):
                continue
            elif polygons_intersect(current_poly, other_poly):
                overlap_found = True
                merged_poly = merge_polygons(current_poly, other_poly)
                if area_of_polygon(other_poly) > current_area:
                    current_class = other_class
                
                current_poly = merged_poly
                current_area = area_of_polygon(merged_poly)
                
                features.pop(i)
                break
    
        current_poly = ensure_polygon_closed(current_poly)

        merged_feature = deepcopy(current)
        merged_feature['geometry']['coordinates'] = [current_poly]
        merged_feature['properties']['classification']['name'] = current_class
        merged_features.append(merged_feature)
        
        
        if overlap_found:
            features.insert(0, merged_feature)
    
    return merged_features