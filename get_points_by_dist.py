from math import sqrt, cos, radians
from functools import cmp_to_key
from classes import *

def dot_product(x1, y1, x2, y2):
    return x1 * x2 + y1 * y2

def distance_to_line(a, b, p):
    x1 = a.x
    y1 = a.y
    x2 = b.x
    y2 = b.y
    x3 = p.x
    y3 = p.y
    numerator = abs((x2 - x1) * (y1 - y3) - (x1 - x3) * (y2 - y1))
    denominator = sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if denominator == 0:
        return sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
    return numerator / denominator

def is_proj_in_segment(a, b, p):
    x1 = a.x
    y1 = a.y
    x2 = b.x
    y2 = b.y
    x3 = p.x
    y3 = p.y
    d1 = dot_product(x3 - x1, y3 - y1, x2 - x1, y2 - y1)
    d2 = dot_product(x3 - x2, y3 - y2, x1 - x2, y1 - y2)
    return d1 >= 0 and d2 >= 0

def get_points_into_route(objects_list, p_start, p_end, max_points=5, max_distance=1000):
    work_objects = []
    for obj in objects_list:
        work_objects.append(obj)
    work_p_start = p_start
    work_p_end = p_end

    mean_lat = 0
    for obj in work_objects:
        mean_lat = mean_lat + obj.y
    mean_lat = mean_lat / len(work_objects)

    lon_to_km = 111.32 * cos(radians(mean_lat))

    for obj in work_objects:
        obj.x = lon_to_km * obj.x
        obj.y = 111.32 * obj.y
    work_p_start.x = lon_to_km * work_p_start.x
    work_p_start.y = 111.32 * work_p_start.y
    work_p_end.x = lon_to_km * work_p_end.x
    work_p_end.y = 111.32 * work_p_end.y
    def comparator(a, b):
        dist_a = distance_to_line(work_p_start, work_p_end, a)
        dist_b = distance_to_line(work_p_start, work_p_end, b)
        if dist_a < dist_b:
            return -1
        elif dist_a > dist_b:
            return 1
        else:
            return 0
    sorted_objects = sorted(work_objects, key=cmp_to_key(comparator))
    ans = []
    for obj in sorted_objects:
        if len(ans) >= max_points: # or distance_to_line(work_p_start, work_p_end, obj) > max_distance:
            break
        ans.append(obj)
        # if is_proj_in_segment(work_p_start, work_p_end, obj):
        #     result_obj = Object(
        #         obj.x / lon_to_km,
        #         obj.y / 111.32,
        #         obj.desc,
        #         obj.other
        #     )
        #     ans.append(result_obj)
    
    return ans


# #нужны данные:
# #coordinates_list, p_start, p_end

# result = get_points_into_route(coordinates_list, p_start, p_end)

# print("Точки, попадающие в маршрут:")
# for obj in result:
#     print(f"Долгота: {obj.x:.6f}, Широта: {obj.y:.6f}")

