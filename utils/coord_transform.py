# -*- coding: utf-8 -*-
import json
import urllib
import math

x_pi = 3.14159265358979324 * 3000.0 / 180.0
pi = 3.1415926535897932384626  # π
a = 6378245.0  
ee = 0.00669342162296594323  


class Convert():
    def __init__(self):
        pass

    def convert(self, lng, lat):
        return lng, lat


class GCJ02ToWGS84(Convert):
    def __init__(self):
        super().__init__()

    def convert(self, lng, lat):
    
        if out_of_china(lng, lat):
            return [lng, lat]
        dlat = _transformlat(lng - 105.0, lat - 35.0)
        dlng = _transformlng(lng - 105.0, lat - 35.0)
        radlat = lat / 180.0 * pi
        magic = math.sin(radlat)
        magic = 1 - ee * magic * magic
        sqrtmagic = math.sqrt(magic)
        dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
        dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
        mglat = lat + dlat
        mglng = lng + dlng
        return lng * 2 - mglng, lat * 2 - mglat


class WGS84ToGCJ02(Convert):
    def __init__(self):
        super().__init__()

    def convert(self, lng, lat):
     
        if out_of_china(lng, lat):
            return [lng, lat]
        dlat = _transformlat(lng - 105.0, lat - 35.0)
        dlng = _transformlng(lng - 105.0, lat - 35.0)
        radlat = lat / 180.0 * pi
        magic = math.sin(radlat)
        magic = 1 - ee * magic * magic
        sqrtmagic = math.sqrt(magic)
        dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
        dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
        mglat = lat + dlat
        mglng = lng + dlng
        return mglng, mglat



def _transformlat(lng, lat):
    ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
          0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lat * pi) + 40.0 *
            math.sin(lat / 3.0 * pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(lat / 12.0 * pi) + 320 *
            math.sin(lat * pi / 30.0)) * 2.0 / 3.0
    return ret


def _transformlng(lng, lat):
    ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
          0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
    ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
            math.sin(2.0 * lng * pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(lng * pi) + 40.0 *
            math.sin(lng / 3.0 * pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(lng / 12.0 * pi) + 300.0 *
            math.sin(lng / 30.0 * pi)) * 2.0 / 3.0
    return ret


def out_of_china(lng, lat):
    return not (lng > 73.66 and lng < 135.05 and lat > 3.86 and lat < 53.55)


# if __name__ == '__main__':
#     lng, lat = 116.527559, 39.807378
#     min_lat = 39.727
#     min_lng = 116.490
#     max_lat = 39.83
#     max_lng = 116.588
#     # lng = 109.642194
#     # lat = 20.123355
#     result1 = gcj02_to_bd09(lng, lat)
#     result2 = bd09_to_gcj02(lng, lat)
#     result3 = wgs84_to_gcj02(lng, lat)
#     result4 = gcj02_to_wgs84(lng, lat)
#     result5 = bd09_to_wgs84(lng, lat)
#     result6 = wgs84_to_bd09(lng, lat)
#     print(gcj02_to_wgs84(min_lng, min_lat))
#     print(gcj02_to_wgs84(max_lng, max_lat))
