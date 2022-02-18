from datetime import datetime
from meteostat import Hourly, Point
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np


class Weather_container:
    def __init__(self, longitude: float, latitude: float, time_interval: str = "4H"):
        self.point = Point(lon=longitude, lat=latitude)
        self.time_interval = time_interval

    def get_weather_df(self, start: datetime, end: datetime, time_interval="4H"):

        data = Hourly(self.point, start=start, end=end)
        data.aggregate(self.time_interval)
        data = data.fetch()

        return data[["temp", "rhum", "prcp", "wspd"]].values.mean(axis=0)


def encode_times(datetime_series: pd.Series):
    """
    Creates a time-encoded numpy array.
    Encoding is given by: HOUR X DAY X MONTH

    Args:
        datetime_series (pd.Series): contains all datetimes
        time_interval (str, optional): Defaults to "4H".
    """
    datetime_series = datetime_series.unique()

    """hour = datetime_series.apply(lambda x: x.hour).values
    day = datetime_series.apply(lambda x: x.weekday()).values
    month = datetime_series.apply(lambda x: x.month).values"""
    hour = np.apply_along_axis(lambda x: x.hour, axis=1, arr=datetime_series)
    day = np.apply_along_axis(lambda x: x.weekday(), axis=0, arr=datetime_series)
    month = np.apply_along_axis(lambda x: x.month, axis=0, arr=datetime_series)

    # hard code a OneHotEncoder that can be used
    encoder = OneHotEncoder()
    x_artificial = np.ones((24, 3))

    time_int = int(time_interval[0])

    # hour encoding:
    x_artificial[:, 0] = hour[-1]
    for i in range(int(24 / time_int)):
        x_artificial[i, 0] = i * time_int

    # weekday encoding:
    x_artificial[:, 1] = day[-1]
    for i in range(7):
        x_artificial[i, 1] = i

    # month encoding:
    x_artificial[:, 2] = month[-1]
    for i in range(1, 13):
        x_artificial[i - 1, 2] = i

    encoder.fit(x_artificial)

    arr = np.zeros((len(datetime_series), 3))
    arr[:, 0] = hour
    arr[:, 1] = day
    arr[:, 2] = month

    return encoder.transform(arr).toarray()


def time_encoder(time_interval="1H"):
    """
    Creates a time-encoded numpy array.
    Encoding is given by: HOUR X DAY X MONTH

    Args:
        datetime_series (pd.Series): contains all datetimes
        time_interval (str, optional): Defaults to "4H".
    """

    # hard code a OneHotEncoder that can be used
    encoder = OneHotEncoder()
    time_int = int(time_interval[0])
    print("time_int")
    if 24 / time_int > 12:
        x_artificial = np.ones((int(24 / time_int), 3))
    else:
        x_artificial = np.ones((24, 3))

    # hour encoding:
    for i in range(int(24 / time_int)):
        x_artificial[i, 0] = i * time_int

    # weekday encoding:
    for i in range(7):
        x_artificial[i, 1] = i

    # month encoding:
    for i in range(1, 13):
        x_artificial[i - 1, 2] = i

    encoder.fit(x_artificial)

    return encoder


def get_weather_info(start: datetime, stop: datetime, point: Point):

    start1 = datetime(2018, 12, 1, 0, 0)

    end1 = datetime(2018, 12, 1, 4, 0)

    data = Hourly(point, start=start1, end=end1)

    data = data.normalize()
    data = data.aggregate("4H")
    data = data.fetch()
    print(data)
    return data[["temp", "rhum", "prcp", "wspd"]]


def get_nearby_point(longitude: float, lattitude: float):

    point = Point(lon=longitude, lat=lattitude)

    return point


if __name__ == "__main__":
    point = get_nearby_point(longitude=-74.002776, lattitude=40.760875)

    data = get_weather_info(0, 0, point)

    print(data.values.mean(axis=0))
