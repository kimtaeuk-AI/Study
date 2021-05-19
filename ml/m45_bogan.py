from pandas import DataFrame, Series
from datetime import datetime
import numpy as np
import pandas as pd

datestrs = ['3/1/2021', '3/2/2021', '3/3/2021', '3/4/2021', '3/5/2021' ]
dates = pd.to_datetime(datestrs)
print(dates)
print("=========================")

ts = Series([1, np.nan, np.nan, 8, 10], index=dates)
print(ts)

ts_intp_linear = ts.interpolate() #interpolate -보강법 ,시계열 데이터에서 쓰면 좋음 
# 시계열 데이터 - 시간의 순서가 있는 데이터

print(ts_intp_linear)