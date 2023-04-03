import numpy as np
import pandas as pd
import requests
import datetime as dt
from datetime import datetime, time
from datetime import timedelta


second = pd.read_csv("C:\\Users\\User\Documents\PostGrad\ML\group proj\BDP_Bikes_Final.csv", on_bad_lines='skip')

#
second = second.iloc[180000:200000]
second.plot(x  = "TIME",y = "bikes")

#second.to_csv("C:\\Users\\User\\Documents\\PostGrad\\ML\\group proj\\DublinBikes.csv")
   
    