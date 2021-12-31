import pytz
from datetime import datetime
import MetaTrader5 as mt5
timezone = pytz.timezone("Etc/UTC")

utc_from = datetime(2021, 11, 1,hour=0, tzinfo=timezone)
utc_to = datetime(2021, 11, 15,hour=0, tzinfo=timezone) #datetime.now(tz=timezone)
t_frame = mt5.TIMEFRAME_M1

class MarketTime:
    def __init__(self,
                 datetime_from=utc_from,
                 datetime_to=utc_to,
                 time_frame=mt5.TIMEFRAME_M1,
                 ):
        self.datetime_from = datetime_from
        self.datetime_to = datetime_to
        self.time_frame = time_frame

    @property
    def time_range(self):
        return self.time_frame, self.datetime_from, self.datetime_to

