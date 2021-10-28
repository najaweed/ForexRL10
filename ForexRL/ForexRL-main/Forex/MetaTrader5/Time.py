import pytz
from datetime import datetime
import MetaTrader5 as mt5
timezone = pytz.timezone("Etc/UTC")

utc_from = datetime(2021, 10, 16, tzinfo=timezone)
utc_to = datetime.now(tz=timezone)  # datetime(2021, 2, 11, hour=13, tzinfo=timezone)


class MarketTime:
    def __init__(self,
                 datetime_from=datetime(2021, 2, 1, tzinfo=timezone),
                 datetime_to=datetime.now(tz=timezone),
                 time_frame=mt5.TIMEFRAME_M5,
                 ):
        self.datetime_from = datetime_from
        self.datetime_to = datetime_to
        self.time_frame = time_frame

    @property
    def time_range(self):
        return self.time_frame, self.datetime_from, self.datetime_to
