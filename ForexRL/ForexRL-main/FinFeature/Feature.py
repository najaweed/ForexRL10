from Forex.MetaTrader5.Market import MarketData
from Forex.MetaTrader5.Time import MarketTime

mrk = MarketData(MarketTime().time_range)
print(mrk.time[0])
print(mrk.log_high_low)