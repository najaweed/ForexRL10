from Forex.MetaTrader5.Market import MarketData, LiveTicks
from Forex.MetaTrader5.Time import MarketTime
from Forex.MetaTrader5.Symbols import Symbols, currencies
from FinFeature.TimeSeries.Kalman import Kalman_smoother

symbols = Symbols(currencies).selected_symbols
mrk = MarketData(MarketTime().time_range, symbols)

from datetime import timedelta

delta_time = timedelta(minutes=3)
date_time_request = MarketTime().time_range[2]
# print(Kalman_smoother(LiveTicks(date_time_request, delta_time, 'EURUSD').ticks.ask))

import matplotlib.pyplot as plt

live_tick = LiveTicks(date_time_request, delta_time, 'EURJPY').ticks

kalman_smooth_ask = Kalman_smoother(live_tick.ask)

plt.plot(live_tick.ask - Kalman_smoother(live_tick.ask))
plt.plot(live_tick.bid - Kalman_smoother(live_tick.bid))
plt.plot(live_tick.ask - live_tick.bid)
plt.plot(Kalman_smoother(live_tick.ask) - Kalman_smoother(live_tick.bid))

plt.show()
