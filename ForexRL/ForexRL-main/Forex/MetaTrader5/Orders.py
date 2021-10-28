import MetaTrader5 as mt5
import pandas as pd
import numpy as np

class OrderManagement:
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()

    def __init__(self,
                 c_symbols,
                 c_point_symbols,
                 ):
        self.symbols = c_symbols
        self.point_symbols = c_point_symbols

    @staticmethod
    def get_opened_positions(p_symbol):
        positions = mt5.positions_get(symbol=p_symbol)
        sym_positions = pd.DataFrame()
        for i_pos, pos in enumerate(positions):
            if pos.symbol == p_symbol:
                df = pd.DataFrame(pos._asdict().items(), columns=['index', 'value'])
                sym_positions[f'val{i_pos}'] = df.set_index('index')

        return sym_positions

    @staticmethod
    def sending_request(request, max_try: int = 10):
        request['volume'] = np.round(request['volume'], 2)
        print('sendig order volume = ', request['volume'], request['symbol'])
        for i in range(max_try):

            order_req = mt5.order_send(request)
            if order_req is not None:

                if order_req.retcode == mt5.TRADE_RETCODE_DONE:

                    return True

                else:
                    # TODO based on error try to fix and send again order
                    request['volume'] = 0.01  # np.round(request['volume'], 2)
                    pass
            if i == max_try - 1:
                print('error to close position ', request['symbol'], order_req.retcode)
                print(request['volume'], order_req.volume)
                return False

    def order_open_position(self,
                            symbol: str,
                            volume_lot: float,
                            price: float = 0.,
                            stop_loss_point: int = 30,
                            take_profit_point: int = 30,
                            price_deviation_point: int = 20,
                            ):
        lot = 0.
        price = price
        deal_side = None
        price_st = stop_loss_point
        price_tp = take_profit_point
        if volume_lot > 0:
            lot = abs(volume_lot)
            deal_side = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask
            price_st = price - stop_loss_point * self.point_symbols[symbol]
            price_tp = price + take_profit_point * self.point_symbols[symbol]

        elif volume_lot < 0:
            lot = abs(volume_lot)
            deal_side = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
            price_st = price + stop_loss_point * self.point_symbols[symbol]
            price_tp = price - take_profit_point * self.point_symbols[symbol]
        lot = np.round(lot, 2)
        if lot < 0.01:
            return None
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": abs(lot),
            "type": deal_side,
            "price": price,
            "sl": price_st,
            "tp": price_tp,
            "deviation": price_deviation_point,
            "magic": 234000,
            "comment": "python script open",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }

        self.sending_request(request)

    def close_opened_position(self,
                              symbol,
                              p_volume_to_close: float = 0.,
                              p_deal_side=0,
                              ):
        df_positions = self.get_opened_positions(symbol)
        deal_side = p_deal_side

        opened_volume = 0.
        if len(df_positions) > 0:
            for col in df_positions.columns:
                if deal_side == df_positions.loc['type', col]:
                    opened_volume += df_positions.loc['volume', col]
        else:
            return None

        volume_to_close = opened_volume if p_volume_to_close == 0. or abs(p_volume_to_close) > opened_volume else abs(
            p_volume_to_close)
        closed_volume = 0.

        for col in df_positions.columns:

            if closed_volume < volume_to_close and deal_side == df_positions.loc['type', col]:
                deal_type = 0.
                deal_price = 0.
                last_deal_type = df_positions.loc['type', col]

                last_deal_volume = df_positions.loc['volume', col]
                if (volume_to_close - closed_volume) < last_deal_volume:
                    last_deal_volume = volume_to_close - closed_volume

                if last_deal_type == mt5.ORDER_TYPE_BUY:
                    deal_type = mt5.ORDER_TYPE_SELL
                    deal_price = mt5.symbol_info_tick(symbol).bid

                elif last_deal_type == mt5.ORDER_TYPE_SELL:
                    deal_type = mt5.ORDER_TYPE_BUY
                    deal_price = mt5.symbol_info_tick(symbol).ask

                position_id = df_positions.loc['ticket', col]
                deviation = 20
                last_deal_volume = abs(last_deal_volume)
                if abs(last_deal_volume) < 0.01:
                    return None
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": symbol,
                    "volume": last_deal_volume,
                    "type": deal_type,
                    "position": position_id,
                    "price": deal_price,
                    "deviation": deviation,
                    "magic": 234000,
                    "comment": "python script close",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_RETURN,
                }

                self.sending_request(request)

    def portfolio_to_order(self,
                           portfolio_volume_t_2,
                           portfolio_volume_t_1
                           ):

        for sym in self.symbols:
            volume_sym_t_1 = portfolio_volume_t_1[sym][0]
            volume_sym_t_2 = portfolio_volume_t_2[sym][0]

            if volume_sym_t_2 == 0.:
                if volume_sym_t_1 > 0.:
                    self.close_opened_position(sym,
                                               p_volume_to_close=volume_sym_t_1,
                                               p_deal_side=0
                                               )
                elif volume_sym_t_1 < 0.:
                    self.close_opened_position(sym,
                                               p_volume_to_close=volume_sym_t_1,
                                               p_deal_side=1
                                               )

            elif volume_sym_t_2 > 0.:
                if volume_sym_t_1 > 0.:
                    diff_vol = volume_sym_t_2 - volume_sym_t_1

                    if diff_vol < 0:
                        self.close_opened_position(sym,
                                                   p_volume_to_close=diff_vol,
                                                   p_deal_side=0
                                                   )

                    elif diff_vol > 0:
                        self.order_open_position(sym, volume_lot=diff_vol)

                elif volume_sym_t_1 < 0.:
                    if self.close_opened_position(sym,
                                                  p_volume_to_close=volume_sym_t_1,
                                                  p_deal_side=1
                                                  ):
                        self.order_open_position(sym, volume_lot=volume_sym_t_2)
                    else:
                        # TODO fix error and try again
                        print('error SELL to BUY', sym)

            elif volume_sym_t_2 < 0.:
                if volume_sym_t_1 < 0.:
                    diff_vol = volume_sym_t_2 - volume_sym_t_1

                    if diff_vol < 0:
                        self.order_open_position(sym, volume_lot=diff_vol)

                    elif diff_vol > 0:
                        self.close_opened_position(sym,
                                                   p_volume_to_close=diff_vol,
                                                   p_deal_side=1
                                                   )

                elif volume_sym_t_1 > 0.:
                    if self.close_opened_position(sym,
                                                  p_volume_to_close=volume_sym_t_1,
                                                  p_deal_side=0
                                                  ):
                        self.order_open_position(sym, volume_lot=volume_sym_t_2)
                    else:
                        # TODO fix error and try again
                        print('error BUY to SELL', sym)

    def is_opposite_order(self):
        is_opposite = [False for _ in range(len(self.symbols))]
        for i_sym, sym in enumerate(self.symbols):
            df_positions = self.get_opened_positions(sym)
            if len(df_positions.columns) > 1:
                type_deals = df_positions.loc['type', :]
                if all(element == type_deals[0] for element in type_deals):
                    is_opposite[i_sym] = False
                else:
                    is_opposite[i_sym] = True
                    print('opposite in ', sym)
        return is_opposite


from Forex.MetaTrader5.Symbols import Symbols, currencies
import numpy as np
import time


def random_portfolio():
    portfolio_volume_lots = pd.DataFrame()
    for sym in symbols.selected_symbols:
        portfolio_volume_lots[sym] = [np.round(np.random.uniform(low=-0.04, high=0.04), 2)]

    return portfolio_volume_lots


symbols = Symbols(currencies)
om = OrderManagement(symbols.selected_symbols, symbols.point_symbols)

r1 = random_portfolio()
print(r1)
for _ in range(40):
    r2 = r1 + random_portfolio() / 2
    print(r2)
    om.portfolio_to_order(r2, r1)
    r1 = r2
    print(om.is_opposite_order())
    print()
    time.sleep(3)
