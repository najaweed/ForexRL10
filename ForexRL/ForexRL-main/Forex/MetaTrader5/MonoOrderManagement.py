import MetaTrader5 as mt5
import pandas as pd
import numpy as np


class MonoOrderManagement:
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        quit()

    def __init__(self,
                 c_symbol,
                 ):
        self.symbol = c_symbol
        self.info = mt5.symbol_info(self.symbol)._asdict()
        self.point = self.info['digits']

    @property
    def get_opened_positions(self):
        positions = mt5.positions_get(symbol=self.symbol)
        sym_positions = []
        for i_pos, pos in enumerate(positions):
            if pos.symbol == self.symbol:
                df = pd.DataFrame(pos._asdict().items(), columns=['index', 'value'])
                print(df)
                sym_positions.append(df)

        return sym_positions

    @staticmethod
    def sending_request(request, max_try: int = 10):
        request['volume'] = np.round(request['volume'], 2)
        print('sending order volume = ', request['volume'], request['symbol'])
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
            print(symbol, mt5.symbol_info_tick(symbol))
            price = mt5.symbol_info_tick(symbol).ask
            price_st = price - stop_loss_point * self.point
            price_tp = price + take_profit_point * self.point

        elif volume_lot < 0:
            lot = abs(volume_lot)
            deal_side = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
            price_st = price + stop_loss_point * self.point
            price_tp = price - take_profit_point * self.point
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
        df_positions = self.get_opened_positions
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


order_manage = MonoOrderManagement('EURUSD.si')
print(order_manage.get_opened_positions)
