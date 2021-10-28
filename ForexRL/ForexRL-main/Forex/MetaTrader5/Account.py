import MetaTrader5 as mt5
import pandas as pd

if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())


class Account:

    def __init__(self,
                 c_symbols=[],
                 ):

        self.symbols = c_symbols
        self.account_info = mt5.account_info()
        if self.account_info is None:
            print('error')

    @staticmethod
    def get_opened_positions(p_symbol):
        positions = mt5.positions_get(symbol=p_symbol)
        sym_positions = pd.DataFrame()
        for i_pos, pos in enumerate(positions):
            if pos.symbol == p_symbol:
                df = pd.DataFrame(pos._asdict().items(), columns=['index', 'value'])
                sym_positions[f'val{i_pos}'] = df.set_index('index')

        return sym_positions

    @property
    def balance(self):
        return self.account_info.balance

    @property
    def margin_locked(self):
        return self.account_info.margin

    @property
    def profit(self):
        return self.account_info.profit

    @property
    def vector_profit(self):
        v_portfolio = pd.DataFrame()
        for sym in self.symbols:
            df_positions = self.get_opened_positions(sym)
            if len(df_positions) > 0:
                v_portfolio[sym] = [sum(df_positions.loc['profit', :])]
            else:
                v_portfolio[sym] = [0.]

        return v_portfolio

    @property
    def portfolio_lot(self):
        v_portfolio = pd.DataFrame()
        for sym in self.symbols:
            df_positions = self.get_opened_positions(sym)
            if len(df_positions) > 0:
                v_portfolio[sym] = [sum(df_positions.loc['volume', :])]
            else:
                v_portfolio[sym] = [0.]

        return v_portfolio


# from Forex.Symbols import Symbols, currencies
#
# symbols = Symbols(currencies)
# pd.set_option('display.max_columns', None)
# print(Account(symbols.selected_symbols).vector_profit)
# print(Account(symbols.selected_symbols).portfolio_lot)
