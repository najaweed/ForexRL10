import MetaTrader5 as mt5

currencies = ['USD', 'EUR', 'GBP', 'AUD', 'NZD', 'JPY', 'CHF', 'CAD']  # , 'HKD', 'XAU', 'XAG', 'BTC', 'ETH']
if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())
    quit()


class Symbols:
    def __init__(self,
                 c_currencies: list = currencies,
                 ):
        self.currencies = c_currencies
        self._all_symbols = self.all_symbols

    @property
    def all_symbols(self):
        symbols_dict = mt5.symbols_get()
        symbols_list = []
        for i in range(len(symbols_dict)):
            symbols_name = symbols_dict[i].name
            symbols_list.append(symbols_name)

        return symbols_list

    @property
    def selected_symbols(self):
        selected_symbol = []
        for i in self.currencies:
            for j in self.currencies:
                if i != j and f'{i}{j}.c' in self._all_symbols:
                    selected_symbol.append(f'{i}{j}.c')
        return selected_symbol

    @property
    def point_symbols(self):
        sym_point = {}
        for sym in self.selected_symbols:
            sym_point[sym] = mt5.symbol_info(sym).point
        return sym_point


symbols = Symbols().selected_symbols
