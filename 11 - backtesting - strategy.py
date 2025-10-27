import backtrader as bt
from datetime import datetime

class SmaStrategy(bt.Strategy):
    params = (('ma_period', 20), )

    def __init__(self):
        self.data_close = self.datas[0].close
        self.order = None
        self.price = None
        self.comm = None
        self.bar_executed = None

        # Use the standard naming and direct indicator construction
        self.sma = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.ma_period)

    def log(self, txt):
        dt = self.datas[0].datetime.date(0).isoformat()
        print(f'{dt}, {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED --- Price: {order.executed.price:.2f}, '
                         f'Cost: {order.executed.value:.2f}, Commission: {order.executed.comm:.2f}')
                self.price = order.executed.price
                self.comm = order.executed.comm
            else:
                self.log(f'SELL EXECUTED ---Price: {order.executed.price:.2f}, '
                         f'Cost: {order.executed.value:.2f}, Commission: {order.executed.comm:.2f}')
                self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Failed')
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f'OPERATION RESULT --- Gross: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}')

    def next(self):
        # Avoid double orders and allow one open order at a time
        if self.order:
            return

        if not self.position:
            if self.data_close[0] > self.sma[0]:
                self.log(f'BUY CREATED --- Price: {self.data_close[0]:.2f}')
                self.order = self.buy()
        else:
            if self.data_close[0] < self.sma[0]:
                self.log(f'SELL CREATED --- Price: {self.data_close[0]:.2f}')
                self.order = self.sell()

if __name__ == "__main__":
    data = bt.feeds.YahooFinanceData(
        dataname='AAPL',
        fromdate=datetime(2018, 1, 1),
        todate=datetime(2018, 12, 31)
    )

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.adddata(data)
    cerebro.broker.set_cash(1000.0)
    cerebro.addstrategy(SmaStrategy)
    cerebro.addobserver(bt.observers.BuySell)
    cerebro.addobserver(bt.observers.Value)
    print(f'Starting Portfolio Value: {cerebro.broker.getvalue():.2f}')
    cerebro.run()
    print(f'Final Portfolio Value: {cerebro.broker.getvalue():.2f}')
    cerebro.plot(iplot=True, volume=False)
