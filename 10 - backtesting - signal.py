from datetime import datetime
import backtrader as bt

# Improved signal strategy using standard Backtrader conventions
class SmaSignal(bt.SignalStrategy):
    params = (('period', 20), )

    def __init__(self):
        # Add the SMA indicator
        sma = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.p.period)
        # Generate signal: 1 (long) if price > SMA, -1 (short) if price < SMA
        self.signal_add(bt.SIGNAL_LONG, self.datas[0] > sma)

if __name__ == "__main__":
    # Use context managers where possible and avoid deprecated patterns
    data = bt.feeds.YahooFinanceData(
        dataname='AAPL',
        fromdate=datetime(2018, 1, 1),
        todate=datetime(2018, 12, 31)
    )

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.adddata(data)
    cerebro.broker.set_cash(1000.0)
    cerebro.addstrategy(SmaSignal)  # Updated to addstrategy instead of add_signal

    cerebro.addobserver(bt.observers.BuySell)
    cerebro.addobserver(bt.observers.Value)

    print(f"Starting Portfolio Value: {cerebro.broker.getvalue():.2f}")
    cerebro.run()
    print(f"Final Portfolio Value: {cerebro.broker.getvalue():.2f}")
    cerebro.plot(iplot=True, volume=False)
