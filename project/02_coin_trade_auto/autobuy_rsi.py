import pandas as pd
import pyupbit
import time

access = "confidential"
secret = "confidential"
upbit = pyupbit.Upbit(access, secret)

# RSI지표 수치를 구하는 함수 어디서 가져온거. (작동 원리는 아직 모름)
def GetRSI(ohlcv,period):
    ohlcv["close"] = ohlcv["close"]
    delta = ohlcv["close"].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    _gain = up.ewm(com=(period - 1), min_periods=period).mean()
    _loss = down.abs().ewm(com=(period - 1), min_periods=period).mean()
    RS = _gain / _loss
    return pd.Series(100 - (100 / (1 + RS)), name="RSI")

# 일봉 정보를 가져옴
ohlcv_df = pyupbit.get_ohlcv("KRW-BTC", interval = "minuete240")

# 14일간의 RSI 
rsi14 = float(GetRSI(ohlcv_df,14).iloc[-1]) # -1 위치를 가져와야 가장 최근 (현재 RSI 값이 가져와짐)

print('BTC bot working: purchasing based when RSI <=30')
print(f'Current RSI: {rsi14}')
if rsi14 <= 30:
    upbit.buy_market_order("KRW-BTC",5000)
    print(f"BTC purchasement completed when RSI was {rsi14}")

# crontab -e
# */30 * * * * python3 /var/pycoinbot/autobuy_rsi.py