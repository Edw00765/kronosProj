import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_ta_classic as ta
import sys
import torch
import os
from pathlib import Path

# --- Kronos setup ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "Kronos"))
from model import Kronos, KronosTokenizer, KronosPredictor

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
print("Loading Kronos model...", flush=True)
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
kronos_model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
predictor = KronosPredictor(kronos_model, tokenizer, max_context=512, device=device)
print("Kronos loaded.", flush=True)

CONTEXT_WINDOW = 400          # bars of history fed to Kronos (must be <= max_context)
FORECAST_HORIZON = 10
KRONOS_UPSIDE_THRESHOLD = 0.6
MONTE_CARLO_SAMPLES = 10      # keep modest -- each sample is a full model call

adx_threshold = 20


def get_kronos_upside_probability(data, idx, n_samples=MONTE_CARLO_SAMPLES):
    """
    Runs Kronos n_samples times over the context window ending at idx,
    and returns the fraction of simulated paths whose final close
    is above the current close (i.e. model's own confidence the price rises).
    """
    if idx < CONTEXT_WINDOW:
        return None  # not enough history yet

    context_df = data.iloc[idx - CONTEXT_WINDOW: idx][['open', 'high', 'low', 'close', 'volume']]
    x_timestamp = data.index[idx - CONTEXT_WINDOW: idx].to_series().reset_index(drop=True)
    y_timestamp = pd.Series(pd.bdate_range(start=data.index[idx], periods=FORECAST_HORIZON))

    current_close = data.iloc[idx]['close']
    final_closes = []

    for _ in range(n_samples):
        pred_df = predictor.predict(
            df=context_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp,
            pred_len=FORECAST_HORIZON, T=1.0, top_p=0.9, sample_count=1, verbose=False
        )
        final_closes.append(pred_df['close'].iloc[-1])

    final_closes = np.array(final_closes)
    return float(np.mean(final_closes > current_close))


def generate_signals(df, short_window, long_window, adx_threshold=20, stop_loss_pct=0.05,
                      ma_type="SMA", price_col="Close", use_kronos_confirmation=True):
    data = df.copy()

    if ma_type == "EMA":
        data['SMA_short'] = data[price_col].ewm(span=short_window, adjust=False).mean()
        data['SMA_long'] = data[price_col].ewm(span=long_window, adjust=False).mean()
    elif ma_type == "EMA_short":
        data['SMA_short'] = data[price_col].ewm(span=short_window, adjust=False).mean()
        data['SMA_long'] = data[price_col].rolling(window=long_window).mean()
    else:
        data['SMA_short'] = data[price_col].rolling(window=short_window).mean()
        data['SMA_long'] = data[price_col].rolling(window=long_window).mean()

    adx_df = ta.adx(data['High'], data['Low'], data[price_col], length=14)
    data['ADX'] = adx_df.iloc[:, 0]
    data['ATR'] = ta.atr(data['High'], data['Low'], data[price_col], length=14)

    data['ma_long_condition'] = data['SMA_short'] > data['SMA_long']
    data['signal'] = 0
    data['exit_reason'] = ""
    data['exit_price'] = np.nan
    data['kronos_prob'] = np.nan  # log it for later analysis, even on bars we don't act on

    # Kronos needs lowercase OHLCV columns
    data_lower = data.rename(columns={
        'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
    })

    in_position = False
    entry_price = None

    signal_col = data.columns.get_loc('signal')
    reason_col = data.columns.get_loc('exit_reason')
    kronos_col = data.columns.get_loc('kronos_prob')

    for i in range(len(data)):
        row = data.iloc[i]
        price = row[price_col]

        if pd.isna(row['SMA_long']) or pd.isna(row['ADX']):
            continue

        if not in_position:
            technical_signal = row['ma_long_condition'] and row['ADX'] > adx_threshold

            if technical_signal:
                if use_kronos_confirmation:
                    prob = get_kronos_upside_probability(data_lower, i)
                    if prob is not None:
                        data.iloc[i, kronos_col] = prob
                        if prob >= KRONOS_UPSIDE_THRESHOLD:
                            in_position = True
                            entry_price = price
                    # if prob is None (not enough history yet), skip -- don't enter blind
                else:
                    in_position = True
                    entry_price = price
        else:
            stop_price = entry_price * (1 - stop_loss_pct)
            hit_stop = price <= stop_price
            trend_reversed = not row['ma_long_condition']

            if hit_stop:
                in_position = False
                entry_price = None
                data.iloc[i, reason_col] = "Stop Loss"
                data.iloc[i, data.columns.get_loc('exit_price')] = min(stop_price, row['Open'])
            elif trend_reversed:
                in_position = False
                entry_price = None
                data.iloc[i, reason_col] = "Trend Reversal"
                data.iloc[i, data.columns.get_loc('exit_price')] = row[price_col]

        data.iloc[i, signal_col] = 1 if in_position else 0

    data['position'] = data['signal'].diff()
    return data


def calculate_trades(data):
    trades = []
    buy_price = None
    buy_date = None

    for date, row in data.iterrows():
        if row["position"] == 1 and buy_price is None:
            buy_price = row["Close"]
            buy_date = date

        elif row["position"] == -1 and buy_price is not None:
            sell_price = row["exit_price"]
            profit = sell_price - buy_price
            percent = (profit / buy_price) * 100

            trades.append({
                "Buy Date": buy_date,
                "Sell Date": date,
                "Buy Price": buy_price,
                "Sell Price": sell_price,
                "Profit": profit,
                "Return %": percent,
                "Exit Type": row["exit_reason"]
            })

            buy_price = None
            buy_date = None

    return pd.DataFrame(trades)


ticker = "TSLA4"
data = pd.read_csv(f"kline_data/{ticker}_data.csv", index_col=0, parse_dates=True)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

signals = generate_signals(data, short_window=20, long_window=50, ma_type="EMA", use_kronos_confirmation=True)

trades = calculate_trades(signals)
total_profit = trades["Profit"].sum()
print(trades)
print(f"Total Profit: {total_profit:.2f}")

actual_buys = trades.set_index("Buy Date")
actual_sells = trades.set_index("Sell Date")

plt.figure(figsize=(15,7))

plt.plot(signals.index, signals["Close"], label="Close Price")
plt.plot(signals.index, signals["SMA_short"], label="20-Day SMA")
plt.plot(signals.index, signals["SMA_long"], label="50-Day SMA")

plt.scatter(
    actual_buys.index,
    actual_buys["Buy Price"],
    marker="^",
    s=100,
    label="Buy"
)
plt.scatter(
    actual_sells.index,
    actual_sells["Sell Price"],
    marker="v",
    s=100,
    label="Sell"
)

for date, row in actual_sells.iterrows():
    plt.text(
        x=date, 
        y=row["Sell Price"] * 0.98,            # Places text 2% below the point to avoid overlap
        s=row["Exit Type"], 
        color="red", 
        fontsize=9, 
        ha="center",                           # Horizontal alignment centered on marker
        va="top",                              # Vertical alignment pointing down
        rotation=45,                           # Rotated slightly for readability on tight charts
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1) # Background for contrast
    )

plt.title(f"{ticker} Moving Average Crossover")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ... (plotting code unchanged)
