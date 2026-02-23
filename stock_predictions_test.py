"""
╔═══════════════════════════════════════════════════════════════════════╗
║   🇮🇳  INDIAN STOCK MARKET ANALYSIS & PREDICTION SYSTEM                ║
║                    WITH BUY/SELL TIMING & EXIT STRATEGY               ║
║                    POWERED BY PYTHON                                  ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
from sklearn.metrics import  r2_score
# from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class IndianStockAnalyzer:
    """
    🇮🇳 Complete Stock Analysis with BUY/SELL Timing & Exit Strategy

    Features:
    ─────────
    ✓ Technical Analysis (8+ indicators)
    ✓ ML-based Price Prediction
    ✓ Specific BUY Entry Points with Dates
    ✓ Multiple EXIT Targets with Timelines
    ✓ Stop Loss & Trailing Stop Loss
    ✓ Position Sizing (Kelly Criterion)
    ✓ Risk/Reward Analysis
    ✓ Price Alert Levels
    """

    def __init__(self, symbol, period='2y'):
        self.symbol = symbol
        self.period = period
        self.data = None
        self.signals = None
        self.model = None
        self.feature_importance = None
        self.risk_metrics = None
        self.training_features = None
        self.predictions = None
        self.trade_plan = None

        print(f"\n{'='*70}")
        print(f"📊 INITIALIZING ANALYZER FOR: {symbol}")
        print(f"{'='*70}")
        print(f"📅 Data Period: {period}")
        print(f"🏢 Exchange: {'NSE' if '.NS' in symbol else 'BSE' if '.BO' in symbol else 'Unknown'}")
        print(f"{'='*70}\n")

    def fetch_data(self):
        """Fetch historical stock data"""
        print("📥 STEP 1: FETCHING HISTORICAL DATA")
        print("-" * 70)
        try:
            stock = yf.Ticker(self.symbol)
            self.data = stock.history(period=self.period)

            if self.data.empty:
                raise ValueError("No data found. Check symbol format (use .NS for NSE)")

            print(f"✅ Data fetched successfully!")
            print(f"   📈 Total Records: {len(self.data)} trading days")
            print(f"   📅 Date Range: {self.data.index[0].date()} to {self.data.index[-1].date()}")
            print(f"   💰 Current Price: ₹{self.data['Close'].iloc[-1]:.2f}")
            return self.data
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None

    def calculate_technical_indicators(self):
        """Calculate all technical indicators"""
        print("\n📊 STEP 2: CALCULATING TECHNICAL INDICATORS")
        print("-" * 70)

        if self.data is None:
            print("❌ Please fetch data first!")
            return

        df = self.data.copy()

        # Daily Returns
        df['Daily_Return'] = df['Close'].pct_change()

        # Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)

        # Volume Analysis
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']

        # Momentum
        df['Momentum'] = df['Close'].pct_change(periods=10)
        df['ROC'] = df['Close'].pct_change(periods=12) * 100

        # ATR (Volatility)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(14).mean()

        # Additional features for ML
        df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
        df['High_Low_Range'] = (df['High'] - df['Low']) / df['Close']
        df['Close_Open_Range'] = (df['Close'] - df['Open']) / df['Open']

        self.data = df

        print(f"✅ {len(df.columns)} indicators calculated!")
        print("-" * 70)

        return df

    def generate_signals(self):
        """Generate Buy/Sell signals with strength"""
        print("\n🎯 STEP 3: GENERATING TRADING SIGNALS")
        print("-" * 70)

        if self.data is None:
            print("❌ Please fetch data first!")
            return

        df = self.data.copy()
        df['Signal'] = 0
        df['Signal_Strength'] = 0

        # Buy Conditions
        ma_buy = (df['SMA_20'] > df['SMA_50']) & (df['SMA_20'].shift(1) <= df['SMA_50'].shift(1))
        rsi_oversold = df['RSI'] < 30
        macd_buy = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
        bb_buy = df['Close'] < df['BB_Lower']

        # Sell Conditions
        ma_sell = (df['SMA_20'] < df['SMA_50']) & (df['SMA_20'].shift(1) >= df['SMA_50'].shift(1))
        rsi_overbought = df['RSI'] > 70
        macd_sell = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))
        bb_sell = df['Close'] > df['BB_Upper']

        # Signal Strength (0-100)
        strength = np.zeros(len(df))
        strength += (df['SMA_20'] > df['SMA_50']).astype(int) * 15
        strength += (df['SMA_50'] > df['SMA_200']).astype(int) * 15
        strength += (df['MACD'] > df['MACD_Signal']).astype(int) * 15
        strength += (df['RSI'] > 50).astype(int) * 10
        strength += (df['Close'] > df['SMA_200']).astype(int) * 15
        strength += (df['Close'] > df['BB_Middle']).astype(int) * 10
        strength += (df['Volume_Ratio'] > 1.5).astype(int) * 10
        strength += (df['Momentum'] > 0).astype(int) * 10
        strength -= (df['SMA_20'] < df['SMA_50']).astype(int) * 15
        strength -= (df['SMA_50'] < df['SMA_200']).astype(int) * 15
        strength -= (df['MACD'] < df['MACD_Signal']).astype(int) * 15
        strength -= (df['RSI'] < 50).astype(int) * 10
        strength -= (df['Close'] < df['SMA_200']).astype(int) * 15
        strength -= (df['Close'] < df['BB_Middle']).astype(int) * 10
        strength -= (df['Volume_Ratio'] < 0.5).astype(int) * 10
        strength -= (df['Momentum'] < 0).astype(int) * 10

        df['Signal_Strength'] = np.clip(strength + 50, 0, 100)

        # Final Signals
        df.loc[ma_buy | macd_buy | rsi_oversold | bb_buy, 'Signal'] = 1
        df.loc[ma_sell | macd_sell | rsi_overbought | bb_sell, 'Signal'] = -1
        df.loc[df['Signal_Strength'] > 70, 'Signal'] = 1
        df.loc[df['Signal_Strength'] < 30, 'Signal'] = -1

        self.signals = df

        print(f"✅ Total BUY Signals: {(df['Signal'] == 1).sum()}")
        print(f"✅ Total SELL Signals: {(df['Signal'] == -1).sum()}")
        print(f"✅ Current Signal Strength: {df['Signal_Strength'].iloc[-1]:.1f}/100")
        print("-" * 70)

        return df

    def calculate_risk_metrics(self):
        """Calculate risk metrics"""
        print("\n⚠️  STEP 4: CALCULATING RISK METRICS")
        print("-" * 70)

        if self.data is None:
            print("❌ Please fetch data first!")
            return

        df = self.data.copy()

        if 'Daily_Return' not in df.columns:
            df['Daily_Return'] = df['Close'].pct_change()

        volatility = df['Daily_Return'].std() * np.sqrt(252) * 100
        sharpe_ratio = (df['Daily_Return'].mean() * 252) / (df['Daily_Return'].std() * np.sqrt(252))
        max_drawdown = ((df['Close'] - df['Close'].cummax()) / df['Close'].cummax()).min() * 100
        var_95 = np.percentile(df['Daily_Return'].dropna(), 5) * 100

        self.risk_metrics = {
            'Annual_Volatility (%)': round(volatility, 2),
            'Sharpe_Ratio': round(sharpe_ratio, 2),
            'Max_Drawdown (%)': round(max_drawdown, 2),
            'VaR_95 (%)': round(var_95, 2),
            'Avg_Daily_Return (%)': round(df['Daily_Return'].mean() * 100, 3),
        }

        print("📊 RISK METRICS:")
        for metric, value in self.risk_metrics.items():
            print(f"   {metric}: {value}")
        print("-" * 70)

        return self.risk_metrics

    def build_prediction_model(self, forecast_days=60):
        """Build ML prediction model"""
        print("\n🤖 STEP 5: BUILDING ML PREDICTION MODEL")
        print("-" * 70)

        if self.data is None:
            print("❌ Please fetch data first!")
            return None

        df = self.data.copy()
        df['Future_Close'] = df['Close'].shift(-forecast_days)
        df = df.dropna()

        if len(df) < 100:
            print("❌ Insufficient data for ML model!")
            return None

        self.training_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 
                                  'RSI', 'MACD', 'Volatility', 'Momentum', 'ATR']

        available_features = [f for f in self.training_features if f in df.columns]

        if len(available_features) < 5:
            print("❌ Insufficient features!")
            return None

        X = df[available_features].values
        y = df['Future_Close'].values

        split_idx = int(len(df) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        print(f"📈 Model Performance: R² = {r2:.2f}")

        if r2 < 0:
            print("⚠️  Warning: Negative R² - Model needs more data")

        self.feature_importance = pd.DataFrame({
            'Feature': available_features,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)

        print("-" * 70)
        print("✅ ML MODEL TRAINED!")
        print("-" * 70)

        return self.model

    def predict_future_price(self, days_ahead=60):
        """Predict future prices with confidence intervals"""
        print("\n🔮 STEP 6: PREDICTING FUTURE PRICES")
        print("-" * 70)

        if self.model is None:
            print("❌ Please build model first!")
            return None

        df = self.data.copy()
        latest = df.iloc[-1:].copy()
        current_price = latest['Close'].values[0]

        print(f"📊 Predicting {days_ahead} days ahead...")
        print(f"💰 Current Price: ₹{current_price:.2f}")

        predictions = []
        lower_bound = []
        upper_bound = []
        current_data = latest.copy()

        # Get historical volatility for confidence intervals
        hist_volatility = df['Daily_Return'].std()

        for i in range(days_ahead):
            available_features = [f for f in self.training_features if f in current_data.columns]

            for feat in self.training_features:
                if feat not in current_data.columns:
                    current_data[feat] = current_data['Close'].values[0]

            X_input = current_data[self.training_features].values
            pred_price = self.model.predict(X_input)[0]
            predictions.append(pred_price)

            # Calculate confidence intervals (±2 standard deviations)
            confidence_range = pred_price * hist_volatility * np.sqrt(i + 1) * 2
            lower_bound.append(max(0, pred_price - confidence_range))
            upper_bound.append(pred_price + confidence_range)

            current_data['Close'] = pred_price
            current_data['Open'] = pred_price * 0.995
            current_data['High'] = pred_price * 1.02
            current_data['Low'] = pred_price * 0.98

        # Create prediction dataframe
        last_date = df.index[-1]
        pred_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead, freq='B')

        self.predictions = pd.DataFrame({
            'Date': pred_dates,
            'Predicted_Price': predictions,
            'Lower_Bound': lower_bound,
            'Upper_Bound': upper_bound
        })
        self.predictions.set_index('Date', inplace=True)

        # Find key levels
        max_pred = max(predictions)
        min_pred = min(predictions)

        print("\n📈 PREDICTION SUMMARY:")
        print(f"   📅 Prediction Period: {pred_dates[0].date()} to {pred_dates[-1].date()}")
        print(f"   💰 Current Price:     ₹{current_price:.2f}")
        print(f"   🔮 Avg Predicted:     ₹{np.mean(predictions):.2f}")
        print(f"   📈 Max Predicted:     ₹{max_pred:.2f} ({((max_pred-current_price)/current_price)*100:.2f}% upside)")
        print(f"   📉 Min Predicted:     ₹{min_pred:.2f} ({((min_pred-current_price)/current_price)*100:.2f}% downside)")
        print("-" * 70)

        return self.predictions

    def create_trade_plan(self):
        """
        🎯 CREATE COMPLETE TRADE PLAN WITH ENTRY & EXIT POINTS

        This is the KEY function that tells you:
        - WHEN TO BUY (entry points with dates)
        - WHEN TO SELL (exit points with targets)
        - STOP LOSS levels
        - POSITION SIZE
        """
        print("\n💼 STEP 7: CREATING TRADE PLAN (ENTRY & EXIT STRATEGY)")
        print("-" * 70)

        if self.signals is None or self.predictions is None:
            print("❌ Please run analysis and prediction first!")
            return None

        df = self.signals
        latest = df.iloc[-1]
        current_price = latest['Close']
        current_date = df.index[-1]
        atr = latest['ATR']
        rsi = latest['RSI']
        signal_strength = latest['Signal_Strength']

        # ─────────────────────────────────────────────────────────────
        # 🟢 BUY STRATEGY - WHEN TO PURCHASE
        # ─────────────────────────────────────────────────────────────
        buy_conditions = []
        buy_signals_list = []

        # Check current signal
        if latest['Signal'] == 1 and signal_strength > 60:
            buy_conditions.append("✅ Strong Buy Signal Active")
        elif latest['Signal'] == 1:
            buy_conditions.append("✅ Buy Signal Active")

        # Check RSI
        if rsi < 30:
            buy_conditions.append(f"✅ RSI Oversold ({rsi:.1f}) - Good entry point")
        elif rsi < 40:
            buy_conditions.append(f"⚠️  RSI Approaching Oversold ({rsi:.1f})")

        # Check Moving Average
        if current_price > latest['SMA_200']:
            buy_conditions.append("✅ Price above 200-day MA (Long-term bullish)")
        else:
            buy_conditions.append("⚠️  Price below 200-day MA (Wait for confirmation)")

        # Check MACD
        if latest['MACD'] > latest['MACD_Signal']:
            buy_conditions.append("✅ MACD Bullish Crossover")

        # Find upcoming buy opportunities from predictions
        for i, (date, row) in enumerate(self.predictions.iterrows()):
            pred_price = row['Predicted_Price']

            # Buy if predicted price is lower than current (dip buying opportunity)
            if pred_price < current_price * 0.95 and len(buy_signals_list) < 3:
                buy_signals_list.append({
                    'Type': 'DIP BUY',
                    'Date': date.date(),
                    'Expected_Price': round(pred_price, 2),
                    'Discount': f"{((current_price - pred_price)/current_price)*100:.1f}%",
                    'Reason': 'Price dip predicted - Accumulate'
                })

        # ─────────────────────────────────────────────────────────────
        # 🔴 EXIT STRATEGY - WHEN TO SELL
        # ─────────────────────────────────────────────────────────────
        exit_targets = []

        # Target 1: Conservative (10% gain)
        target_1_price = current_price * 1.10
        target_1_date = self._find_predicted_date(target_1_price)
        exit_targets.append({
            'Level': 'Target 1 (Conservative)',
            'Price': round(target_1_price, 2),
            'Gain': '10%',
            'Expected_Date': target_1_date,
            'Action': 'Book 30% profit'
        })

        # Target 2: Moderate (20% gain)
        target_2_price = current_price * 1.20
        target_2_date = self._find_predicted_date(target_2_price)
        exit_targets.append({
            'Level': 'Target 2 (Moderate)',
            'Price': round(target_2_price, 2),
            'Gain': '20%',
            'Expected_Date': target_2_date,
            'Action': 'Book 40% profit'
        })

        # Target 3: Aggressive (30% gain)
        target_3_price = current_price * 1.30
        target_3_date = self._find_predicted_date(target_3_price)
        exit_targets.append({
            'Level': 'Target 3 (Aggressive)',
            'Price': round(target_3_price, 2),
            'Gain': '30%',
            'Expected_Date': target_3_date,
            'Action': 'Book remaining profit'
        })

        # Stop Loss Levels
        stop_loss_1 = current_price - (atr * 2)  # Tight stop
        stop_loss_2 = current_price - (atr * 3)  # Wide stop
        stop_loss_3 = current_price * 0.90  # 10% hard stop

        # ─────────────────────────────────────────────────────────────
        # POSITION SIZING
        # ─────────────────────────────────────────────────────────────
        # Kelly Criterion for position size
        try:
            win_rate = (df['Signal'] == 1).sum() / len(df)
            buy_signals_df = df[df['Signal'] == 1]
            avg_gain = buy_signals_df['Daily_Return'].mean() if len(buy_signals_df) > 0 else 0.01
            avg_loss = abs(df[df['Signal'] == -1]['Daily_Return'].mean()) if (df['Signal'] == -1).any() else 0.01

            if avg_loss > 0 and avg_gain > 0:
                kelly = (win_rate * avg_gain - (1 - win_rate) * avg_loss) / avg_gain
                position_size = max(5, min(25, kelly * 100))  # Cap between 5-25%
            else:
                position_size = 10
        except Exception as e:
            print(f"❌ Error calculating position size: {e}")
            position_size = 10

        # ─────────────────────────────────────────────────────────────
        # CREATE TRADE PLAN
        # ─────────────────────────────────────────────────────────────
        self.trade_plan = {
            'Stock': self.symbol,
            'Analysis_Date': current_date.date(),
            'Current_Price': round(current_price, 2),

            # BUY Strategy
            'BUY_RECOMMENDATION': {
                'Current_Signal': '🟢 BUY' if latest['Signal'] == 1 else '🟠 HOLD' if latest['Signal'] == 0 else '🔴 SELL',
                'Signal_Strength': f"{signal_strength:.1f}/100",
                'Conditions': buy_conditions,
                'Upcoming_Buy_Opportunities': buy_signals_list,
                'Position_Size': f"{position_size:.1f}% of capital",
                'Entry_Zone_Low': round(current_price * 0.98, 2),
                'Entry_Zone_High': round(current_price * 1.02, 2),
            },

            # EXIT Strategy
            'EXIT_RECOMMENDATION': {
                'Targets': exit_targets,
                'Stop_Loss_Levels': {
                    'Tight_Stop': {
                        'Price': round(stop_loss_1, 2),
                        'Loss': f"{((current_price - stop_loss_1)/current_price)*100:.1f}%",
                        'When': "If you're a short-term trader"
                    },
                    'Wide_Stop': {
                        'Price': round(stop_loss_2, 2),
                        'Loss': f"{((current_price - stop_loss_2)/current_price)*100:.1f}%",
                        'When': "If you're a swing trader"
                    },
                    'Hard_Stop': {
                        'Price': round(stop_loss_3, 2),
                        'Loss': '10%',
                        'When': 'Maximum loss limit - EXIT immediately'
                    }
                },
                'Trailing_Stop': {
                    'Method': 'Move stop loss up by 2×ATR for every 5% gain',
                    'Current_ATR': round(atr, 2)
                }
            },

            # Risk Analysis
            'RISK_ANALYSIS': {
                'Risk_Reward_Ratio': f"1:{round((target_1_price - current_price)/(current_price - stop_loss_1), 1)}",
                'Max_Recommended_Loss': f"{position_size * 0.1:.1f}% of total capital",
                'Volatility': f"{self.risk_metrics['Annual_Volatility (%)'] if self.risk_metrics else 'N/A'}%",
                'RSI': f"{rsi:.1f} ({'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'})"
            },

            # Timeline
            'TIMELINE': {
                'Short_Term (1-2 weeks)': self.predictions['Predicted_Price'].iloc[min(10, len(self.predictions)-1)],
                'Medium_Term (1 month)': self.predictions['Predicted_Price'].iloc[min(20, len(self.predictions)-1)],
                'Long_Term (2 months)': self.predictions['Predicted_Price'].iloc[-1]
            }
        }

        # ─────────────────────────────────────────────────────────────
        # PRINT TRADE PLAN
        # ─────────────────────────────────────────────────────────────
        print("\n" + "=" * 70)
        print(f"💼 COMPLETE TRADE PLAN FOR {self.symbol}")
        print("=" * 70)

        print("\n🟢 WHEN TO BUY (ENTRY STRATEGY):")
        print("-" * 70)
        print(f"   Current Signal: {self.trade_plan['BUY_RECOMMENDATION']['Current_Signal']}")
        print(f"   Signal Strength: {self.trade_plan['BUY_RECOMMENDATION']['Signal_Strength']}")
        print(f"   Position Size: {self.trade_plan['BUY_RECOMMENDATION']['Position_Size']}")
        print(f"   Entry Zone: ₹{self.trade_plan['BUY_RECOMMENDATION']['Entry_Zone_Low']} - ₹{self.trade_plan['BUY_RECOMMENDATION']['Entry_Zone_High']}")
        print("\n   Buy Conditions Met:")
        for condition in self.trade_plan['BUY_RECOMMENDATION']['Conditions']:
            print(f"      {condition}")

        if self.trade_plan['BUY_RECOMMENDATION']['Upcoming_Buy_Opportunities']:
            print("\n   📅 Upcoming Buy Opportunities:")
            for opp in self.trade_plan['BUY_RECOMMENDATION']['Upcoming_Buy_Opportunities']:
                print(f"      {opp['Type']} on {opp['Date']} @ ₹{opp['Expected_Price']} ({opp['Discount']} discount)")

        print("\n" + "=" * 70)
        print("🔴 WHEN TO EXIT (EXIT STRATEGY):")
        print("-" * 70)
        print("\n   🎯 Profit Targets:")
        for target in self.trade_plan['EXIT_RECOMMENDATION']['Targets']:
            print(f"      {target['Level']}:")
            print(f"         Price: ₹{target['Price']} ({target['Gain']} gain)")
            print(f"         Expected Date: {target['Expected_Date']}")
            print(f"         Action: {target['Action']}")

        print("\n   🛑 Stop Loss Levels:")
        for stop_name, stop_info in self.trade_plan['EXIT_RECOMMENDATION']['Stop_Loss_Levels'].items():
            print(f"      {stop_name}:")
            print(f"         Price: ₹{stop_info['Price']} ({stop_info['Loss']} loss)")
            print(f"         When: {stop_info['When']}")

        print("\n   📊 Trailing Stop Strategy:")
        print(f"      {self.trade_plan['EXIT_RECOMMENDATION']['Trailing_Stop']['Method']}")
        print(f"      Current ATR: ₹{self.trade_plan['EXIT_RECOMMENDATION']['Trailing_Stop']['Current_ATR']}")

        print("\n" + "=" * 70)
        print("📈 PRICE PREDICTION TIMELINE:")
        print("-" * 70)
        for period, price in self.trade_plan['TIMELINE'].items():
            change = ((price - current_price) / current_price) * 100
            emoji = "📈" if change > 0 else "📉"
            print(f"   {emoji} {period}: ₹{price:.2f} ({change:+.2f}%)")

        print("\n" + "=" * 70)
        print("⚖️  RISK ANALYSIS:")
        print("-" * 70)
        for metric, value in self.trade_plan['RISK_ANALYSIS'].items():
            print(f"   {metric}: {value}")

        print("\n" + "=" * 70)
        print("✅ TRADE PLAN CREATED SUCCESSFULLY!")
        print("=" * 70)

        return self.trade_plan

    def _find_predicted_date(self, target_price):
        """Find when price is predicted to reach target"""
        for date, row in self.predictions.iterrows():
            if row['Predicted_Price'] >= target_price:
                return date.date()
        return "Not reached within prediction period"

    def plot_dashboard_with_predictions(self):
        """Create dashboard with price predictions"""
        print("\n📊 STEP 8: CREATING VISUALIZATION DASHBOARD")
        print("-" * 70)

        if self.data is None:
            print("❌ Please fetch data first!")
            return

        import matplotlib
        matplotlib.use('Agg')

        fig, axes = plt.subplots(4, 2, figsize=(16, 20))
        fig.suptitle(f'Stock Analysis & Prediction Dashboard - {self.symbol}', fontsize=16, fontweight='bold')

        df = self.signals if self.signals is not None else self.data

        # 1. Price & Moving Averages
        ax1 = axes[0, 0]
        ax1.plot(df.index, df['Close'], label='Close Price', linewidth=2)
        if 'SMA_20' in df.columns:
            ax1.plot(df.index, df['SMA_20'], label='SMA 20', linestyle='--', alpha=0.7)
        if 'SMA_50' in df.columns:
            ax1.plot(df.index, df['SMA_50'], label='SMA 50', linestyle='--', alpha=0.7)
        ax1.set_title('Price & Moving Averages')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Volume
        ax2 = axes[0, 1]
        ax2.bar(df.index, df['Volume'], alpha=0.5, color='gray')
        ax2.set_title('Volume')
        ax2.grid(True, alpha=0.3)

        # 3. RSI
        ax3 = axes[1, 0]
        if 'RSI' in df.columns:
            ax3.plot(df.index, df['RSI'], label='RSI', color='purple')
            ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5)
            ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5)
        ax3.set_title('RSI')
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)

        # 4. MACD
        ax4 = axes[1, 1]
        if 'MACD' in df.columns:
            ax4.plot(df.index, df['MACD'], label='MACD', color='blue')
        if 'MACD_Signal' in df.columns:
            ax4.plot(df.index, df['MACD_Signal'], label='Signal', color='orange')
        ax4.set_title('MACD')
        ax4.grid(True, alpha=0.3)

        # 5. Price Prediction
        ax5 = axes[2, 0]
        ax5.plot(df.index, df['Close'], label='Historical Price', linewidth=2, color='blue')
        if self.predictions is not None:
            ax5.plot(self.predictions.index, self.predictions['Predicted_Price'], 
                    label='Predicted Price', linewidth=2, color='green', linestyle='--')
            ax5.fill_between(self.predictions.index, 
                           self.predictions['Lower_Bound'], 
                           self.predictions['Upper_Bound'], 
                           alpha=0.3, color='green', label='Confidence Interval')
        ax5.set_title('Price Prediction (Next 60 Days)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 6. Signal Strength
        ax6 = axes[2, 1]
        if 'Signal_Strength' in df.columns:
            ax6.plot(df.index, df['Signal_Strength'], label='Signal Strength', color='green', linewidth=2)
            ax6.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='Strong Buy')
            ax6.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='Strong Sell')
        ax6.set_title('Signal Strength')
        ax6.set_ylim(0, 100)
        ax6.grid(True, alpha=0.3)

        # 7. Buy/Sell Signals
        ax7 = axes[3, 0]
        ax7.plot(df.index, df['Close'], label='Close Price', linewidth=2)
        if 'Signal' in df.columns:
            buy_signals = df[df['Signal'] == 1]
            sell_signals = df[df['Signal'] == -1]
            ax7.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', s=100, label='Buy')
            ax7.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', s=100, label='Sell')
        ax7.set_title('Buy/Sell Signals')
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        # 8. Feature Importance
        ax8 = axes[3, 1]
        if self.feature_importance is not None and len(self.feature_importance) > 0:
            top_features = self.feature_importance.head(8)
            ax8.barh(top_features['Feature'], top_features['Importance'], color='steelblue')
            ax8.set_title('Top 8 Feature Importance')
        else:
            ax8.text(0.5, 0.5, 'ML Model Required', ha='center', va='center')
            ax8.set_title('Feature Importance')
        ax8.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f'{self.symbol.replace(".NS", "").replace(".BO", "")}_prediction_dashboard.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Dashboard saved as '{filename}'")
        plt.close()

        return filename

    def run_complete_analysis(self, forecast_days=60):
        """Run complete analysis with trade plan"""
        print("\n" + "=" * 70)
        print(f"🚀 STARTING COMPLETE ANALYSIS FOR {self.symbol}")
        print("=" * 70)

        self.fetch_data()
        self.calculate_technical_indicators()
        self.generate_signals()
        self.calculate_risk_metrics()
        self.build_prediction_model(forecast_days=forecast_days)
        self.predict_future_price(days_ahead=forecast_days)
        self.create_trade_plan()
        self.plot_dashboard_with_predictions()

        print("\n" + "=" * 70)
        print("✅ ANALYSIS COMPLETE!")
        print("=" * 70)

        # Save data
        if self.signals is not None:
            self.signals.to_csv(f'{self.symbol.replace(".NS", "")}_analysis_data.csv')
        if self.predictions is not None:
            self.predictions.to_csv(f'{self.symbol.replace(".NS", "")}_predictions.csv')
        if self.trade_plan is not None:
            import json
            with open(f'{self.symbol.replace(".NS", "")}_trade_plan.json', 'w') as f:
                # Convert dates to strings for JSON
                trade_plan_serializable = self._make_serializable(self.trade_plan)
                json.dump(trade_plan_serializable, f, indent=2)
            print(f"💾 Trade plan saved to '{self.symbol.replace('.NS', '')}_trade_plan.json'")

        return self.trade_plan

    def _make_serializable(self, obj):
        """Convert numpy types to Python types for JSON"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(i) for i in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║   🇮🇳  INDIAN STOCK MARKET ANALYSIS & PREDICTION SYSTEM          ║
    ║              WITH BUY/SELL TIMING & EXIT STRATEGY                 ║
    ║                    POWERED BY PYTHON                              ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)

    # Change symbol as needed
    symbol = 'FEDERALBANK.NS'

    print(f"📊 Analyzing: {symbol}")

    # Create analyzer and run complete analysis
    analyzer = IndianStockAnalyzer(symbol=symbol, period='2y')
    trade_plan = analyzer.run_complete_analysis(forecast_days=60)

    print("\n" + "=" * 70)
    print("⚠️  IMPORTANT DISCLAIMER:")
    print("=" * 70)
    print("   📌 This tool is for EDUCATIONAL PURPOSES ONLY")
    print("   📌 Do NOT use for actual trading without proper research")
    print("   📌 Past performance does NOT guarantee future results")
    print("   📌 Stock market involves RISK of capital loss")
    print("   📌 Consult a SEBI-registered financial advisor")
    print("   📌 Always use stop loss and proper position sizing")
    print("=" * 70)