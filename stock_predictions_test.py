"""
Indian Stock Market Analysis & Prediction System
Features:
- Data fetching from NSE/Yahoo Finance
- Technical Analysis (RSI, MACD, Moving Averages, Bollinger Bands)
- Buy/Sell Signal Generation
- ML-based Price Prediction
- Risk Management & Position Sizing
- Visualization Dashboard
"""

# Install required packages first:
# pip install yfinance pandas numpy matplotlib seaborn scikit-learn ta datetime

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class IndianStockAnalyzer:
    """Comprehensive Stock Analysis for Indian Market"""
    
    def __init__(self, symbol, period='2y'):
        """
        Initialize the analyzer
        symbol: Stock symbol with .NS for NSE (e.g., 'RELIANCE.NS', 'TCS.NS')
        period: Data period ('1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
        """
        self.symbol = symbol
        self.period = period
        self.data = None
        self.signals = None
        self.model = None
        self.scaler = MinMaxScaler()
        
    def fetch_data(self):
        """Fetch historical stock data from Yahoo Finance"""
        print(f"📥 Fetching data for {self.symbol}...")
        try:
            stock = yf.Ticker(self.symbol)
            self.data = stock.history(period=self.period)
            
            if self.data.empty:
                raise ValueError("No data found. Check symbol format (use .NS for NSE)")
            
            print(f"✅ Data fetched successfully! {len(self.data)} records")
            print(f"📅 Date range: {self.data.index[0].date()} to {self.data.index[-1].date()}")
            return self.data
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None
    
    def calculate_technical_indicators(self):
        """Calculate key technical indicators"""
        if self.data is None:
            print("❌ Please fetch data first!")
            return
        
        df = self.data.copy()
        
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
        
        # RSI (Relative Strength Index)
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
        
        # Price Momentum
        df['Momentum'] = df['Close'].pct_change(periods=10)
        df['ROC'] = df['Close'].pct_change(periods=12) * 100
        
        # ATR (Average True Range) for volatility
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(14).mean()
        
        self.data = df
        print("✅ Technical indicators calculated!")
        return df
    
    def generate_signals(self):
        """Generate Buy/Sell/Hold signals based on multiple indicators"""
        if self.data is None:
            print("❌ Please fetch data first!")
            return
        
        df = self.data.copy()
        
        # Initialize signals
        df['Signal'] = 0  # 0=Hold, 1=Buy, -1=Sell
        df['Signal_Strength'] = 0  # 0-100 score
        
        # Condition 1: Moving Average Crossover
        ma_buy = (df['SMA_20'] > df['SMA_50']) & (df['SMA_20'].shift(1) <= df['SMA_50'].shift(1))
        ma_sell = (df['SMA_20'] < df['SMA_50']) & (df['SMA_20'].shift(1) >= df['SMA_50'].shift(1))
        
        # Condition 2: RSI Signals
        rsi_oversold = df['RSI'] < 30
        rsi_overbought = df['RSI'] > 70
        
        # Condition 3: MACD Crossover
        macd_buy = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
        macd_sell = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))
        
        # Condition 4: Bollinger Bands
        bb_buy = df['Close'] < df['BB_Lower']
        bb_sell = df['Close'] > df['BB_Upper']
        
        # Condition 5: Price above 200 SMA (long-term trend)
        trend_bullish = df['Close'] > df['SMA_200']
        trend_bearish = df['Close'] < df['SMA_200']
        
        # Calculate Signal Strength (0-100)
        strength = np.zeros(len(df))
        
        # Bullish factors
        strength += (df['SMA_20'] > df['SMA_50']).astype(int) * 15
        strength += (df['SMA_50'] > df['SMA_200']).astype(int) * 15
        strength += (df['MACD'] > df['MACD_Signal']).astype(int) * 15
        strength += (df['RSI'] > 50).astype(int) * 10
        strength += (df['Close'] > df['SMA_200']).astype(int) * 15
        strength += (df['Close'] > df['BB_Middle']).astype(int) * 10
        strength += (df['Volume_Ratio'] > 1.5).astype(int) * 10
        strength += (df['Momentum'] > 0).astype(int) * 10
        
        # Bearish factors (subtract)
        strength -= (df['SMA_20'] < df['SMA_50']).astype(int) * 15
        strength -= (df['SMA_50'] < df['SMA_200']).astype(int) * 15
        strength -= (df['MACD'] < df['MACD_Signal']).astype(int) * 15
        strength -= (df['RSI'] < 50).astype(int) * 10
        strength -= (df['Close'] < df['SMA_200']).astype(int) * 15
        strength -= (df['Close'] < df['BB_Middle']).astype(int) * 10
        strength -= (df['Volume_Ratio'] < 0.5).astype(int) * 10
        strength -= (df['Momentum'] < 0).astype(int) * 10
        
        # Normalize to 0-100
        df['Signal_Strength'] = np.clip(strength + 50, 0, 100)
        
        # Generate final signals
        df.loc[ma_buy | macd_buy | rsi_oversold | bb_buy, 'Signal'] = 1
        df.loc[ma_sell | macd_sell | rsi_overbought | bb_sell, 'Signal'] = -1
        
        # Strong signals only when multiple indicators agree
        df.loc[df['Signal_Strength'] > 70, 'Signal'] = 1
        df.loc[df['Signal_Strength'] < 30, 'Signal'] = -1
        
        self.signals = df
        print("✅ Trading signals generated!")
        return df
    
    def build_prediction_model(self, forecast_days=30):
        """Build ML model for price prediction"""
        if self.data is None:
            print("❌ Please fetch data first!")
            return None
        
        df = self.data.copy()
        
        # Create features
        df['Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        df['High_Low_Range'] = (df['High'] - df['Low']) / df['Close']
        df['Close_Open_Range'] = (df['Close'] - df['Open']) / df['Open']
        
        # Target: Future price
        df['Future_Close'] = df['Close'].shift(-forecast_days)
        
        # Drop NaN values
        df = df.dropna()
        
        if len(df) < 100:
            print("❌ Insufficient data for ML model!")
            return None
        
        # Features for prediction
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 
                   'RSI', 'MACD', 'Volatility', 'Momentum', 'ATR']
        
        X = df[features].values
        y = df['Future_Close'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Build Random Forest model
        self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"✅ ML Model trained successfully!")
        print(f"📊 Model Performance: MSE={mse:.2f}, R²={r2:.2f}")
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        return self.model
    
    def predict_future_price(self, days_ahead=30):
        """Predict future stock price"""
        if self.model is None:
            print("❌ Please build prediction model first!")
            return None
        
        df = self.data.copy()
        
        # Get latest data point
        latest = df.iloc[-1:].copy()
        
        # Create future predictions
        predictions = []
        current_data = latest.copy()
        
        for i in range(days_ahead):
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_20', 'SMA_50', 
                       'RSI', 'MACD', 'Volatility', 'Momentum', 'ATR']
            
            # Prepare input
            X_input = current_data[features].values
            
            # Predict
            pred_price = self.model.predict(X_input)[0]
            predictions.append(pred_price)
            
            # Update current data for next iteration (simplified)
            current_data['Close'] = pred_price
            current_data['Open'] = pred_price * 0.995
            current_data['High'] = pred_price * 1.02
            current_data['Low'] = pred_price * 0.98
        
        # Create prediction dataframe
        last_date = df.index[-1]
        pred_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days_ahead, freq='B')
        
        pred_df = pd.DataFrame({
            'Date': pred_dates,
            'Predicted_Price': predictions
        })
        pred_df.set_index('Date', inplace=True)
        
        print(f"✅ Price prediction for next {days_ahead} days generated!")
        return pred_df
    
    def calculate_risk_metrics(self):
        """Calculate risk management metrics"""
        if self.data is None:
            print("❌ Please fetch data first!")
            return
        
        df = self.data.copy()
        
        # Daily returns
        df['Daily_Return'] = df['Close'].pct_change()
        
        # Risk metrics
        volatility = df['Daily_Return'].std() * np.sqrt(252) * 100
        sharpe_ratio = (df['Daily_Return'].mean() * 252) / (df['Daily_Return'].std() * np.sqrt(252))
        max_drawdown = ((df['Close'] - df['Close'].cummax()) / df['Close'].cummax()).min() * 100
        
        # VaR (Value at Risk) - 95% confidence
        var_95 = np.percentile(df['Daily_Return'].dropna(), 5) * 100
        
        # Beta (vs NIFTY - simplified)
        # In production, you'd compare with NIFTY returns
        
        risk_metrics = {
            'Annual_Volatility (%)': round(volatility, 2),
            'Sharpe_Ratio': round(sharpe_ratio, 2),
            'Max_Drawdown (%)': round(max_drawdown, 2),
            'VaR_95 (%)': round(var_95, 2),
            'Avg_Daily_Return (%)': round(df['Daily_Return'].mean() * 100, 3),
            'Best_Day (%)': round(df['Daily_Return'].max() * 100, 2),
            'Worst_Day (%)': round(df['Daily_Return'].min() * 100, 2)
        }
        
        print("📊 Risk Metrics:")
        for metric, value in risk_metrics.items():
            print(f"  {metric}: {value}")
        
        return risk_metrics
    
    def get_trade_recommendation(self):
        """Get current trade recommendation"""
        if self.signals is None:
            print("❌ Please generate signals first!")
            return
        
        latest = self.signals.iloc[-1]
        
        signal = latest['Signal']
        strength = latest['Signal_Strength']
        current_price = latest['Close']
        rsi = latest['RSI']
        macd = latest['MACD']
        atr = latest['ATR']
        
        # Determine action
        if signal == 1 and strength > 60:
            action = "🟢 STRONG BUY"
            confidence = "High"
        elif signal == 1:
            action = "🟡 BUY"
            confidence = "Medium"
        elif signal == -1 and strength < 40:
            action = "🔴 STRONG SELL"
            confidence = "High"
        elif signal == -1:
            action = "🟠 SELL"
            confidence = "Medium"
        else:
            action = "⚪ HOLD"
            confidence = "Neutral"
        
        # Calculate position size (Kelly Criterion simplified)
        win_rate = (self.signals['Signal'] == 1).sum() / len(self.signals)
        avg_gain = self.signals[self.signals['Signal'] == 1]['Daily_Return'].mean() if (self.signals['Signal'] == 1).any() else 0
        avg_loss = abs(self.signals[self.signals['Signal'] == -1]['Daily_Return'].mean()) if (self.signals['Signal'] == -1).any() else 0.01
        
        if avg_loss > 0:
            kelly_fraction = (win_rate * avg_gain - (1 - win_rate) * avg_loss) / avg_gain
            position_size = max(0, min(0.25, kelly_fraction)) * 100  # Cap at 25%
        else:
            position_size = 10
        
        # Stop loss and target
        stop_loss = current_price - (atr * 2)
        target_1 = current_price + (atr * 1.5)
        target_2 = current_price + (atr * 3)
        
        recommendation = {
            'Stock': self.symbol,
            'Current_Price': round(current_price, 2),
            'Action': action,
            'Confidence': confidence,
            'Signal_Strength': round(strength, 2),
            'RSI': round(rsi, 2),
            'MACD': round(macd, 4),
            'Position_Size (%)': round(position_size, 2),
            'Stop_Loss': round(stop_loss, 2),
            'Target_1': round(target_1, 2),
            'Target_2': round(target_2, 2),
            'Risk_Reward_Ratio': round((target_1 - current_price) / (current_price - stop_loss), 2)
        }
        
        print("\n" + "="*60)
        print(f"📈 TRADE RECOMMENDATION for {self.symbol}")
        print("="*60)
        for key, value in recommendation.items():
            print(f"{key:25}: {value}")
        print("="*60)
        
        return recommendation
    
    def plot_dashboard(self):
        """Create comprehensive visualization dashboard"""
        if self.data is None:
            print("❌ Please fetch data first!")
            return
        
        fig, axes = plt.subplots(4, 2, figsize=(16, 20))
        fig.suptitle(f'📊 Stock Analysis Dashboard - {self.symbol}', fontsize=16, fontweight='bold')
        
        df = self.data
        
        # 1. Price & Moving Averages
        ax1 = axes[0, 0]
        ax1.plot(df.index, df['Close'], label='Close Price', linewidth=2)
        ax1.plot(df.index, df['SMA_20'], label='SMA 20', linestyle='--', alpha=0.7)
        ax1.plot(df.index, df['SMA_50'], label='SMA 50', linestyle='--', alpha=0.7)
        ax1.plot(df.index, df['SMA_200'], label='SMA 200', linestyle='--', alpha=0.7)
        ax1.set_title('Price & Moving Averages')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Volume
        ax2 = axes[0, 1]
        ax2.bar(df.index, df['Volume'], alpha=0.5, color='gray')
        ax2.axhline(y=df['Volume_SMA'].iloc[-1], color='red', linestyle='--', label='Avg Volume')
        ax2.set_title('Volume')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. RSI
        ax3 = axes[1, 0]
        ax3.plot(df.index, df['RSI'], label='RSI', color='purple')
        ax3.axhline(y=70, color='red', linestyle='--', alpha=0.5, label='Overbought')
        ax3.axhline(y=30, color='green', linestyle='--', alpha=0.5, label='Oversold')
        ax3.fill_between(df.index, 30, 70, alpha=0.1)
        ax3.set_title('RSI (Relative Strength Index)')
        ax3.set_ylim(0, 100)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. MACD
        ax4 = axes[1, 1]
        ax4.plot(df.index, df['MACD'], label='MACD', color='blue')
        ax4.plot(df.index, df['MACD_Signal'], label='Signal', color='orange')
        ax4.bar(df.index, df['MACD_Hist'], label='Histogram', alpha=0.5, color='gray')
        ax4.set_title('MACD')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Bollinger Bands
        ax5 = axes[2, 0]
        ax5.plot(df.index, df['Close'], label='Close', linewidth=2)
        ax5.plot(df.index, df['BB_Upper'], label='Upper Band', linestyle='--', color='red')
        ax5.plot(df.index, df['BB_Middle'], label='Middle Band', linestyle='--', color='green')
        ax5.plot(df.index, df['BB_Lower'], label='Lower Band', linestyle='--', color='red')
        ax5.fill_between(df.index, df['BB_Upper'], df['BB_Lower'], alpha=0.1)
        ax5.set_title('Bollinger Bands')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Signal Strength
        ax6 = axes[2, 1]
        ax6.plot(df.index, df['Signal_Strength'], label='Signal Strength', color='green', linewidth=2)
        ax6.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='Strong Buy Zone')
        ax6.axhline(y=30, color='red', linestyle='--', alpha=0.5, label='Strong Sell Zone')
        ax6.fill_between(df.index, 30, 70, alpha=0.1)
        ax6.set_title('Signal Strength (0-100)')
        ax6.set_ylim(0, 100)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Buy/Sell Signals
        ax7 = axes[3, 0]
        buy_signals = df[df['Signal'] == 1]
        sell_signals = df[df['Signal'] == -1]
        ax7.plot(df.index, df['Close'], label='Close Price', linewidth=2)
        ax7.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', 
                   s=100, label='Buy Signal', zorder=5)
        ax7.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', 
                   s=100, label='Sell Signal', zorder=5)
        ax7.set_title('Buy/Sell Signals')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Feature Importance (if model exists)
        ax8 = axes[3, 1]
        if hasattr(self, 'feature_importance'):
            top_features = self.feature_importance.head(8)
            ax8.barh(top_features['Feature'], top_features['Importance'], color='steelblue')
            ax8.set_title('Top 8 Feature Importance (ML Model)')
            ax8.set_xlabel('Importance')
        else:
            ax8.text(0.5, 0.5, 'Build ML Model\nfor Feature Importance', 
                    ha='center', va='center', fontsize=12, style='italic')
            ax8.set_title('Feature Importance')
        ax8.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.symbol}_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        print(f"✅ Dashboard saved as '{self.symbol}_analysis_dashboard.png'")
        plt.show()
    
    def run_complete_analysis(self, forecast_days=30):
        """Run complete analysis pipeline"""
        print("\n" + "="*60)
        print(f"🚀 STARTING COMPLETE ANALYSIS FOR {self.symbol}")
        print("="*60 + "\n")
        
        # Step 1: Fetch Data
        self.fetch_data()
        
        # Step 2: Calculate Technical Indicators
        self.calculate_technical_indicators()
        
        # Step 3: Generate Signals
        self.generate_signals()
        
        # Step 4: Calculate Risk Metrics
        self.calculate_risk_metrics()
        
        # Step 5: Build ML Model
        self.build_prediction_model(forecast_days=forecast_days)
        
        # Step 6: Get Trade Recommendation
        recommendation = self.get_trade_recommendation()
        
        # Step 7: Predict Future Prices
        if self.model:
            predictions = self.predict_future_price(days_ahead=forecast_days)
            print(f"\n📈 Price Prediction (Next {forecast_days} days):")
            print(f"  Current Price: ₹{recommendation['Current_Price']}")
            print(f"  Predicted Price: ₹{round(predictions['Predicted_Price'].iloc[-1], 2)}")
            change = ((predictions['Predicted_Price'].iloc[-1] - recommendation['Current_Price']) 
                     / recommendation['Current_Price']) * 100
            print(f"  Expected Change: {change:.2f}%")
        
        # Step 8: Create Dashboard
        self.plot_dashboard()
        
        print("\n" + "="*60)
        print("✅ ANALYSIS COMPLETE!")
        print("="*60)
        
        return recommendation


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║   🇮🇳  INDIAN STOCK MARKET ANALYSIS & PREDICTION SYSTEM  ║
    ║                    Powered by Python                      ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Example: Analyze popular Indian stocks
    stocks_to_analyze = [
        'RELIANCE.NS',  # Reliance Industries
        'TCS.NS',       # TCS
        'INFY.NS',      # Infosys
        'HDFCBANK.NS',  # HDFC Bank
        'ICICIBANK.NS', # ICICI Bank
    ]
    
    # Analyze one stock (uncomment to analyze multiple)
    symbol = 'RELIANCE.NS'  # Change this to any NSE stock (add .NS suffix)
    
    # Create analyzer instance
    analyzer = IndianStockAnalyzer(symbol=symbol, period='2y')
    
    # Run complete analysis
    recommendation = analyzer.run_complete_analysis(forecast_days=30)
    
    # Save results to CSV
    if analyzer.signals is not None:
        analyzer.signals.to_csv(f'{symbol}_analysis_data.csv')
        print(f"\n💾 Full analysis data saved to '{symbol}_analysis_data.csv'")
    
    print("\n" + "="*60)
    print("⚠️  DISCLAIMER:")
    print("   This is for educational purposes only.")
    print("   Do NOT use for actual trading without proper research.")
    print("   Past performance does not guarantee future results.")
    print("   Consult a SEBI-registered financial advisor.")
    print("="*60)
