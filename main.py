import asyncio
import os
from dotenv import load_dotenv
from binance.client import Client
from binance.enums import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Load environment variables
load_dotenv()

# === BINANCE CONFIGURATION ===
SYMBOL = "XRPUSDT"
TAKER_FEE = 0.0004      # 0.04% futures taker
MAKER_FEE = 0.0002      # 0.02% futures maker
MAKER_FILL_RATIO = 0.30
BLENDED_FEE = 0.00026   # Blended fee rate
SLIPPAGE = 0.0002
TOTAL_COST = 0.00046    # Total trading cost

# Position sizing
MIN_POSITION = 1500
MAX_POSITION = 5000
MAX_ACCOUNT_RISK = 0.15
EMERGENCY_STOP = 0.015

# ADX thresholds
ADX_TREND_MIN = 20
ADX_STRONG = 25
ADX_EXTREME = 40

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamlinedTradingBot:
    def __init__(self):
        self.client = Client(
            api_key=os.getenv('BINANCE_API_KEY'),
            api_secret=os.getenv('BINANCE_API_SECRET'),
            testnet=True,  # Using Binance testnet
            tld='com'  # Ensure correct TLD
        )
        # Set testnet base URL
        self.client.API_URL = 'https://testnet.binancefuture.com'
        
        self.position_size = 0
        self.entry_price = 0
        self.in_position = False
        
    def get_klines(self, interval="15m", limit=100):
        """Fetch kline data for XRPUSDT"""
        try:
            klines = self.client.futures_klines(
                symbol=SYMBOL,
                interval=interval,
                limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype({
                'open': float, 'high': float, 'low': float, 'close': float, 'volume': float
            })
            
            return df.sort_values('timestamp')
        except Exception as e:
            logger.error(f"Error fetching klines: {e}")
            return None

    def calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high = df['high']
        low = df['low'] 
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        return atr

    def calculate_adx(self, df, period=14):
        """Calculate ADX for trend strength"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate +DI and -DI
        up = high.diff()
        down = -low.diff()
        
        plus_dm = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([pd.Series(tr1), pd.Series(tr2), pd.Series(tr3)], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean().iloc[-1]
        
        return adx if not pd.isna(adx) else 0

    def calculate_position_size(self, account_balance, confidence_score):
        """Dynamic position sizing based on confidence"""
        base_size = MIN_POSITION
        
        # Adjust based on confidence (0.8x to 1.5x multiplier)
        confidence_multiplier = 0.8 + (confidence_score * 0.7)
        adjusted_size = base_size * confidence_multiplier
        
        # Cap at max position and account risk
        max_allowed = account_balance * MAX_ACCOUNT_RISK
        position_size = min(adjusted_size, MAX_POSITION, max_allowed)
        
        return position_size

    def calculate_break_even(self, position_size):
        """Calculate break-even PnL and minimum profit target"""
        break_even_pnl = position_size * TOTAL_COST
        min_profit_target = break_even_pnl + 15.0
        return break_even_pnl, min_profit_target

    def generate_signal(self, df):
        """Simple trend following signal based on moving averages and ADX"""
        if len(df) < 50:
            return None, 0
            
        # Calculate indicators
        close = df['close']
        ma_fast = close.rolling(window=10).mean().iloc[-1]
        ma_slow = close.rolling(window=20).mean().iloc[-1]
        current_price = close.iloc[-1]
        
        adx = self.calculate_adx(df)
        
        # Signal generation
        signal = None
        confidence = 0
        
        if adx > ADX_TREND_MIN:
            if ma_fast > ma_slow and current_price > ma_fast:
                signal = "LONG"
                confidence = min(0.9, adx / 50)
            elif ma_fast < ma_slow and current_price < ma_fast:
                signal = "SHORT" 
                confidence = min(0.9, adx / 50)
                
        return signal, confidence

    def place_limit_order(self, side, qty, price):
        """Place limit order (IOC equivalent)"""
        try:
            if side == "BUY":
                order = self.client.futures_create_order(
                    symbol=SYMBOL,
                    side=SIDE_BUY,
                    type=ORDER_TYPE_LIMIT,
                    timeInForce=TIME_IN_FORCE_IOC,
                    quantity=qty,
                    price=str(price)
                )
            else:
                order = self.client.futures_create_order(
                    symbol=SYMBOL,
                    side=SIDE_SELL,
                    type=ORDER_TYPE_LIMIT,
                    timeInForce=TIME_IN_FORCE_IOC,
                    quantity=qty,
                    price=str(price)
                )
            return order
        except Exception as e:
            logger.error(f"Limit order error: {e}")
            return None

    def place_market_order(self, side, qty):
        """Fallback market order"""
        try:
            if side == "BUY":
                order = self.client.futures_create_order(
                    symbol=SYMBOL,
                    side=SIDE_BUY,
                    type=ORDER_TYPE_MARKET,
                    quantity=qty
                )
            else:
                order = self.client.futures_create_order(
                    symbol=SYMBOL,
                    side=SIDE_SELL,
                    type=ORDER_TYPE_MARKET,
                    quantity=qty
                )
            return order
        except Exception as e:
            logger.error(f"Market order error: {e}")
            return None

    def execute_trade(self, signal, position_size, current_price, confidence):
        """Limit-first execution strategy"""
        # Convert position size to quantity (USDT to XRP)
        qty = round(position_size / current_price, 1)
        
        if signal == "LONG":
            side = "BUY"
            # Try limit order first
            limit_offset = 0.0003 if confidence > 0.85 else 0.0006
            limit_price = round(current_price * (1 - limit_offset), 4)
            
            result = self.place_limit_order(side, qty, limit_price)
            
            # If limit fails, use market order
            if not result:
                result = self.place_market_order(side, qty)
                
        elif signal == "SHORT":
            side = "SELL"
            limit_offset = 0.0003 if confidence > 0.85 else 0.0006
            limit_price = round(current_price * (1 + limit_offset), 4)
            
            result = self.place_limit_order(side, qty, limit_price)
            
            if not result:
                result = self.place_market_order(side, qty)
        
        return result

    def check_exit_conditions(self, current_price, atr):
        """Check if position should be closed"""
        if not self.in_position:
            return False
            
        # Calculate PnL
        if self.position_size > 0:  # Long position
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:  # Short position
            pnl_pct = (self.entry_price - current_price) / self.entry_price
            
        # Emergency stop
        if pnl_pct <= -EMERGENCY_STOP:
            logger.info(f"Emergency stop triggered: {pnl_pct:.4f}")
            return True
            
        # Trailing stop based on ATR
        if pnl_pct > 0.01:  # Only trail if in profit
            trailing_distance = atr * 2
            if self.position_size > 0:  # Long
                stop_price = current_price - trailing_distance
                if current_price <= stop_price:
                    return True
            else:  # Short
                stop_price = current_price + trailing_distance
                if current_price >= stop_price:
                    return True
                    
        return False

    def get_account_balance(self):
        """Get futures account balance"""
        try:
            account = self.client.futures_account()
            balance = float(account['totalWalletBalance'])
            return balance
        except:
            return 10000  # Default for testing

    def close_position(self):
        """Close current position"""
        try:
            # Get current position
            positions = self.client.futures_position_information(symbol=SYMBOL)
            
            for pos in positions:
                if float(pos['positionAmt']) != 0:
                    qty = abs(float(pos['positionAmt']))
                    side = SIDE_SELL if float(pos['positionAmt']) > 0 else SIDE_BUY
                    
                    order = self.client.futures_create_order(
                        symbol=SYMBOL,
                        side=side,
                        type=ORDER_TYPE_MARKET,
                        quantity=qty
                    )
                    
                    if order:
                        self.in_position = False
                        self.position_size = 0
                        logger.info("Position closed successfully")
                        return True
            return False
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False

    async def run_trading_loop(self):
        """Main trading loop"""
        logger.info(f"Starting streamlined Binance trading bot for {SYMBOL}")
        
        while True:
            try:
                # Get market data
                df = self.get_klines()
                if df is None:
                    await asyncio.sleep(30)
                    continue
                    
                current_price = df['close'].iloc[-1]
                atr = self.calculate_atr(df)
                
                # Check exit conditions first
                if self.check_exit_conditions(current_price, atr):
                    self.close_position()
                
                # Generate new signals if not in position
                if not self.in_position:
                    signal, confidence = self.generate_signal(df)
                    
                    if signal and confidence > 0.7:  # Minimum confidence threshold
                        account_balance = self.get_account_balance()
                        position_size = self.calculate_position_size(account_balance, confidence)
                        
                        # Validate fee efficiency
                        break_even_pnl, min_profit = self.calculate_break_even(position_size)
                        fee_pct = (break_even_pnl / position_size) * 100
                        
                        if fee_pct <= 3.0:  # Fee efficiency check
                            result = self.execute_trade(signal, position_size, current_price, confidence)
                            
                            if result:
                                self.in_position = True
                                self.entry_price = current_price
                                self.position_size = position_size if signal == "LONG" else -position_size
                                
                                logger.info(f"Trade executed: {signal} ${position_size} @ {current_price:.4f}")
                                logger.info(f"Break-even PnL: ${break_even_pnl:.2f}, Min profit: ${min_profit:.2f}")
                
                await asyncio.sleep(15)  # 15-second cycle
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(30)

# === STARTUP ===
if __name__ == "__main__":
    bot = StreamlinedTradingBot()
    asyncio.run(bot.run_trading_loop())