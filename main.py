import asyncio
import os
from pybit.unified_trading import HTTP
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# === CORE CONFIGURATION ===
SYMBOL = "XRPUSDT"
TAKER_FEE = 0.00055
MAKER_FEE = 0.0001
MAKER_FILL_RATIO = 0.30
BLENDED_FEE = 0.000415
SLIPPAGE = 0.0002
TOTAL_COST = 0.000615

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
        self.session = HTTP(
            testnet=True,  # Set to False for live trading
            api_key=os.getenv('BYBIT_API_KEY'),
            api_secret=os.getenv('BYBIT_API_SECRET')
        )
        self.position_size = 0
        self.entry_price = 0
        self.in_position = False
        
    def get_klines(self, interval="15", limit=100):
        """Fetch kline data for XRPUSDT"""
        try:
            response = self.session.get_kline(
                category="linear",
                symbol=SYMBOL,
                interval=interval,
                limit=limit
            )
            if response['retCode'] == 0:
                data = response['result']['list']
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                df = df.astype({'open': float, 'high': float, 'low': float, 'close': float, 'volume': float})
                return df.sort_values('timestamp')
            return None
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
        confidence_multiplier = 0.8 + (confidence_score * 0.7)  # Maps 0-1 to 0.8-1.5
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
                confidence = min(0.9, adx / 50)  # Scale ADX to confidence
            elif ma_fast < ma_slow and current_price < ma_fast:
                signal = "SHORT" 
                confidence = min(0.9, adx / 50)
                
        return signal, confidence

    def place_limit_order(self, side, qty, price):
        """Place limit order with IOC (Immediate-or-Cancel)"""
        try:
            response = self.session.place_order(
                category="linear",
                symbol=SYMBOL,
                side=side,
                orderType="Limit",
                qty=str(qty),
                price=str(price),
                timeInForce="IOC"  # Immediate-or-Cancel for maker attempts
            )
            return response
        except Exception as e:
            logger.error(f"Limit order error: {e}")
            return None

    def place_market_order(self, side, qty):
        """Fallback market order"""
        try:
            response = self.session.place_order(
                category="linear",
                symbol=SYMBOL,
                side=side,
                orderType="Market",
                qty=str(qty)
            )
            return response
        except Exception as e:
            logger.error(f"Market order error: {e}")
            return None

    def execute_trade(self, signal, position_size, current_price, confidence):
        """Limit-first execution strategy"""
        if signal == "LONG":
            side = "Buy"
            # Try limit order first
            limit_offset = 0.0003 if confidence > 0.85 else 0.0006
            limit_price = current_price * (1 - limit_offset)
            
            result = self.place_limit_order(side, position_size, limit_price)
            
            # If limit fails, use market order
            if not result or result.get('retCode') != 0:
                result = self.place_market_order(side, position_size)
                
        elif signal == "SHORT":
            side = "Sell"
            limit_offset = 0.0003 if confidence > 0.85 else 0.0006
            limit_price = current_price * (1 + limit_offset)
            
            result = self.place_limit_order(side, position_size, limit_price)
            
            if not result or result.get('retCode') != 0:
                result = self.place_market_order(side, position_size)
        
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
        """Get account balance"""
        try:
            response = self.session.get_wallet_balance(accountType="UNIFIED")
            if response['retCode'] == 0:
                balance = float(response['result']['list'][0]['totalWalletBalance'])
                return balance
            return 10000  # Default for testing
        except:
            return 10000

    async def run_trading_loop(self):
        """Main trading loop"""
        logger.info(f"Starting streamlined trading bot for {SYMBOL}")
        
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
                    # Close position logic here
                    self.in_position = False
                    self.position_size = 0
                    logger.info("Position closed")
                
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
                            
                            if result and result.get('retCode') == 0:
                                self.in_position = True
                                self.entry_price = current_price
                                self.position_size = position_size if signal == "LONG" else -position_size
                                
                                logger.info(f"Trade executed: {signal} {position_size} @ {current_price:.4f}")
                                logger.info(f"Break-even PnL: ${break_even_pnl:.2f}, Min profit: ${min_profit:.2f}")
                
                await asyncio.sleep(15)  # 15-second cycle
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(30)

# === STARTUP ===
if __name__ == "__main__":
    bot = StreamlinedTradingBot()
    asyncio.run(bot.run_trading_loop())