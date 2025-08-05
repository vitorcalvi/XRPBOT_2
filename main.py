import asyncio
import os
from dotenv import load_dotenv
from binance.client import Client
from binance.enums import *
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import requests

load_dotenv()

# === TELEGRAM NOTIFIER CLASS ===
class TelegramNotifier:
    """Streamlined Telegram notifications for scalping bot"""
    
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.symbol = "XRPUSDT"
        self.enabled = bool(self.bot_token and self.chat_id)
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
    
    async def send_message(self, message: str) -> bool:
        """Send message to Telegram"""
        if not self.enabled:
            return False
        
        try:
            response = requests.post(
                f"{self.base_url}/sendMessage",
                json={
                    'chat_id': self.chat_id,
                    'text': message,
                    'parse_mode': 'HTML',
                    'disable_web_page_preview': True
                },
                timeout=10
            )
            return response.status_code == 200
        except:
            return False
    
    async def send_scalp_entry(self, signal, price, position_size, confidence, rsi, mfi):
        """Send scalping entry notification"""
        emoji = "🟢 LONG" if signal == "LONG" else "🔴 SHORT"
        
        message = f"""⚡ <b>SCALP ENTRY</b> {emoji}

<b>Symbol:</b> {self.symbol}
<b>Price:</b> ${price:.4f}
<b>Size:</b> ${position_size:.0f}
<b>Confidence:</b> {confidence:.0%}

<b>Indicators:</b>
RSI: {rsi:.1f} | MFI: {mfi:.1f}
HHLL: {'Uptrend' if signal == 'LONG' else 'Downtrend'}

<b>Targets:</b>
Profit: {price * (1.008 if signal == 'LONG' else 0.992):.4f} (0.8%)
Stop: {price * (0.996 if signal == 'LONG' else 1.004):.4f} (0.4%)

🕒 {datetime.now().strftime('%H:%M:%S')}"""
        
        await self.send_message(message)
    
    async def send_scalp_exit(self, signal, exit_price, pnl_pct, reason, duration_seconds):
        """Send scalping exit notification"""
        emoji = "🟢 WIN" if pnl_pct >= 0 else "🔴 LOSS"
        pnl_text = f"+{pnl_pct:.2%}" if pnl_pct >= 0 else f"{pnl_pct:.2%}"
        duration_text = self._format_duration(duration_seconds)
        
        message = f"""⚡ <b>SCALP EXIT</b> {emoji}

<b>Symbol:</b> {self.symbol}
<b>Price:</b> ${exit_price:.4f}
<b>PnL:</b> {pnl_text}
<b>Duration:</b> {duration_text}
<b>Reason:</b> {reason}

🕒 {datetime.now().strftime('%H:%M:%S')}"""
        
        await self.send_message(message)
    
    def _format_duration(self, duration_seconds):
        """Format duration"""
        if duration_seconds < 60:
            return f"{duration_seconds:.0f}s"
        elif duration_seconds < 3600:
            return f"{duration_seconds / 60:.1f}m"
        else:
            return f"{duration_seconds / 3600:.1f}h"
    
    async def send_bot_status(self, status: str, message_text: str = ""):
        """Send bot status"""
        headlines = {
            'started': '🚀 SCALPING BOT STARTED',
            'stopped': '🛑 SCALPING BOT STOPPED',
            'error': '❌ SCALPING ERROR'
        }
        
        headline = headlines.get(status.lower(), '📊 SCALPING STATUS')
        info_line = f"\n<b>Info:</b> {message_text}" if message_text else ""
        
        message = f"""<b>{headline}</b>

<b>Symbol:</b> {self.symbol}
<b>Strategy:</b> 3m RSI+MFI+HHLL Scalping
<b>Timeframe:</b> 3-minute
<b>Targets:</b> 0.8% profit | 0.4% stop{info_line}

🕒 {datetime.now().strftime('%H:%M:%S')}"""
        
        await self.send_message(message)

# === SCALPING CONFIGURATION ===
SYMBOL = "XRPUSDT"
TIMEFRAME = "3m"  # 3-minute scalping
TAKER_FEE = 0.0004
MAKER_FEE = 0.0002
TOTAL_COST = 0.00046

# Scalping parameters
MIN_POSITION = 500   # Smaller positions for scalping
MAX_POSITION = 2000
SCALP_PROFIT_TARGET = 0.008  # 0.8% profit target
SCALP_STOP_LOSS = 0.004      # 0.4% stop loss
QUICK_EXIT_TIME = 300        # 5 minutes max hold

# Indicator settings
RSI_PERIOD = 14
RSI_OVERSOLD = 25
RSI_OVERBOUGHT = 75
MFI_PERIOD = 14
MFI_OVERSOLD = 20
MFI_OVERBOUGHT = 80
HHLL_LOOKBACK = 10

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScalpingBot:
    def __init__(self):
        self.client = Client(
            api_key=os.getenv('BINANCE_API_KEY'),
            api_secret=os.getenv('BINANCE_API_SECRET'),
            testnet=True
        )
        self.telegram = TelegramNotifier()
        self.in_position = False
        self.position_side = None
        self.entry_price = 0
        self.entry_time = None
        self.position_qty = 0
        
    def get_klines(self, limit=100):
        """Get 3-minute klines"""
        try:
            klines = self.client.futures_klines(
                symbol=SYMBOL,
                interval=TIMEFRAME,
                limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            return df.sort_values('timestamp')
        except Exception as e:
            logger.error(f"Error fetching klines: {e}")
            return None

    def calculate_rsi(self, prices, period=RSI_PERIOD):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_mfi(self, df, period=MFI_PERIOD):
        """Calculate Money Flow Index"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        # Positive and negative money flow
        positive_mf = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_mf = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        # Money flow ratio
        positive_mf_sum = positive_mf.rolling(window=period).sum()
        negative_mf_sum = negative_mf.rolling(window=period).sum()
        
        mfr = positive_mf_sum / negative_mf_sum
        mfi = 100 - (100 / (1 + mfr))
        
        return mfi

    def detect_hhll(self, df, lookback=HHLL_LOOKBACK):
        """Detect Higher Highs/Lower Lows pattern"""
        highs = df['high'].rolling(window=lookback).max()
        lows = df['low'].rolling(window=lookback).min()
        
        current_high = df['high'].iloc[-1]
        current_low = df['low'].iloc[-1]
        prev_high = highs.iloc[-2] if len(highs) > 1 else current_high
        prev_low = lows.iloc[-2] if len(lows) > 1 else current_low
        
        # Pattern detection
        higher_high = current_high > prev_high
        higher_low = current_low > prev_low
        lower_high = current_high < prev_high
        lower_low = current_low < prev_low
        
        # Trend confirmation
        uptrend = higher_high and higher_low
        downtrend = lower_high and lower_low
        
        return {
            'uptrend': uptrend,
            'downtrend': downtrend,
            'higher_high': higher_high,
            'higher_low': higher_low,
            'lower_high': lower_high,
            'lower_low': lower_low
        }

    def generate_scalping_signal(self, df):
        """Generate scalping signals using RSI + MFI + HHLL"""
        if len(df) < max(RSI_PERIOD, MFI_PERIOD, HHLL_LOOKBACK) + 5:
            return None, 0
            
        # Calculate indicators
        rsi = self.calculate_rsi(df['close'])
        mfi = self.calculate_mfi(df)
        hhll = self.detect_hhll(df)
        
        current_rsi = rsi.iloc[-1]
        current_mfi = mfi.iloc[-1]
        
        if pd.isna(current_rsi) or pd.isna(current_mfi):
            return None, 0
        
        signal = None
        confidence = 0
        
        # LONG Signal: RSI oversold + MFI oversold + Uptrend confirmation
        if (current_rsi < RSI_OVERSOLD and 
            current_mfi < MFI_OVERSOLD and 
            hhll['uptrend']):
            
            signal = "LONG"
            # Confidence based on how oversold and trend strength
            rsi_strength = (RSI_OVERSOLD - current_rsi) / RSI_OVERSOLD
            mfi_strength = (MFI_OVERSOLD - current_mfi) / MFI_OVERSOLD
            confidence = min(0.95, (rsi_strength + mfi_strength) / 2 + 0.3)
            
        # SHORT Signal: RSI overbought + MFI overbought + Downtrend confirmation
        elif (current_rsi > RSI_OVERBOUGHT and 
              current_mfi > MFI_OVERBOUGHT and 
              hhll['downtrend']):
            
            signal = "SHORT"
            rsi_strength = (current_rsi - RSI_OVERBOUGHT) / (100 - RSI_OVERBOUGHT)
            mfi_strength = (current_mfi - MFI_OVERBOUGHT) / (100 - MFI_OVERBOUGHT)
            confidence = min(0.95, (rsi_strength + mfi_strength) / 2 + 0.3)
        
        return signal, confidence

    def check_scalping_exit(self, df, current_price):
        """Check scalping exit conditions including HHLL pattern reversal"""
        if not self.in_position:
            return False, "No position"
            
        # Time-based exit (5 minutes max)
        if self.entry_time:
            time_diff = (datetime.now() - self.entry_time).seconds
            if time_diff > QUICK_EXIT_TIME:
                return True, "Time exit"
        
        # Calculate PnL
        if self.position_side == "LONG":
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:  # SHORT
            pnl_pct = (self.entry_price - current_price) / self.entry_price
            
        # Profit target
        if pnl_pct >= SCALP_PROFIT_TARGET:
            return True, f"Profit target hit: {pnl_pct:.3f}"
            
        # Stop loss
        if pnl_pct <= -SCALP_STOP_LOSS:
            return True, f"Stop loss hit: {pnl_pct:.3f}"
            
        # HHLL Pattern Reversal Exit (KEY FEATURE)
        if len(df) >= HHLL_LOOKBACK + 5:
            hhll = self.detect_hhll(df)
            
            # Exit LONG position if uptrend breaks (Lower High or Lower Low detected)
            if self.position_side == "LONG":
                if hhll['downtrend'] or (not hhll['higher_high'] and not hhll['higher_low']):
                    return True, f"HHLL uptrend reversal: {pnl_pct:.3f}"
                    
            # Exit SHORT position if downtrend breaks (Higher High or Higher Low detected)  
            elif self.position_side == "SHORT":
                if hhll['uptrend'] or (not hhll['lower_high'] and not hhll['lower_low']):
                    return True, f"HHLL downtrend reversal: {pnl_pct:.3f}"
        
        return False, "Hold"

    def calculate_scalp_position_size(self, confidence):
        """Calculate position size for scalping"""
        try:
            account = self.client.futures_account()
            balance = float(account['totalWalletBalance'])
        except:
            balance = 10000
            
        # Smaller base size for scalping
        base_size = MIN_POSITION * confidence
        max_allowed = balance * 0.1  # 10% max for scalping
        
        return min(base_size, MAX_POSITION, max_allowed)

    def execute_scalp_order(self, signal, position_size, current_price):
        """Execute scalping order with tight spreads"""
        qty = round(position_size / current_price, 1)
        
        try:
            if signal == "LONG":
                # Try limit order very close to market
                limit_price = round(current_price * 0.9998, 4)  # 0.02% below market
                
                order = self.client.futures_create_order(
                    symbol=SYMBOL,
                    side=SIDE_BUY,
                    type=ORDER_TYPE_LIMIT,
                    timeInForce=TIME_IN_FORCE_IOC,
                    quantity=qty,
                    price=str(limit_price)
                )
                
                # Fallback to market if limit fails
                if not order:
                    order = self.client.futures_create_order(
                        symbol=SYMBOL,
                        side=SIDE_BUY,
                        type=ORDER_TYPE_MARKET,
                        quantity=qty
                    )
                    
            else:  # SHORT
                limit_price = round(current_price * 1.0002, 4)  # 0.02% above market
                
                order = self.client.futures_create_order(
                    symbol=SYMBOL,
                    side=SIDE_SELL,
                    type=ORDER_TYPE_LIMIT,
                    timeInForce=TIME_IN_FORCE_IOC,
                    quantity=qty,
                    price=str(limit_price)
                )
                
                if not order:
                    order = self.client.futures_create_order(
                        symbol=SYMBOL,
                        side=SIDE_SELL,
                        type=ORDER_TYPE_MARKET,
                        quantity=qty
                    )
            
            return order
            
        except Exception as e:
            logger.error(f"Order execution error: {e}")
            return None

    def close_scalp_position(self):
        """Close scalping position quickly"""
        try:
            positions = self.client.futures_position_information(symbol=SYMBOL)
            
            for pos in positions:
                pos_amt = float(pos['positionAmt'])
                if pos_amt != 0:
                    qty = abs(pos_amt)
                    side = SIDE_SELL if pos_amt > 0 else SIDE_BUY
                    
                    # Market order for quick exit
                    order = self.client.futures_create_order(
                        symbol=SYMBOL,
                        side=side,
                        type=ORDER_TYPE_MARKET,
                        quantity=qty
                    )
                    
                    if order:
                        self.in_position = False
                        self.position_side = None
                        self.entry_price = 0
                        self.entry_time = None
                        return True
                        
            return False
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False

    async def run_scalping_loop(self):
        """Main 3-minute scalping loop"""
        logger.info(f"🚀 Starting 3-minute RSI+MFI+HHLL scalping bot for {SYMBOL}")
        logger.info(f"📊 Timeframe: {TIMEFRAME}")
        logger.info(f"💰 Position range: ${MIN_POSITION}-${MAX_POSITION}")
        logger.info(f"🎯 Profit target: {SCALP_PROFIT_TARGET*100:.1f}% | Stop loss: {SCALP_STOP_LOSS*100:.1f}%")
        
        while True:
            try:
                # Get fresh market data
                df = self.get_klines(limit=50)
                if df is None:
                    await asyncio.sleep(10)
                    continue
                    
                current_price = df['close'].iloc[-1]
                
                # Check exit conditions first
                if self.in_position:
                    should_exit, reason = self.check_scalping_exit(df, current_price)
                    if should_exit:
                        if self.close_scalp_position():
                            # Calculate PnL and duration for notification
                            if self.position_side == "LONG":
                                pnl_pct = (current_price - self.entry_price) / self.entry_price
                            else:
                                pnl_pct = (self.entry_price - current_price) / self.entry_price
                            
                            duration = (datetime.now() - self.entry_time).total_seconds() if self.entry_time else 0
                            
                            logger.info(f"🔄 Position closed: {reason}")
                            
                            # Send Telegram exit notification
                            await self.telegram.send_scalp_exit(
                                self.position_side, current_price, pnl_pct, reason, duration
                            )
                        else:
                            logger.error("❌ Failed to close position")
                
                # Look for new scalping opportunities
                if not self.in_position:
                    signal, confidence = self.generate_scalping_signal(df)
                    
                    if signal and confidence > 0.6:  # Lower threshold for scalping
                        position_size = self.calculate_scalp_position_size(confidence)
                        
                        # Execute scalping trade
                        order = self.execute_scalp_order(signal, position_size, current_price)
                        
                        if order:
                            self.in_position = True
                            self.position_side = signal
                            self.entry_price = current_price
                            self.entry_time = datetime.now()
                            self.position_qty = position_size / current_price
                            
                            # Calculate targets
                            if signal == "LONG":
                                target_price = current_price * (1 + SCALP_PROFIT_TARGET)
                                stop_price = current_price * (1 - SCALP_STOP_LOSS)
                            else:
                                target_price = current_price * (1 - SCALP_PROFIT_TARGET)
                                stop_price = current_price * (1 + SCALP_STOP_LOSS)
                            
                            # Get current indicators for notification
                            rsi = self.calculate_rsi(df['close']).iloc[-1]
                            mfi = self.calculate_mfi(df).iloc[-1]
                            
                            logger.info(f"📈 {signal} scalp executed:")
                            logger.info(f"   Entry: ${current_price:.4f}")
                            logger.info(f"   Size: ${position_size:.0f}")
                            logger.info(f"   Target: ${target_price:.4f}")
                            logger.info(f"   Stop: ${stop_price:.4f}")
                            logger.info(f"   Confidence: {confidence:.2f}")
                            
                            # Send Telegram notification
                            await self.telegram.send_scalp_entry(
                                signal, current_price, position_size, confidence, rsi, mfi
                            )
                
                # Short cycle for scalping
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Scalping loop error: {e}")
                await asyncio.sleep(15)

# === STARTUP ===
if __name__ == "__main__":
    bot = ScalpingBot()
    asyncio.run(bot.run_scalping_loop())