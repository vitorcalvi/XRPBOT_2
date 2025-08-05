import os
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class TelegramNotifier:
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.symbol = os.getenv('TRADING_SYMBOL', 'ADAUSDT')
        self.enabled = bool(self.bot_token and self.chat_id)

    async def send_message(self, message: str) -> bool:
        if not self.enabled:
            return False
        try:
            response = requests.post(
                f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
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

    async def send_trade_entry(self, signal_data, price, quantity, strategy_info):
        emoji = "🟢 LONG" if signal_data['action'].upper() == 'BUY' else "🔴 SHORT"
        message = f"""
📥 <b>TRADE ENTRY</b> {emoji}

<b>🔹 Symbol:</b> {self.symbol}
<b>💰 Price:</b> ${price:.2f} │ <b>📦 Qty:</b> {quantity}
<b>🛑 Stop Loss:</b> ${signal_data['structure_stop']:.2f}

<b>🧠 Strategy:</b> {signal_data['signal_type']}
<b>📈 RSI:</b> {signal_data['rsi']:.1f} │ <b>💧 MFI:</b> {signal_data['mfi']:.1f}
<b>🎯 Confidence:</b> {signal_data.get('confidence', 0):.0f}%

🕒 <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}
"""
        await self.send_message(message)

    async def send_trade_exit(self, exit_data, price, pnl, duration, strategy_info):
        emoji = "🟢 WIN" if pnl >= 0 else "🔴 LOSS"
        pnl_text = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        trigger = exit_data.get("trigger", "").replace('_', ' ').title()

        message = f"""
📤 <b>TRADE EXIT</b> {emoji}

<b>🔹 Symbol:</b> {self.symbol}
<b>💸 Exit Price:</b> ${price:.2f}
<b>📊 PnL:</b> {pnl_text}
<b>⏱ Duration:</b> {duration:.1f}s
<b>🎯 Trigger:</b> {trigger}

<b>📈 RSI:</b> {exit_data.get('rsi', 0):.1f} │ <b>💧 MFI:</b> {exit_data.get('mfi', 0):.1f}
🕒 <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}
"""
        await self.send_message(message)

    async def send_bot_status(self, status: str, message_text: str = ""):
        headlines = {
            'started': '🚀 BOT STARTED',
            'stopped': '🛑 BOT STOPPED',
            'error': '❌ ERROR',
            'warning': '⚠️ WARNING'
        }
        headline = headlines.get(status.lower(), '📊 BOT STATUS')
        extra = f"\n🗒 <b>Message:</b> {message_text}" if message_text else ""

        message = f"""
<b>{headline}</b>

<b>🔹 Symbol:</b> {self.symbol}
<b>🧠 Strategy:</b> RSI + MFI (Fixed $10K){extra}

🕒 <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}
"""
        await self.send_message(message)