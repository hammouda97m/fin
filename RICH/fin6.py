import json
import os
import time
import math
import csv
from datetime import datetime
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
from eth_utils import event_signature_to_log_topic
from dotenv import load_dotenv, find_dotenv
import requests


#countdown so you don't wait 30 minutes
countdown = 1500

print(f"Starting in {countdown} seconds...")
time.sleep(countdown)


# === Config ===

load_dotenv(find_dotenv())

BOT_NUMBER = 6

PRIVATE_KEY = os.getenv(f"PRIVATE_KEY_{BOT_NUMBER}")
WALLET_ADDRESS = os.getenv(f"WALLET_ADDRESS_{BOT_NUMBER}")

with open("prediction_abi.json", "r") as f:
    ABI = json.load(f)
web3 = Web3(
    Web3.HTTPProvider("https://falling-yolo-pond.bsc.quiknode.pro/6eb51af3491eaccbc70865ba6f966621820df0aa/"))

web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

if not web3.is_connected():
    raise Exception("❌ Failed to connect to BSC")
CONTRACT_ADDRESS = Web3.to_checksum_address("0x18B2A687610328590Bc8F2e5fEdDe3b582A49cdA")

contract = web3.eth.contract(address=CONTRACT_ADDRESS, abi=ABI)

# === NEW: Chainlink Price Feed ===
CHAINLINK_BNB_USD = Web3.to_checksum_address("0x0567F2323251f0Aab15c8dFb1967E4e8A7D42aeE")
CHAINLINK_ABI = [
    {
        "inputs": [],
        "name": "latestRoundData",
        "outputs": [
            {"name": "roundId", "type": "uint80"},
            {"name": "answer", "type": "int256"},
            {"name": "startedAt", "type": "uint256"},
            {"name": "updatedAt", "type": "uint256"},
            {"name": "answeredInRound", "type": "uint80"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

price_feed = web3.eth.contract(address=CHAINLINK_BNB_USD, abi=CHAINLINK_ABI)

# === NEW: Price Caching Variables ===
last_price_fetch = 0
cached_price = 0

# === NEW: BTC TRACKING VARIABLES ===
last_btc_fetch = 0
cached_btc_data = {"price": 0, "change_5min": 0}

# Topics for events
bet_bull_topic = event_signature_to_log_topic("BetBull(address,uint256,uint256)")
bet_bear_topic = event_signature_to_log_topic("BetBear(address,uint256,uint256)")

# Global flags
bet_placed_last_round = False
last_bet_epochs = []

# NEW: Loss prevention variables
skip_rounds_remaining = 0
last_loss_epoch = None

# NEW: Streak tracking variables
current_streak = 0
current_streak_type = None  # 'BULL' or 'BEAR'
rounds_history = []  # Store last 24 rounds info

# Cache file for storing rounds data
CACHE_FILE = "rounds_cache.json"
# NEW: Shared loss prevention file
SHARED_LOSS_FILE = "shared_loss_state.json"

# === UPDATED ML INSIGHTS INTEGRATION ===

# Updated time patterns from your new analysis (hourly bull win rates)
HOURLY_BULL_RATES = {
    0: 0.518, 1: 0.505, 2: 0.514, 3: 0.528, 4: 0.509, 5: 0.528,
    6: 0.509, 7: 0.500, 8: 0.517, 9: 0.520, 10: 0.492, 11: 0.520,
    12: 0.503, 13: 0.509, 14: 0.512, 15: 0.514, 16: 0.507, 17: 0.506,
    18: 0.514, 19: 0.511, 20: 0.515, 21: 0.529, 22: 0.524, 23: 0.504
}

# Updated feature importance weights from your new model
FEATURE_WEIGHTS = {
    "price_volatility": 0.154,
    "bet_ratio": 0.151,
    "bear_bets_amount": 0.140,
    "bull_bets_amount": 0.140,
    "total_bets_amount": 0.132,
    "total_bet_log": 0.131,
    "hour": 0.093,
    "day_of_week": 0.058
}

# Key thresholds from your analysis
BULL_WIN_BET_RATIO = 1.117  # Bulls win when bet_ratio ≈ 1.117
BEAR_WIN_BET_RATIO = 1.051  # Bears win when bet_ratio ≈ 1.051

# Store price history for volatility calculation
price_history = []


# === NEW: BET SNAPSHOT CSV FUNCTION ===
def save_bet_snapshot_csv(bet_data, decision, confidence, ml_score, bet_amount_bnb, current_epoch):
    """Save all bet details to CSV for later analysis"""
    folder = "bet_history"
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, "bets.csv")
    file_exists = os.path.isfile(file_path)

    # Get current prices and BTC data
    live_price = get_live_bnb_price()
    btc_data = get_btc_data()

    # Try to get lock price from previous epoch
    lock_price = 0
    price_diff = 0
    try:
        previous_epoch = current_epoch - 1
        previous_round = contract.functions.rounds(previous_epoch).call()
        lock_price = previous_round[4] / 1e8
        price_diff = live_price - lock_price
    except:
        pass

    # Collect all possible snapshot details
    snapshot = {
        "datetime": datetime.now().isoformat(),
        "epoch": current_epoch,
        "decision": decision,
        "confidence": confidence,
        "bet_amount_bnb": bet_amount_bnb,
        "ml_score": ml_score,
        "bull_amount": bet_data.get("bull_amount", 0),
        "bear_amount": bet_data.get("bear_amount", 0),
        "total_amount": bet_data.get("total_amount", 0),
        "bull_payout": bet_data.get("bull_payout", 0),
        "bear_payout": bet_data.get("bear_payout", 0),
        "bull_percent": bet_data.get("bull_percent", 0),
        "bear_percent": bet_data.get("bear_percent", 0),
        "bet_ratio": bet_data.get("bull_amount", 0) / bet_data.get("bear_amount", 1) if bet_data.get("bear_amount",
                                                                                                     0) > 0 else 2.0,
        "whale_bet_side": bet_data.get("whale_bet_side", ""),
        "max_bet_on_bull": bet_data.get("max_bet_on_bull", 0),
        "max_bet_on_bear": bet_data.get("max_bet_on_bear", 0),
        "bull_whales": bet_data.get("bull_whales", 0),
        "bear_whales": bet_data.get("bear_whales", 0),
        "current_streak": current_streak,
        "current_streak_type": current_streak_type,
        "skip_rounds_remaining": skip_rounds_remaining,
        "last_loss_epoch": last_loss_epoch,
        "live_bnb_price": live_price,
        "lock_price_previous": lock_price,
        "price_difference": price_diff,
        "btc_price": btc_data.get("price", 0),
        "btc_5min_change": btc_data.get("change_5min", 0),
        "btc_influence": calculate_btc_influence(),
        "real_price_volatility": calculate_real_price_volatility(),
        "price_momentum": calculate_price_momentum(),
        "current_hour": datetime.now().hour,
        "current_day_of_week": datetime.now().weekday(),
        "hourly_bull_rate": HOURLY_BULL_RATES.get(datetime.now().hour, 0.5),
        "total_bet_log": math.log(bet_data.get("total_amount", 1) + 1),
        "streak_valid": should_use_streak_for_ml(current_epoch) if current_streak >= 4 else True,
        "last_round_price_change": rounds_history[-1]['price_change_usdt'] if len(rounds_history) >= 1 else 0,
        "big_move_detected": abs(rounds_history[-1]['price_change_usdt']) > 1.75 if len(rounds_history) >= 1 else False,
        "wallet_number": BOT_NUMBER,
        "claimed": "no",
        "result": "",
        "reward_amount": 0,
    }

    # Write to CSV
    with open(file_path, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=snapshot.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(snapshot)

    print(f"💾 Bet snapshot saved to {file_path}")


def load_shared_loss_state():
    """Load shared loss state from file"""
    try:
        if os.path.exists(SHARED_LOSS_FILE):
            with open(SHARED_LOSS_FILE, 'r') as f:
                return json.load(f)
        else:
            return {"skip_rounds_remaining": 0, "last_loss_epoch": None, "loss_bot": None}
    except Exception as e:
        print(f"⚠️ Error loading shared loss state: {e}")
        return {"skip_rounds_remaining": 0, "last_loss_epoch": None, "loss_bot": None}

def save_shared_loss_state(skip_rounds, loss_epoch, bot_name="Unknown"):
    """Save shared loss state to file"""
    try:
        loss_data = {
            "skip_rounds_remaining": skip_rounds,
            "last_loss_epoch": loss_epoch,
            "loss_bot": bot_name,
            "timestamp": datetime.now().isoformat()
        }
        with open(SHARED_LOSS_FILE, 'w') as f:
            json.dump(loss_data, f, indent=2)
        print(f"💾 Shared loss state saved: {skip_rounds} rounds remaining")
    except Exception as e:
        print(f"⚠️ Error saving shared loss state: {e}")



# === NEW: CHAINLINK PRICE FUNCTIONS ===

def get_live_bnb_price():
    """Get live BNB price from Chainlink with 8-second caching"""
    global last_price_fetch, cached_price

    current_time = time.time()

    # UPDATED: Return cached price if less than 8 seconds old (was 10)
    if current_time - last_price_fetch < 8 and cached_price > 0:
        return cached_price

    # Fetch new price
    try:
        latest_data = price_feed.functions.latestRoundData().call()
        cached_price = latest_data[1] / 1e8  # Chainlink uses 8 decimals
        last_price_fetch = current_time
        return cached_price
    except Exception as e:
        print(f"⚠️ Price fetch error: {e}")
        return cached_price if cached_price > 0 else 0


# === NEW: BTC 5-MINUTE TRACKING FUNCTIONS ===

def get_btc_data():
    """Get BTC price and 5-minute change with 30-second caching"""
    global last_btc_fetch, cached_btc_data

    current_time = time.time()

    # Return cached data if less than 30 seconds old
    if current_time - last_btc_fetch < 30 and cached_btc_data["price"] > 0:
        return cached_btc_data

    try:
        # Get current price + 5-minute data from Binance (free, fast)
        response = requests.get("https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT", timeout=5)
        data = response.json()

        current_price = float(data["lastPrice"])

        # Get 5-minute kline data for 5-min change calculation
        kline_response = requests.get(
            "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=6",
            timeout=5
        )
        klines = kline_response.json()

        # Current price vs 5 minutes ago
        price_5min_ago = float(klines[0][4])  # Close price 5 minutes ago
        change_5min = ((current_price - price_5min_ago) / price_5min_ago) * 100

        cached_btc_data = {
            "price": current_price,
            "change_5min": change_5min
        }
        last_btc_fetch = current_time

        return cached_btc_data

    except Exception as e:
        print(f"⚠️ BTC 5min fetch error: {e}")
        return cached_btc_data if cached_btc_data["price"] > 0 else {"price": 0, "change_5min": 0}


def calculate_btc_influence():
    """Calculate BTC influence based on 5-MINUTE change (-0.3 to +0.3)"""
    btc_data = get_btc_data()

    if btc_data["price"] == 0:
        return 0.0

    change_5min = btc_data["change_5min"]

    # 5-minute thresholds (much smaller than 24h)
    if change_5min > 1.5:  # Strong 5min pump (>1.5%)
        return 0.3
    elif change_5min > 0.8:  # Medium 5min pump (0.8-1.5%)
        return 0.2
    elif change_5min > 0.3:  # Small 5min pump (0.3-0.8%)
        return 0.1
    elif change_5min < -1.5:  # Strong 5min dump (<-1.5%)
        return -0.3
    elif change_5min < -0.8:  # Medium 5min dump (-1.5 to -0.8%)
        return -0.2
    elif change_5min < -0.3:  # Small 5min dump (-0.8 to -0.3%)
        return -0.1
    else:  # Sideways (-0.15% to +0.15%)
        return 0.0


# === NEW: SMART STREAK VALIDATION ===

def should_use_streak_for_ml(current_epoch):
    """Check if streak should be used based on price direction vs streak direction"""
    global current_streak, current_streak_type

    if current_streak < 4:
        return True  # No significant streak to validate

    try:
        # FIXED: Use previous epoch lock price (current is always $0)
        previous_epoch = current_epoch - 1
        previous_round = contract.functions.rounds(previous_epoch).call()
        lock_price = previous_round[4] / 1e8

        # Get live price
        live_price = get_live_bnb_price()
        price_difference = live_price - lock_price

        # Check if price direction matches streak direction
        if current_streak_type == "BULL" and price_difference > 0:
            return True  # Bull streak + price up = valid streak
        elif current_streak_type == "BEAR" and price_difference < 0:
            return True  # Bear streak + price down = valid streak
        else:
            print(
                f"🚫 STREAK INVALIDATED: {current_streak_type} streak x{current_streak} but price difference: {price_difference:+.4f}")
            return False  # Streak contradicts current price = invalid

    except Exception as e:
        print(f"⚠️ Error validating streak: {e}")
        return True  # Default to using streak if error occurs


# === NEW: ROUNDS HISTORY FUNCTIONS ===

def load_cached_rounds():
    """Load rounds data from cache file"""
    global rounds_history

    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                cached_data = json.load(f)
                rounds_history = cached_data.get('rounds_history', [])
                print(f"📂 Loaded {len(rounds_history)} rounds from cache")
        else:
            print("📂 No cache file found, starting fresh")
            rounds_history = []
    except Exception as e:
        print(f"⚠️ Error loading cache: {e}")
        rounds_history = []


def save_rounds_to_cache():
    """Save rounds data to cache file"""
    try:
        cache_data = {
            'rounds_history': rounds_history,
            'last_updated': datetime.now().isoformat()
        }
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache_data, f, indent=2)
        print(f"💾 Saved {len(rounds_history)} rounds to cache")
    except Exception as e:
        print(f"⚠️ Error saving cache: {e}")


def fetch_single_round_data(epoch):
    """Fetch data for a single round"""
    try:
        round_data = contract.functions.rounds(epoch).call()

        start_ts = round_data[1]
        lock_ts = round_data[2]
        close_ts = round_data[3]
        lock_price = round_data[4] / 1e8  # Oracle price (8 decimals) - this is already in USDT
        close_price = round_data[5] / 1e8 if round_data[5] > 0 else 0  # Oracle price in USDT
        total_amount = round_data[8] / 1e18
        bull_amount = round_data[9] / 1e18
        bear_amount = round_data[10] / 1e18

        if close_price == 0 or close_ts == 0:
            return None  # Skip incomplete rounds

        # Calculate payouts
        bull_payout = total_amount / bull_amount if bull_amount > 0 else 0
        bear_payout = total_amount / bear_amount if bear_amount > 0 else 0

        # Use oracle prices directly (they're already in USDT)
        lock_price_usdt = lock_price
        close_price_usdt = close_price
        price_change_usdt = close_price_usdt - lock_price_usdt

        # Determine winner
        winner = "BULL" if close_price > lock_price else "BEAR"

        round_info = {
            'epoch': epoch,
            'lock_price': lock_price,
            'close_price': close_price,
            'lock_price_usdt': lock_price_usdt,
            'close_price_usdt': close_price_usdt,
            'price_change_usdt': price_change_usdt,
            'bull_payout': bull_payout,
            'bear_payout': bear_payout,
            'winner': winner,
            'total_amount': total_amount
        }

        return round_info

    except Exception as e:
        print(f"⚠️ Error fetching round {epoch}: {e}")
        return None


def fetch_round_history():
    """Fetch the last 24 completed rounds data with caching"""
    global rounds_history, current_streak, current_streak_type

    try:
        # Load existing cache
        load_cached_rounds()

        current_epoch = contract.functions.currentEpoch().call()
        target_epochs = list(range(current_epoch - 24, current_epoch))  # Last 24 epochs

        # Get epochs we already have cached
        cached_epochs = {round_info['epoch'] for round_info in rounds_history}

        # Find which epochs we need to fetch
        epochs_to_fetch = [epoch for epoch in target_epochs if epoch not in cached_epochs and epoch > 0]

        print(f"📊 Current epoch: {current_epoch}")
        print(f"📂 Cached rounds: {len(cached_epochs)}")
        print(f"🔄 Need to fetch: {len(epochs_to_fetch)} new rounds")

        # Fetch missing rounds
        new_rounds_fetched = 0
        for epoch in epochs_to_fetch:
            print(f"🔄 Fetching round {epoch}...")
            round_info = fetch_single_round_data(epoch)
            if round_info:
                rounds_history.append(round_info)
                new_rounds_fetched += 1

        # Sort by epoch and keep only last 24
        rounds_history.sort(key=lambda x: x['epoch'])
        rounds_history = rounds_history[-24:]  # Keep only last 24

        # Update streak tracking
        if len(rounds_history) >= 2:
            current_streak = 1
            current_streak_type = rounds_history[-1]['winner']

            # Count consecutive wins from the end
            for i in range(len(rounds_history) - 2, -1, -1):
                if rounds_history[i]['winner'] == current_streak_type:
                    current_streak += 1
                else:
                    break

        # Save updated cache
        if new_rounds_fetched > 0:
            save_rounds_to_cache()
            print(f"✅ Fetched {new_rounds_fetched} new rounds")
        else:
            print("✅ All rounds up to date from cache")

        # Display streak info
        if current_streak >= 4:
            streak_emoji = "🟢" if current_streak_type == "BULL" else "🔴"
            print(f"🔥 CURRENT STREAK: {streak_emoji} {current_streak_type} x{current_streak}")

        print(f"📋 Total rounds in history: {len(rounds_history)}")

    except Exception as e:
        print(f"⚠️ Error in fetch_round_history: {e}")


# === NEW: ENHANCED ML FUNCTIONS ===

def calculate_real_price_volatility():
    """Calculate volatility using real BNB prices from rounds history"""
    if len(rounds_history) < 5:
        return 0.0

    # Get last 10 close prices
    prices = [r['close_price_usdt'] for r in rounds_history[-10:]]
    if len(prices) < 2:
        return 0.0

    # Calculate price changes
    changes = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
    avg_change = sum(changes) / len(changes)
    avg_price = sum(prices) / len(prices)

    return avg_change / avg_price if avg_price > 0 else 0.0


def calculate_price_momentum():
    """Calculate if BNB is trending up or down"""
    if len(rounds_history) < 5:
        return 0.0

    # Compare recent vs older prices
    recent_avg = sum(r['close_price_usdt'] for r in rounds_history[-3:]) / 3
    older_avg = sum(r['close_price_usdt'] for r in rounds_history[-8:-5]) / 3

    momentum = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0

    # Return momentum score (-1 to +1)
    return max(-1.0, min(1.0, momentum * 10))


def calculate_price_volatility():
    """LEGACY: Calculate price volatility from fake price movements (kept for compatibility)"""
    if len(price_history) < 10:
        return 0.0

    # Simple volatility calculation using standard deviation
    prices = price_history[-10:]  # Last 10 data points
    mean_price = sum(prices) / len(prices)
    variance = sum((p - mean_price) ** 2 for p in prices) / len(prices)
    return math.sqrt(variance) / mean_price if mean_price > 0 else 0.0


def send_telegram_message(message):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        response = requests.post(url, data=payload)
        if not response.ok:
            print(f"⚠️ Telegram error: {response.text}")
    except Exception as e:
        print(f"⚠️ Telegram exception: {e}")


def get_bet_size_category(amount_bnb):
    """Categorize bet sizes based on your analysis"""
    if amount_bnb < 0.1:
        return "Small"
    elif amount_bnb < 0.5:
        return "Medium"
    elif amount_bnb < 2.0:
        return "Large"
    else:
        return "Huge"


def calculate_ml_prediction_score(bet_data):
    """
    ENHANCED: Calculate prediction score with SMART streak validation + BTC INFLUENCE
    Returns: score (-1 to 1), where positive = bull bias, negative = bear bias
    """
    global rounds_history, current_streak, current_streak_type

    current_hour = datetime.now().hour
    current_day = datetime.now().weekday()  # 0=Monday, 6=Sunday

    bull_amount = bet_data["bull_amount"]
    bear_amount = bet_data["bear_amount"]
    total_amount = bet_data["total_amount"]
    current_epoch = bet_data.get("current_epoch", 0)

    if total_amount == 0:
        return 0.0

    # Calculate features
    bet_ratio = bull_amount / bear_amount if bear_amount > 0 else 2.0
    total_bet_log = math.log(total_amount + 1)

    # NEW: Use real price volatility instead of fake
    real_volatility = calculate_real_price_volatility()

    score = 0.0

    # 1. Bet ratio feature (most important: 15.1%)
    # From your updated analysis: bulls win when bet_ratio > 1.117
    if bet_ratio > BULL_WIN_BET_RATIO:
        score += 0.3 * FEATURE_WEIGHTS["bet_ratio"]
    elif bet_ratio < BEAR_WIN_BET_RATIO:
        score -= 0.3 * FEATURE_WEIGHTS["bet_ratio"]

    # 2. NEW: Real price volatility (15.4% importance)
    # Higher volatility = more uncertainty = lower confidence
    volatility_factor = min(real_volatility * 2, 0.2)  # Cap at 0.2
    score += volatility_factor * FEATURE_WEIGHTS["price_volatility"]

    # 3. Bet amounts (14.0% each)
    # Larger total amounts favor bulls based on your data
    if total_amount > 5.0:  # Large pool
        score += 0.2 * FEATURE_WEIGHTS["total_bets_amount"]
    elif total_amount < 1.0:  # Small pool
        score -= 0.1 * FEATURE_WEIGHTS["total_bets_amount"]

    # 4. Time-based patterns (9.3% importance)
    hourly_bull_rate = HOURLY_BULL_RATES.get(current_hour, 0.5)
    hour_bias = (hourly_bull_rate - 0.5) * 2  # Convert to -1 to 1 scale
    score += hour_bias * FEATURE_WEIGHTS["hour"]

    # 5. Day of week (5.8% importance) - simplified
    # Weekend bias (rough approximation)
    if current_day in [5, 6]:  # Saturday, Sunday
        score += 0.05 * FEATURE_WEIGHTS["day_of_week"]

    # 6. FLIPPED: Whale count pressure (contrarian)
    bull_whales = bet_data.get("bull_whales", 0)
    bear_whales = bet_data.get("bear_whales", 0)

    if bull_whales > bear_whales:
        score -= 0.1  # FLIPPED: bet against bull whales
    elif bear_whales > bull_whales:
        score += 0.1  # FLIPPED: bet against bear whales

    # === UPDATED: SMART STREAK REVERSAL LOGIC ===
    if current_streak >= 6:
        # NEW: Check if streak should be used based on price direction
        if should_use_streak_for_ml(current_epoch):
            streak_bias = min(current_streak * 0.1, 0.4)  # Max 0.4 bias
            if current_streak_type == "BULL":
                score -= streak_bias  # Bet against bull streak
            elif current_streak_type == "BEAR":
                score += streak_bias  # Bet against bear streak
        # If streak is invalid (contradicts price), streak_bias = 0 (no effect on ML)

    # === NEW: PRICE MOMENTUM ===
    if len(rounds_history) >= 5:
        momentum_score = calculate_price_momentum()
        score += momentum_score * 0.15

    # === NEW: BIG MOVE REVERSAL ===
    if len(rounds_history) >= 1:
        last_move = abs(rounds_history[-1]['price_change_usdt'])
        if last_move > 1.75:
            # Expect reversal after big move
            if rounds_history[-1]['price_change_usdt'] > 0:
                score -= 0.2  # Big up move = bear bias
            else:
                score += 0.2  # Big down move = bull bias

    # === NEW: BTC INFLUENCE ===
    btc_influence = calculate_btc_influence()
    score += btc_influence

    # Debug info
    if abs(btc_influence) > 0.05:
        btc_data = get_btc_data()
        btc_direction = "📈" if btc_influence > 0 else "📉"
        print(f"\n💰 BTC Influence: {btc_direction} {btc_influence:+.2f} (5min: {btc_data['change_5min']:+.2f}%)")

    return max(-1.0, min(1.0, score))  # Clamp between -1 and 1


def enhanced_decision_logic(bet_data, ml_score):
    """
    UPDATED: Enhanced decision making based on new 10k rounds analysis
    MODIFIED: Now bets AGAINST whales and AGAINST bet ratio patterns
    """
    global skip_rounds_remaining

    # 🎯 NEW: Check shared loss state FIRST
    shared_state = load_shared_loss_state()
    if shared_state["skip_rounds_remaining"] > 0:
        # Update local state from shared state
        skip_rounds_remaining = shared_state["skip_rounds_remaining"]

        # Decrease shared counter
        new_skip_count = max(0, shared_state["skip_rounds_remaining"] - 1)
        save_shared_loss_state(new_skip_count, shared_state["last_loss_epoch"], "SKIP_COUNTDOWN")

        return None, f"SKIP (SHARED Loss prevention: {new_skip_count + 1} rounds remaining, caused by {shared_state.get('loss_bot', 'Unknown')})"

    bull_amount = bet_data["bull_amount"]
    bear_amount = bet_data["bear_amount"]
    total_amount = bet_data["total_amount"]
    bull_payout = bet_data["bull_payout"]
    bear_payout = bet_data["bear_payout"]
    whale_bet_side = bet_data["whale_bet_side"]
    max_bet_on_bull = bet_data["max_bet_on_bull"]
    max_bet_on_bear = bet_data["max_bet_on_bear"]

    # Calculate bet ratio
    bet_ratio = bull_amount / bear_amount if bear_amount > 0 else 2.0

    # Priority 1: SMART whale logic - Follow whale UNLESS ML very weak
    if whale_bet_side:
        if abs(ml_score) <= 0.06:  # ML very weak = bet against whale
            opposite_side = "bear" if whale_bet_side == "bull" else "bull"
            confidence = "HIGH (Against whale - ML too weak)"
            return opposite_side, confidence
        else:  # ML confident = follow whale
            same_side = whale_bet_side  # "bull" or "bear"
            confidence = "HIGH (Follow whale - ML agrees)"
            return same_side, confidence


    # Priority 3: MODIFIED - FLIPPED BETTING LOGIC (bet against the patterns)
    if total_amount > 2.0 and abs(bull_payout - bear_payout) >= 1.2:

        # FLIPPED LOGIC: Bet against the bet ratio + large bet patterns
        payout_decision = None

        # FLIPPED: When bet_ratio > 1.0 AND large bet on bull >= 0.7 BNB -> bet BEAR
        if bet_ratio > 1.0 and max_bet_on_bull >= 0.7:
            payout_decision = "bear"  # FLIPPED from "bull"
        # FLIPPED: When bet_ratio < 1.0 AND large bet on bear >= 0.7 BNB -> bet BULL
        elif bet_ratio < 1.0 and max_bet_on_bear >= 0.7:
            payout_decision = "bull"  # FLIPPED from "bear"
        else:
            payout_decision = None  # No clear edge

        # Get ML-based decision (require stronger threshold for agreement)
        ml_decision = None
        if ml_score > 0.05:  # ML suggests bull (UPDATED from 0.05)
            ml_decision = "bull"
        elif ml_score < -0.05:  # ML suggests bear (UPDATED from -0.05)
            ml_decision = "bear"

        # NEW: Dynamic strategy based on ML strength
        if abs(ml_score) > 0.16:  # STRONG ML = Trust it, ignore contrarian
            final_decision = ml_decision
            strategy_type = "Strong ML (Override contrarian)"
        elif payout_decision and payout_decision == ml_decision:  # WEAK ML = Use contrarian
            final_decision = payout_decision
            strategy_type = "Contrarian + ML consensus"
        else:
            final_decision = None  # Skip if no agreement

        if final_decision:
            # Both methods agree! Now check time-based confidence boost
            current_hour = datetime.now().hour
            bad_bull_hours = [10]  # Hour with lowest bull rate (0.492)
            good_bull_hours = [21, 3, 5]  # Hours with highest bull rates (0.529, 0.528, 0.528)

            base_confidence = "HIGH" if abs(ml_score) > 0.2 else "MEDIUM"

            # NEW: Add streak information to confidence (with validation)
            streak_info = ""
            if current_streak >= 7:
                current_epoch = bet_data.get("current_epoch", 0)
                if should_use_streak_for_ml(current_epoch):
                    streak_emoji = "🟢" if current_streak_type == "BULL" else "🔴"
                    streak_info = f" + {streak_emoji}Streak x{current_streak}"
                else:
                    streak_info = f" + ❌Streak x{current_streak} (Invalid)"

            # Time-based confidence adjustments
            if payout_decision == "bull":
                if current_hour in good_bull_hours:
                    confidence = f"HIGH (Against bet ratio {bet_ratio:.2f} + ML + Good bull hour {current_hour}{streak_info})"
                elif current_hour in bad_bull_hours:
                    confidence = f"MEDIUM (Against bet ratio {bet_ratio:.2f} + ML, but bad bull hour {current_hour}{streak_info})"
                else:
                    confidence = f"{base_confidence} (Against bet ratio {bet_ratio:.2f} + ML consensus{streak_info})"
            else:  # bear decision
                if current_hour in bad_bull_hours:
                    confidence = f"HIGH (Against bet ratio {bet_ratio:.2f} + ML + Bad bull hour {current_hour}{streak_info})"
                elif current_hour in good_bull_hours:
                    confidence = f"MEDIUM (Against bet ratio {bet_ratio:.2f} + ML, but good bull hour {current_hour}{streak_info})"
                else:
                    confidence = f"{base_confidence} (Against bet ratio {bet_ratio:.2f} + ML consensus{streak_info})"

            return payout_decision, confidence

        else:
            # Methods disagree or no clear payout edge - SKIP
            if payout_decision is None:
                return None, f"SKIP (No clear bet ratio edge: {bet_ratio:.2f}, max bull: {max_bet_on_bull:.3f}, max bear: {max_bet_on_bear:.3f})"
            elif ml_decision is None:
                return None, f"SKIP (Bet ratio says {payout_decision}, but ML neutral: {ml_score:.3f})"
            else:
                return None, f"SKIP (Bet ratio says {payout_decision}, ML says {ml_decision})"

    # Priority 4: Pure ML decision (only for very strong signals) - UNCHANGED
    if abs(ml_score) > 0.26:  # Very strong ML signal
        decision = "bull" if ml_score > 0 else "bear"
        streak_info = ""
        if current_streak >= 6:
            current_epoch = bet_data.get("current_epoch", 0)
            if should_use_streak_for_ml(current_epoch):
                streak_emoji = "🟢" if current_streak_type == "BULL" else "🔴"
                streak_info = f" + {streak_emoji}Streak x{current_streak}"
            else:
                streak_info = f" + ❌Streak x{current_streak} (Invalid)"
        return decision, f"MEDIUM (Very strong ML signal: {ml_score:.3f}{streak_info})"

    # Priority 5: Time + Large bet combination (more restrictive) - UNCHANGED
    if total_amount > 2.0:  # Decent pool size
        current_hour = datetime.now().hour
        good_bull_hours = [21, 3, 5]  # Best performing hours
        bad_bull_hours = [10]  # Worst performing hour

        # Enhanced with large bet detection
        if current_hour in good_bull_hours and ml_score > 0.15 and max_bet_on_bull >= 0.7:
            return "bull", f"LOW (Good bull hour {current_hour} + ML + Large bull bet {max_bet_on_bull:.3f})"
        elif current_hour in bad_bull_hours and ml_score < -0.15 and max_bet_on_bear >= 0.7:
            return "bear", f"LOW (Bad bull hour {current_hour} + ML + Large bear bet {max_bet_on_bear:.3f})"

    return None, f"No edge detected (ratio: {bet_ratio:.2f}, ML: {ml_score:.3f})"


def fetch_bets(start_block, end_block, current_epoch):
    bet_bull_logs = web3.eth.get_logs({
        "fromBlock": start_block,
        "toBlock": end_block,
        "address": CONTRACT_ADDRESS,
        "topics": [bet_bull_topic]
    })
    bet_bear_logs = web3.eth.get_logs({
        "fromBlock": start_block,
        "toBlock": end_block,
        "address": CONTRACT_ADDRESS,
        "topics": [bet_bear_topic]
    })
    bet_bull_events = [
        contract.events.BetBull().process_log(log)
        for log in bet_bull_logs
        if contract.events.BetBull().process_log(log)['args']['epoch'] == current_epoch
    ]
    bet_bear_events = [
        contract.events.BetBear().process_log(log)
        for log in bet_bear_logs
        if contract.events.BetBear().process_log(log)['args']['epoch'] == current_epoch
    ]

    # Prepare a list of all bets as dictionaries
    all_bets = []

    for event in bet_bull_events:
        amount_bnb = event['args']['amount'] / 1e18
        all_bets.append({
            "side": "Bull",
            "amount_bnb": amount_bnb,
            "user": event['args']['sender'],
            "epoch": event['args']['epoch']
        })

    for event in bet_bear_events:
        amount_bnb = event['args']['amount'] / 1e18
        all_bets.append({
            "side": "Bear",
            "amount_bnb": amount_bnb,
            "user": event['args']['sender'],
            "epoch": event['args']['epoch']
        })

    bull_amount = sum(bet["amount_bnb"] for bet in all_bets if bet["side"] == "Bull")
    bear_amount = sum(bet["amount_bnb"] for bet in all_bets if bet["side"] == "Bear")
    total_amount = bull_amount + bear_amount

    # Count whales (bets ≥ 0.8 BNB)
    bull_whales = sum(1 for b in all_bets if b["side"] == "Bull" and b["amount_bnb"] >= 0.80)
    bear_whales = sum(1 for b in all_bets if b["side"] == "Bear" and b["amount_bnb"] >= 0.80)

    bull_payout = total_amount / bull_amount if bull_amount > 0 else 0
    bear_payout = total_amount / bear_amount if bear_amount > 0 else 0

    bull_percent = (bull_amount / total_amount) * 100 if total_amount > 0 else 0
    bear_percent = (bear_amount / total_amount) * 100 if total_amount > 0 else 0

    # Detect whale bet (updated threshold)
    whale_threshold = 2
    whale_bet_side = None
    for bet in all_bets:
        if bet["amount_bnb"] >= whale_threshold:
            whale_bet_side = bet["side"].lower()
            break

    # NEW: Calculate max bet on each side (key insight from analysis)
    max_bet_on_bull = max([bet["amount_bnb"] for bet in all_bets if bet["side"] == "Bull"], default=0)
    max_bet_on_bear = max([bet["amount_bnb"] for bet in all_bets if bet["side"] == "Bear"], default=0)

    # Update price history (simplified - using total pool as price proxy)
    if total_amount > 0:
        price_history.append(total_amount)
        if len(price_history) > 20:
            price_history.pop(0)

    return {
        "bull_amount": bull_amount,
        "bear_amount": bear_amount,
        "total_amount": total_amount,
        "bull_payout": bull_payout,
        "bear_payout": bear_payout,
        "bull_percent": bull_percent,
        "bear_percent": bear_percent,
        "whale_bet_side": whale_bet_side,
        "max_bet_on_bull": max_bet_on_bull,  # NEW
        "max_bet_on_bear": max_bet_on_bear,  # NEW
        "bull_whales": bull_whales,
        "bear_whales": bear_whales,
        "all_bets": all_bets,
        "current_epoch": current_epoch  # NEW: Pass epoch for streak validation
    }


def place_bet(decision, current_epoch, confidence, bet_data, ml_score):
    global last_bet_epochs
    if decision == "bull":
        function = contract.functions.betBull(current_epoch)
        side = "UP (Bull)"
    else:
        function = contract.functions.betBear(current_epoch)
        side = "DOWN (Bear)"

    balance_wei = web3.eth.get_balance(WALLET_ADDRESS)

    # Updated dynamic bet sizing based on new confidence levels
    if confidence.startswith("HIGH"):
        bet_multiplier = 0.06  # 6% of balance for high confidence (increased)
    elif confidence.startswith("MEDIUM"):
        bet_multiplier = 0.04  # 4% of balance for medium confidence (increased)
    else:
        bet_multiplier = 0.02  # 2% of balance for low confidence

    bet_amount_wei = int(balance_wei * bet_multiplier)
    min_bet_wei = web3.to_wei('0.001', 'ether')

    if bet_amount_wei < min_bet_wei:
        bet_amount_wei = min_bet_wei

    bet_amount_bnb = web3.from_wei(bet_amount_wei, 'ether')
    print(f"\n🎯 Placing bet of {bet_amount_bnb:.6f} BNB on: {side}")
    print(f"🔍 Confidence: {confidence}")

    nonce = web3.eth.get_transaction_count(WALLET_ADDRESS)

    tx = function.build_transaction({
        'from': WALLET_ADDRESS,
        'value': bet_amount_wei,
        'gas': 200000,
        'gasPrice': web3.to_wei('0.1', 'gwei'),
        'nonce': nonce
    })

    signed_tx = web3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = web3.eth.send_raw_transaction(signed_tx.raw_transaction)
    print(f"🚀 Transaction sent! TX Hash: {web3.to_hex(tx_hash)}")
    send_telegram_message(
        f"🚀 Bet placed on WALLET{BOT_NUMBER}, Round:{current_epoch}\nSide: {side}\nAmount: {bet_amount_bnb:.6f} BNB\nConfidence: {confidence}\nTX: https://bscscan.com/tx/{web3.to_hex(tx_hash)}"
    )

    last_bet_epochs.append(current_epoch)
    if len(last_bet_epochs) > 2:
        last_bet_epochs.pop(0)

    # NEW: Save bet snapshot to CSV
    try:
        save_bet_snapshot_csv(bet_data, decision, confidence, ml_score, bet_amount_bnb, current_epoch)
    except Exception as e:
        print(f"⚠️ Could not save bet snapshot: {e}")



def main_loop():
    global bet_placed_last_round
    print("⏳ Waiting for current round initialization...")

    # NEW: Initialize rounds history at startup
    print("🔄 Fetching rounds history...")
    fetch_round_history()

    while True:
        try:
            current_epoch = contract.functions.currentEpoch().call()
            current_round = contract.functions.rounds(current_epoch).call()
            start_ts = current_round[1]
            end_ts = current_round[2]
            if start_ts > 0 and end_ts > 0:
                break
        except Exception:
            pass
        time.sleep(1)

    print(f"🔄 Epoch: {current_epoch}")
    print(f"🕒 Start: {datetime.fromtimestamp(start_ts)}")
    print(f"🕒 End:   {datetime.fromtimestamp(end_ts)}")

    # Show skip status
    if skip_rounds_remaining > 0:
        print(f"⚠️ LOSS PREVENTION ACTIVE: Skipping {skip_rounds_remaining} more rounds (except whale bets)")

    # NEW: Show current streak with validation
    if current_streak >= 3:
        streak_emoji = "🟢" if current_streak_type == "BULL" else "🔴"
        if should_use_streak_for_ml(current_epoch):
            print(f"🔥 CURRENT STREAK: {streak_emoji} {current_streak_type} x{current_streak} ✅ VALID")
        else:
            print(
                f"🔥 CURRENT STREAK: {streak_emoji} {current_streak_type} x{current_streak} ❌ INVALID (Price contradicts)")

    # NEW: Show live BNB price using PREVIOUS epoch lock price
    live_price = get_live_bnb_price()
    try:
        # FIXED: Use previous epoch lock price (current is always $0)
        previous_epoch = current_epoch - 1
        previous_round = contract.functions.rounds(previous_epoch).call()
        lock_price = previous_round[4] / 1e8

        price_diff = live_price - lock_price
        price_emoji = "📈" if price_diff >= 0 else "📉"
        print(f"💰 Live BNB: ${live_price:.4f} | Lock: ${lock_price:.4f} (Prev) | Diff: {price_emoji} {price_diff:+.4f}")
    except:
        print(f"💰 Live BNB Price: ${live_price:.4f}")

    # NEW: Show BTC status
    btc_data = get_btc_data()
    if btc_data['price'] > 0:
        btc_emoji = "📈" if btc_data['change_5min'] >= 0 else "📉"
        print(f"₿ BTC: ${btc_data['price']:,.0f} | 5min: {btc_emoji} {btc_data['change_5min']:+.2f}%")

    def get_block_by_timestamp(target_timestamp, before=True):
        latest_block = web3.eth.block_number
        low = 1
        high = latest_block
        while low < high:
            mid = (low + high) // 2
            block = web3.eth.get_block(mid)
            block_time = block.timestamp
            if block_time < target_timestamp:
                low = mid + 1
            else:
                high = mid
        return (low - 1) if before and low > 1 else low

    start_block = get_block_by_timestamp(start_ts, before=False)
    print(f"✅ Monitoring bets from block: {start_block}")

    snapshot = None
    decision = None
    confidence = None

    print(
        "\n⏳ SMART STREAK + BTC ML monitoring with 8-second price updates + 5-min BTC tracking every 0.0001 seconds until 5.000001 seconds before round ends...\n")

    while True:
        now = int(time.time())
        time_left = end_ts - now
        if time_left <= 5.000001:
            break

        try:
            latest_block = web3.eth.block_number
            snapshot = fetch_bets(start_block, latest_block, current_epoch)

            bull_amount = snapshot["bull_amount"]
            bear_amount = snapshot["bear_amount"]
            total_amount = snapshot["total_amount"]
            bull_percent = snapshot["bull_percent"]
            bear_percent = snapshot["bear_percent"]
            bet_ratio = bull_amount / bear_amount if bear_amount > 0 else 2.0

            # Calculate ML prediction score
            ml_score = calculate_ml_prediction_score(snapshot)

            # Get enhanced decision
            decision, confidence = enhanced_decision_logic(snapshot, ml_score)

            # Enhanced display with new metrics + LIVE PRICE + BTC
            current_hour = datetime.now().hour
            hourly_rate = HOURLY_BULL_RATES.get(current_hour, 0.5)

            # NEW: Get live price and difference in monitoring loop
            live_price = get_live_bnb_price()
            price_info = ""
            try:
                previous_epoch = current_epoch - 1
                previous_round = contract.functions.rounds(previous_epoch).call()
                lock_price = previous_round[4] / 1e8
                price_diff = live_price - lock_price
                price_emoji = "📈" if price_diff >= 0 else "📉"
                price_info = f" | Price: ${live_price:.2f} {price_emoji}{price_diff:+.2f}"
            except:
                price_info = f" | Price: ${live_price:.2f}"

            # NEW: Get BTC data for display
            btc_data = get_btc_data()
            btc_display = f" | BTC: ${btc_data['price']:,.0f} ({btc_data['change_5min']:+.2f}%)" if btc_data[
                                                                                                        'price'] > 0 else ""

            skip_indicator = f" | SKIP: {skip_rounds_remaining}" if skip_rounds_remaining > 0 else ""

            # NEW: Enhanced streak indicator with validation
            streak_indicator = ""
            if current_streak >= 3:
                if should_use_streak_for_ml(current_epoch):
                    streak_indicator = f" | 🔥{current_streak_type}x{current_streak}✅"
                else:
                    streak_indicator = f" | 🔥{current_streak_type}x{current_streak}❌"

            print(
                f"\r📊 Bull: {bull_percent:.2f}% | Bear: {bear_percent:.2f}% | Ratio: {bet_ratio:.2f} | Pool: {total_amount:.4f}{price_info}{btc_display} | Max Bull: {snapshot['max_bet_on_bull']:.3f} | Max Bear: {snapshot['max_bet_on_bear']:.3f} | ML: {ml_score:.3f}{skip_indicator}{streak_indicator}",
                end="")

            if decision:
                print(f" | 🎯 {decision.upper()} ({confidence})")

        except Exception as e:
            print(f"\n⚠️ Error during fetch: {e}")
            send_telegram_message(f"⚠️ Error during fetch: {e}")

        time.sleep(0.0001)

    print("\n\n✅ FINAL SMART STREAK + BTC ANALYSIS (8-second price + 5-min BTC + streak validation)")
    if snapshot:
        ml_score = calculate_ml_prediction_score(snapshot)
        decision, confidence = enhanced_decision_logic(snapshot, ml_score)
        bet_ratio = snapshot['bull_amount'] / snapshot['bear_amount'] if snapshot['bear_amount'] > 0 else 2.0

        print(f"🔄 Epoch: {current_epoch}")
        print(f"📈 Bull: {snapshot['bull_percent']:.2f}% ({snapshot['bull_amount']:.4f} BNB)")
        print(f"📉 Bear: {snapshot['bear_percent']:.2f}% ({snapshot['bear_amount']:.4f} BNB)")
        print(f"💰 Total Pool: {snapshot['total_amount']:.4f} BNB")
        print(f"⚖️ Bet Ratio: {bet_ratio:.3f} (Bull/Bear)")
        print(f"🎯 Max Bull Bet: {snapshot['max_bet_on_bull']:.4f} BNB")
        print(f"🎯 Max Bear Bet: {snapshot['max_bet_on_bear']:.4f} BNB")
        print(f"📊 Payouts -> Bull: {snapshot['bull_payout']:.2f}x | Bear: {snapshot['bear_payout']:.2f}x")
        print(
            f"🤖 ML Score: {ml_score:.3f} ({'Bull bias' if ml_score > 0 else 'Bear bias' if ml_score < 0 else 'Neutral'})")
        print(
            f"🕒 Current Hour: {datetime.now().hour} (Historical bull rate: {HOURLY_BULL_RATES.get(datetime.now().hour, 0.5):.1%})")

        # NEW: Show BTC influence
        btc_data = get_btc_data()
        if btc_data['price'] > 0:
            btc_influence = calculate_btc_influence()
            if abs(btc_influence) > 0.05:
                btc_emoji = "📈" if btc_influence > 0 else "📉"
                print(
                    f"₿ BTC Impact: {btc_emoji} {btc_influence:+.2f} (${btc_data['price']:,.0f}, 5min: {btc_data['change_5min']:+.2f}%)")

        # NEW: Enhanced streak display with validation
        if current_streak >= 3:
            streak_emoji = "🟢" if current_streak_type == "BULL" else "🔴"
            if should_use_streak_for_ml(current_epoch):
                print(f"🔥 Current Streak: {streak_emoji} {current_streak_type} x{current_streak} ✅ VALID (Used in ML)")
            else:
                print(
                    f"🔥 Current Streak: {streak_emoji} {current_streak_type} x{current_streak} ❌ INVALID (Ignored in ML)")

        # NEW: Show current price difference using PREVIOUS epoch
        try:
            # FIXED: Use previous epoch lock price
            previous_epoch = current_epoch - 1
            previous_round = contract.functions.rounds(previous_epoch).call()
            lock_price = previous_round[4] / 1e8

            live_price = get_live_bnb_price()
            price_diff = live_price - lock_price
            price_emoji = "📈" if price_diff >= 0 else "📉"
            print(
                f"💰 Price Difference: {price_emoji} {price_diff:+.4f} USDT (Live: ${live_price:.4f} vs Lock: ${lock_price:.4f} Prev)")
        except Exception as e:
            print(f"⚠️ Could not calculate price difference: {e}")

        if len(rounds_history) >= 1:
            last_move = rounds_history[-1]['price_change_usdt']
            if abs(last_move) > 1.75:
                move_emoji = "📈" if last_move > 0 else "📉"
                print(f"💥 Last Big Move: {move_emoji} ${last_move:+.4f} USDT")

        momentum = calculate_price_momentum()
        if abs(momentum) > 0.1:
            momentum_emoji = "📈" if momentum > 0 else "📉"
            print(f"🌊 Price Momentum: {momentum_emoji} {momentum:.3f}")

        if snapshot['whale_bet_side']:
            print(f"🐋 Whale detected betting: {snapshot['whale_bet_side'].upper()} -> Bot will bet OPPOSITE")

        if skip_rounds_remaining > 0:
            print(f"⚠️ Loss Prevention: {skip_rounds_remaining} rounds remaining to skip")

    if decision:
        place_bet(decision, current_epoch, confidence, snapshot, ml_score)
        bet_placed_last_round = True
    else:
        print(f"\n⚠️ Skipping bet. {confidence}")
        bet_placed_last_round = False


if __name__ == "__main__":
    print("🚀 SMART STREAK + BTC PancakeSwap Prediction Bot with ENHANCED ML")
    print("📊 NEW: Bet AGAINST whales + smart streaks + big moves + bet ratios")
    print("🛡️ Feature: Skip 8 rounds after loss (except anti-whale bets)")
    print("🧠 NEW: Smart streak validation (ignores contradictory streaks)")
    print("⏰ NEW: 8-second price updates (was 10 seconds)")
    print("💰 NEW: Real-time price difference tracking")
    print("₿ NEW: BTC 5-minute tracking & influence")
    print("🌊 NEW: Price momentum tracking")
    print("💥 NEW: Big move reversal (>$1.75)")
    print("📂 NEW: 24 rounds history caching")
    print("💾 NEW: Bet snapshots saved to bet_history/bets.csv")
    print("=" * 80)

    while True:
        try:
            # Moved reward check to the beginning of the loop here

            main_loop()

        except Exception as e:
            print(f"⚠️ Error in main loop: {e}")

        print("\n⏳ Waiting 1780 seconds before restarting the loop...\n")
        for i in range(1780, 0, -1):
            print(f"\r⏳ Restarting in {i} seconds...", end="")
            time.sleep(1)
        print()
