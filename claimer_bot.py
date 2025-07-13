import json
import os
import time
import csv
import pandas as pd
from datetime import datetime
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
from dotenv import load_dotenv, find_dotenv
import requests

# === CONFIG ===
load_dotenv(find_dotenv())

# Load all 6 wallets
WALLETS = {}
for i in range(1, 7):
    private_key = os.getenv(f"PRIVATE_KEY_{i}")
    wallet_address = os.getenv(f"WALLET_ADDRESS_{i}")
    if private_key and wallet_address:
        WALLETS[i] = {
            "private_key": private_key,
            "address": wallet_address
        }

print(f"🔑 Loaded {len(WALLETS)} wallets: {list(WALLETS.keys())}")

# === WEB3 SETUP ===
with open("prediction_abi.json", "r") as f:
    ABI = json.load(f)

web3 = Web3(
    Web3.HTTPProvider("https://thrilling-rough-shape.bsc.quiknode.pro/ce859309d10cbdb7eab6b562035cfc118471ebf8/"))
web3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

CONTRACT_ADDRESS = Web3.to_checksum_address("0x18B2A687610328590Bc8F2e5fEdDe3b582A49cdA")
contract = web3.eth.contract(address=CONTRACT_ADDRESS, abi=ABI)

# === FILES ===
BET_HISTORY_FILE = "bet_history/bets.csv"
SHARED_LOSS_FILE = "shared_loss_state.json"


# === TELEGRAM ===
def send_telegram_message(message):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        requests.post(url, data=payload, timeout=5)
    except:
        pass


# === SHARED LOSS FUNCTIONS ===
def save_shared_loss_state(skip_rounds, loss_epoch, bot_name="Claimer"):
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
        print(f"💾 SHARED LOSS: {skip_rounds} rounds remaining")
    except Exception as e:
        print(f"⚠️ Error saving shared loss: {e}")


# === NEW: EPOCH CHECK FUNCTION ===
def is_epoch_finished(epoch):
    """Check if epoch is finished and has results"""
    try:
        round_data = contract.functions.rounds(epoch).call()
        close_timestamp = round_data[3]  # closeTimestamp
        close_price = round_data[5]  # closePrice

        # Epoch finished if both exist
        return close_timestamp > 0 and close_price > 0
    except:
        return False


# === MAIN FUNCTIONS ===
def read_bet_history():
    """Read unclaimed bets from CSV"""
    try:
        if not os.path.exists(BET_HISTORY_FILE):
            print("📂 No bet history file found")
            return []

        df = pd.read_csv(BET_HISTORY_FILE)

        # Find unclaimed bets
        unclaimed = df[df['claimed'] == 'no'].copy()
        print(f"📊 Found {len(unclaimed)} unclaimed bets")

        return unclaimed.to_dict('records')
    except Exception as e:
        print(f"⚠️ Error reading bet history: {e}")
        return []


def check_and_claim_bet(bet_record):
    """Check and claim reward for a single bet"""
    try:
        epoch = int(bet_record['epoch'])
        wallet_num = int(bet_record['wallet_number'])

        # 🎯 NEW: Skip if epoch not finished
        if not is_epoch_finished(epoch):
            print(f"⏳ Epoch {epoch} still LIVE - skipping until next cycle")
            return None  # Don't process yet

        if wallet_num not in WALLETS:
            print(f"❌ Wallet {wallet_num} not found!")
            return None

        wallet = WALLETS[wallet_num]
        wallet_address = wallet['address']
        private_key = wallet['private_key']

        print(f"🔍 Checking FINISHED epoch {epoch} for wallet {wallet_num}...")

        # Check if claimable
        claimable_amount = contract.functions.claimable(epoch, wallet_address).call()

        if claimable_amount > 0:
            # WIN! Claim it
            reward_bnb = web3.from_wei(claimable_amount, 'ether')
            print(f"🎉 WIN! Claiming {reward_bnb:.6f} BNB for wallet {wallet_num}")

            # Claim transaction
            nonce = web3.eth.get_transaction_count(wallet_address)
            tx = contract.functions.claim([epoch]).build_transaction({
                'from': wallet_address,
                'gas': 200000,
                'gasPrice': web3.eth.gas_price + web3.to_wei('0.1', 'gwei'),
                'nonce': nonce
            })

            signed_tx = web3.eth.account.sign_transaction(tx, private_key)
            tx_hash = web3.eth.send_raw_transaction(signed_tx.raw_transaction)

            print(f"🚀 Claim TX: {web3.to_hex(tx_hash)}")
            send_telegram_message(f"🎉 WIN! Wallet{wallet_num} claimed {reward_bnb:.6f} BNB from epoch {epoch}")

            # Reset shared loss state on ANY win
            save_shared_loss_state(0, None, f"WIN_W{wallet_num}")

            return {
                "result": "win",
                "reward_amount": float(reward_bnb),
                "claimed": "yes"
            }
        else:
            # LOSS
            print(f"❌ LOSS detected for wallet {wallet_num}, epoch {epoch}")

            # Set shared loss state (skip 10 rounds)
            save_shared_loss_state(10, epoch, f"LOSS_W{wallet_num}")

            send_telegram_message(f"❌ LOSS! Wallet{wallet_num} lost in epoch {epoch}. ALL BOTS skip 10 rounds!")

            return {
                "result": "loss",
                "reward_amount": 0,
                "claimed": "yes"
            }

    except Exception as e:
        print(f"⚠️ Error checking bet {epoch}: {e}")
        return None


def update_bet_history(bet_record, result_data):
    """Update CSV with claim results"""
    try:
        # Read current CSV
        df = pd.read_csv(BET_HISTORY_FILE)

        # Find the row to update
        mask = (df['epoch'] == bet_record['epoch']) & (df['wallet_number'] == bet_record['wallet_number'])

        if mask.any():
            # Update the row
            df.loc[mask, 'result'] = result_data['result']
            df.loc[mask, 'reward_amount'] = result_data['reward_amount']
            df.loc[mask, 'claimed'] = result_data['claimed']

            # Save back to CSV
            df.to_csv(BET_HISTORY_FILE, index=False)
            print(f"📝 Updated CSV: {result_data['result']} for epoch {bet_record['epoch']}")
        else:
            print(f"⚠️ Could not find bet record to update")

    except Exception as e:
        print(f"⚠️ Error updating CSV: {e}")


def claimer_loop():
    """Main claimer loop"""
    print(f"\n🤖 CLAIMER BOT STARTING - {datetime.now()}")

    # Read unclaimed bets
    unclaimed_bets = read_bet_history()

    if not unclaimed_bets:
        print("✅ No unclaimed bets found")
        return

    # Process each unclaimed bet
    processed = 0
    skipped = 0
    for bet_record in unclaimed_bets:
        print(f"\n📋 Processing bet {processed + skipped + 1}/{len(unclaimed_bets)}")

        # Check and claim
        result_data = check_and_claim_bet(bet_record)

        if result_data:
            # Update CSV
            update_bet_history(bet_record, result_data)
            processed += 1
        else:
            skipped += 1

        # Small delay between claims
        time.sleep(2)

    print(f"\n✅ CLAIMER CYCLE COMPLETE: {processed} processed, {skipped} skipped (live epochs)")


# === MAIN ===
if __name__ == "__main__":
    print("🤖 CLAIMER BOT - HANDLES ALL WALLET CLAIMS")
    print("📋 Checks bet_history/bets.csv every 300 seconds")
    print("🎯 Claims rewards and updates loss streaks")
    print("⏳ NEW: Skips live epochs to prevent false losses")
    print("=" * 50)

    while True:
        try:
            claimer_loop()
        except Exception as e:
            print(f"⚠️ Claimer error: {e}")
            send_telegram_message(f"⚠️ Claimer bot error: {e}")

        print(f"\n⏳ WAITING 300 seconds...")
        for i in range(300, 0, -1):
            print(f"\r⏳ Next cycle in {i}s...", end="")
            time.sleep(1)
        print()