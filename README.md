<h1>PancakeSwap Prediction Bot</h1>
<p>This is a sophisticated cryptocurrency trading bot for PancakeSwap's prediction game on Binance Smart Chain (BSC). Here's what it does:</p>
<h3><br />Core Purpose</h3>
<p><br />The bot automatically places bets on whether BNB price will go UP (Bull) or DOWN (Bear) in 5-minute prediction rounds, using machine learning insights and market analysis.</p>
<h3><br />Key Components</h3>
<p><br /><strong>1. Connection Setup</strong></p>
<p>Connects to BSC blockchain via QuikNode<br />Interacts with PancakeSwap prediction contract at a specific address<br />Uses Chainlink oracle for real-time BNB price feeds<br />Monitors Bitcoin prices from Binance API</p>
<p><strong>2. Data Collection</strong></p>
<p>Round History: Caches last 24 completed rounds to identify patterns<br />Bet Monitoring: Tracks all bets placed in current round (bull vs bear amounts)<br />Price Tracking: Updates BNB price every 8 seconds, BTC every 30 seconds<br />Whale Detection: Identifies large bets (&ge;2 BNB) that might influence outcomes</p>
<p><strong>3. Machine Learning Strategy</strong></p>
<p><br />The bot uses multiple factors weighted by importance:</p>
<h3><br />Top Features:</h3>
<p>Price Volatility (15.4%): Higher volatility = lower confidence<br />Bet Ratio (15.1%): Bull/Bear amount ratio - looks for imbalances<br />Bet Amounts (14%): Pool size affects predictions<br />Time Patterns (9.3%): Certain hours favor bulls (e.g., hour 21: 52.9% bull wins)</p>
<p><strong>Advanced Signals:</strong></p>
<p>Streak Validation: Tracks winning streaks (bull/bear) but validates if they align with actual price movement<br />BTC Influence: 5-minute BTC changes affect BNB predictions<br />Price Momentum: Recent trend direction (last 8 rounds)<br />Big Move Reversal: After large price moves (&gt;$1.75), expects reversal</p>
<p><strong>4. Contrarian Strategy</strong></p>
<p><br />The bot has a contrarian approach - it often bets AGAINST:</p>
<p>Whale positions (when ML is weak)<br />Crowd behavior patterns<br />Extended winning streaks (mean reversion)</p>
<p><strong>5. Decision Logic Priority</strong></p>
<p>Whale Detection: If whale bets and ML is uncertain, bet opposite<br />Bet Ratio Analysis: When bet ratio shows clear imbalance + large bets, bet against the crowd<br />ML Strong Signals: Only when ML score &gt;0.26 (very confident)<br />Time Combinations: Good hours + favorable conditions</p>
<p><strong>6. Risk Management</strong></p>
<p>Dynamic Bet Sizing: 2-6% of wallet based on confidence (HIGH/MEDIUM/LOW)<br />Loss Prevention: After a loss, bot can skip 8 rounds (shared across multiple bot instances)<br />Skip Conditions: Doesn't bet when signals conflict or edges unclear</p>
<p><strong>7. Data Logging</strong></p>
<p><br />Saves every bet decision to CSV with comprehensive data:</p>
<p>Bet details, ML scores, prices, BTC data<br />Pool composition, whale activity<br />Streaks, momentum, volatility<br />Used for later analysis and strategy refinement</p>
<p><strong>8. Monitoring Loop</strong></p>
<p>Waits for round to start<br />Continuously monitors bets every 0.0001 seconds<br />Shows real-time: pool percentages, ratios, prices, ML score<br />Makes final decision 5 seconds before round ends<br />Places bet via smart contract transaction</p>
<h3>Key Innovations</h3>
<p>-Smart streak validation - ignores streaks that contradict price movement<br />-5-minute BTC tracking - more relevant than 24h changes<br />-Contrarian whale betting - fades large positions<br />-Real-time price differential - compares live vs lock price<br />-Shared loss prevention - multiple bots coordinate to avoid consecutive losses</p>
<p>The bot essentially tries to exploit market inefficiencies by combining multiple signals while being cautious about overconfidence, using a sophisticated ML-inspired scoring system trained on 10,000+ historical rounds.</p>
