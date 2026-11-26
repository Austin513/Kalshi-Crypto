# Can you price 15-minute Bitcoin bets better than a bell curve?

## Why I did this

Kalshi runs a market every 15 minutes that asks one question: **will Bitcoin be higher at the end of these 15 minutes than at the start?** You can buy "yes" or "no", and the price tells you what the crowd thinks the odds are.

To put your own price on a bet like that, you need a decent sense of how far BTC tends to move in 15 minutes. The textbook answer is to assume price moves follow a bell curve. But anyone who's watched Bitcoin knows it doesn't behave that way. It sits still for a while, then suddenly jumps.

So I wanted to find out:

1. **How wrong is the bell curve, really?** And in which direction?
2. **Can I do better** just by using what BTC has actually done in the past, or by noticing when the market's been jumpy lately?
3. **Do any of my guesses beat the people trading on Kalshi?**

The whole analysis, with charts, is in [`notebooks/btc_tails.ipynb`](notebooks/btc_tails.ipynb). It's saved with all its results, so you can read it right on GitHub without running anything.

## What I did

**Got the data.** I downloaded every 1-minute Bitcoin price from Binance for six months, May through October 2026. That's 264,960 minutes. I checked for gaps and duplicates and there weren't any.

**Chopped it into 15-minute pieces.** Each piece starts on the quarter hour (:00, :15, :30, :45), just like Kalshi's markets. For each one I measured how much BTC went up or down. That left me with 17,647 pieces to work with.

**Put the real moves next to a bell curve** to see where they disagree.

**Tried four ways of guessing** "what are the odds BTC ends above some price?"

- **A. Bell curve.** The textbook approach.
- **B. Just count.** Look back at history and ask: how often did BTC move at least this far? No assumptions about shape.
- **C. Count, but pay attention to recent jumpiness.** Same idea as B, but it also looks at how wild the last 1 to 4 hours have been.
- **D. Bell curve, but pay attention to recent jumpiness.** A small neural network that makes the bell curve wider when the market's been wild and narrower when it's been calm.

**Kept the test honest.** All four learned from May through August. I used September to make a few setup choices. October was locked away until the end and only used once, as the final exam.

**Went up against the market.** I pulled October's Kalshi prices for the same 15-minute windows and checked who was closer to what actually happened: me or the traders.

## What I found

### The bell curve gets the shape wrong

A typical 15-minute move is around 0.2%. Next to a bell curve, real Bitcoin is:

- **Calmer most of the time.** Moves bigger than 0.25% (either way) happened about 2,300 times. A bell curve says it should've been about 3,700.
- **Wilder when it does move.** Moves bigger than 0.5% happened 466 times, more than double the 210 a bell curve predicts. Moves bigger than 0.75% happened 154 times. The bell curve says that should've happened about 3 times.

So the bell curve thinks medium-sized moves are more common than they are, and it seriously underestimates the big ones.

### Bitcoin doesn't lean up or down in any reliable way

There's a tiny tilt toward bigger up-moves than down-moves overall, but it isn't consistent. It was strongest in August and actually flipped the other way in October. Take out the 5 most extreme 15-minute windows and it mostly disappears.

### My guesses got better, but not by a lot

Here's how each method did on October, the month none of them had seen. The score measures how close the predicted odds came to what actually happened. Lower is better.

| method | score | compared with the bell curve |
|---|---|---|
| A. Bell curve | 0.2482 | (starting point) |
| B. Just count | 0.2399 | about 3% better |
| C. Count + recent jumpiness | 0.2327 | about 6% better |
| D. Bell curve + recent jumpiness | 0.2335 | about 6% better |

Using real history helped. Paying attention to recent jumpiness helped about as much again. To check whether these improvements could just be luck, I reran the comparison many times on reshuffled sets of October's days. They held up. They're real, but small.

### The market was better than all of them

On 2,936 October Kalshi markets, the market's own price one minute in predicted the result better than any of my four methods. Method D came closest, but it still lost, and not by a margin you could blame on chance. Basically, Kalshi traders already know what my models know, and then some.

### I tried four ways to make money, and none of them worked

Being less accurate on average doesn't automatically mean you can't profit. You only need to find the occasional bet where the price is wrong by more than the fee. So I downloaded Kalshi's prices for every minute of every window in September and October and tried four strategies:

- **S1. Bet when my best model disagrees with the price at the start of the window.**
- **S2. Live fair value.** Every minute, work out the "right" price from where Bitcoin is on Binance compared with the starting price, how much time is left, and how jumpy the last hour was. Bet when Kalshi's price lags behind.
- **S3. Look for price ranges the market gets wrong.** For example, do contracts priced at 90¢ actually win more than 90% of the time?
- **S4. A regression that combines all of the above** with the market price itself, and bets when it disagrees with the market.

To keep it fair, each strategy's rules (how big a disagreement before betting, which minutes, which price ranges) were picked using **September only**, then tested once on **October**. Each bet is 1 contract at the listed price, after Kalshi's fee.

| strategy | made money in September? | bets in October | win rate | October profit per bet |
|---|---|---|---|---|
| S1. Model vs. price at the start | yes (+$11.75) | 303 | 39% | −4.6¢ |
| S2. Live fair value | yes (+$11.64) | 2,095 | 55% | −2.5¢ |
| S3. Mispriced price ranges | yes (+$30.60) | 719 | 85% | −1.2¢ |
| S4. Regression on top of the market | no (−$9.26), so I wouldn't trade it | 432 | 50% | −4.2¢ |

Every one of them lost money in October. The three that looked good in September didn't hold up. September's "winners" were mostly luck that happened to fit that month. S2 was the clearest failure. It lost money even before fees, and its losses after fees were too consistent to blame on bad luck. When I tracked accuracy minute by minute, the market's price was more accurate than my live fair value at every single minute of the window, in both months.

Two caveats work in my favor here, which makes the losses more convincing, not less. I assumed every bet got filled at the listed price, and I couldn't confirm October's exact fees. A real trader would likely do a bit worse.

## Conclusion

Bitcoin's 15-minute moves really don't look like a bell curve. Most of the time BTC is quieter than the bell curve expects, and when it moves, it moves much harder. If you price bets on BTC using a bell curve, you'll be off, mostly on the big moves.

Fixing that is straightforward and it does help. Counting what actually happened in the past beats the bell curve. Noticing whether the market has been calm or wild over the last few hours helps about as much again. Together that's roughly a 6% improvement. It's real, but modest.

The catch is that none of this beats the market. Kalshi's prices were more accurate than anything I built, at every minute of the window. I tried four different ways to turn my models into a profit, and all four lost money in October. My takeaway is that this market already prices Bitcoin's jumpiness well, and the public data doesn't hold an easy edge. Finding one would probably take things this data doesn't have: the full order book (how many contracts sit at each price), confirmed fees, and faster price data than one snapshot a minute.

## Everything in the analysis

**Plots**
- Histogram of 15-minute moves with a bell curve drawn on top
- Q–Q plot (real moves vs. where a bell curve says they should fall)
- How often moves of each size happened vs. what a bell curve predicts (up and down sides)
- Gap between the bell curve and "just count" at each strike, with luck ranges
- Calibration plots for all four models (when a model says 30%, does it happen 30% of the time?)
- Calibration of Kalshi's price vs. my models
- Accuracy at each minute of the window: Kalshi's price vs. my live fair value
- Running October profit for each of the four strategies

**Models and regressions**
- A. Bell curve (normal distribution fit to past moves)
- B. "Just count" (share of past moves bigger than the target)
- C. Logistic regression using the target, recent volatility (last 1 and 4 hours) and the last two 15-minute moves. Three versions were compared and the best on September was kept.
- D. Small PyTorch neural network that sets the bell curve's center and width from recent volatility
- Live fair value model: past moves, scaled by the last hour's volatility, for each minute left in the window
- S4 logistic regression combining the market price, live fair value, the last minute's move, the spread and time left

**Analysis**
- Data checks: missing minutes, duplicates, misaligned timestamps
- Summary stats (typical move, skew, fat tails), overall and by month, plus the biggest moves
- What-if check with the 5 biggest moves removed
- Train on May–Aug, tune on September, test on October
- Scoring with Brier score and log loss, by strike and overall
- Luck checks that resample whole days of the test month
- Check that a higher target never gets a higher probability
- How often Binance's price gives the same result as Kalshi's official settlement
- Four trading strategies with rules picked on September, tested on October, including a "filled one minute late" stress test

**Outputs**
- Score tables for every model, overall and by strike
- Model vs. model and model vs. market comparison tables with luck ranges
- Kalshi coverage table (which markets existed and had prices)
- Strategy results table: bets, win rate, profit per bet before and after fees, luck range
- A summary of the key numbers at the end of the notebook

## Running it yourself

You'll need Python 3.14.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python -m pytest -q tests                          # quick sanity checks, no data needed
.venv/bin/python -m btc_tails.download_binance               # BTC prices, about 12 MB
.venv/bin/python -m btc_tails.build_db                       # builds data/btc.sqlite
.venv/bin/python -m btc_tails.kalshi 2026-10-01 2026-11-01   # Kalshi data (slow, about 1.5 hours, because Kalshi limits request speed)
.venv/bin/python -m btc_tails.kalshi 2026-09-01 2026-11-01 --full   # every minute of every window, for the strategy tests (about 15 minutes)
.venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/btc_tails.ipynb
```

Anything already downloaded gets skipped, so running it again is quick. Everything is saved in `data/`, which Git ignores.

**Opening the notebook in VS Code or Jupyter:** pick the Python from this project's `.venv` folder (`.venv/bin/python`) as the kernel. If you pick a different one, the notebook won't be able to find its packages.
