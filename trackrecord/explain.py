"""A "How & why" note under every computed number on the site.

Each tile (`<div class="tile"><div class="tl">label</div><div class="tv">value</div>…`) and every
stat on a home/area card gets a click-to-expand note saying how the number was calculated —
inputs, formula, where in the code — and why it is on the page: what decision it informs.

The notes live in one registry keyed by the tile's label (a regex) and, where the same label
means different things on different pages, by the page (a regex on the URL path).  Pages are
annotated after rendering (`annotate`) so no template has to change, and a test asserts that
every tile on every page has a note — a number without one is a build failure, not a gap.
"""
from __future__ import annotations

import html
import re

# (page regex or None, label regex, how, why).  First match wins; more specific pages first.
NOTES: list[tuple[str | None, str, str, str]] = [
    # ---------------------------------------------------------------- manager memo
    ("memo", r"^return /yr$",
     "The monthly returns are chained together, and the total growth is converted to a yearly rate. The market figure is the US stock market as a whole, weighted by company size, over exactly the same months.",
     "This is the headline a manager quotes. It is here to be compared with the market line beneath it. The raw gap is the starting point, not the conclusion; the alpha tile is what is left after risk is accounted for."),
    ("memo", r"^ff3 alpha /yr$",
     "The record's monthly return above the risk-free rate is regressed on three things anyone can buy cheaply: the market, small companies and cheap companies. The part left over, the intercept, is the alpha, and it is multiplied by twelve to give a yearly figure. The t-statistic and p-value use standard errors that allow for months that move together.",
     "This is the single most important number in the memo: the return that market, size and value exposure cannot explain. A t-statistic above 2 is the usual bar for calling it evidence of skill rather than noise. Below that, the record is consistent with luck."),
    ("memo", r"^information ratio$",
     "The average monthly gap between the record and the market is divided by how much that gap varies from month to month, and the result is scaled to a yearly figure. The tracking error is that variation on its own, expressed per year.",
     "This is the return earned per unit of the risk a client actually takes by choosing this manager over an index fund. Around 0.5 is good over a long record. It is also what the fund-of-funds allocator maximises."),
    ("memo", r"^scores$",
     "Both scores use fixed scales that are never re-tuned, so they stay comparable across managers. The alpha-maxing score is 50 plus 10 times the excess return over the US market in percent per year. The wealth-management score combines evidence of skill (30%), risk-adjusted return (25%), downside protection (25%) and consistency across rolling five-year windows (20%).",
     "There is one number for a client who wants the most return and one for a client who wants to keep what they have. The same manager can score well on one and badly on the other, which is the point."),
    ("memo", r"^volatility$",
     "Volatility is the standard deviation of the monthly returns, scaled to a yearly figure. Beta (β) is how much the record moved for each 1% move in the market, from a regression of the record's excess return on the market's.",
     "This says how bumpy the ride was, next to the market's. A record with more volatility than the market needs more return to justify itself, and beta says how much of that volatility is simply market exposure."),
    ("memo", r"^max drawdown$",
     "One dollar is invested at the start and tracked month by month. The drawdown is the largest fall from any peak to the following low, and the dates of that peak and low are shown. Because it is measured on month-ends, it understates falls that happened within a month.",
     "This is the worst experience a client would have lived through. It is the number most likely to make a real client redeem at the bottom, so a mandate has to be sized so that this loss is survivable."),
    ("memo", r"^down / up capture$",
     "Down capture is the record's average return in the months the market fell, divided by the market's average return in those months. Up capture is the same for the months the market rose. A value of 1.0 means moving one-for-one with the market.",
     "Asymmetry is where a defensive manager earns their fee: less than 1 on the downside with close to 1 on the upside. Equal capture both ways is just the market with leverage."),
    ("memo", r"^expected shortfall 95%$",
     "The monthly returns are sorted, the worst 5% are taken, and their average is the expected shortfall. The market's figure is computed the same way over the same months, and the worst single month is shown alongside.",
     "This is what a bad month costs on average, not just the single worst one. It feeds the downside-protection part of the wealth-management score and is the tail-risk number a risk committee asks for."),
    ("memo", r"^sharpe$",
     "The average monthly return above the one-month Treasury bill is divided by the standard deviation of the monthly returns, and the result is scaled to a yearly figure. The market's Sharpe ratio is computed over the same months.",
     "This is return per unit of total risk. Comparing it with the market's over the same period asks whether the manager was paid for the risk taken at all, before asking whether the extra was skill."),

    # ---------------------------------------------------------------- simulation summary
    (r"^/sim/", r"^alpha-maxing score$",
     "The score is 50 plus 10 times the blend's excess return over the US market in percent per year, limited to the range 0 to 100. The blend's monthly returns are the weighted mix of the chosen managers and asset classes, run through the same calculations as a single manager.",
     "This is the return-first client's score, on the same fixed scale used for every individual manager, so a blend can be compared directly with the managers inside it."),
    (r"^/sim/", r"^wealth-management score$",
     "The score is a weighted sum of four fixed parts: evidence of skill (30%), risk-adjusted return (25%), downside protection (25%) and consistency across rolling five-year windows (20%).",
     "This is the keep-what-you-have client's score. Blending managers usually raises it even when it lowers the alpha-maxing score, and that trade-off is what the simulation is for."),
    (r"^/sim/", r"^ff3 alpha$",
     "The blend's monthly return above the risk-free rate is regressed on the market, small companies and cheap companies. The part left over is the alpha, scaled to a yearly figure. The three loadings beside it show how much of each the blend carries.",
     "This asks whether the combination still has return that cheap exposures cannot explain. Diversifying across managers reduces the noise in this estimate, so its t-statistic is the fairest comparison between a blend and any one of its parts."),
    (r"^/sim/", r"^sharpe · ir$",
     "The Sharpe ratio is the average return above the Treasury bill divided by the volatility, scaled to a yearly figure. The information ratio is the average gap over the market divided by how much that gap varies, and that variation is the tracking error. The drawdown is the largest fall from a peak to a following low on month-ends.",
     "These are the two ratios an allocator sizes a mandate on: return per unit of total risk, and per unit of risk relative to the index. The drawdown says how it would have felt."),
    (r"^/sim/", r"^gross /yr$",
     "Each sleeve's monthly return is weighted by its share of the blend, the months are chained together, and the total growth is converted to a yearly rate before any fees. The market is the US stock market over the same months.",
     "This is the starting point for the fee arithmetic underneath: what the managers produced before the client paid for it."),
    (r"^/sim/", r"^net of manager fees /yr$",
     "Each sleeve's monthly return is reduced by its management fee divided by twelve, and by the performance fee on any gains above that sleeve's previous high point. The sleeves are then blended and converted to a yearly rate as above. The fee drag is the gross figure minus this one.",
     "This is what the client keeps. The gap to the gross number is the cost of the managers, and the memo's fee questions are written against it."),
    (r"^/sim/", r"^max drawdown$",
     "One dollar is tracked month by month through the blend, and the drawdown is the largest fall from any peak to the following low. It is shown before fees, after fees, and for the market.",
     "This is the loss a client would have had to sit through. The after-fee figure is the honest one, because fees are still charged while the portfolio is under water."),
    (r"^/sim/", r"^best / worst month$",
     "These are the highest and lowest of the blend's monthly returns before fees over the simulated period.",
     "They give a quick sense of the range. A worst month far larger than the typical one is a warning that the drawdown and shortfall numbers are driven by a single event."),

    # ---------------------------------------------------------------- manager dashboard
    ("dashboard", r"^names for half the outperformance$",
     "For every stock the manager ever held, its contribution to the portfolio's return each month (its weight times its return) is compared with what the same weight in the market would have made, and the differences are added up over the months it was held. Each stock's share of the outperformance is its contribution divided by the total of all positive contributions. The winners are sorted by share, and the tile counts how many are needed to reach 50%.",
     "This measures how concentrated the source of the returns is. A record built on two or three names is a different risk from one built on fifty, even when the alpha is the same."),
    ("dashboard", r"^names for 80%$",
     "The same contributions are sorted from the largest winner down, and the tile counts how many names it takes for their combined share to reach 80% of the outperformance, out of every name ever held.",
     "This is the breadth of the skill. If a handful of names made almost all of it, the memo has to ask whether those were repeatable decisions or one thesis."),
    ("dashboard", r"^top five's share$",
     "The five largest positive contributors' shares of the outperformance are added together. A share is a stock's contribution above the market, summed over the months it was held, divided by the total of all positive contributions.",
     "This is the single-thesis test. Above roughly two-thirds, the record is mostly one or two calls, and the questions for the manager should be about how those were found rather than about the process in general."),
    ("dashboard", r"^winners / losers$",
     "The positive contributions above the market are added up, and so are the negative ones, in percentage points of cumulative outperformance. They are simple sums, so together they equal the total gap to the market.",
     "This shows how much the losers gave back. Two records with the same total can be many small wins or huge wins minus big losses, and the second is fragile."),
    ("dashboard", r"^before \d{4}$",
     "The alpha for the months before the split year is estimated from a single regression of the record's excess return on the market's, with an extra term that lets the alpha differ after the split. Both halves therefore come out of the same fit, each with its own standard error.",
     "This asks whether the skill was there from the start or only in one era. The first half is the part the manager was less likely to be marketing on."),
    ("dashboard", r"^\d{4} onward$",
     "This is the same regression's alpha for the months from the split year on: the first-half alpha plus the difference term, scaled to a yearly figure. The split year is the same for every manager so that records are cut at the same date.",
     "The recent half is what a client would actually get. If it is much lower than the first half, the record may describe a strategy that no longer works or a market that got crowded."),
    ("dashboard", r"^difference",
     "This is the difference term itself: the second-half alpha minus the first-half alpha, scaled to a yearly figure, with its standard error and p-value from the same regression.",
     "This is a formal test of decay. With standard errors this size, halves of a record with constant true alpha will differ by more than two standard errors about one time in twenty, so a decline inside that range is not evidence of anything."),

    # ---------------------------------------------------------------- signal research
    (r"alpha-lab", r"^signals with .*evidence",
     "Each signal is tested two ways every month: the rank correlation between the signal and the following month's return, and the return of the top tenth of the universe minus the bottom tenth. Both give a t-statistic over the whole sample, and a signal is counted here only if one of them reaches 2. The count is out of the eight characteristics tested; the fitted models are not included, because they are built from these same eight.",
     "This is the result the page exists to produce: how many of the candidates survive contact with the data. A research process that cannot return a small number here is not testing anything."),
    (r"alpha-lab", r"^best (?:of the eight|signal)",
     "For the signal with the strongest evidence, the tile shows the top tenth of the universe minus the bottom tenth, annualized, with its t-statistic, its Sharpe ratio and its monthly rank correlation. Portfolios are rebuilt every month-end from what was known at the time, and each tenth is equally weighted.",
     "This is the one candidate worth trading, and the size of the number is why: it sets what a portfolio built on it could plausibly earn before costs and constraints. The construction page takes this signal and nothing else."),
    (r"alpha-lab", r"^learned model versus simple average$",
     "The two models are scored on identical months, and the difference between their monthly rank correlations is averaged over the period, giving the t-statistic shown. Anything short of 2 means the sample cannot tell them apart.",
     "It says whether a fitted model earns its complexity against a baseline with nothing to fit. On this sample it does not, which is the honest answer and the reason the portfolio is built on a single signal instead."),
    (r"alpha-lab", r"^learned model$",
     "A model that learns from all eight signals at once is trained on next-month returns, and its predictions are tested the same way as a single signal: ranked against the following month's returns, month by month. The first 36 months are held out entirely, and the model is refit every twelve months using only earlier data, so it never sees a month it is predicting.",
     "This is where learning is tried honestly. Compared with the simple average on identical months, it answers whether combining the signals cleverly adds anything on this universe."),
    (r"alpha-lab", r"^simple average$",
     "Each of the eight signals is standardized within the month and signed so that higher is better, and the eight are averaged with equal weights. The result is tested exactly like a single signal, by its monthly rank correlation with the following month's returns. Nothing is fitted.",
     "This is the baseline every model has to beat. It has no settings to overfit, so if a fitted model cannot beat it out of sample, the fitted model is not earning its complexity."),
    (r"alpha-lab", r"^head to head$",
     "Month by month, the learned model's IC minus the simple average's IC is taken on the same months, and the tile shows the average difference, its t-statistic, and the share of months the learned model was ahead.",
     "A paired comparison is the only fair one, since both see the same months. A t-statistic below 2 means the two are indistinguishable on this sample, whatever the averages say."),
    (r"alpha-lab", r"^does the learned model beat the simple average\?$",
     "The verdict comes from the head-to-head t-statistic. The learned model wins only if its average advantage clears two standard errors; a deficit of that size means the simple average wins; anything in between is no difference.",
     "This is the one-line answer a portfolio manager wants before deciding whether a learned model is worth running in production. On this universe and period, the honest answer is what the tile says."),

    # ---------------------------------------------------------------- factor risk model
    (r"risk-model", r"^factors$",
     "This counts everything the monthly regression estimates: the market, eight style factors (momentum, reversal, low volatility, size, value, profitability, cash flow and earnings yield, each a standardized exposure per stock built from prices and company accounts) and the industry groups.",
     "This is the size of the model: few enough factors to estimate every month from a few hundred stocks, and enough to capture the exposures a quantitative portfolio actually takes."),
    (r"risk-model", r"^industries$",
     "Each stock belongs to one industry group, taken from its sector classification. Each group is a yes-or-no exposure in the monthly regression, with the industry returns constrained to average to zero so that they are measured relative to the market.",
     "Industry membership explains more of a stock's month than any style does. A risk model without it would mistake industry bets for stock selection."),
    (r"risk-model", r"^of monthly returns explained$|^explanatory power$",
     "Each month, every stock's return is regressed on its style exposures and industry membership, and the share of that month's spread of returns the regression explains is recorded. The tile is the average of those monthly shares.",
     "This is how much of the cross-section of returns the factors explain. Commercial models manage 30% to 40% on US stocks, so this number says how close a public-data model gets. It matters for attribution: the higher it is, the more of a portfolio's return the model can assign to factors."),
    (r"risk-model", r"^total risk$",
     "This is the model's forecast of the portfolio's yearly volatility. It combines the portfolio's factor exposures with the factor covariance, and the portfolio's weights with each stock's specific variance, and takes the square root of the sum.",
     "A risk report starts with this number. The two tiles beside it split it into the part that comes from factor bets and the part that comes from individual stocks."),
    (r"risk-model", r"^factor$",
     "This is the factor part of the forecast on its own, expressed as a yearly volatility, with its share of the total variance.",
     "This is the risk that comes from tilts such as size, value and industry, which can be hedged or is being taken deliberately. A portfolio that meant to be stock-picking but shows mostly factor risk has a problem."),
    (r"risk-model", r"^specific$",
     "This is the stock-specific part of the forecast, expressed as a yearly volatility, with its share of the variance. Each stock's specific variance is estimated from its own past regression residuals, with more weight on recent months, and shrunk toward the median across stocks.",
     "This is the risk that only diversification reduces. It is what an active stock-picker should be taking, and its share is the model's view of how active the portfolio really is."),
    (r"risk-model", r"^bias statistic, random portfolios$",
     "Every month, 100 random portfolios of 50 stocks are formed. For each, the return over the following month is divided by the volatility the model predicted for it, giving a standardized outcome. The bias statistic is the standard deviation of all those outcomes. A value of 1.00 means predicted risk matched realized risk; below 1 the model over-predicts, and above 1 it under-predicts.",
     "This is the standard acceptance test for a risk model, and it is how commercial models are validated. A model that fails it will size positions and tracking-error budgets wrongly, whatever its explanatory power."),
    (r"risk-model", r"^bias statistic, whole universe$",
     "This is the same test on one portfolio, the whole universe with equal weights, each month: the standard deviation of realized return divided by predicted volatility across months.",
     "This checks that the model is calibrated for a broad portfolio as well as for concentrated random ones. The two together bracket the portfolios it will be used on."),
    (r"risk-model", r"^median specific risk$",
     "For the latest month, each stock's specific volatility is estimated from its own recent regression residuals, scaled to a yearly figure, and the tile shows the median across stocks.",
     "This is the typical risk of one name that has nothing to do with the market or its industry. It is the scale against which position limits are set: at 30% specific risk, a 2% active weight is 60 basis points of tracking error from one stock."),

    # ---------------------------------------------------------------- portfolio construction
    (r"construction", r"^active return, constrained$|^active return per year$",
     "This is the constrained portfolio's return minus the benchmark's, converted to a yearly rate over all the rebalances, after deducting 10 basis points of cost on every dollar traded. Each quarter, an optimiser chooses the weights that maximise the signal while respecting a 4% cap per position, a 2% band around the benchmark weight, 3% sector bands, 30% active share and 20% one-way turnover.",
     "This is what the signal is worth once it has to live inside a mandate. The unconstrained tile shows what the constraints cost, and this is the number a portfolio manager would actually have delivered."),
    (r"construction", r"^tracking error$",
     "The realized figure is the standard deviation of the monthly gap between the portfolio and the benchmark, scaled to a yearly rate. The predicted figure is the risk model's forecast of that gap for the target weights at each rebalance, averaged across rebalances.",
     "This is the risk budget the mandate is written in. Realized close to predicted is also a live test of the risk model: if they diverge, the model rather than the portfolio needs attention."),
    (r"construction", r"^information ratio$",
     "The yearly active return is divided by the realized tracking error. The hit rate beside it is the share of months in which the portfolio beat the benchmark.",
     "This is the efficiency of the active risk. Institutional active mandates are usually judged on it, and 0.5 sustained is good. The fund-of-funds allocator uses the same quantity."),
    (r"construction", r"^turnover$",
     "At each rebalance, the absolute change in every weight is added up and halved, giving one-way turnover, and the tile shows the average across rebalances next to the 20% budget the optimiser was given. The name count is the average number of stocks held.",
     "Turnover is cost and market impact. The constraint binds in most quarters, which is why the trade list is short and why the realized return is shown after cost."),
    (r"construction", r"^unconstrained top decile$",
     "Each quarter the top 10% of stocks by the signal are held with equal weights, with no caps, bands, active-share or turnover limits, and the same 10 basis points of trading cost. Its tracking error, information ratio, turnover and name count are shown beside it.",
     "This is the upper bound of what the signal could deliver. The gap to the constrained tile is the price of the mandate's rules, and the tracking error shows why those rules exist."),
    (r"construction", r"^benchmark$",
     "The benchmark is everything the managers own, added together: every manager's disclosed positions summed in dollars each quarter and held until the next filing. Its monthly return is chained and converted to a yearly rate, with its volatility and largest fall over the same months.",
     "The constructor is measured against what the managers collectively own, rather than a published index, because the signal comes from their filings. Everything above is relative to this."),
    (r"construction", r"^rebalances$",
     "This is the number of quarter-ends, 45 days after each filing date, on which the optimiser was run and a trade list was produced.",
     "This is the sample size behind the active-return and information-ratio figures. With around 50 quarters, the information ratio's standard error is about 0.3, which is why no claim of skill is made from it."),

    # ---------------------------------------------------------------- independent verification
    (r"r-verify", r"^managers agreeing$|^managers$",
     "For every manager, a second implementation written from scratch recalculates every headline statistic from the same aligned monthly returns and factors. A manager agrees when every one of its numbers is within one millionth of the site's own figure.",
     "An independent implementation catching an error in the first one is the only verification that means anything."),
    (r"r-verify", r"^numbers compared$|^numbers checked$",
     "This counts every manager-and-quantity pair compared: for each of four factor models, the alpha, its standard error, t-statistic, p-value, confidence interval, explanatory power and every loading, plus the yearly return, volatility, Sharpe ratio, largest drawdown and the alpha-maxing score.",
     "The check is not on a sample of numbers but on every number that appears on a dashboard or in a memo."),
    (r"r-verify", r"^largest difference$",
     "This is the largest absolute difference between the two implementations across all compared numbers, with the manager and quantity it came from.",
     "At one part in ten trillion, the difference is the order in which the computer added up the terms, not a difference of method. A real discrepancy would be many orders of magnitude larger and would appear in the disagreements tile."),
    (r"r-verify", r"^disagreements$",
     "This is the number of compared values whose difference exceeds one millionth.",
     "This is the pass-or-fail figure. Zero is the only acceptable value, and the test suite asserts it."),

    # ---------------------------------------------------------------- holdings research
    (r"13f-signals", r"^managers$",
     "This is the number of managers whose quarterly holdings filings are in the panel: the managers who run a portfolio of long positions, after excluding multi-strategy, macro and market-making firms whose disclosed positions are not their portfolio.",
     "This is the breadth of the evidence. Every signal here is a comparison across these managers, so the count bounds how much any result can be trusted."),
    (r"13f-signals", r"^quarters$",
     "This is the number of quarterly formation dates since 2013: the end of the month in which each quarter's filings become public, which is February, May, August and November because of the 45-day filing deadline.",
     "This is the length of the backtest in independent decisions. Around 50 quarters is short, so a result needs a spread large enough to be economically interesting before it can be statistically convincing."),
    (r"13f-signals", r"^positions$",
     "This is the total number of manager, stock and quarter combinations in the holdings panel after cleaning: securities matched to a traded stock, and share counts corrected for reporting errors and stock splits.",
     "This is the raw material for every portfolio on the page. Its size is what keeps the best-ideas and crowding portfolios well populated even when they are formed from one position per manager."),

    # ---------------------------------------------------------------- manager decay model
    (r"decay", r"^learned model$",
     "At every filing date, a model that learns from all the features at once is fit on every earlier manager-quarter whose twelve-month outcome was already known, and then scores that date's managers on their chance of lagging the market over the next year. The AUC is computed within the date, asking whether it ranked the eventual laggards above the rest, and the tile is the average of those AUCs across dates.",
     "This asks whether a manager's filings can predict who will fade. Scoring within each date removes the effect of a bad year that hits everyone, which is what makes 0.5 the honest baseline here."),
    (r"decay", r"^regression$|^logistic regression$",
     "A logistic regression on the standardized features (concentration, turnover, crowding, purchases and sales, trailing excess return, drawdown and recent alpha) is fit and tested the same way as the learned model, walk-forward with the same embargo. The tile is its average AUC by date. The Brier skill beside it is how much better its probabilities were than always predicting the base rate.",
     "This is the simple baseline the learned model must beat. If neither beats last year's laggards, the features carry nothing that last year's return did not already say."),
    (r"decay", r"^last year's laggards$|^last year's laggards lag again\?$",
     "No model is fit. At each date the managers are ranked by their excess return over the past twelve months, lowest first, and the AUC within the date measures how well that ranking predicted who lagged over the next twelve months.",
     "This is the question a trustee actually asks, whether to fire the ones that just did badly, as a number. It is also the benchmark for the two fitted models."),
    (r"decay", r"^can the filings predict next year\?$",
     "The verdict comes from the best t-statistic across the three predictors' average AUC by date, tested against 0.5. It says nothing is detectable unless that t-statistic clears 2.",
     "This is the one-line answer for the manager-selection process. On around 90 managers and a dozen independent years the test is weak, and the page says so; it is a scouting result, not a validated model."),
    (r"decay", r"^leave-managers-out, learned model$",
     "The data are split into groups by manager: the model is trained on some managers' whole histories and tested on the others'. Training and test therefore share calendar years. The AUC is computed over all the held-out rows pooled together.",
     "This is the leak, worked as a teaching example. It is how such a model would usually be validated, and it looks better than it is because the model learns which years were bad from the managers it trained on."),
    (r"decay", r"^leave-managers-out, logistic$",
     "The logistic regression is validated with the same split by manager, and the tile shows its pooled AUC.",
     "This shows that the leak flatters a simple model too. The optimism is in the validation design, not in the learned model's flexibility."),
    (r"decay", r"^walk-forward, learned model$",
     "This is the learned model's AUC over all its walk-forward predictions pooled together, where every test row was scored by a model that saw nothing from its year. It is pooled rather than within-date so that it can be compared directly with the leave-managers-out figure.",
     "This is the honest number on the same scale as the leaky one. The difference between them is the tile to the right."),
    (r"decay", r"^difference$",
     "This is the leave-managers-out AUC minus the walk-forward AUC for the learned model.",
     "This is what the leak is worth: the amount a manager-selection model would overstate its skill if it were validated the usual way. It is the reason the site reports walk-forward numbers only."),

    # ---------------------------------------------------------------- fund of funds
    (r"fund-of-funds", r"^candidates$",
     "These are the managers with a usable alpha and standard error, meaning enough months of history to be scored, entering the allocation.",
     "This is the opportunity set, and the spread of true skill is estimated across exactly these managers."),
    (r"fund-of-funds", r"^selected$",
     "This is the number of managers given a non-zero weight by the allocation, which maximises the blend's expected information ratio over long-only weights with a cap per manager, using the discounted alphas and the measured overlap between managers' returns.",
     "This is how many managers diversification actually rewards once alphas are discounted for noise. It is usually far fewer than a fund of funds holds."),
    (r"fund-of-funds", r"^blend ir, shrunk$|^expected information ratio$",
     "This is the expected information ratio of the chosen blend using the discounted alphas: the blend's alpha divided by its residual volatility, per year. Each alpha is discounted by how noisy it is relative to the spread of true skill across managers, and that spread is estimated from the data.",
     "This is the honest expectation for the fund of funds, not the back-tested one. When the estimated spread of true skill comes out at zero, as it does on this universe, every discounted alpha is zero and the ratio with it, which is the finding."),
    (r"fund-of-funds", r"^past top \d+$",
     "Managers are ranked by the t-statistic of their alpha fitted on the months before the split date, the top N are taken, and the tile shows the alpha of their equal-weight blend over the months after the split, with its t-statistic. The figure beside it is the same blend's alpha before the split.",
     "This is the selection test that matters: does picking on past alpha produce future alpha? The research literature says barely, and this reports what the public data show."),
    (r"fund-of-funds", r"^past bottom \d+$",
     "This is the same construction for the N managers with the lowest first-half alpha t-statistic.",
     "If the bottom group does as well as the top after the split, past alpha carried no information about future alpha."),
    (r"fund-of-funds", r"^top − bottom$",
     "The second-half alpha of the top group minus the bottom group's, from one regression on the difference of their monthly returns, with a t-statistic over the second-half months.",
     "This is the formal test. A t-statistic above 2 would say that past alpha predicted future alpha; below it, manager selection on the return record alone is not supported."),
    (r"fund-of-funds", r"^rank persistence$",
     "This is the rank correlation between each manager's first-half alpha and second-half alpha across all candidates.",
     "This uses every manager rather than just the extremes. Near zero means the ordering reshuffles, so the memo's verdicts must lean on process rather than on the ranking."),
]

# Model and signal names as the data files store them, and as the site says them.
PLAIN = [("xgboost (walk-forward)", "Learned model (walk-forward)"), ("Linear composite (equal-weight z)", "Simple average"),
         ("Persistence (no model)", "Last year's laggards"), ("Logistic regression", "Logistic regression"), ("xgboost", "learned model")]


def plain(text: str) -> str:
    """Rename a label read from a data file into the site's plain vocabulary."""
    for a, b in PLAIN:
        text = text.replace(a, b)
    return text


_COMPILED = [(re.compile(p, re.I) if p else None, re.compile(l, re.I), h, w) for p, l, h, w in NOTES]


def _norm(label: str) -> str:
    s = html.unescape(re.sub(r"<[^>]+>", "", label))
    return re.sub(r"\s+", " ", s).strip().lower()


def lookup(label: str, scope: str = "") -> tuple[str, str] | None:
    """(how, why) for a tile label on a page, or None."""
    lab = _norm(label)
    for page_re, lab_re, how, why in _COMPILED:
        if (page_re is None or page_re.search(scope)) and lab_re.search(lab):
            return how, why
    return None


def note_html(label: str, scope: str = "") -> str:
    """The click-to-expand note, or '' when the label has no entry."""
    hit = lookup(label, scope)
    if not hit:
        return ""
    how, why = hit
    return (f'<details class="how hw"><summary>How &amp; why</summary><div class="howb">'
            f'<p><b>How.</b> {how}</p><p><b>Why.</b> {why}</p></div></details>')


_TILE = re.compile(r"<div class=([\"'])tile(?: [^\"']*)?\1[^>]*>", re.I)
_LABEL = re.compile(r"<div class=([\"'])tl\1>(.*?)</div>", re.I | re.S)
_TAG = re.compile(r"<div\b|</div>", re.I)


def _tile_end(doc: str, start: int) -> int:
    """Index of the tile's closing </div> (nesting-aware)."""
    depth = 0
    for m in _TAG.finditer(doc, start):
        depth += 1 if m.group(0).lower().startswith("<div") else -1
        if depth == 0:
            return m.start()
    return -1


def annotate(doc: str, scope: str = "") -> str:
    """Append a note to every tile that has a label, has none yet, and has an entry for this page."""
    out, pos = [], 0
    for m in _TILE.finditer(doc):
        if m.start() < pos:
            continue
        end = _tile_end(doc, m.start())
        if end < 0:
            break
        tile = doc[m.start():end]
        lab = _LABEL.search(tile)
        if lab and "<details" not in tile:
            note = note_html(lab.group(2), scope)
            if note:
                out.append(doc[pos:end]); out.append(note); pos = end
    out.append(doc[pos:])
    return "".join(out)


def tiles(doc: str) -> list[str]:
    """The HTML of every labelled tile in a rendered page (for the coverage test)."""
    out = []
    for m in _TILE.finditer(doc):
        t = doc[m.start():_tile_end(doc, m.start())]
        if _LABEL.search(t):
            out.append(t)
    return out


def tile_labels(doc: str) -> list[str]:
    return [_norm(_LABEL.search(t).group(2)) for t in tiles(doc)]


CSS = """<style>
details.how{margin-top:10px;font-size:13px}
details.how summary{cursor:pointer;color:var(--navy,#1b2a41);font:600 9.5px/1 "Helvetica Neue",Helvetica,Arial,sans-serif;letter-spacing:.16em;text-transform:uppercase;list-style:none;display:inline-flex;align-items:center;gap:6px}
details.how summary::-webkit-details-marker{display:none}
details.how summary::before{content:"";width:5px;height:5px;border-right:1px solid currentColor;border-bottom:1px solid currentColor;transform:rotate(-45deg)}
details.how[open] summary::before{transform:rotate(45deg)}
details.how summary:focus-visible{outline:2px solid var(--gold,#b08d57);outline-offset:2px}
details.how .howb{margin-top:8px;line-height:1.5;color:var(--ink2,#3a3f47)}
details.how .howb p{margin:0 0 6px}
@media(prefers-color-scheme:dark){details.how summary{color:var(--gold-l,#c9b48a)}}
</style>"""
