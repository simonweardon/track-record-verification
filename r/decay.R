#!/usr/bin/env Rscript
# Manager decay model — the R twin of trackrecord/decay.py.
#
# Rebuilds the manager × formation-date panel from the raw inputs with data.table:
#   data/reference/compact/holdings13f.csv.gz   576k 13F positions → books, concentration, turnover,
#                                               new / sold names, book growth, crowding
#   data/funds/<slug>/statements.csv            each clone's monthly values → trailing and forward returns
#   <out>/factors.csv                           US market and FF3 factors (exported by Python from Ken French)
# then re-runs the walk-forward — persistence, glm (binomial) and xgboost — with the same embargo,
# and writes panel_r.csv and summary_r.csv for Python to compare.
#
#   Rscript r/decay.R data/research/decay
#
# The panel and the logistic predictions should agree with Python to rounding.  xgboost differs
# slightly by construction (subsampling uses each library's own random stream); its AUC is reported
# side by side, not asserted equal.
suppressPackageStartupMessages(library(data.table))
has_xgb <- requireNamespace("xgboost", quietly = TRUE)

args <- commandArgs(trailingOnly = TRUE)
out  <- if (length(args) >= 1) args[1] else "data/research/decay"

HORIZON <- 12; MIN_TRAIN_DATES <- 12; MIN_BOOK <- 5
FILING  <- c("n_positions", "hhi", "top1_w", "top10_w", "turnover", "new_frac", "sold_frac", "book_growth", "crowding")
RETURNS <- c("excess_12", "excess_36", "vol_12", "mdd_12", "alpha36_t")
FEATS   <- c(FILING, RETURNS)

month_end <- function(d) as.Date(format(as.Date(d) + 32 - as.integer(format(as.Date(d), "%d")), "%Y-%m-01")) - 1
formation <- function(period) month_end(month_end(as.Date(period) + 1) + 1)     # quarter end + 2 months, month end
prev_quarter <- function(period) {                                             # quarter end three months earlier
  d <- as.Date(period); y <- as.integer(format(d, "%Y")); m <- as.integer(format(d, "%m")) - 3
  y <- ifelse(m <= 0, y - 1, y); m <- ifelse(m <= 0, m + 12, m)
  month_end(as.Date(sprintf("%d-%02d-01", y, m)))
}

# ---- books: one row per (slug, period, cusip), as load_books() ---------------------------------
h <- fread(cmd = "gzip -dc data/reference/compact/holdings13f.csv.gz", colClasses = list(character = c("period", "filed", "cusip", "cik", "putcall", "cls")))
h <- h[putcall == "" & value > 0]
filed <- h[, .(filed = max(filed)), by = .(slug, period)]
g <- h[, .(value = sum(value), shares = sum(shares)), by = .(slug, period, cusip)]
g <- filed[g, on = .(slug, period)]
g[, w_all := value / sum(value), by = .(slug, period)]
setorder(g, slug, period, -value)
g[, rank := seq_len(.N), by = .(slug, period)]
g[, n_book := .N, by = .(slug, period)]
g[, formation := formation(period)]
g <- g[as.Date(filed) <= formation & n_book >= MIN_BOOK]
g[, prev_period := format(prev_quarter(period), "%Y-%m-%d")]
have <- unique(g[, .(slug, prev_period = period)])[, has_prev := TRUE]
g <- have[g, on = .(slug, prev_period)][is.na(has_prev), has_prev := FALSE]
g <- g[, .(slug, prev_period = period, cusip, shares_prev = shares)][g, on = .(slug, prev_period, cusip)]
g[, new := has_prev & is.na(shares_prev)]
cat(sprintf("books: %d managers, %d quarters, %d positions\n", uniqueN(g$slug), uniqueN(g$period), nrow(g)))

# ---- filing features --------------------------------------------------------------------------
g[, holders := uniqueN(slug), by = .(period, cusip)]
f <- g[, .(n_positions = .N, hhi = sum(w_all^2), top1_w = max(w_all), top10_w = sum(w_all[rank <= 10]),
           crowding = sum(w_all * holders), value = sum(value), new_frac = mean(new), has_prev = has_prev[1],
           prev_period = prev_period[1], formation = formation[1], filed = filed[1]), by = .(slug, period)]
prev <- g[, .(slug, prev_period = period, cusip, w_prev = w_all)]
ov <- prev[g[, .(slug, period, prev_period, cusip, w_all)], on = .(slug, prev_period, cusip)]
ov[is.na(w_prev), w_prev := 0]
f <- ov[, .(overlap = sum(pmin(w_all, w_prev))), by = .(slug, period)][f, on = .(slug, period)]
f[, turnover := ifelse(has_prev, 1 - overlap, NA_real_)]
sets <- g[, .(names = list(cusip)), by = .(slug, period)]
f <- sets[, .(slug, period, cur = names)][f, on = .(slug, period)]
f <- sets[, .(slug, prev_period = period, prv = names)][f, on = .(slug, prev_period)]
f[, sold_frac := mapply(function(hp, a, b) if (!hp || length(b) == 0) NA_real_ else length(setdiff(b, a)) / length(b), has_prev, cur, prv)]
f <- f[, .(slug, prev_period = period, value_prev = value)][f, on = .(slug, prev_period)]
f[, growth := log(value / value_prev)]
f[, growth := growth - round(growth / log(1000)) * log(1000)]
f[, book_growth := ifelse(has_prev, pmin(pmax(growth, -2), 2), NA_real_)]
f[, n_positions := log(n_positions)]
f[, new_frac := ifelse(has_prev, new_frac, NA_real_)]
f <- f[, c("slug", "period", "formation", "filed", FILING), with = FALSE]

# ---- clone returns and factors ----------------------------------------------------------------
fac <- fread(file.path(out, "factors.csv"))
fac[, month := as.Date(month)]
files <- list.files("data/funds", "statements.csv", recursive = TRUE, full.names = TRUE)
R <- rbindlist(lapply(files, function(p) {
  s <- fread(p)[order(period_end)]
  data.table(slug = basename(dirname(p)), month = month_end(as.Date(s$period_end)), r = c(NA, diff(s$ending_value) / head(s$ending_value, -1)))
}))[!is.na(r)]
months <- sort(unique(c(R$month, fac$month)))
R <- CJ(slug = unique(R$slug), month = months)[R, on = .(slug, month), r := i.r][fac, on = "month", `:=`(mkt = i.MKT, mkt_rf = i.MKT_RF, smb = i.SMB, hml = i.HML, rf = i.RF)]
setorder(R, slug, month)
roll_sum <- function(x, n) { s <- frollsum(x, n, align = "right"); s }        # NA unless the whole window is present
mdd <- function(x) { c <- cumprod(1 + x); min(c / cummax(c) - 1) }
R[, `:=`(lp = log1p(r), lm = log1p(mkt))]
R[, `:=`(excess_12 = expm1(roll_sum(lp, 12)) - expm1(roll_sum(lm, 12)),
         excess_36 = expm1(roll_sum(lp, 36)) - expm1(roll_sum(lm, 36)),
         vol_12 = frollapply(r, 12, sd) * sqrt(12),
         mdd_12 = frollapply(r, 12, mdd)), by = slug]
R[, `:=`(fwd_excess = shift(expm1(roll_sum(lp, HORIZON)) - expm1(roll_sum(lm, HORIZON)), HORIZON, type = "lead"),
         fwd_return = shift(expm1(roll_sum(lp, HORIZON)), HORIZON, type = "lead")), by = slug]

# Newey-West HAC t of the FF3 intercept, exactly as verify.R / statsmodels (Bartlett, L = floor(0.75 n^(1/3)), no correction)
hac_t <- function(y, X) {
  Xc <- cbind(1, X); n <- nrow(Xc)
  XtXi <- solve(crossprod(Xc)); b <- as.vector(XtXi %*% crossprod(Xc, y)); u <- as.vector(y - Xc %*% b)
  L <- floor(0.75 * n^(1 / 3)); xu <- Xc * u; S <- crossprod(xu)
  for (l in seq_len(L)) { s <- crossprod(xu[(l + 1):n, , drop = FALSE], xu[1:(n - l), , drop = FALSE]); S <- S + (1 - l / (L + 1)) * (s + t(s)) }
  V <- XtXi %*% S %*% XtXi; b[1] / sqrt(V[1, 1])
}
R[, y_ex := r - rf]
R[, alpha36_t := {
  out <- rep(NA_real_, .N)
  for (i in seq_len(.N)) if (i >= 36) {
    w <- (i - 35):i
    if (!anyNA(y_ex[w]) && !anyNA(mkt_rf[w]) && !anyNA(smb[w]) && !anyNA(hml[w])) out[i] <- hac_t(y_ex[w], cbind(mkt_rf[w], smb[w], hml[w]))
  }
  out
}, by = slug]

# ---- panel ------------------------------------------------------------------------------------
P <- R[, c("slug", "month", RETURNS, "fwd_excess", "fwd_return"), with = FALSE][f, on = .(slug, month = formation)]
setnames(P, "month", "formation")
P <- P[!is.na(excess_12)]
P[, y := ifelse(is.na(fwd_excess), NA_real_, as.numeric(fwd_excess < 0))]
setorder(P, formation, slug)
cat(sprintf("panel: %d managers, %d formation dates, %d rows, %d labelled, base rate %.0f%%\n",
            uniqueN(P$slug), uniqueN(P$formation), nrow(P), sum(!is.na(P$y)), 100 * mean(P$y, na.rm = TRUE)))

# ---- walk-forward -----------------------------------------------------------------------------
P[, `:=`(p_persist = NA_real_, p_base = NA_real_, p_logistic = NA_real_, p_xgboost = NA_real_)]
add_months <- function(d, n) month_end(as.Date(format(as.Date(d), "%Y-%m-01")) + 31 * n)
P[, closes := add_months(formation, HORIZON)]
xgb_params <- list(objective = "binary:logistic", max_depth = 3, eta = 0.05, subsample = 0.8, colsample_bytree = 0.8, min_child_weight = 20, lambda = 5, nthread = 4)
n_fits <- 0
for (F in sort(unique(P$formation))) {
  train <- P[closes <= F & !is.na(y)]
  if (uniqueN(train$formation) < MIN_TRAIN_DATES) next
  test_i <- which(P$formation == F); test <- P[test_i]
  P[test_i, p_persist := frank(-fifelse(is.na(excess_12), 0, excess_12), ties.method = "average") / .N]
  P[test_i, p_base := mean(train$y)]
  mu <- train[, lapply(.SD, mean, na.rm = TRUE), .SDcols = FEATS]; sdv <- train[, lapply(.SD, sd, na.rm = TRUE), .SDcols = FEATS]
  z <- function(d) { m <- as.matrix(d[, FEATS, with = FALSE]); m <- sweep(sweep(m, 2, as.numeric(mu)), 2, ifelse(as.numeric(sdv) == 0, 1, as.numeric(sdv)), "/"); m[is.na(m)] <- 0; m }
  Xtr <- z(train); Xte <- z(test)
  fit <- suppressWarnings(glm.fit(cbind(1, Xtr), train$y, family = binomial()))
  b <- fit$coefficients; b[is.na(b)] <- 0                                       # an aliased column gets weight 0
  P[test_i, p_logistic := as.numeric(1 / (1 + exp(-(cbind(1, Xte) %*% b))))]
  if (has_xgb) {
    set.seed(0)
    dtr <- xgboost::xgb.DMatrix(as.matrix(train[, FEATS, with = FALSE]), label = train$y, missing = NA)
    m <- xgboost::xgb.train(params = xgb_params, data = dtr, nrounds = 200, verbose = 0)
    P[test_i, p_xgboost := predict(m, xgboost::xgb.DMatrix(as.matrix(test[, FEATS, with = FALSE]), missing = NA))]
  }
  n_fits <- n_fits + 1
}
cat(sprintf("walk-forward: %d fits%s\n", n_fits, if (has_xgb) "" else " (xgboost not installed: glm only)"))

# ---- evaluation: pooled AUC and mean cross-sectional AUC ----------------------------------------
auc <- function(y, s) { ok <- !is.na(s); y <- y[ok] == 1; s <- s[ok]; n1 <- sum(y); n0 <- sum(!y)
  if (n1 == 0 || n0 == 0) return(NA_real_); r <- rank(s); (sum(r[y]) - n1 * (n1 + 1) / 2) / (n1 * n0) }
T <- P[!is.na(y) & !is.na(p_logistic)]
models <- c("persist", "logistic", if (has_xgb) "xgboost")
S <- rbindlist(lapply(models, function(m) {
  col <- paste0("p_", m)
  bd <- T[, .(auc = auc(y, get(col))), by = formation]
  data.table(model = m, n_rows = nrow(T), n_dates = nrow(bd), auc_pooled = auc(T$y, T[[col]]), auc_cs_mean = mean(bd$auc, na.rm = TRUE))
}))
fwrite(S, file.path(out, "summary_r.csv"))
fwrite(P[, c("slug", "period", "formation", FEATS, "fwd_excess", "fwd_return", "y", "p_persist", "p_base", "p_logistic", "p_xgboost"), with = FALSE],
       file.path(out, "panel_r.csv"))
for (i in seq_len(nrow(S))) cat(sprintf("R %-9s pooled AUC %.3f, mean cross-sectional AUC %.3f over %d dates\n", S$model[i], S$auc_pooled[i], S$auc_cs_mean[i], S$n_dates[i]))
