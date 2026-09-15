#!/usr/bin/env Rscript
# Independent reproduction of the headline numbers — the R twin of trackrecord/validation.py.
#
# Reads only output/<fund>/phase4/aligned_data.csv (the aligned excess returns and factors)
# and recomputes, in data.table + base R, what Python reported in the same folder:
#   CAPM / FF3 / Carhart4 / FF5 alpha with Newey-West HAC standard errors and t
#   annualized return, volatility, Sharpe, max drawdown
#   the alpha-maxing score
# then compares against regressions.csv, metrics.csv and scores.csv and writes verify_r.csv.
#
#   Rscript r/verify.R output/funds/appaloosa/phase4 [factor_set]
#
# HAC matches statsmodels' cov_type="HAC" exactly: Bartlett weights w_l = 1 - l/(L+1),
# L = floor(0.75 * n^(1/3)), NO small-sample correction, and normal (not t) p-values and CIs.
# Annual cells get plain OLS with t-distribution p-values, as Python does.
suppressPackageStartupMessages(library(data.table))

args <- commandArgs(trailingOnly = TRUE)
dir  <- if (length(args) >= 1) args[1] else "output/funds/appaloosa/phase4"

MODELS <- list(CAPM      = c("MKT_RF"),
               FF3       = c("MKT_RF", "SMB", "HML"),
               Carhart4  = c("MKT_RF", "SMB", "HML", "MOM"),
               FF5       = c("MKT_RF", "SMB", "HML", "RMW", "CMA"))

al   <- fread(file.path(dir, "aligned_data.csv"))
regs <- fread(file.path(dir, "regressions.csv"))
mets <- fread(file.path(dir, "metrics.csv"))
scrs <- fread(file.path(dir, "scores.csv"))

# aligned_data.csv holds the primary factor set — the one regressions.csv lists first.
fset <- if (length(args) >= 2) args[2] else as.character(regs$factor_set[1])
# monthly cells look like 2017-07; anything else is treated as annual, exactly as Python does
ppy  <- if (grepl("^[0-9]{4}-[0-9]{2}$", as.character(al$cell[1]))) 12 else 1

# ---- OLS with Newey-West HAC errors ------------------------------------------------
# V = (X'X)^-1 S (X'X)^-1 with S = S_0 + sum_l w_l (S_l + S_l'), S_l = sum_t u_t u_{t-l} x_t x_{t-l}'
fit_hac <- function(y, X, ppy) {
  Xc <- cbind(1, as.matrix(X)); n <- nrow(Xc); k <- ncol(Xc)
  XtXi <- solve(crossprod(Xc))
  b    <- as.vector(XtXi %*% crossprod(Xc, y))
  u    <- as.vector(y - Xc %*% b)
  if (ppy > 1) {
    L  <- floor(0.75 * n^(1 / 3))
    xu <- Xc * u
    S  <- crossprod(xu)
    for (l in seq_len(L)) {
      s <- crossprod(xu[(l + 1):n, , drop = FALSE], xu[1:(n - l), , drop = FALSE])
      S <- S + (1 - l / (L + 1)) * (s + t(s))
    }
    V  <- XtXi %*% S %*% XtXi
    se <- sqrt(diag(V)); tv <- b / se
    p  <- 2 * pnorm(-abs(tv)); q <- qnorm(0.975)          # normal: statsmodels use_t=False under HAC
  } else {
    L  <- 0
    V  <- sum(u^2) / (n - k) * XtXi
    se <- sqrt(diag(V)); tv <- b / se
    p  <- 2 * pt(-abs(tv), df = n - k); q <- qt(0.975, df = n - k)
  }
  rss <- sum(u^2); tss <- sum((y - mean(y))^2)
  list(n = n, hac_lag = L, coef = b, se = se, t = tv, p = p,
       ci_low = b - q * se, ci_high = b + q * se,
       r2 = 1 - rss / tss, r2_adj = 1 - (rss / (n - k)) / (tss / (n - 1)),
       resid_sd = sqrt(rss / (n - k)))
}

# ---- metrics (mirroring validation.risk_metrics / validation.scores) ----------------
ann_ret <- function(r, ppy) prod(1 + r)^(1 / (length(r) / ppy)) - 1
max_dd  <- function(r) { r[is.na(r)] <- 0; idx <- cumprod(1 + r); min(idx / cummax(idx) - 1) }

rows <- list()
add  <- function(quantity, model, r_value, py_value)
  rows[[length(rows) + 1]] <<- data.table(quantity, model, r_value, py_value,
                                          abs_diff = abs(r_value - py_value))

for (m in names(MODELS)) {
  facs <- MODELS[[m]]
  if (!all(facs %in% names(al))) next
  sub  <- al[complete.cases(al[, c("y", facs), with = FALSE]), c("y", facs), with = FALSE]
  if (nrow(sub) < length(facs) + 3) next
  f   <- fit_hac(sub$y, sub[, facs, with = FALSE], ppy)
  py  <- regs[factor_set == fset & model == m]
  if (!nrow(py)) next
  add("alpha_annual", m, f$coef[1] * ppy, py$alpha_annual)
  add("se_annual",    m, f$se[1] * ppy,   py$se_annual)
  add("t_alpha",      m, f$t[1],          py$t)
  add("p_alpha",      m, f$p[1],          py$p)
  add("ci_low_annual",  m, f$ci_low[1] * ppy,  py$ci_low_annual)
  add("ci_high_annual", m, f$ci_high[1] * ppy, py$ci_high_annual)
  add("r2",           m, f$r2,            py$r2)
  add("hac_lag",      m, f$hac_lag,       py$hac_lag)
  for (i in seq_along(facs)) add(paste0("b_", facs[i]), m, f$coef[i + 1], py[[paste0("b_", facs[i])]])
}

# risk metrics use the cells with a portfolio return, a benchmark return and an RF
d <- al[!is.na(r_p) & !is.na(r_b) & !is.na(RF)]
pyn <- function(key) {                       # portfolio column of metrics.csv, matched by prefix
  r <- mets[startsWith(metric, key)]
  as.numeric(r$portfolio[1])
}
add("annualized return",     "-", ann_ret(d$r_p, ppy),              pyn("annualized return"))
add("annualized volatility", "-", sd(d$r_p) * sqrt(ppy),            pyn("annualized volatility"))
add("Sharpe",                "-", mean(d$y) / sd(d$y) * sqrt(ppy),  pyn("Sharpe (excess over RF)"))
add("max drawdown",          "-", max_dd(d$r_p),                    pyn("max drawdown"))

# the alpha-maxing score drops only r_p / r_b (RF is not needed for an excess-over-market number)
e      <- al[!is.na(r_p) & !is.na(r_b)]
excess <- ann_ret(e$r_p, ppy) - ann_ret(e$r_b, ppy)
add("excess return over market", "-", excess, as.numeric(scrs[score == "Alpha-maxing score"]$input[1]))
add("alpha-maxing score", "-", min(100, max(0, 50 + 10 * excess * 100)),
    as.numeric(scrs[score == "Alpha-maxing score"]$value[1]))

out <- rbindlist(rows)
out[, rel_diff := abs_diff / pmax(abs(py_value), 1e-12)]
fwrite(out, file.path(dir, "verify_r.csv"))

worst <- out[which.max(abs_diff)]
cat(sprintf("R verify [%s, %s factors, %d cells]: %d quantities, max |R - Python| %.3e (%s %s)\n",
            basename(dirname(dir)), fset, nrow(al), nrow(out), worst$abs_diff, worst$quantity, worst$model))
