#!/usr/bin/env Rscript
# Portfolio construction LP — the R twin of trackrecord/construct.py::solve_lp.
#
# Same inputs, same problem, solved with Rglpk (GLPK simplex) instead of HiGHS:
#   maximise  alpha'w − cost·Σ(buy + sell)
#   s.t.      Σw = 1;  w − buy + sell = w0;  Σ(buy + sell) ≤ 2·turnover
#             w − over + under = b;  Σ(over + under) ≤ 2·active_share
#             sector bands vs benchmark;  0 ≤ w ≤ min(max(max_weight, b), b + active_band), w ≥ b − active_band
#             names leaving the universe forced to 0
#
#   Rscript r/construct.R data/research/construction/inputs_latest
#
# Reads universe.csv (ticker, alpha, bench, sector, w0), leaving.csv (ticker, w0) and
# constraints.csv (key,value); writes weights_r.csv.  The Python side compares the two solutions.
suppressPackageStartupMessages({ library(data.table); library(Rglpk) })

args <- commandArgs(trailingOnly = TRUE)
dir  <- if (length(args) >= 1) args[1] else "data/research/construction/inputs_latest"

uni <- fread(file.path(dir, "universe.csv"))
lv  <- if (file.exists(file.path(dir, "leaving.csv"))) fread(file.path(dir, "leaving.csv")) else data.table(ticker = character(), w0 = numeric())
cc  <- fread(file.path(dir, "constraints.csv"))            # key,value — no jsonlite dependency
con <- as.list(setNames(cc$value, cc$key))

# ---- assemble the variable table: in-universe names first, then names being forced out
dt <- rbind(uni[, .(ticker, alpha, bench, sector, w0, in_universe = TRUE)],
            lv[nrow(lv) > 0 & !ticker %in% uni$ticker, .(ticker, alpha = 0, bench = 0, sector = "Leaving", w0, in_universe = FALSE)],
            fill = TRUE)
dt[is.na(sector) | sector == "", sector := "Unknown"]
n    <- nrow(dt)
cost <- con$cost_bps / 1e4

# ---- objective over x = [w, buy, sell, over, under]  (Rglpk maximises when max = TRUE)
Z   <- rep(0, n)
obj <- c(dt$alpha, rep(-cost, n), rep(-cost, n), Z, Z)

# ---- constraints
I <- diag(n)
rows <- list(); dir_ <- character(); rhs <- numeric()
add <- function(row, d, r) { rows[[length(rows) + 1]] <<- row; dir_ <<- c(dir_, d); rhs <<- c(rhs, r) }
add(c(rep(1, n), rep(0, 4 * n)), "==", 1)                                       # fully invested
for (i in seq_len(n)) add(c(I[i, ], -I[i, ], I[i, ], Z, Z), "==", dt$w0[i])      # w − buy + sell = w0
for (i in seq_len(n)) add(c(I[i, ], Z, Z, -I[i, ], I[i, ]), "==", dt$bench[i])   # w − over + under = b
add(c(Z, rep(1, 2 * n), Z, Z), "<=", 2 * con$turnover)                          # turnover budget
add(c(Z, Z, Z, rep(1, 2 * n)), "<=", 2 * con$active_share)                      # active-share cap
for (s in sort(unique(dt$sector))) {                                             # sector bands
  m  <- as.numeric(dt$sector == s); bs <- sum(dt$bench[dt$sector == s])
  add(c(m, rep(0, 4 * n)), "<=", bs + con$sector_band)
  add(c(m, rep(0, 4 * n)), ">=", bs - con$sector_band)
}
mat <- do.call(rbind, rows)

# ---- bounds: active band and name cap on w; leaving names pinned at 0; buy/sell ≥ 0
lo <- ifelse(dt$in_universe, pmax(0, dt$bench - con$active_band), 0)
hi <- ifelse(dt$in_universe, pmin(pmax(con$max_weight, dt$bench), dt$bench + con$active_band), 0)
bounds <- list(lower = list(ind = seq_len(5 * n), val = c(lo, rep(0, 4 * n))),
               upper = list(ind = seq_len(5 * n), val = c(hi, rep(Inf, 4 * n))))

sol <- Rglpk_solve_LP(obj, mat, dir_, rhs, bounds = bounds, max = TRUE)
if (sol$status != 0) stop("GLPK status ", sol$status)

w <- sol$solution[seq_len(n)]; w[w < 1e-9] <- 0; w <- w / sum(w)
out <- data.table(ticker = dt$ticker, weight = w)[weight > 0][order(-weight)]
fwrite(out, file.path(dir, "weights_r.csv"))
cat(sprintf("Rglpk: %d names, expected alpha %.4f, turnover %.3f, objective %.4f\n",
            nrow(out), sum(w * dt$alpha), sum(abs(w - dt$w0)) / 2, sol$optimum))
