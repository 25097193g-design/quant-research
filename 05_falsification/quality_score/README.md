# Quality Score — A Fundamental Screen That Failed as a Factor

> 中文版：[README.zh.md](README.zh.md) · Reports: `评分报告_v1.md`, `v4因子证伪复盘_2023-2026.md`, `弱环修复报告_v2.md`

---

## What it is

An 8-dimension fundamental score over the A-share universe, iterated v1 → v4:

profitability · moat · cash generation · growth · leverage · operations · safety · valuation

plus a **red-flag veto** (three consecutive years of ROE < 8%, or two consecutive years of negative operating cash flow).

| Version | Universe |
|---|---|
| v1 | 7,478 names (incl. BSE and pre-IPO coaching names) |
| v4 | 4,885 names — adds multi-year trend and a growth dimension |

## What it turned out to be

**A screen, not a factor.**

- **As a ranking signal: IC ≈ 0** over 2023–26 monthly cross-sections. It is not a market-cap proxy (ρ = 0.18 with size) and not a negative-carry signal — it is uninformative.
- **As a red-flag veto: some value.** Red-flagged names returned **−6.9%** against **−2.1%** for the rest over the validation window, i.e. the flag avoids some losses even though the score cannot rank.

That distinction is why `01_factor_framework/` uses quality as a **pool filter** and takes its alpha from price-volume factors instead.

## The falsification process (the actually useful part)

`v4因子证伪复盘_2023-2026.md` records five stages of "hypothesis → test → conclusion", where the conclusion was overturned more than once:

1. **Endpoint bug** — a month-start anchor missed 2026-06, inflating medians by 14–18pp (from −7% to +11%). Fixed by switching to month-end.
2. **Look-ahead is small** — removing look-ahead changes quantile spreads by < 0.5%, because quality is a slow-moving variable and 2024/2025 scores are highly correlated.
3. **The real culprit was the validation regime** — moving the validation start from 2025-06 to 2025-01 flips the conclusion entirely.
4. **Direct test** — monthly IC with and without market-cap neutralisation ⇒ ≈ 0.
5. **Final ruling** — invalid as a factor.

`弱环修复报告_v2.md` documents the clean-up of the weak links in the scoring system (data coverage, industry classification, score distribution).

## Files

| File | Content |
|---|---|
| `评分报告_v1.md` | v1 scoring report (universe, dimensions, distribution) |
| `v4因子证伪复盘_2023-2026.md` | The five-stage falsification review |
| `弱环修复报告_v2.md` | Weak-link repair report |

## Data

`akshare` + `rqdatac` fundamentals, point-in-time. Not redistributable, so only the reports are published here.
