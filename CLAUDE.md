# CLAUDE.md

Guidance for Claude Code (and future contributors) working in this repository.

## What this application does

**Fair Value Betting — NFL** is a Streamlit web app that helps users find positive-expected-value
(+EV) NFL bets. It pulls live odds from many sportsbooks, computes a weighted no-vig "fair" line
per market, and surfaces where the best available sportsbook price beats that fair line — plus
line shopping, arbitrage detection, and NFL player/fantasy research tools.

This is the **NFL** edition of a multi-sport product family. Sibling apps exist for MLB and NCAAF
(linked from the sidebar: fairvaluebetting.com, mlb.fairvaluebetting.com, ncaaf.fairvaluebetting.com).
Expect architectural patterns here to be shared with (and sometimes ported from) those sibling repos —
comments in the code reference "the MLB pattern" in a few places.

## Tech stack

- **Frontend/runtime:** Streamlit (single-page app, `st.tabs` for navigation)
- **Auth & database:** Supabase (Postgres + auth). `supabase-py` client.
- **Session persistence:** `streamlit-cookies-manager` (`EncryptedCookieManager`) stores
  Supabase access/refresh tokens in an encrypted cookie for "keep me logged in."
- **Data processing:** pandas
- **NFL stats data:** `nflreadpy` (nflverse) for historical player/team stats
- **Odds data:** [The Odds API](https://the-odds-api.com/) (`fetch_odds_nfl.py`, external cron/script — not run by the Streamlit app itself)
- **Fantasy ADP data:** static `data/fantasy_adp.xlsx` (multi-sheet: PPR / Half PPR / STD)
- **Live scores:** ESPN's public (unofficial) scoreboard/search JSON endpoints
- **Email:** MailerSend (`mailer_send.py`, standalone script, not wired into the app)
- **Notebook:** `EVSportsBet_NFL.ipynb` — exploratory/scratch notebook, not part of the running app

## Architecture

```
Odds API ──(fetch_odds_nfl.py, run externally on a schedule)──> Supabase (odds_snapshots, odds_lines)
                                                                        │
                                                              app.py (Streamlit)
                                                                        │
                                                        core/data_sources.py (cached reads)
                                                                        │
                                                          core/pipeline.py (fair value engine)
                                                                        │
                                          tabs/*.py (one module per tab, calls into core/*)
```

**Key idea:** `fetch_odds_nfl.py` is a separate, out-of-band script (presumably run on a cron/
scheduler outside this app — e.g. GitHub Actions, a serverless cron, etc.) that writes raw odds
into Supabase. The Streamlit app **never calls the Odds API directly** for the fair-value/EV
tables — it only reads what's already in Supabase via `core/data_sources.py`. Only
`core/lineup_data.py` (player props for Lineup Analysis) touches `ODDS_API_KEY` directly from
within the app.

nflverse/ESPN data, by contrast, **is** fetched live from within the running app (cached via
`st.cache_data`).

## Important files and directories

| Path | Purpose |
|---|---|
| `app.py` | Entry point. Auth (Supabase + cookies), branding, sidebar (How to use / Glossary / Feedback / Disclaimer), tab registration, `run_app()`. |
| `core/pipeline.py` | **The core fair-value engine.** One generic `MarketConfig`-driven pipeline shared by Moneyline/Spread/Total: build books → weighted no-vig consensus → best prices → market intelligence → display rows. |
| `core/odds_math.py` | Pure math/formatting helpers (American↔decimal↔implied prob, EV%, Kelly fraction, date/odds formatting). No Streamlit, no I/O — shared by pipeline and every tab. |
| `core/arbitrage_engine.py` | Arbitrage detection. Runs **on top of** `run_market_pipeline`'s output (best price per side across all books) — not a separate raw-odds fetch. |
| `core/data_sources.py` | Supabase odds readers (`@st.cache_data`, 300s TTL) + all NFL week/date-window logic (season start date, week index inference, date-range resolution for the UI selectors). |
| `core/nflverse_data.py` | nflreadpy-backed player/team season & weekly stats, position-specific metric maps, recency-weighted blending for props. |
| `core/nfl_defense_data.py` | Opponent defensive context (yards/TDs allowed per position), derived from nflverse team_stats. |
| `core/nfl_live_scores.py` | ESPN scoreboard polling for live game status. |
| `core/nfl_player_card.py`, `nfl_player_context.py`, `nfl_player_search.py` | Shared UI-building-block components reused across Lineup Analysis and Prop Leaderboard (see "Naming conventions" — the `render_*` pattern). |
| `core/lineup_data.py` | ESPN player search, team/position maps, player-prop market definitions; also the app's other direct consumer of `ODDS_API_KEY`. |
| `core/fantasy_data.py` | Loads/cleans/parses `data/fantasy_adp.xlsx` into ranking tables; regex-based parsing of the raw sheet's "Player (Bye)" / "POS" columns. |
| `core/prop_hit_rate_dashboard.py` | Shared prop hit-rate / streak dashboard, reused by NFL Prop Leaderboard (comment notes it's also shared with an MLB equivalent). |
| `tabs/*.py` | One module per tab; each exposes a `render(...)` function called from `app.py`. |
| `fetch_odds_nfl.py` | **External** odds-ingestion script — hits The Odds API, writes `odds_snapshots` + `odds_lines` to Supabase. Not imported by the Streamlit app. Run via `python3 fetch_odds_nfl.py`. |
| `mailer_send.py` | Standalone MailerSend transactional-email script. Not imported by the app; has hardcoded test emails in its `__main__` block — treat as a manual/CLI utility only. |
| `data/fantasy_adp.xlsx` | Static fantasy ADP source data (sheets: PPR, Half PPR, STD). |
| `.streamlit/config.toml` | Theme (light, primary `#4A79BD`). |
| `.streamlit/assets/` | `logo.png`, `favicon.png`. |
| `EVSportsBet_NFL.ipynb` | Scratch/exploration notebook — not part of the production app path. |

## Major features and tabs

Registered in `app.py` (`run_app()`), in this order:

1. **Market Movers** (`tabs/market_movers.py`) — weekly snapshot: game/book/market counts, latest odds pull time, top 3 EV plays.
2. **Fair Value Model** (`tabs/fair_value_model.py`) — the main product: filterable table of Best Odds vs. Fair Odds across Moneyline/Spread/Total, by date range, EV%, and fair-win%. **Gated behind sign-in** (`authed` check).
3. **Matchup Center** (`tabs/matchup_center.py`) — single-game deep dive.
4. **Lineup Analysis** (`tabs/lineup_comparison.py`) — player usage/props/matchup research; compare up to 4 players.
5. **Fantasy Draft** (`tabs/fantasy_draft.py`) — consensus ADP rankings from `data/fantasy_adp.xlsx`, filterable by position.
6. **Prop Research** (`tabs/prop_leaderboard.py`) — individual player props + best current hit-rate streaks.
7. **Sportsbook Screener** (`tabs/sportsbook_screener.py`) — pure line shopping.
8. **Parlay Builder** (`tabs/parlay_builder.py`) — multi-leg parlay comparison across books.
9. **Arbitrage Tracker** (`tabs/arbitrage_tracker.py`) — guaranteed-profit two-sided price mismatches, built on `core/arbitrage_engine.py`.

`tabs/player_research.py` exists but is **not registered in `app.py`** — it's an explicit stub
("pending a data source decision"). Don't assume it's live; don't wire it in without being asked.

## APIs and external data sources

- **The Odds API** (`the-odds-api.com`) — raw sportsbook odds. Fetched only by `fetch_odds_nfl.py`
  (batch, external) and `core/lineup_data.py` (player props, live from the app). Requires `ODDS_API_KEY`.
- **Supabase** — system of record for odds (`odds_snapshots`, `odds_lines` tables) and auth/user
  accounts (plus a `feedback` table for the sidebar feedback form). Requires `SUPABASE_URL` +
  `SUPABASE_ANON_KEY` (app, read-only-ish via RLS presumably) and `SUPABASE_KEY` (ingestion script,
  service-role or write-capable key — **do not confuse the two**).
- **nflreadpy / nflverse** — historical player & team stats (season, weekly). No API key; installed as a Python package.
- **ESPN public/unofficial endpoints** — live scoreboard (`site.api.espn.com/.../scoreboard`) and
  player search (`site.web.api.espn.com/.../search`). No key, but unofficial/undocumented — subject
  to breaking without notice; treat responses defensively (existing code already wraps in try/except
  and checks status codes).
- **MailerSend** — transactional email, `mailer_send.py` only. Requires `MAILERSEND_API_KEY`.

### Environment variables (must never be committed)

`SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_KEY`, `COOKIE_SECRET`, `ODDS_API_KEY`, `MAILERSEND_API_KEY`.

## Important calculations and model logic

This is the product's core IP — see "Areas that should not be modified casually" below.

- **Implied probability from American odds:** `core/odds_math.py::american_to_implied_prob`.
- **No-vig fair probability:** each book's two-sided prices are devigged individually
  (`core/pipeline.py::_implied_prob_no_vig`) *before* aggregating — not devigged after averaging.
- **Weighted consensus:** `core/pipeline.py::_consensus_engine` aggregates devigged probabilities
  across books using `BOOK_WEIGHTS` (sharp books like Pinnacle weighted highest at 1.50, down to
  0.30 for unlisted/recreational books).
- **Consensus confidence rating:** `core/pipeline.py::compute_dispersion` /
  `consensus_rating` — standard deviation across **anchor books only**
  (`ANCHOR_BOOKS = {pinnacle, fanduel, draftkings, caesars, bet365, kalshi}`) → "Very Strong" /
  "Strong" / "Moderate" / "Weak" / "Little" label shown in market intelligence.
- **EV% and Kelly stake sizing:** `core/odds_math.py::expected_value_pct`, `kelly_fraction` — driven
  off the fair probability vs. the best available price. Displayed stake = `bankroll * kelly_multiplier * kelly_fraction`.
- **Arbitrage:** `core/arbitrage_engine.py` — only evaluates markets with exactly two distinct picks
  at the same line; requires `sum(implied_probabilities) < 1.0`; stake split is proportional to
  `1/decimal_odds` per leg (equal-payout construction).
- **NFL week/date windows:** `core/data_sources.py` — season start = "Thursday after Labor Day."
  Week 0 is preseason (a forward-only window, deliberately never looks backward, to avoid pulling in
  last season's Super Bowl). All date logic is Eastern-Time-anchored (`EASTERN = ZoneInfo("America/New_York")`).
- **Prop projections / recency weighting:** `core/nflverse_data.py` — `RECENCY_DECAY = 0.85`,
  `_BLEND_SCHEDULE` blends season average with recent-game average, ramping weight toward recent
  games as sample size grows (floors at `_BLEND_FLOOR_GAMES = 6`).

## UI and design conventions

- Light theme, primary color `#4A79BD`, defined once in `.streamlit/config.toml` — don't hardcode
  colors in Python; extend the theme file if a new brand color is genuinely needed.
- Layout is `wide`, sidebar starts `expanded` (`app.py::st.set_page_config`).
- Sidebar is the fixed chrome: logo → sibling-site links → auth panel → How to use / Glossary /
  Feedback / Disclaimer expanders. Tabs are the main content area.
- Every tab module exposes a single `render(...)` function, called from `app.py::run_app()`. Keep
  new tabs to this contract.
- A **local `_sc_name(book)` sportsbook-display-name helper is duplicated** in at least
  `tabs/market_movers.py` and `tabs/fair_value_model.py` (and `core/pipeline.py::ANCHOR_DISPLAY` has
  a smaller version of the same map). This is existing duplication, not a new pattern to copy — if
  you're touching sportsbook display names, prefer consolidating into `core/` rather than adding a
  fourth copy, but don't do a speculative refactor unasked.
- Shared, reusable UI pieces belong in `core/` (e.g. `nfl_player_card.py`, `nfl_player_search.py`,
  `prop_hit_rate_dashboard.py`) and are explicitly commented as "one implementation, not two copies
  that can drift" — follow that precedent for new cross-tab UI.
- Tooltips/help text are centralized in small `TIPS` dicts near the top of a tab module
  (see `tabs/fair_value_model.py::TIPS`) rather than inlined at each call site.

## Naming conventions

- `render(...)` — every tab's public entry point; `render_<thing>(...)` — reusable UI component
  functions in `core/` (e.g. `render_nfl_player_card`, `render_nfl_player_search`,
  `render_opponent_defense_single`, `render_prop_hit_rate_dashboard`).
- Leading-underscore first parameter (`_supabase`) on functions wrapped in `@st.cache_data` — this
  is intentional, not a typo: Streamlit's cache tries to hash all args, and a leading underscore
  tells it to skip hashing that one (the Supabase client isn't hashable). Keep this convention for
  any new cached function that takes the client.
- Local scratch variables inside tab `render()` functions are prefixed with a single underscore
  (`_fr1`, `_week_start`, `_snap_games`, …) — a convention to visually separate "local render-scope
  temp vars" from real data/columns. Not enforced by tooling, but followed consistently — match it
  in new tab code.
- `MarketConfig` names (`"moneyline"`, `"spread"`, `"total"`) are the canonical internal market keys,
  matched against `db_market_key` in Supabase and `odds_api_market` from The Odds API
  (`"h2h"`, `"spreads"`, `"totals"`) — these three strings are a contract across `fetch_odds_nfl.py`,
  Supabase schema, and `core/pipeline.py::MARKETS`. Don't rename casually.
- DataFrame display columns use Title Case with units in parens (`"EV%"`, `"Fair Win %"`,
  `"Kelly (u)"`, `"Stake ($)"`, `"Best Odds"`) — internal/raw computation columns are prefixed
  `_` (`_ev_raw`, `_fair_raw`) or `mi_` for market-intelligence fields (`mi_rating`, `mi_std_dev`,
  `mi_fair_odds_a`). `format_display_df` strips the `_`-prefixed raw columns before display.

## Dependencies between important files

- `core/pipeline.py` depends on `core/odds_math.py` only — pure functions, no Streamlit/Supabase
  imports. Keep it that way; if pipeline code starts needing `st.*`, that's a sign it's drifting
  into UI concerns and belongs in a tab instead.
- `core/arbitrage_engine.py` depends on **the shape of `run_market_pipeline`'s output**
  (`Game`, `commence_time`, `Market`, `Pick`, `Line`, `Best Odds`, `Best Book` columns) — not on raw
  odds. Changing `build_display_rows`' output column names/shape in `core/pipeline.py` will silently
  break arbitrage detection (it soft-fails to an empty list rather than raising, via the
  `required.issubset(...)` check — so a broken contract here won't throw, it'll just quietly return
  no opportunities).
- `core/data_sources.py::get_date_window` / `infer_current_week_index` / `nfl_week_window_utc` are
  called from essentially every tab that shows games (`fair_value_model`, `market_movers`,
  `matchup_center`, `lineup_comparison`, etc.) — this is the single source of truth for "what week/day
  is it." Do not duplicate week-inference logic locally in a tab.
- `core/nfl_player_context.py`, `core/nfl_player_card.py`, `core/nfl_player_search.py` are explicitly
  shared between `tabs/lineup_comparison.py` and `tabs/prop_leaderboard.py` — a change to one affects
  both tabs' rendering.
- `tabs/*.py` → `core/pipeline.py` (`MARKETS`, `run_market_pipeline`) → `core/odds_math.py` is the
  main data-flow chain for every odds-driven tab. `core/data_sources.py` sits alongside it for
  fetching + windowing raw lines before they reach the pipeline.
- `fetch_odds_nfl.py` and `app.py` share the Supabase schema (`odds_snapshots`, `odds_lines`)
  implicitly through column names — there's no shared schema file/ORM model, so a column rename on
  one side requires manually updating the other and `core/data_sources.py`'s `.select(...)` list.

## Areas that should not be modified casually

- **`core/pipeline.py`** — the fair-value/consensus engine (`BOOK_WEIGHTS`, `_consensus_engine`,
  `ANCHOR_BOOKS`, `consensus_rating` thresholds, `_fair_odds_american`). This is the product's actual
  edge. Do not change weighting, devig methodology, anchor-book membership, or rating thresholds
  unless explicitly asked — these encode deliberate product/business decisions, not incidental
  implementation details.
- **`core/odds_math.py`** — EV%/Kelly formulas. These are standard, well-defined formulas; changing
  them changes every displayed number app-wide.
- **`core/arbitrage_engine.py`** — stake-split math (`1/decimal` proportional split) and the
  "exactly 2 picks" / `total_implied >= 1.0` gating are load-bearing for correctness of a feature
  that directly implies real-money stakes to users.
- **NFL week/date-window logic in `core/data_sources.py`** (`thursday_after_labor_day_utc`,
  `nfl_week_window_utc`, `infer_current_week_index`) — has already been tuned for edge cases
  (preseason forward-only window, per comments referencing a past bug where last season's Super Bowl
  leaked into "upcoming"). Changing it risks reintroducing that class of bug.
- **Auth flow in `app.py`** (Supabase session handling, cookie read/write, `save_session`/
  `clear_session`) — security-sensitive; a subtle bug here can leak sessions or lock users out.
- **`BOOK_WEIGHTS` / `ANCHOR_BOOKS`** — business-tuned constants, not defaults to "clean up."
- **Environment variable names and the two distinct Supabase keys** (`SUPABASE_ANON_KEY` for the
  app vs. `SUPABASE_KEY` for the ingestion script) — do not consolidate or rename without confirming
  which privilege level each is actually meant to carry.
- **`data/fantasy_adp.xlsx` parsing regexes in `core/fantasy_data.py`** — brittle by nature (parsing
  a semi-structured spreadsheet export); a regex tweak that looks like a simplification can silently
  drop or misparse players. Test against the real file before/after any change here.

## Known rough edges (context, not necessarily "fix this")

- `core/nfl_live_scores.py` has `DEBUG_LIVE = True` with debug `st.caption`/`st.error` calls that are
  currently live in production output (comment says "temporary — remove once confirmed working").
  Don't assume this is intentional permanent behavior, but also don't silently flip it without
  flagging it — it may be there deliberately during an active debugging session.
- `tabs/player_research.py` is an intentional stub, not wired into `app.py`.
- Sportsbook display-name maps (`_sc_name`) are duplicated across a few tab files rather than
  centralized — see "UI and design conventions" above.

## Testing and validation procedures

There is **no automated test suite** in this repository (no `tests/` directory, no `pytest`/`unittest`
config, nothing in `requirements.txt` for testing). Validation is manual:

1. **Syntax/import check** — at minimum, `python3 -m py_compile <changed_file>.py` before considering
   a change done; for larger changes, actually import the module (`python3 -c "import core.pipeline"`)
   to catch import-time errors.
2. **Run the app locally** and click through any tab you touched:
   ```bash
   streamlit run app.py
   ```
   Requires the environment variables listed above to be set (the app hard-stops with `st.error` +
   `st.stop()` if `SUPABASE_URL`/`SUPABASE_ANON_KEY`/`COOKIE_SECRET` are missing).
3. **For `core/pipeline.py` / `core/odds_math.py` / `core/arbitrage_engine.py` changes** — these are
   pure-Python and can be exercised directly (construct a small DataFrame, call the function, inspect
   output) without running Streamlit at all. Preferred for verifying model-logic changes precisely,
   since the UI won't show you intermediate values.
4. **For `fetch_odds_nfl.py` changes** — do not run it against production Supabase/Odds API credentials
   speculatively; it writes real rows and consumes real API quota. Confirm the change is safe (e.g. by
   reading the diff carefully, or a dry-run/print-only check) before executing, and mention this
   explicitly if the user asks you to run it.
5. Always re-check `PipelineTrace` (`core/pipeline.py`) output/behavior didn't regress if you touch
   `run_market_pipeline` — it's the built-in diagnostic path for "why is this table empty," used by
   `debug_mode` in `tabs/fair_value_model.py`.

## Development Rules

- Always inspect the existing implementation before modifying it.
- Preserve existing functionality unless explicitly requested otherwise.
- Prefer targeted changes over rewriting entire files.
- Reuse existing components and patterns whenever practical.
- Maintain visual and functional consistency across the application.
- Do not change model logic, formulas, weighting methodologies, APIs, or data sources unless
  specifically requested.
- Do not expose or commit API keys, credentials, tokens, `.env` files, or other secrets.
- Test or validate changes before considering a task complete.
- Review the final Git diff for unintended changes.

## Git Workflow

The `main` branch is the primary working and production branch. Unless explicitly told otherwise:

1. Work directly on `main`.
2. Do not create feature branches or pull requests.
3. Confirm the current branch is `main` before starting.
4. Pull the latest changes from `origin/main` before making changes.
5. Make and validate the requested changes.
6. Review the Git diff for unintended modifications.
7. Commit completed changes directly to `main` using a concise, descriptive commit message.
8. Push the commit to `origin/main`.
9. Never force-push or rewrite Git history.
10. Never discard unrelated uncommitted changes.
11. If unrelated uncommitted changes already exist when starting a task, stop and tell the user before proceeding.
12. If validation fails or a potentially serious issue is identified, do not push the changes — explain the issue first.
