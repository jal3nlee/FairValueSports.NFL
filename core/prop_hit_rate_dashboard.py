# core/prop_hit_rate_dashboard.py
# Shared "Prop Hit Rate" dashboard — ported from MLB Player Research's
# existing Prop Hit Rate card (hit rate, hit count, average, streak,
# hit/miss bar chart with threshold line). MLB's own file is untouched;
# this is a new component NFL calls into, built to visually/functionally
# mirror it.
import streamlit as st
import pandas as pd


def _current_streak(game_log_newest_first: list[dict], line: float, side: str) -> tuple[int, str] | tuple[None, None]:
    """game_log_newest_first: most recent game FIRST. Walks forward from
    the most recent game, counting consecutive same-result games. Pushes
    (exact line matches) break the streak rather than counting either way."""
    streak_type = None
    count = 0
    for g in game_log_newest_first:
        if g["value"] == line:
            break  # push — streak doesn't extend through it
        kind = "hit" if (g["value"] > line if side == "Over" else g["value"] < line) else "miss"
        if streak_type is None:
            streak_type, count = kind, 1
        elif kind == streak_type:
            count += 1
        else:
            break
    if streak_type is None:
        return None, None
    return count, streak_type


def render_prop_hit_rate_dashboard(
    stat_label: str,
    side: str,
    line: float,
    game_log_newest_first: list[dict],
    sample_label: str,
    week_label_key: str = "week",
    opponent_key: str = "opponent",
    value_key: str = "value",
):
    """
    game_log_newest_first: list of dicts with at least week_label_key,
    opponent_key, value_key — MOST RECENT GAME FIRST (matches
    get_player_game_log's existing return order). Uses only games
    actually present in the log — no fabricated zeros for missing weeks.
    """
    if not game_log_newest_first:
        st.caption("No current-season game data is available for this prop yet.")
        return

    hits = misses = pushes = 0
    for g in game_log_newest_first:
        if g[value_key] == line:
            pushes += 1
        elif (g[value_key] > line if side == "Over" else g[value_key] < line):
            hits += 1
        else:
            misses += 1
    decided = hits + misses
    if decided == 0:
        st.caption("Every game in this sample was an exact push — no Over/Under result to summarize.")
        return

    hit_rate = hits / decided * 100
    avg_val = sum(g[value_key] for g in game_log_newest_first) / len(game_log_newest_first)

    streak_count, streak_type = _current_streak(game_log_newest_first, line, side)

    with st.container(border=True):
        st.markdown(
            f"<div style='font-size:0.85rem;opacity:0.65;letter-spacing:0.04em;"
            f"text-transform:uppercase;margin-bottom:2px'>"
            f"{side.upper()} {line:g} {stat_label.upper()}</div>",
            unsafe_allow_html=True,
        )
        _sc1, _sc2 = st.columns([1.3, 2])
        with _sc1:
            st.markdown(
                f"<div style='font-size:2.6rem;font-weight:800;line-height:1'>{hit_rate:.0f}%</div>"
                f"<div style='opacity:0.6;font-size:0.85rem;margin-top:2px'>Hit Rate</div>",
                unsafe_allow_html=True,
            )
        with _sc2:
            st.markdown(
                f"<div style='margin-top:0.4rem;font-size:0.95rem'>"
                f"Hit in {hits} of {decided if decided == len(game_log_newest_first) else f'the last {decided}'} games</div>",
                unsafe_allow_html=True,
            )
            _m1, _m2 = st.columns(2)
            _m1.metric("Average", f"{avg_val:.1f}")
            if streak_count is not None:
                _streak_word = "hit" + ("s" if streak_count != 1 else "") if streak_type == "hit" else "miss" + ("es" if streak_count != 1 else "")
                _m2.metric("Current streak", f"{streak_count} {_streak_word}")
            else:
                _m2.metric("Current streak", "—")

        st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)
        try:
            import altair as alt
            _chart_rows = list(reversed(game_log_newest_first))  # chronological, oldest -> newest for the chart
            _chart_df = pd.DataFrame(_chart_rows)
            _chart_df["Label"] = _chart_df[week_label_key].apply(lambda w: f"W{w}")
            _chart_df["Result"] = _chart_df[value_key].apply(
                lambda v: "Push" if v == line else ("Hit" if (v > line if side == "Over" else v < line) else "Miss")
            )
            _bars = (
                alt.Chart(_chart_df).mark_bar(size=18)
                .encode(
                    x=alt.X("Label:N", sort=None, title=None, axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y(f"{value_key}:Q", title=stat_label),
                    color=alt.Color(
                        "Result:N",
                        scale=alt.Scale(domain=["Hit", "Miss", "Push"], range=["#2ecc71", "#7a7a7a", "#f0ad4e"]),
                        legend=alt.Legend(title=None, orient="top"),
                    ),
                    tooltip=["Label", f"{opponent_key}", f"{value_key}", "Result"],
                )
            )
            _rule = (
                alt.Chart(pd.DataFrame({"y": [line]}))
                .mark_rule(color="#e74c3c", strokeDash=[4, 3]).encode(y="y:Q")
            )
            st.altair_chart((_bars + _rule).properties(height=220), use_container_width=True)
        except Exception:
            _chart_rows = list(reversed(game_log_newest_first))
            _indicators = " ".join(
                "🟢H" if (g[value_key] > line if side == "Over" else g[value_key] < line)
                else ("🟠P" if g[value_key] == line else "⚪M")
                for g in _chart_rows
            )
            st.caption(_indicators)
