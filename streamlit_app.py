"""
streamlit_app.py — IPL AI Analyst
Extended with Advanced Scoring Engine: 8 algorithms across 5 sports.
  1. Dream11 Pts/Match      — Fantasy Cricket
  2. Batting Pressure Index — Cricket Analytics
  3. Player Impact (PI)     — ESPNcricinfo-inspired
  4. Value Above Replacement— Baseball WAR
  5. Cricket PER            — Basketball Hollinger PER
  6. Clutch Index           — NFL/NBA high-leverage
  7. Dynamic Elo Rating     — Chess / Tennis
  8. Consistency Score      — Tennis reliability

Run locally:  streamlit run streamlit_app.py
Deploy:       Push to GitHub → connect on share.streamlit.io
"""

import streamlit as st
import json
import sqlite3
import re
import random
import pandas as pd
import numpy as np
from pathlib import Path

# ── Page config ────────────────────────────────────
st.set_page_config(page_title="IPL AI Analyst", page_icon="🏏", layout="wide")

st.markdown("""
<style>
    .stApp { max-width: 1100px; margin: 0 auto; }
    div[data-testid="stExpander"] details summary span p { font-size: 0.9rem; color: #888; }
    .stat-card { background: var(--secondary-background-color); border-radius: 8px; padding: 16px; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Database layer ─────────────────────────────────
DB_PATH = Path(__file__).parent / "ipl.db"

SCHEMA_DESCRIPTION = """
DATABASE: IPL Ball-by-Ball Cricket Database (2008–2025, 18 seasons)

TABLES:

1. deliveries (278,171 rows) — Every ball bowled in every IPL match.
   Columns: match_id INT, inning INT (1 or 2), over INT (0-19), ball INT (1-6+),
   over_ball REAL, batting_team TEXT, bowling_team TEXT, batsman TEXT,
   non_striker TEXT, bowler TEXT, batsman_runs INT (0-6), extras INT,
   is_wide INT (0/1), is_no_ball INT (0/1), byes INT (0/1), leg_byes INT (0/1),
   penalty INT (0/1), dismissal_kind TEXT (NULL if not out, else: 'caught',
   'bowled', 'lbw', 'run out', 'stumped', 'caught and bowled', 'hit wicket',
   'retired hurt', 'retired out', 'obstructing the field'),
   player_dismissed TEXT, date TEXT (YYYY-MM-DD),
   batsman_type TEXT ('Right hand Bat'/'Left hand Bat'),
   bowler_type TEXT (e.g. 'Right arm Fast', 'Legbreak Googly').

   NOTES:
   - total_runs for a delivery = batsman_runs + extras
   - Powerplay: overs 0-5. Middle: overs 6-15. Death: overs 16-19.
   - Bowler wickets: dismissal_kind NOT IN ('run out','retired hurt','obstructing the field','retired out')
   - Strike rate = (runs / balls_faced) * 100. Balls faced excludes wides (is_wide=0).
   - Economy rate = (runs_conceded / balls_bowled) * 6.

2. matches (1,169 rows) — One row per match.
   Columns: match_id INT PK, season TEXT, date TEXT, venue TEXT, city TEXT,
   team1 TEXT, team2 TEXT, toss_winner TEXT, toss_decision TEXT ('bat'/'field'),
   winner TEXT, winner_runs INT, winner_wickets INT, player_of_match TEXT,
   method TEXT, year INT, match_stage TEXT ('League','Qualifier 1','Qualifier 2',
   'Eliminator','Final').

3. players (772 rows) — Player master data.
   Columns: player_id INT PK, player_name TEXT, player_full_name TEXT,
   bat_style TEXT, bowl_style TEXT, is_keeper INT (0/1).

4. playing_xi (26,137 rows) — Squad for each match.
   Columns: match_id INT, team TEXT, player_name TEXT.

5. teams (16 rows), 6. team_aliases (46 rows), 7. wicket_fielders (10,540 rows)
   Columns(wicket_fielders): match_id, inning, over, ball, wicket_kind, player_out, fielder_name.

JOIN KEYS: deliveries.match_id → matches.match_id.
Players join on player_name (short form like 'V Kohli').
SQLite syntax: use LOWER() + LIKE, not ILIKE. For year filtering, use matches.year (INT).
"""


def run_query(sql: str, params: tuple = ()) -> list[dict]:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description] if cur.description else []
    rows = cur.fetchall()
    conn.close()
    return [dict(zip(cols, row)) for row in rows]


# ── Claude integration ─────────────────────────────
def get_client():
    import anthropic
    api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
    if not api_key:
        st.error("Set `ANTHROPIC_API_KEY` in `.streamlit/secrets.toml` or Streamlit Cloud secrets.")
        st.stop()
    return anthropic.Anthropic(api_key=api_key)

MODEL = "claude-sonnet-4-20250514"

def call_claude(system: str, user_msg: str, max_tokens: int = 1000) -> str:
    client = get_client()
    resp = client.messages.create(
        model=MODEL, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": user_msg}],
    )
    return resp.content[0].text.strip()


# ── /ask logic ─────────────────────────────────────
def ask_ipl(question: str) -> dict:
    sql = call_claude(
        system=(
            "You are an expert SQL analyst for IPL cricket data. "
            "Given the schema below and a user question, write a single "
            "SQLite SELECT query that answers the question.\n\n"
            "RULES:\n"
            "- Return ONLY the raw SQL. No markdown, no explanation, no backticks.\n"
            "- Use SQLite syntax (LOWER() + LIKE instead of ILIKE).\n"
            "- Limit results to 50 rows max.\n"
            "- Exclude wides when counting balls faced.\n"
            "- For bowler wickets, exclude: run out, retired hurt, retired out, obstructing the field.\n"
            f"\n{SCHEMA_DESCRIPTION}"
        ),
        user_msg=question,
    )
    sql = re.sub(r'^```\w*\n?', '', sql)
    sql = re.sub(r'\n?```$', '', sql).strip()
    try:
        data = run_query(sql)
    except Exception as e:
        return {"sql": sql, "data": [], "answer": f"SQL execution failed: {e}", "error": True}
    data_trunc = data[:50]
    answer = call_claude(
        system=(
            "You are a cricket analyst. Given a question, the SQL run, and results, "
            "provide a concise, insightful answer. Use specific numbers. Under 200 words."
        ),
        user_msg=f"Question: {question}\n\nSQL: {sql}\n\nResults ({len(data_trunc)} rows):\n{json.dumps(data_trunc, indent=2, default=str)}",
        max_tokens=800,
    )
    return {"sql": sql, "data": data_trunc, "answer": answer, "row_count": len(data)}


# ── /scout logic ───────────────────────────────────
def pull_player_stats(name: str) -> dict:
    stats = {}
    basic = run_query("SELECT * FROM players WHERE LOWER(player_name) = LOWER(?)", (name,))
    if not basic:
        basic = run_query("SELECT * FROM players WHERE LOWER(player_name) LIKE LOWER(?)", (f"%{name}%",))
    if not basic:
        return {}
    stats["basic"] = basic[0]
    pn = basic[0]["player_name"]
    stats["batting_by_season"] = run_query("""
        SELECT m.year, COUNT(DISTINCT d.match_id) as matches,
               SUM(d.batsman_runs) as runs,
               SUM(CASE WHEN d.is_wide=0 THEN 1 ELSE 0 END) as balls,
               ROUND(SUM(d.batsman_runs)*100.0/NULLIF(SUM(CASE WHEN d.is_wide=0 THEN 1 ELSE 0 END),0),1) as sr,
               SUM(CASE WHEN d.dismissal_kind IS NOT NULL AND d.player_dismissed=? THEN 1 ELSE 0 END) as outs
        FROM deliveries d JOIN matches m ON d.match_id=m.match_id
        WHERE d.batsman=? GROUP BY m.year ORDER BY m.year
    """, (pn, pn))
    stats["batting_by_phase"] = run_query("""
        SELECT CASE WHEN over BETWEEN 0 AND 5 THEN 'Powerplay'
                    WHEN over BETWEEN 6 AND 15 THEN 'Middle' ELSE 'Death' END as phase,
               SUM(batsman_runs) as runs,
               SUM(CASE WHEN is_wide=0 THEN 1 ELSE 0 END) as balls,
               ROUND(SUM(batsman_runs)*100.0/NULLIF(SUM(CASE WHEN is_wide=0 THEN 1 ELSE 0 END),0),1) as sr
        FROM deliveries WHERE batsman=? GROUP BY phase ORDER BY MIN(over)
    """, (pn,))
    stats["batting_vs_type"] = run_query("""
        SELECT CASE WHEN bowler_type LIKE '%Fast%' OR bowler_type LIKE '%Medium%'
                    OR bowler_type LIKE '%Pace%' THEN 'Pace' ELSE 'Spin' END as bowl_cat,
               SUM(batsman_runs) as runs,
               SUM(CASE WHEN is_wide=0 THEN 1 ELSE 0 END) as balls,
               ROUND(SUM(batsman_runs)*100.0/NULLIF(SUM(CASE WHEN is_wide=0 THEN 1 ELSE 0 END),0),1) as sr
        FROM deliveries WHERE batsman=? GROUP BY bowl_cat
    """, (pn,))
    stats["bowling_by_season"] = run_query("""
        SELECT m.year, COUNT(DISTINCT d.match_id) as matches,
               SUM(CASE WHEN d.is_wide=0 AND d.is_no_ball=0 THEN 1 ELSE 0 END) as legal_balls,
               SUM(d.batsman_runs+d.extras) as runs_conceded,
               SUM(CASE WHEN d.dismissal_kind IS NOT NULL
                    AND d.dismissal_kind NOT IN ('run out','retired hurt','retired out','obstructing the field')
                    THEN 1 ELSE 0 END) as wickets,
               ROUND(SUM(d.batsman_runs+d.extras)*6.0/NULLIF(SUM(CASE WHEN d.is_wide=0 AND d.is_no_ball=0 THEN 1 ELSE 0 END),0),2) as economy
        FROM deliveries d JOIN matches m ON d.match_id=m.match_id
        WHERE d.bowler=? GROUP BY m.year ORDER BY m.year
    """, (pn,))
    stats["pom_count"] = run_query("SELECT COUNT(*) as awards FROM matches WHERE player_of_match=?", (pn,))[0]["awards"]
    return stats

def scout_player(p1: str, p2: str = None) -> dict:
    s1 = pull_player_stats(p1)
    if not s1.get("basic"):
        return {"error": f"Player '{p1}' not found."}
    s2 = pull_player_stats(p2) if p2 else None
    if p2 and not (s2 or {}).get("basic"):
        return {"error": f"Player '{p2}' not found."}
    if p2:
        prompt = f"Compare these two IPL players.\n\nPLAYER 1: {p1}\n{json.dumps(s1, indent=2, default=str)}\n\nPLAYER 2: {p2}\n{json.dumps(s2, indent=2, default=str)}"
    else:
        prompt = f"Produce a scouting report.\n\nPLAYER: {p1}\n{json.dumps(s1, indent=2, default=str)}"
    report = call_claude(
        system=(
            "You are an elite cricket scout writing for a franchise coaching staff. "
            "Sections: 1. Overview 2. Batting Profile 3. Bowling Profile (if applicable) "
            "4. Key Matchups & Vulnerabilities 5. Verdict. "
            "Use specific numbers. Be direct and analytical."
        ),
        user_msg=prompt, max_tokens=1500,
    )
    return {"stats1": s1, "stats2": s2, "report": report}


# ── /fantasy logic ─────────────────────────────────
def fantasy_advice(team1: str, team2: str, venue: str = None) -> dict:
    data = {}
    for label, team in [("team1_form", team1), ("team2_form", team2)]:
        data[label] = run_query("""
            WITH rm AS (SELECT DISTINCT d.match_id, m.date FROM deliveries d
                        JOIN matches m ON d.match_id=m.match_id
                        WHERE d.batting_team=? ORDER BY m.date DESC LIMIT 5)
            SELECT d.batsman, COUNT(DISTINCT d.match_id) as inn, SUM(d.batsman_runs) as runs,
                   ROUND(SUM(d.batsman_runs)*100.0/NULLIF(SUM(CASE WHEN d.is_wide=0 THEN 1 ELSE 0 END),0),1) as sr
            FROM deliveries d WHERE d.match_id IN (SELECT match_id FROM rm)
              AND d.batting_team=? GROUP BY d.batsman ORDER BY runs DESC
        """, (team, team))
    for label, team in [("team1_bowling", team1), ("team2_bowling", team2)]:
        data[label] = run_query("""
            WITH rm AS (SELECT DISTINCT d.match_id, m.date FROM deliveries d
                        JOIN matches m ON d.match_id=m.match_id
                        WHERE d.bowling_team=? ORDER BY m.date DESC LIMIT 5)
            SELECT d.bowler, SUM(CASE WHEN d.dismissal_kind IS NOT NULL
                    AND d.dismissal_kind NOT IN ('run out','retired hurt','retired out','obstructing the field')
                    THEN 1 ELSE 0 END) as wkts,
                   ROUND(SUM(d.batsman_runs+d.extras)*6.0/NULLIF(SUM(CASE WHEN d.is_wide=0 AND d.is_no_ball=0 THEN 1 ELSE 0 END),0),2) as econ
            FROM deliveries d WHERE d.match_id IN (SELECT match_id FROM rm)
              AND d.bowling_team=? GROUP BY d.bowler ORDER BY wkts DESC
        """, (team, team))
    data["h2h"] = run_query("""
        SELECT winner, COUNT(*) as wins FROM matches
        WHERE (team1=? AND team2=?) OR (team1=? AND team2=?) GROUP BY winner
    """, (team1, team2, team2, team1))
    if venue:
        data["venue_top_scorers"] = run_query("""
            SELECT d.batsman, SUM(d.batsman_runs) as runs FROM deliveries d
            JOIN matches m ON d.match_id=m.match_id
            WHERE LOWER(m.venue) LIKE LOWER(?) GROUP BY d.batsman ORDER BY runs DESC LIMIT 10
        """, (f"%{venue}%",))
    rec = call_claude(
        system=(
            "You are a fantasy cricket expert. Recommend a Fantasy XI (11 players from both squads). "
            "Structure: 1. CAPTAIN (2x) 2. VICE-CAPTAIN (1.5x) 3. Full XI 4. Differential pick "
            "5. Players to AVOID. Be specific with stats."
        ),
        user_msg=f"Match: {team1} vs {team2}" + (f" at {venue}" if venue else "") +
                 f"\n\nData:\n{json.dumps(data, indent=2, default=str)}",
        max_tokens=1500,
    )
    return {"data": data, "recommendation": rec}


# ══════════════════════════════════════════════════
#  QUIZ ENGINE — Narrative / Funda Style
# ══════════════════════════════════════════════════

# Each template has:
#   sql        — fetches the correct answer + primary stat
#   context_sql— fetches 3-4 supporting facts about the answer (for Claude to weave into the story)
#   prompt_hint— tells Claude what kind of narrative angle to take

QUIZ_TEMPLATES = [
    {
        "id": "most_sixes_career",
        "sql": """SELECT batsman as answer, SUM(CASE WHEN batsman_runs=6 THEN 1 ELSE 0 END) as val
                  FROM deliveries GROUP BY batsman ORDER BY val DESC LIMIT 1""",
        "context_sql": """SELECT SUM(batsman_runs) as career_runs,
                  COUNT(DISTINCT match_id) as matches,
                  ROUND(SUM(batsman_runs)*100.0/NULLIF(SUM(CASE WHEN is_wide=0 THEN 1 ELSE 0 END),0),1) as sr,
                  SUM(CASE WHEN batsman_runs=4 THEN 1 ELSE 0 END) as fours
                  FROM deliveries WHERE batsman=?""",
        "stat_label": "sixes in IPL history",
        "prompt_hint": "Focus on the sheer scale of destruction — the sixes total is almost mythical. Hint at the player's size, Caribbean origins, and the terror he caused opening the batting.",
    },
    {
        "id": "most_runs_career",
        "sql": """SELECT batsman as answer, SUM(batsman_runs) as val
                  FROM deliveries GROUP BY batsman ORDER BY val DESC LIMIT 1""",
        "context_sql": """SELECT COUNT(DISTINCT match_id) as matches,
                  SUM(CASE WHEN batsman_runs=6 THEN 1 ELSE 0 END) as sixes,
                  ROUND(SUM(batsman_runs)*100.0/NULLIF(SUM(CASE WHEN is_wide=0 THEN 1 ELSE 0 END),0),1) as sr,
                  COUNT(DISTINCT m.year) as seasons
                  FROM deliveries d JOIN matches m ON d.match_id=m.match_id WHERE d.batsman=?""",
        "stat_label": "runs — all-time leading scorer",
        "prompt_hint": "Tell the story of longevity and consistency. The player has outlasted entire generations of IPL cricketers. Hint at the number of seasons and the consistency required to accumulate this volume.",
    },
    {
        "id": "most_wickets_career",
        "sql": """SELECT bowler as answer,
                  SUM(CASE WHEN dismissal_kind IS NOT NULL AND dismissal_kind NOT IN
                  ('run out','retired hurt','retired out','obstructing the field') THEN 1 ELSE 0 END) as val
                  FROM deliveries GROUP BY bowler ORDER BY val DESC LIMIT 1""",
        "context_sql": """SELECT COUNT(DISTINCT match_id) as matches,
                  ROUND(SUM(batsman_runs+extras)*6.0/NULLIF(SUM(CASE WHEN is_wide=0 AND is_no_ball=0 THEN 1 ELSE 0 END),0),2) as economy,
                  SUM(CASE WHEN dismissal_kind='bowled' THEN 1 ELSE 0 END) as bowled_count
                  FROM deliveries WHERE bowler=?""",
        "stat_label": "wickets — all-time leading wicket-taker",
        "prompt_hint": "Focus on the art of taking wickets in T20 — this player made it a science. Hint at the yorker, the slower ball, and what it was like to face them in the final over.",
    },
    {
        "id": "most_pom",
        "sql": """SELECT player_of_match as answer, COUNT(*) as val FROM matches
                  WHERE player_of_match IS NOT NULL GROUP BY player_of_match ORDER BY val DESC LIMIT 1""",
        "context_sql": """SELECT SUM(batsman_runs) as career_runs,
                  COUNT(DISTINCT match_id) as matches,
                  SUM(CASE WHEN batsman_runs=6 THEN 1 ELSE 0 END) as sixes,
                  ROUND(SUM(batsman_runs)*100.0/NULLIF(SUM(CASE WHEN is_wide=0 THEN 1 ELSE 0 END),0),1) as sr
                  FROM deliveries WHERE batsman=?""",
        "stat_label": "Player of the Match awards",
        "prompt_hint": "Player of the Match is about match-winning performances, not just stats. Build the story around what it means to win a game almost single-handedly, and how often this player did it.",
    },
    {
        "id": "most_titles",
        "sql": """SELECT winner as answer, COUNT(*) as val FROM matches
                  WHERE match_stage='Final' AND winner IS NOT NULL GROUP BY winner ORDER BY val DESC LIMIT 1""",
        "context_sql": """SELECT COUNT(*) as total_wins FROM matches WHERE winner=?""",
        "stat_label": "IPL titles",
        "prompt_hint": "This is about a dynasty. Tell the story of a franchise that kept winning when others could not. Hint at the city, the home ground, the colours — but not the name.",
        "answer_type": "team",
    },
    {
        "id": "highest_individual_score",
        "sql": """SELECT batsman as answer, SUM(batsman_runs) as val
                  FROM deliveries GROUP BY batsman, match_id ORDER BY val DESC LIMIT 1""",
        "context_sql": """SELECT SUM(batsman_runs) as career_runs,
                  COUNT(DISTINCT match_id) as matches,
                  SUM(CASE WHEN batsman_runs=6 THEN 1 ELSE 0 END) as sixes,
                  ROUND(SUM(batsman_runs)*100.0/NULLIF(SUM(CASE WHEN is_wide=0 THEN 1 ELSE 0 END),0),1) as sr
                  FROM deliveries WHERE batsman=?""",
        "stat_label": "runs in a single IPL innings — highest individual score",
        "prompt_hint": "This innings is legendary. Set the scene: it was 2013, it was Bangalore, it was a Wednesday evening that the bowling attack would prefer to forget. Don't name the player — describe the carnage.",
    },
    {
        "id": "best_economy_career",
        "sql": """SELECT bowler as answer,
                  ROUND(SUM(batsman_runs+extras)*6.0/NULLIF(SUM(CASE WHEN is_wide=0 AND is_no_ball=0 THEN 1 ELSE 0 END),0),2) as val
                  FROM deliveries GROUP BY bowler
                  HAVING SUM(CASE WHEN is_wide=0 AND is_no_ball=0 THEN 1 ELSE 0 END) >= 1000
                  ORDER BY val LIMIT 1""",
        "context_sql": """SELECT COUNT(DISTINCT match_id) as matches,
                  SUM(CASE WHEN dismissal_kind IS NOT NULL AND dismissal_kind NOT IN
                  ('run out','retired hurt','retired out','obstructing the field') THEN 1 ELSE 0 END) as wickets
                  FROM deliveries WHERE bowler=?""",
        "stat_label": "economy rate among bowlers with 1000+ balls — the most miserly",
        "prompt_hint": "Economy rate in T20 is almost criminally hard to keep low. Build the story around the idea of a bowler who gave batters almost nothing — ever. Hint at their bowling style and nationality.",
    },
    {
        "id": "most_fours_career",
        "sql": """SELECT batsman as answer, SUM(CASE WHEN batsman_runs=4 THEN 1 ELSE 0 END) as val
                  FROM deliveries GROUP BY batsman ORDER BY val DESC LIMIT 1""",
        "context_sql": """SELECT SUM(batsman_runs) as career_runs,
                  COUNT(DISTINCT match_id) as matches,
                  SUM(CASE WHEN batsman_runs=6 THEN 1 ELSE 0 END) as sixes,
                  ROUND(SUM(batsman_runs)*100.0/NULLIF(SUM(CASE WHEN is_wide=0 THEN 1 ELSE 0 END),0),1) as sr
                  FROM deliveries WHERE batsman=?""",
        "stat_label": "fours in IPL history",
        "prompt_hint": "Fours are about timing and placement — not brute force. This player has spent more time stroking the ball to the boundary than anyone in IPL history. Hint at their classical technique and long career.",
    },
    {
        "id": "highest_sr_career",
        "sql": """SELECT batsman as answer,
                  ROUND(SUM(batsman_runs)*100.0/NULLIF(SUM(CASE WHEN is_wide=0 THEN 1 ELSE 0 END),0),1) as val
                  FROM deliveries GROUP BY batsman
                  HAVING SUM(batsman_runs) >= 2000
                  ORDER BY val DESC LIMIT 1""",
        "context_sql": """SELECT SUM(batsman_runs) as career_runs,
                  COUNT(DISTINCT match_id) as matches,
                  SUM(CASE WHEN batsman_runs=6 THEN 1 ELSE 0 END) as sixes
                  FROM deliveries WHERE batsman=?""",
        "stat_label": "strike rate among batters with 2000+ runs — fastest scoring",
        "prompt_hint": "A strike rate this high with this many runs is almost physically impossible. Tell the story of a player who hits so hard that the match situation is almost irrelevant to them.",
    },
    {
        "id": "most_ducks",
        "sql": """WITH innings AS (
                    SELECT batsman, match_id,
                           SUM(batsman_runs) as runs,
                           MAX(CASE WHEN player_dismissed=batsman THEN 1 ELSE 0 END) as got_out
                    FROM deliveries WHERE is_wide=0 GROUP BY batsman, match_id)
                  SELECT batsman as answer, SUM(CASE WHEN runs=0 AND got_out=1 THEN 1 ELSE 0 END) as val
                  FROM innings GROUP BY batsman ORDER BY val DESC LIMIT 1""",
        "context_sql": """SELECT SUM(batsman_runs) as career_runs,
                  COUNT(DISTINCT match_id) as matches,
                  SUM(CASE WHEN batsman_runs=6 THEN 1 ELSE 0 END) as sixes,
                  ROUND(SUM(batsman_runs)*100.0/NULLIF(SUM(CASE WHEN is_wide=0 THEN 1 ELSE 0 END),0),1) as sr
                  FROM deliveries WHERE batsman=?""",
        "stat_label": "ducks in IPL — dismissed for zero more than anyone",
        "prompt_hint": "Here is the paradox: the same player who has been dismissed for zero more than almost anyone is also among the most feared batters in IPL history. Tell that contradiction. Hint at the high-risk, high-reward style.",
    },
    {
        "id": "most_stumpings",
        "sql": """SELECT fielder_name as answer, COUNT(*) as val FROM wicket_fielders
                  WHERE wicket_kind='stumped' GROUP BY fielder_name ORDER BY val DESC LIMIT 1""",
        "context_sql": """SELECT COUNT(DISTINCT match_id) as matches FROM wicket_fielders WHERE fielder_name=?""",
        "stat_label": "stumpings — most by any wicketkeeper",
        "prompt_hint": "Stumpings are the wicketkeeper's art — gloves, reflexes, and reading the spinner. Tell the story of a keeper who has done it more than anyone, and what that says about their partnership with the spinners.",
    },
    {
        "id": "final_winner_year",
        "sql": """SELECT winner as answer, year as val FROM matches
                  WHERE match_stage='Final' AND winner IS NOT NULL
                  ORDER BY RANDOM() LIMIT 1""",
        "context_sql": """SELECT winner_runs, winner_wickets, player_of_match, venue FROM matches
                  WHERE match_stage='Final' AND winner=? ORDER BY year DESC LIMIT 1""",
        "stat_label": "IPL title winner",
        "prompt_hint": "Tell the story of this final without naming the winning team. Describe the venue, the margin of victory (runs or wickets), the player of the match, and the significance of the win.",
        "answer_type": "team",
    },
    {
        "id": "most_runs_season",
        "sql": """SELECT d.batsman as answer, SUM(d.batsman_runs) as val, m.year as extra
                  FROM deliveries d JOIN matches m ON d.match_id=m.match_id
                  GROUP BY d.batsman, m.year ORDER BY val DESC LIMIT 1""",
        "context_sql": """SELECT COUNT(DISTINCT d.match_id) as matches,
                  SUM(CASE WHEN d.batsman_runs=6 THEN 1 ELSE 0 END) as sixes,
                  ROUND(SUM(d.batsman_runs)*100.0/NULLIF(SUM(CASE WHEN d.is_wide=0 THEN 1 ELSE 0 END),0),1) as sr
                  FROM deliveries d JOIN matches m ON d.match_id=m.match_id
                  WHERE d.batsman=? AND m.year=?""",
        "stat_label": "runs in a single IPL season — the record",
        "prompt_hint": "One season, one player, one extraordinary run of form. Tell the story of what it looks like when a batter is completely in the zone for an entire IPL season. Hint at the year and the total without naming the player.",
        "extra_context_param": True,
    },
    {
        "id": "most_catches_fielder",
        "sql": """SELECT fielder_name as answer, COUNT(*) as val FROM wicket_fielders
                  WHERE wicket_kind='caught' GROUP BY fielder_name ORDER BY val DESC LIMIT 1""",
        "context_sql": """SELECT COUNT(DISTINCT match_id) as matches FROM wicket_fielders WHERE fielder_name=?""",
        "stat_label": "catches — most by any fielder in IPL",
        "prompt_hint": "Catching in T20 is explosive and instinctive. Tell the story of a player whose hands have ended more batting careers than almost any bowler — without a ball in hand. Hint at their position and athleticism.",
    },
    {
        "id": "most_team_wins",
        "sql": """SELECT winner as answer, COUNT(*) as val FROM matches
                  WHERE winner IS NOT NULL GROUP BY winner ORDER BY val DESC LIMIT 1""",
        "context_sql": """SELECT COUNT(*) as titles FROM matches WHERE match_stage='Final' AND winner=?""",
        "stat_label": "wins — most by any franchise in IPL history",
        "prompt_hint": "This is the franchise that has simply won more than anyone. Tell the story through their dominance — city, home ground, colours, the captains who led them — without naming the team.",
        "answer_type": "team",
    },
    # ── Category 1: Venue Specialists ─────────────────────
    {
        "id": "venue_sr_wankhede",
        "sql": """SELECT d.batsman as answer,
                  ROUND(SUM(d.batsman_runs)*100.0/SUM(CASE WHEN d.is_wide=0 THEN 1 ELSE 0 END),1) as val,
                  SUM(CASE WHEN d.is_wide=0 THEN 1 ELSE 0 END) as balls,
                  COUNT(DISTINCT d.match_id) as matches
                  FROM deliveries d JOIN matches m ON d.match_id=m.match_id
                  WHERE LOWER(m.venue) LIKE '%wankhede%' AND d.is_wide=0
                  GROUP BY d.batsman HAVING balls >= 150 ORDER BY val DESC LIMIT 8""",
        "context_sql": """SELECT SUM(d.batsman_runs) as venue_runs,
                  SUM(CASE WHEN d.is_wide=0 THEN 1 ELSE 0 END) as venue_balls,
                  COUNT(DISTINCT d.match_id) as venue_matches,
                  ROUND(SUM(d.batsman_runs)*100.0/NULLIF(SUM(CASE WHEN d.is_wide=0 THEN 1 ELSE 0 END),0),1) as overall_sr
                  FROM deliveries d WHERE d.batsman=?""",
        "stat_label": "strike rate at Wankhede Stadium among batters with 150+ balls there",
        "prompt_hint": "Some grounds suit certain players. This batter and Wankhede have a relationship the groundstaff would rather forget. Hint at the contrast between their overall career and what they do at this specific venue — the numbers at Wankhede are disproportionately violent.",
    },
    {
        "id": "venue_runs_chinnaswamy",
        "sql": """SELECT d.batsman as answer, SUM(d.batsman_runs) as val,
                  COUNT(DISTINCT d.match_id) as matches
                  FROM deliveries d JOIN matches m ON d.match_id=m.match_id
                  WHERE LOWER(m.venue) LIKE '%chinnaswamy%'
                  GROUP BY d.batsman HAVING matches >= 5 ORDER BY val DESC LIMIT 8""",
        "context_sql": """SELECT SUM(d.batsman_runs) as venue_runs,
                  COUNT(DISTINCT d.match_id) as venue_matches,
                  ROUND(SUM(d.batsman_runs)*100.0/NULLIF(SUM(CASE WHEN d.is_wide=0 THEN 1 ELSE 0 END),0),1) as venue_sr
                  FROM deliveries d JOIN matches m ON d.match_id=m.match_id
                  WHERE LOWER(m.venue) LIKE '%chinnaswamy%' AND d.batsman=?""",
        "stat_label": "runs at M Chinnaswamy Stadium — the highest-scoring ground in IPL history",
        "prompt_hint": "Chinnaswamy is the stadium that turns decent scores into impossible ones. Short boundaries, fast outfield, thin Bangalore air. One batter has feasted here more than anyone. Hint at the fact that they also call this city home — and what it means to dominate your own fortress.",
    },
    {
        "id": "venue_economy_eden",
        "sql": """SELECT d.bowler as answer,
                  ROUND(SUM(d.batsman_runs+d.extras)*6.0/NULLIF(SUM(CASE WHEN d.is_wide=0 AND d.is_no_ball=0 THEN 1 ELSE 0 END),0),2) as val,
                  SUM(CASE WHEN d.is_wide=0 AND d.is_no_ball=0 THEN 1 ELSE 0 END) as balls
                  FROM deliveries d JOIN matches m ON d.match_id=m.match_id
                  WHERE LOWER(m.venue) LIKE '%eden%' AND d.is_wide=0
                  GROUP BY d.bowler HAVING balls >= 150 ORDER BY val LIMIT 8""",
        "context_sql": """SELECT SUM(CASE WHEN d.dismissal_kind IS NOT NULL AND d.dismissal_kind NOT IN
                  ('run out','retired hurt','retired out','obstructing the field') THEN 1 ELSE 0 END) as wickets,
                  COUNT(DISTINCT d.match_id) as matches,
                  ROUND(SUM(d.batsman_runs+d.extras)*6.0/NULLIF(SUM(CASE WHEN d.is_wide=0 AND d.is_no_ball=0 THEN 1 ELSE 0 END),0),2) as career_economy
                  FROM deliveries d JOIN matches m ON d.match_id=m.match_id
                  WHERE LOWER(m.venue) LIKE '%eden%' AND d.bowler=?""",
        "stat_label": "economy rate at Eden Gardens among bowlers with 150+ balls there",
        "prompt_hint": "Eden Gardens is supposed to be a batsman's paradise — 66,000 people willing them to hit. This bowler turned it into their personal laboratory. Hint at the dew, the pitch, and how this bowler found a way to be miserly in the one ground where miserliness seems impossible.",
    },
    # ── Category 2: Season Dominance ──────────────────────────
    {
        "id": "most_wickets_season",
        "sql": """SELECT d.bowler as answer,
                  SUM(CASE WHEN d.dismissal_kind IS NOT NULL AND d.dismissal_kind NOT IN
                  ('run out','retired hurt','retired out','obstructing the field') THEN 1 ELSE 0 END) as val,
                  m.year as extra
                  FROM deliveries d JOIN matches m ON d.match_id=m.match_id
                  GROUP BY d.bowler, m.year ORDER BY val DESC LIMIT 8""",
        "context_sql": """SELECT COUNT(DISTINCT d.match_id) as matches,
                  ROUND(SUM(d.batsman_runs+d.extras)*6.0/NULLIF(SUM(CASE WHEN d.is_wide=0 AND d.is_no_ball=0 THEN 1 ELSE 0 END),0),2) as season_economy
                  FROM deliveries d JOIN matches m ON d.match_id=m.match_id
                  WHERE d.bowler=? AND m.year=?""",
        "stat_label": "wickets in a single IPL season — the record",
        "prompt_hint": "Most bowlers have one purple patch in a season and fade. This one was relentless across an entire edition — a wicket every few overs, match after match. Hint at the year, the number, and the fact that the team still couldn't convert that individual brilliance into a trophy.",
        "extra_context_param": True,
    },
    {
        "id": "best_economy_season",
        "sql": """SELECT d.bowler as answer,
                  ROUND(SUM(d.batsman_runs+d.extras)*6.0/NULLIF(SUM(CASE WHEN d.is_wide=0 AND d.is_no_ball=0 THEN 1 ELSE 0 END),0),2) as val,
                  m.year as extra,
                  SUM(CASE WHEN d.is_wide=0 AND d.is_no_ball=0 THEN 1 ELSE 0 END) as balls
                  FROM deliveries d JOIN matches m ON d.match_id=m.match_id
                  GROUP BY d.bowler, m.year HAVING balls >= 200 ORDER BY val LIMIT 8""",
        "context_sql": """SELECT COUNT(DISTINCT d.match_id) as matches,
                  SUM(CASE WHEN d.dismissal_kind IS NOT NULL AND d.dismissal_kind NOT IN
                  ('run out','retired hurt','retired out','obstructing the field') THEN 1 ELSE 0 END) as wickets
                  FROM deliveries d JOIN matches m ON d.match_id=m.match_id
                  WHERE d.bowler=? AND m.year=?""",
        "stat_label": "economy rate in a single IPL season (min 200 balls) — the most miserly single-season performance",
        "prompt_hint": "Keeping T20 batters under 6 an over for a full season is considered nearly impossible. This bowler did it while barely breaking a sweat. Hint at their bowling style — the kind that makes batters look foolish rather than unlucky — and the year it happened.",
        "extra_context_param": True,
    },
    # ── Category 3: Head-to-Head ───────────────────────────────
    {
        "id": "most_runs_vs_mi",
        "sql": """SELECT d.batsman as answer, SUM(d.batsman_runs) as val,
                  COUNT(DISTINCT d.match_id) as matches
                  FROM deliveries d
                  WHERE d.bowling_team='Mumbai Indians'
                  GROUP BY d.batsman HAVING matches >= 8 ORDER BY val DESC LIMIT 8""",
        "context_sql": """SELECT SUM(d.batsman_runs) as runs_vs_mi,
                  COUNT(DISTINCT d.match_id) as matches_vs_mi,
                  ROUND(SUM(d.batsman_runs)*100.0/NULLIF(SUM(CASE WHEN d.is_wide=0 THEN 1 ELSE 0 END),0),1) as sr_vs_mi
                  FROM deliveries d WHERE d.bowling_team='Mumbai Indians' AND d.batsman=?""",
        "stat_label": "runs against Mumbai Indians — more than any other batter in IPL history",
        "prompt_hint": "Mumbai Indians have the best bowling attack money can buy. They have tried everything against this batter — pace, spin, bouncers, yorkers. Nothing has worked consistently. Hint at the volume of runs and the number of encounters, and what it says about a batter who has made one of the great franchises feel helpless.",
    },
    {
        "id": "most_runs_vs_csk",
        "sql": """SELECT d.batsman as answer, SUM(d.batsman_runs) as val,
                  COUNT(DISTINCT d.match_id) as matches
                  FROM deliveries d
                  WHERE d.bowling_team IN ('Chennai Super Kings','Rising Pune Supergiant','Rising Pune Supergiants')
                  GROUP BY d.batsman HAVING matches >= 8 ORDER BY val DESC LIMIT 8""",
        "context_sql": """SELECT SUM(d.batsman_runs) as runs_vs_csk,
                  COUNT(DISTINCT d.match_id) as matches_vs_csk,
                  ROUND(SUM(d.batsman_runs)*100.0/NULLIF(SUM(CASE WHEN d.is_wide=0 THEN 1 ELSE 0 END),0),1) as sr_vs_csk
                  FROM deliveries d
                  WHERE d.bowling_team IN ('Chennai Super Kings','Rising Pune Supergiant','Rising Pune Supergiants')
                  AND d.batsman=?""",
        "stat_label": "runs against Chennai Super Kings — more than any other batter",
        "prompt_hint": "CSK are famous for knowing their opponents inside out — they plan, they adapt, they contain. Against this batter, those plans have repeatedly come unstuck. Hint at the number of times they have met, the runs that have piled up, and the quiet indignity of being the one franchise a batter has simply owned.",
    },
    # ── Category 4: Clutch / High-Stakes ──────────────────────
    {
        "id": "most_runs_playoffs",
        "sql": """SELECT d.batsman as answer, SUM(d.batsman_runs) as val,
                  COUNT(DISTINCT d.match_id) as matches
                  FROM deliveries d JOIN matches m ON d.match_id=m.match_id
                  WHERE m.match_stage NOT IN ('League') AND d.is_wide=0
                  GROUP BY d.batsman HAVING matches >= 8 ORDER BY val DESC LIMIT 8""",
        "context_sql": """SELECT SUM(d.batsman_runs) as playoff_runs,
                  COUNT(DISTINCT d.match_id) as playoff_matches,
                  ROUND(SUM(d.batsman_runs)*100.0/NULLIF(SUM(CASE WHEN d.is_wide=0 THEN 1 ELSE 0 END),0),1) as playoff_sr
                  FROM deliveries d JOIN matches m ON d.match_id=m.match_id
                  WHERE m.match_stage NOT IN ('League') AND d.batsman=?""",
        "stat_label": "runs in IPL knockout matches — more than any batter in history",
        "prompt_hint": "League cricket and knockout cricket are different sports. One rewards consistency, the other rewards nerve. This batter has been at their best precisely when the margin for error disappears. Hint at the number of playoff matches they have played and what their runs record in those games reveals about their temperament.",
    },
    {
        "id": "most_wickets_playoffs",
        "sql": """SELECT d.bowler as answer,
                  SUM(CASE WHEN d.dismissal_kind IS NOT NULL AND d.dismissal_kind NOT IN
                  ('run out','retired hurt','retired out','obstructing the field') THEN 1 ELSE 0 END) as val,
                  COUNT(DISTINCT d.match_id) as matches
                  FROM deliveries d JOIN matches m ON d.match_id=m.match_id
                  WHERE m.match_stage NOT IN ('League')
                  GROUP BY d.bowler HAVING matches >= 8 ORDER BY val DESC LIMIT 8""",
        "context_sql": """SELECT SUM(CASE WHEN d.dismissal_kind IS NOT NULL AND d.dismissal_kind NOT IN
                  ('run out','retired hurt','retired out','obstructing the field') THEN 1 ELSE 0 END) as playoff_wickets,
                  COUNT(DISTINCT d.match_id) as playoff_matches,
                  ROUND(SUM(d.batsman_runs+d.extras)*6.0/NULLIF(SUM(CASE WHEN d.is_wide=0 AND d.is_no_ball=0 THEN 1 ELSE 0 END),0),2) as playoff_economy
                  FROM deliveries d JOIN matches m ON d.match_id=m.match_id
                  WHERE m.match_stage NOT IN ('League') AND d.bowler=?""",
        "stat_label": "wickets in IPL playoff matches — more than any bowler in history",
        "prompt_hint": "Knockout cricket is where reputations are made or buried. This bowler chose the former — match after match in Qualifiers, Eliminators, and Finals, picking up wickets at a rate their league season never fully predicted. Hint at the sheer number of high-stakes appearances and what it says about their ability to be at their best when it costs the most.",
    },
    # ── Category 5: Longevity ──────────────────────────────────
    {
        "id": "most_seasons",
        "sql": """SELECT d.batsman as answer, COUNT(DISTINCT m.year) as val,
                  COUNT(DISTINCT d.match_id) as matches
                  FROM deliveries d JOIN matches m ON d.match_id=m.match_id
                  GROUP BY d.batsman ORDER BY val DESC LIMIT 8""",
        "context_sql": """SELECT SUM(d.batsman_runs) as career_runs,
                  COUNT(DISTINCT m.year) as seasons,
                  COUNT(DISTINCT d.batting_team) as franchises
                  FROM deliveries d JOIN matches m ON d.match_id=m.match_id WHERE d.batsman=?""",
        "stat_label": "IPL seasons played — more than any other cricketer",
        "prompt_hint": "Franchises change rosters every year. Younger players arrive hungry. Careers in T20 burn fast and bright. This player has been there for all of it — present in the first season, still playing nearly two decades later. Hint at the longevity, the franchises they have outlasted, and what it takes to remain wanted across 18 editions of the same competition.",
    },
    {
        "id": "most_franchises",
        "sql": """SELECT batsman as answer, COUNT(DISTINCT batting_team) as val,
                  COUNT(DISTINCT match_id) as matches
                  FROM deliveries GROUP BY batsman ORDER BY val DESC LIMIT 8""",
        "context_sql": """SELECT COUNT(DISTINCT batting_team) as franchises,
                  SUM(batsman_runs) as career_runs,
                  COUNT(DISTINCT match_id) as matches
                  FROM deliveries WHERE batsman=?""",
        "stat_label": "different IPL franchises played for — more than any other player",
        "prompt_hint": "Most IPL careers are defined by one or two franchises. This player has been picked up, released, and picked up again by franchise after franchise — not because they failed, but because every team believed they could use what this player offers. Hint at the mercenary nature of the career and the sheer number of team colours they have worn.",
    },
    # ── Category 6: Role Specialists ──────────────────────────
    {
        "id": "highest_sr_lower_order",
        "sql": """WITH first_ball AS (
                    SELECT match_id, batting_team, batsman,
                           MIN(over*10+ball) as first_seq
                    FROM deliveries WHERE is_wide=0
                    GROUP BY match_id, batting_team, batsman
                  ),
                  positions AS (
                    SELECT match_id, batting_team, batsman,
                           DENSE_RANK() OVER (PARTITION BY match_id, batting_team ORDER BY first_seq) as bat_pos
                    FROM first_ball
                  )
                  SELECT d.batsman as answer,
                         ROUND(SUM(d.batsman_runs)*100.0/COUNT(*),1) as val,
                         COUNT(*) as balls,
                         COUNT(DISTINCT d.match_id) as matches
                  FROM deliveries d
                  JOIN positions p ON d.match_id=p.match_id AND d.batsman=p.batsman
                  WHERE p.bat_pos >= 7 AND d.is_wide=0
                  GROUP BY d.batsman HAVING balls >= 150
                  ORDER BY val DESC LIMIT 8""",
        "context_sql": """SELECT SUM(batsman_runs) as career_runs,
                  COUNT(DISTINCT match_id) as matches,
                  ROUND(SUM(batsman_runs)*100.0/NULLIF(SUM(CASE WHEN is_wide=0 THEN 1 ELSE 0 END),0),1) as career_sr,
                  SUM(CASE WHEN batsman_runs=6 THEN 1 ELSE 0 END) as sixes
                  FROM deliveries WHERE batsman=?""",
        "stat_label": "strike rate batting at position 7 or lower (min 150 balls) — the most explosive finisher",
        "prompt_hint": "They do not arrive to build an innings. They arrive to end one. Coming in at seven or lower, with fewer balls available than most batters face in the powerplay, this player has still managed to score faster than almost anyone. Hint at the specific role — the designated weapon of the final overs — and the strike rate that makes opposition captains rearrange their bowling plans the moment this player walks to the crease.",
    },
    # ── Category 7: Nearly Men ─────────────────────────────────
    {
        "id": "most_runs_no_title",
        "sql": """SELECT d.batsman as answer, SUM(d.batsman_runs) as val,
                  COUNT(DISTINCT m.year) as seasons
                  FROM deliveries d JOIN matches m ON d.match_id=m.match_id
                  WHERE d.batsman NOT IN (
                      SELECT DISTINCT p.player_name FROM playing_xi p
                      JOIN matches m2 ON p.match_id=m2.match_id AND p.team=m2.winner
                      WHERE m2.match_stage='Final'
                  )
                  GROUP BY d.batsman ORDER BY val DESC LIMIT 8""",
        "context_sql": """SELECT SUM(batsman_runs) as career_runs,
                  COUNT(DISTINCT match_id) as matches,
                  SUM(CASE WHEN batsman_runs=6 THEN 1 ELSE 0 END) as sixes,
                  COUNT(DISTINCT batting_team) as franchises
                  FROM deliveries WHERE batsman=?""",
        "stat_label": "runs in IPL history without ever winning the title — the greatest nearly man",
        "prompt_hint": "Use the contradiction technique. By every individual batting measure, this player is among the greatest in IPL history. By the only measure that franchises ultimately care about, they have never crossed the line. Hint at the seasons, the runs, the franchises tried — and the one thing that has always been just out of reach.",
    },
    {
        "id": "most_wickets_no_title",
        "sql": """SELECT d.bowler as answer,
                  SUM(CASE WHEN d.dismissal_kind IS NOT NULL AND d.dismissal_kind NOT IN
                  ('run out','retired hurt','retired out','obstructing the field') THEN 1 ELSE 0 END) as val,
                  COUNT(DISTINCT m.year) as seasons
                  FROM deliveries d JOIN matches m ON d.match_id=m.match_id
                  WHERE d.bowler NOT IN (
                      SELECT DISTINCT p.player_name FROM playing_xi p
                      JOIN matches m2 ON p.match_id=m2.match_id AND p.team=m2.winner
                      WHERE m2.match_stage='Final'
                  )
                  GROUP BY d.bowler ORDER BY val DESC LIMIT 8""",
        "context_sql": """SELECT SUM(CASE WHEN dismissal_kind IS NOT NULL AND dismissal_kind NOT IN
                  ('run out','retired hurt','retired out','obstructing the field') THEN 1 ELSE 0 END) as wickets,
                  COUNT(DISTINCT match_id) as matches,
                  ROUND(SUM(batsman_runs+extras)*6.0/NULLIF(SUM(CASE WHEN is_wide=0 AND is_no_ball=0 THEN 1 ELSE 0 END),0),2) as economy,
                  COUNT(DISTINCT batting_team) as teams_bowled_for
                  FROM deliveries WHERE bowler=?""",
        "stat_label": "wickets in IPL history without ever winning the title — the most decorated titleless bowler",
        "prompt_hint": "More wickets than almost anyone who has ever bowled in the IPL. Across more than a decade of seasons, more franchises than most players know. The title has never come. Use the elimination trail — hint at the wicket count, the longevity, and the cruel gap between individual excellence and team success.",
    },
    # ── Category 8: Bowling Matchup ───────────────────────────
    {
        "id": "best_economy_vs_lefties",
        "sql": """SELECT d.bowler as answer,
                  ROUND(SUM(d.batsman_runs+d.extras)*6.0/NULLIF(SUM(CASE WHEN d.is_wide=0 AND d.is_no_ball=0 THEN 1 ELSE 0 END),0),2) as val,
                  SUM(CASE WHEN d.is_wide=0 AND d.is_no_ball=0 THEN 1 ELSE 0 END) as balls
                  FROM deliveries d
                  WHERE d.batsman_type='Left hand Bat'
                  GROUP BY d.bowler HAVING balls >= 200 ORDER BY val LIMIT 8""",
        "context_sql": """SELECT ROUND(SUM(CASE WHEN d.batsman_type='Left hand Bat' THEN d.batsman_runs+d.extras ELSE 0 END)*6.0/
                  NULLIF(SUM(CASE WHEN d.batsman_type='Left hand Bat' AND d.is_wide=0 AND d.is_no_ball=0 THEN 1 ELSE 0 END),0),2) as econ_vs_lhb,
                  ROUND(SUM(CASE WHEN d.batsman_type='Right hand Bat' THEN d.batsman_runs+d.extras ELSE 0 END)*6.0/
                  NULLIF(SUM(CASE WHEN d.batsman_type='Right hand Bat' AND d.is_wide=0 AND d.is_no_ball=0 THEN 1 ELSE 0 END),0),2) as econ_vs_rhb,
                  COUNT(DISTINCT d.match_id) as matches
                  FROM deliveries d WHERE d.bowler=?""",
        "stat_label": "economy rate against left-handed batters (min 200 balls) — the best left-hand-bat specialist bowler",
        "prompt_hint": "Use the wrong assumption technique. Left-handers are supposed to be the difficult matchup for this type of bowler. Opposition teams build their left-heavy lineups specifically to counter certain bowling styles. This bowler has made that plan look foolish — their economy against left-handers is actually lower than against right-handers. Hint at the bowling type and the tactical headache they create for any captain who tries to game the matchup.",
    },

    # ── Phase-based templates ──────────────────────────
    {
        "id": "best_pp_sr_batter",
        "sql": """SELECT batsman as answer,
                  ROUND(SUM(batsman_runs)*100.0/COUNT(*),1) as val,
                  COUNT(*) as balls
                  FROM deliveries WHERE over BETWEEN 0 AND 5 AND is_wide=0
                  GROUP BY batsman HAVING balls >= 300 ORDER BY val DESC LIMIT 1""",
        "context_sql": """SELECT SUM(batsman_runs) as pp_runs,
                  COUNT(*) as pp_balls,
                  ROUND(SUM(batsman_runs)*100.0/COUNT(*),1) as pp_sr,
                  COUNT(DISTINCT match_id) as matches
                  FROM deliveries WHERE over BETWEEN 0 AND 5 AND is_wide=0 AND batsman=?""",
        "stat_label": "strike rate in the powerplay (overs 1-6) among batters with 300+ balls",
        "prompt_hint": "The powerplay is when the field is up and the bowling is fresh. Most batters try to survive it. This player treats it as their personal showcase. Hint at the fear they put into opposition captains at the toss, and how quickly they can change the entire complexion of a match in 6 overs.",
    },
    {
        "id": "best_death_sr_batter",
        "sql": """SELECT batsman as answer,
                  ROUND(SUM(batsman_runs)*100.0/COUNT(*),1) as val,
                  COUNT(*) as balls
                  FROM deliveries WHERE over BETWEEN 16 AND 19 AND is_wide=0
                  GROUP BY batsman HAVING balls >= 200 ORDER BY val DESC LIMIT 1""",
        "context_sql": """SELECT SUM(batsman_runs) as death_runs,
                  COUNT(*) as death_balls,
                  ROUND(SUM(batsman_runs)*100.0/COUNT(*),1) as death_sr,
                  COUNT(DISTINCT match_id) as matches,
                  SUM(CASE WHEN batsman_runs=6 THEN 1 ELSE 0 END) as sixes
                  FROM deliveries WHERE over BETWEEN 16 AND 19 AND is_wide=0 AND batsman=?""",
        "stat_label": "strike rate in the death overs (overs 17-20) among batters with 200+ balls",
        "prompt_hint": "Most batters who come in at the death are already bracing themselves. This player arrives with a different mindset entirely. At a strike rate that borders on the mathematically implausible, they have redefined what is possible in the final 4 overs. Hint at their audacity and the specific carnage they cause — without naming them.",
    },
    {
        "id": "best_pp_economy_bowler",
        "sql": """SELECT bowler as answer,
                  ROUND(SUM(batsman_runs+extras)*6.0/NULLIF(SUM(CASE WHEN is_wide=0 AND is_no_ball=0 THEN 1 ELSE 0 END),0),2) as val,
                  SUM(CASE WHEN is_wide=0 AND is_no_ball=0 THEN 1 ELSE 0 END) as balls
                  FROM deliveries WHERE over BETWEEN 0 AND 5
                  GROUP BY bowler HAVING balls >= 300 ORDER BY val LIMIT 1""",
        "context_sql": """SELECT SUM(CASE WHEN dismissal_kind IS NOT NULL AND dismissal_kind NOT IN
                  ('run out','retired hurt','retired out','obstructing the field') THEN 1 ELSE 0 END) as wickets,
                  COUNT(DISTINCT match_id) as matches,
                  ROUND(SUM(batsman_runs+extras)*6.0/NULLIF(SUM(CASE WHEN is_wide=0 AND is_no_ball=0 THEN 1 ELSE 0 END),0),2) as pp_economy
                  FROM deliveries WHERE over BETWEEN 0 AND 5 AND bowler=?""",
        "stat_label": "economy rate in the powerplay among bowlers with 300+ balls — the most miserly new-ball bowler",
        "prompt_hint": "A new-ball bowler who can contain in T20 is worth their weight in gold. When the field restrictions are on and the batters are looking to launch, this bowler made a career of saying no. Hint at their pace, their precision, and the country that produced them.",
    },
    {
        "id": "best_death_economy_bowler",
        "sql": """SELECT bowler as answer,
                  ROUND(SUM(batsman_runs+extras)*6.0/NULLIF(SUM(CASE WHEN is_wide=0 AND is_no_ball=0 THEN 1 ELSE 0 END),0),2) as val,
                  SUM(CASE WHEN is_wide=0 AND is_no_ball=0 THEN 1 ELSE 0 END) as balls
                  FROM deliveries WHERE over BETWEEN 16 AND 19
                  GROUP BY bowler HAVING balls >= 200 ORDER BY val LIMIT 1""",
        "context_sql": """SELECT SUM(CASE WHEN dismissal_kind IS NOT NULL AND dismissal_kind NOT IN
                  ('run out','retired hurt','retired out','obstructing the field') THEN 1 ELSE 0 END) as wickets,
                  COUNT(DISTINCT match_id) as matches,
                  ROUND(SUM(batsman_runs+extras)*6.0/NULLIF(SUM(CASE WHEN is_wide=0 AND is_no_ball=0 THEN 1 ELSE 0 END),0),2) as death_economy
                  FROM deliveries WHERE over BETWEEN 16 AND 19 AND bowler=?""",
        "stat_label": "economy rate in the death overs among bowlers with 200+ balls — the best death bowler",
        "prompt_hint": "The death overs are where T20 matches are won and lost. Batters swing for the fences, captains panic, and most bowlers pray. This one thrived. Tell the story of a bowler whose death-over economy reads like a misprint. Hint at their weapon — the yorker, the slower ball — without naming them.",
    },
    {
        "id": "most_runs_middle_overs",
        "sql": """SELECT batsman as answer, SUM(batsman_runs) as val,
                  COUNT(DISTINCT match_id) as matches
                  FROM deliveries WHERE over BETWEEN 6 AND 15 AND is_wide=0
                  GROUP BY batsman ORDER BY val DESC LIMIT 1""",
        "context_sql": """SELECT SUM(batsman_runs) as mid_runs,
                  COUNT(*) as mid_balls,
                  ROUND(SUM(batsman_runs)*100.0/COUNT(*),1) as mid_sr,
                  COUNT(DISTINCT match_id) as matches
                  FROM deliveries WHERE over BETWEEN 6 AND 15 AND is_wide=0 AND batsman=?""",
        "stat_label": "runs in the middle overs (overs 7-15) — the engine room specialist",
        "prompt_hint": "The middle overs get no glamour. No power play buzz, no death-over fireworks. Just 10 overs of building, rotating, and occasionally accelerating. This player has accumulated more runs in this thankless phase than anyone in IPL history. Hint at their consistency, their longevity, and the way they quietly take a match away from the opposition.",
    },
    {
        "id": "best_middle_economy_bowler",
        "sql": """SELECT bowler as answer,
                  ROUND(SUM(batsman_runs+extras)*6.0/NULLIF(SUM(CASE WHEN is_wide=0 AND is_no_ball=0 THEN 1 ELSE 0 END),0),2) as val,
                  SUM(CASE WHEN is_wide=0 AND is_no_ball=0 THEN 1 ELSE 0 END) as balls
                  FROM deliveries WHERE over BETWEEN 6 AND 15
                  GROUP BY bowler HAVING balls >= 500 ORDER BY val LIMIT 1""",
        "context_sql": """SELECT SUM(CASE WHEN dismissal_kind IS NOT NULL AND dismissal_kind NOT IN
                  ('run out','retired hurt','retired out','obstructing the field') THEN 1 ELSE 0 END) as wickets,
                  COUNT(DISTINCT match_id) as matches,
                  ROUND(SUM(batsman_runs+extras)*6.0/NULLIF(SUM(CASE WHEN is_wide=0 AND is_no_ball=0 THEN 1 ELSE 0 END),0),2) as mid_economy
                  FROM deliveries WHERE over BETWEEN 6 AND 15 AND bowler=?""",
        "stat_label": "economy rate in the middle overs among bowlers with 500+ balls — the most miserly mid-innings bowler",
        "prompt_hint": "Batters look at the middle overs as the time to build. This bowler turned that window into a slow strangulation. With flight, variation, or relentless accuracy, they made run-scoring feel like hard labour for 10 overs. Hint at their bowling type and the fear they induced in opposing batters during that phase.",
    },
]


def get_player_context(player: str, context_sql: str, extra_param=None) -> dict:
    """Fetch supporting stats about the answer player/team."""
    try:
        if extra_param:
            rows = run_query(context_sql, (player, extra_param))
        else:
            rows = run_query(context_sql, (player,))
        return rows[0] if rows else {}
    except Exception:
        return {}


def generate_quiz_question() -> dict:
    """Generate a narrative/funda style question using Claude as the storyteller."""
    # ── Shuffled deck with cooling window ──────────────────
    # Guarantees all N templates appear before any repeat.
    # Last COOLING templates from previous round are held back
    # so they can't appear first in the next shuffle.
    COOLING = 5
    if "quiz_deck" not in st.session_state or not st.session_state.quiz_deck:
        # Build a fresh shuffled deck, excluding recently seen templates
        seen_ids = st.session_state.get("quiz_recent_ids", [])
        available = [t for t in QUIZ_TEMPLATES if t["id"] not in seen_ids]
        if not available:           # full reset once everything has cooled
            available = QUIZ_TEMPLATES[:]
        deck = available[:]
        random.shuffle(deck)
        st.session_state.quiz_deck = deck
    template = st.session_state.quiz_deck.pop(0)
    # Track recently used IDs for the cooling window
    recent = st.session_state.get("quiz_recent_ids", [])
    recent.append(template["id"])
    st.session_state.quiz_recent_ids = recent[-COOLING:]  # keep only last COOLING
    # ────────────────────────────────────────────────────────

    rows = run_query(template["sql"])
    if not rows:
        return None

    # Pick randomly from top 8 so the same template gives different answers each session
    correct      = random.choice(rows)
    answer_text  = correct["answer"]
    stat_value   = correct["val"]
    extra        = correct.get("extra", "")

    # Fetch context stats about the answer
    extra_param  = extra if template.get("extra_context_param") else None
    context      = get_player_context(answer_text, template["context_sql"], extra_param)

    # Build the payload for Claude
    payload = {
        "answer":      answer_text,
        "stat":        f"{stat_value} {template['stat_label']}",
        "extra":       extra,
        "context":     context,
        "prompt_hint": template["prompt_hint"],
    }

    system_prompt = """You are a cricket quiz writer in the tradition of KVizzing with the Comedians — India's premier narrative quiz show. Your questions make people think, not just recall.

PHILOSOPHY
Every question must lead to an "aha" moment — the answer should feel inevitable in hindsight, never arbitrary. The puzzle is always a pattern, contradiction, or consequence hiding in the data. A smart cricket fan who has never seen this stat before should be able to reason their way to the answer. Numbers and stats may appear as supporting context, but they must never be the puzzle itself. The cognitive work is inference, not arithmetic.

STRUCTURE — follow exactly:
- Exactly 3 sentences, then a final question. No more, no less.
- Sentence 1: the hook — the anomaly, the consequence, or the contradiction. Never explain cricket. Never give away the answer. Start with what makes this person or team singular.
- Sentences 2-3: the clues, woven naturally. Write them so they could plausibly point to 2-3 candidates — then the combination makes only one possible.
- Final line: a short direct question. "Who is this player?" / "Which team?" / "Who is this?"
- DO NOT name the answer anywhere in the question.

TECHNIQUE — pick exactly one per question based on what the underlying data naturally suggests:

1. THE CONTRADICTION — two facts that seem to oppose each other, resolved by one answer.
   Best for: players with paradoxical careers (e.g. most ducks AND most sixes).
   Example: "He has been dismissed for a duck more times than almost anyone in IPL history. He has also cleared the boundary more than almost anyone. Both facts belong to the same man. Who is this?"

2. THE ELIMINATION TRAIL — three clues in descending specificity. Each narrows the field.
   Best for: records, franchise history, multi-season dominance.
   Example: "He has played for four different IPL franchises. He has never won a title. Yet by one measure, he is the most valuable batter the competition has ever seen. Who is this?"

3. THE WRONG ASSUMPTION — open with what the reader will assume, then flip it.
   Best for: counter-intuitive stats, specialists who break the mould.
   Example: "You'd expect the IPL's all-time leading middle-overs run scorer to be a grinder — someone who accumulates, rotates, waits. This player's strike rate in those same overs is 130. Who is this?"

4. THE PEER COMPARISON — anchor the answer against a known reference point without naming either.
   Best for: stats whose scale only lands when compared to something familiar.
   Example: "Most IPL openers dream of one purple patch. This one has had more Player of the Match awards than some franchises have won games. Who is this?"

FIRST SENTENCE RULE — CRITICAL:
Never explain cricket to the reader. Never open with context or background. Start with the impact, the anomaly, the consequence.
Wrong: "The powerplay was meant to give bowlers an advantage..."
Right: "Some openers survive the powerplay. This one weaponises it."

TEAM QUESTIONS — additional rules:
- Never use colours, city, home ground, or trophy count in the first sentence. Instant giveaway. These belong only in sentences 2-3.
- Lead with the human story inside the franchise OR the rivals' experience of facing them — choose whichever the data suggests more naturally.
- Human story direction: "Three captains, three different eras, one identical outcome — a trophy at the end of the season."
- Rivals' experience direction: "Every franchise has beaten them at least once. None have figured out how to do it consistently."

EXAMPLE of the right tone (player question):
"Stumpings are the keeper's art — gloves, instinct, and reading the spinner before the batter does. Over 166 matches, this keeper turned that partnership into a science, picking off 47 victims who ventured just a step too far. No flashy run-outs, no diving catches — just quiet, precise, devastating. Who is this?"

Return ONLY a valid JSON object with exactly these keys:
  "question": 3 sentences + final question
  "fun_fact": one crisp sentence of surprising context, revealed after the answer
  "punchline": one short punchy line for the reveal — the "aha" moment
No markdown, no backticks, no preamble."""

    try:
        raw = call_claude(
            system=system_prompt,
            user_msg=json.dumps(payload, default=str),
            max_tokens=500,
        )
        raw = re.sub(r'^```\w*\n?', '', raw)
        raw = re.sub(r'\n?```$', '', raw).strip()
        result = json.loads(raw)
    except Exception:
        # Fallback to a simple question
        result = {
            "question": f"This player holds the IPL record for {stat_value} {template['stat_label']}. Who is this?",
            "fun_fact":  f"The answer is {answer_text}.",
            "punchline": f"{answer_text} — a true IPL legend.",
        }

    result["answer"]  = answer_text
    result["stat"]    = f"{stat_value} {template['stat_label']}"
    return result

# ══════════════════════════════════════════════════
#  SCORING ENGINE — 8 Algorithms Across 5 Sports
# ══════════════════════════════════════════════════

# ── 1. Dream11 Avg Points / Match ─────────────────
@st.cache_data(ttl=3600)
def score_dream11(min_matches: int = 10) -> pd.DataFrame:
    """
    Official Dream11 T20 point system averaged per match.
    Batting: 1/run + 1/boundary(4) + 2/six + milestone bonus (30+=4, 50+=8, 100+=16) - 2/duck.
    Bowling: 25/wkt + 8/maiden + 8/lbw-or-bowled + haul bonus (3W=4, 4W=8, 5W=16).
    Fielding: 8/catch + 12/stumping + 6/run-out.
    Inspired by: Fantasy Cricket (Dream11, 200M+ users).
    """
    bat = pd.DataFrame(run_query("""
        SELECT match_id, batsman AS player,
            SUM(batsman_runs) AS runs,
            SUM(CASE WHEN batsman_runs=4 THEN 1 ELSE 0 END) AS fours,
            SUM(CASE WHEN batsman_runs=6 THEN 1 ELSE 0 END) AS sixes,
            MAX(CASE WHEN player_dismissed=batsman
                AND dismissal_kind NOT IN ('run out','retired hurt','retired out','obstructing the field')
                THEN 1 ELSE 0 END) AS got_out
        FROM deliveries WHERE is_wide=0
        GROUP BY match_id, batsman
    """))
    if bat.empty:
        return pd.DataFrame()
    milestone = np.where(bat.runs >= 100, 16, np.where(bat.runs >= 50, 8, np.where(bat.runs >= 30, 4, 0)))
    duck = ((bat.runs == 0) & (bat.got_out == 1)).astype(int)
    bat['bat_pts'] = bat.runs + bat.fours + bat.sixes * 2 + milestone - duck * 2

    bowl = pd.DataFrame(run_query("""
        SELECT match_id, bowler AS player,
            SUM(CASE WHEN dismissal_kind IS NOT NULL
                AND dismissal_kind NOT IN ('run out','retired hurt','retired out','obstructing the field')
                THEN 1 ELSE 0 END) AS wkts,
            SUM(CASE WHEN dismissal_kind IN ('bowled','lbw') THEN 1 ELSE 0 END) AS bowled_lbw
        FROM deliveries WHERE is_wide=0 AND is_no_ball=0
        GROUP BY match_id, bowler
    """))
    maiden_df = pd.DataFrame(run_query("""
        SELECT match_id, bowler AS player, COUNT(*) AS maidens FROM (
            SELECT match_id, bowler, over,
                   SUM(batsman_runs+extras) AS r, COUNT(*) AS b
            FROM deliveries WHERE is_wide=0 AND is_no_ball=0
            GROUP BY match_id, bowler, over HAVING b>=6 AND r=0
        ) GROUP BY match_id, bowler
    """))
    if not bowl.empty:
        if not maiden_df.empty:
            bowl = bowl.merge(maiden_df, on=['match_id','player'], how='left')
            bowl['maidens'] = bowl['maidens'].fillna(0)
        else:
            bowl['maidens'] = 0
        haul_bonus = np.where(bowl.wkts >= 5, 16, np.where(bowl.wkts >= 4, 8, np.where(bowl.wkts >= 3, 4, 0)))
        bowl['bowl_pts'] = bowl.wkts * 25 + bowl.maidens * 8 + bowl.bowled_lbw * 8 + haul_bonus

    field = pd.DataFrame(run_query("""
        SELECT match_id, fielder_name AS player,
            SUM(CASE WHEN wicket_kind IN ('caught','caught and bowled') THEN 1 ELSE 0 END) AS catches,
            SUM(CASE WHEN wicket_kind='stumped' THEN 1 ELSE 0 END) AS stumpings,
            SUM(CASE WHEN wicket_kind='run out' THEN 1 ELSE 0 END) AS runouts
        FROM wicket_fielders
        GROUP BY match_id, fielder_name
    """))
    if not field.empty:
        field['field_pts'] = field.catches * 8 + field.stumpings * 12 + field.runouts * 6

    parts = [(bat, 'bat_pts'), (bowl, 'bowl_pts'), (field, 'field_pts')]
    frames = [df[['match_id','player', c]] for df, c in parts if not df.empty and c in df.columns]
    if not frames:
        return pd.DataFrame()
    combined = frames[0]
    for f in frames[1:]:
        combined = combined.merge(f, on=['match_id','player'], how='outer')
    combined = combined.fillna(0)
    combined['total'] = combined[[c for c in ['bat_pts','bowl_pts','field_pts'] if c in combined.columns]].sum(axis=1)
    agg = combined.groupby('player').agg(
        matches=('match_id','nunique'), avg_pts=('total','mean'), best=('total','max')
    ).reset_index()
    agg = agg[agg.matches >= min_matches]
    agg['avg_pts'] = agg.avg_pts.round(1)
    agg['best'] = agg.best.astype(int)
    return agg.sort_values('avg_pts', ascending=False).reset_index(drop=True)


# ── 2. Batting Pressure Index (BPI) ────────────────
@st.cache_data(ttl=3600)
def score_bpi(min_balls: int = 200) -> pd.DataFrame:
    """
    BPI = weighted_runs / balls × 100
    Weights: chasing(inning 2) × 1.5  |  death overs × 2.0  |  powerplay × 1.3  |  middle × 1.0
    BPI Premium = BPI − raw SR, revealing how much the batter elevates under pressure.
    Inspired by: CricViz / Jarrod Kimber situation-weighted batting analysis.
    """
    df = pd.DataFrame(run_query("""
        SELECT batsman AS player, over, batsman_runs AS runs, match_id, inning
        FROM deliveries WHERE is_wide=0
    """))
    if df.empty:
        return pd.DataFrame()
    df['phase_w'] = np.where(df.over <= 5, 1.3, np.where(df.over >= 16, 2.0, 1.0))
    df['chase_w'] = np.where(df.inning == 2, 1.5, 1.0)
    df['weighted'] = df.runs * df.phase_w * df.chase_w
    agg = df.groupby('player').agg(
        balls=('runs','count'), raw_runs=('runs','sum'),
        weighted=('weighted','sum'), matches=('match_id','nunique')
    ).reset_index()
    agg = agg[agg.balls >= min_balls]
    agg['bpi'] = (agg.weighted / agg.balls * 100).round(1)
    agg['raw_sr'] = (agg.raw_runs / agg.balls * 100).round(1)
    agg['bpi_premium'] = (agg.bpi - agg.raw_sr).round(1)
    return agg[['player','matches','balls','raw_sr','bpi','bpi_premium']].sort_values('bpi', ascending=False).reset_index(drop=True)


# ── 3. Player Impact Score (PI) ─────────────────────
@st.cache_data(ttl=3600)
def score_player_impact(min_balls: int = 200) -> pd.DataFrame:
    """
    Actual contribution vs historical phase baseline (runs per ball by phase).
    PI_bat  = Σ(runs_scored − expected_rpb) per ball faced.
    PI_bowl = Σ(expected_rpb − runs_conceded) per legal ball bowled.
    PI_total / matches = impact per match (positive = above average).
    Inspired by: ESPNcricinfo Player Impact metric.
    Phase baselines (from full 278K ball dataset): PP=1.25, MID=1.27, DEATH=1.61 rpb.
    """
    bl = pd.DataFrame(run_query("""
        SELECT CASE WHEN over BETWEEN 0 AND 5 THEN 'PP'
                    WHEN over BETWEEN 6 AND 15 THEN 'MID' ELSE 'DEATH' END AS phase,
               AVG(CAST(batsman_runs AS FLOAT)) AS bat_rpb,
               AVG(CAST(batsman_runs+extras AS FLOAT)) AS bowl_rpb
        FROM deliveries WHERE is_wide=0 AND is_no_ball=0 GROUP BY phase
    """))
    bat_bl = dict(zip(bl.phase, bl.bat_rpb))
    bowl_bl = dict(zip(bl.phase, bl.bowl_rpb))

    bat = pd.DataFrame(run_query("""
        SELECT batsman AS player,
               CASE WHEN over BETWEEN 0 AND 5 THEN 'PP'
                    WHEN over BETWEEN 6 AND 15 THEN 'MID' ELSE 'DEATH' END AS phase,
               COUNT(*) AS balls, SUM(batsman_runs) AS runs
        FROM deliveries WHERE is_wide=0 AND is_no_ball=0
        GROUP BY batsman, phase
    """))
    if bat.empty:
        return pd.DataFrame()
    bat['exp'] = bat.apply(lambda r: r.balls * bat_bl.get(r.phase, 1.25), axis=1)
    bat['surplus'] = bat.runs - bat.exp
    bat_agg = bat.groupby('player').agg(bat_balls=('balls','sum'), pi_bat=('surplus','sum')).reset_index()

    bowl = pd.DataFrame(run_query("""
        SELECT bowler AS player,
               CASE WHEN over BETWEEN 0 AND 5 THEN 'PP'
                    WHEN over BETWEEN 6 AND 15 THEN 'MID' ELSE 'DEATH' END AS phase,
               COUNT(*) AS balls, SUM(batsman_runs+extras) AS runs_conceded
        FROM deliveries WHERE is_wide=0 AND is_no_ball=0
        GROUP BY bowler, phase
    """))
    bowl_agg = None
    if not bowl.empty:
        bowl['exp'] = bowl.apply(lambda r: r.balls * bowl_bl.get(r.phase, 1.28), axis=1)
        bowl['savings'] = bowl.exp - bowl.runs_conceded   # positive = better than baseline
        bowl_agg = bowl.groupby('player').agg(bowl_balls=('balls','sum'), pi_bowl=('savings','sum')).reset_index()

    result = bat_agg.copy()
    if bowl_agg is not None:
        result = result.merge(bowl_agg, on='player', how='outer').fillna(0)
    else:
        result['bowl_balls'] = 0
        result['pi_bowl'] = 0.0
    result['pi_total'] = result.pi_bat + result.pi_bowl

    matches = pd.DataFrame(run_query("SELECT player_name AS player, COUNT(DISTINCT match_id) AS matches FROM playing_xi GROUP BY player_name"))
    result = result.merge(matches, on='player', how='left')
    result['matches'] = result['matches'].fillna(1).astype(int)
    result = result[result.bat_balls >= min_balls]
    result['pi_per_match'] = (result.pi_total / result.matches).round(2)
    result['pi_bat_pm']    = (result.pi_bat  / result.matches).round(2)
    result['pi_bowl_pm']   = (result.pi_bowl / result.matches).round(2)
    return result[['player','matches','bat_balls','pi_bat_pm','pi_bowl_pm','pi_per_match']].sort_values('pi_per_match', ascending=False).reset_index(drop=True)


# ── 4. Value Above Replacement (VAR) ───────────────
@st.cache_data(ttl=3600)
def score_var(min_balls: int = 200) -> pd.DataFrame:
    """
    Baseball WAR (Bill James / FanGraphs) adapted for T20.
    Replacement level = 25th percentile IPL player by runs-per-ball (batting) or
    75th percentile by economy (bowling — replacing with a cheaper option).
    VAR_bat  = (player_rpb − repl_rpb) × career_balls
    VAR_bowl = (repl_econ − player_econ) × legal_balls + wicket_premium
    Total VAR = combined run-impact above a replacement-level IPL player.
    Inspired by: Baseball WAR (Wins Above Replacement).
    """
    bat = pd.DataFrame(run_query("""
        SELECT batsman AS player,
               SUM(batsman_runs) AS runs,
               SUM(CASE WHEN is_wide=0 THEN 1 ELSE 0 END) AS balls,
               COUNT(DISTINCT match_id) AS matches
        FROM deliveries GROUP BY batsman
    """))
    bat = bat[bat.balls >= min_balls].copy()
    bat['rpb'] = bat.runs / bat.balls
    repl_rpb = float(bat.rpb.quantile(0.25))   # 25th pct ≈ 1.21 runs/ball
    bat['var_bat'] = ((bat.rpb - repl_rpb) * bat.balls).round(1)

    bowl = pd.DataFrame(run_query("""
        SELECT bowler AS player,
               SUM(batsman_runs+extras) AS runs_conceded,
               SUM(CASE WHEN is_wide=0 AND is_no_ball=0 THEN 1 ELSE 0 END) AS legal_balls,
               SUM(CASE WHEN dismissal_kind IS NOT NULL AND
                   dismissal_kind NOT IN ('run out','retired hurt','retired out','obstructing the field')
                   THEN 1 ELSE 0 END) AS wickets
        FROM deliveries GROUP BY bowler
    """))
    bowl = bowl[bowl.legal_balls >= min_balls].copy()
    bowl['rpb']  = bowl.runs_conceded / bowl.legal_balls
    bowl['wpb']  = bowl.wickets / bowl.legal_balls
    repl_bowl_rpb = float(bowl.rpb.quantile(0.75))   # replacement concedes more
    repl_wpb      = float(bowl.wpb.quantile(0.25))   # replacement takes fewer wickets
    bowl['var_bowl'] = (
        (repl_bowl_rpb - bowl.rpb) * bowl.legal_balls +  # runs saved
        (bowl.wpb - repl_wpb) * bowl.legal_balls * 25    # wicket premium
    ).round(1)

    result = bat[['player','balls','matches','var_bat']].rename(columns={'balls':'bat_balls','matches':'bat_matches'})
    result = result.merge(bowl[['player','legal_balls','var_bowl']], on='player', how='outer').fillna(0)
    result['matches']    = result['bat_matches'].clip(lower=1)
    result['total_var']  = (result.var_bat + result.var_bowl).round(1)
    result['var_pm']     = (result.total_var / result.matches).round(2)
    return result[['player','matches','var_bat','var_bowl','total_var','var_pm']].sort_values('total_var', ascending=False).reset_index(drop=True)


# ── 5. Cricket PER (Basketball Player Efficiency) ──
@st.cache_data(ttl=3600)
def score_cricket_per(min_matches: int = 10) -> pd.DataFrame:
    """
    Adapted from John Hollinger's NBA PER (2003 — Basketball Prospectus).
    PER = (runs + 4s×0.5 + 6s×1.5 + wkts×20 + catches×8 + stumpings×12 − wides×2 − no-balls) / matches
    Normalized so league mean = 15.0 (matching NBA PER convention).
    Allrounders shine here; pure batters or bowlers get penalized for missing the other dimension.
    Inspired by: NBA Player Efficiency Rating (John Hollinger).
    """
    bat_stats = pd.DataFrame(run_query("""
        SELECT p.player_name AS player, COUNT(DISTINCT p.match_id) AS matches,
               COALESCE(SUM(d.batsman_runs),0) AS runs,
               COALESCE(SUM(CASE WHEN d.batsman_runs=4 THEN 1 ELSE 0 END),0) AS fours,
               COALESCE(SUM(CASE WHEN d.batsman_runs=6 THEN 1 ELSE 0 END),0) AS sixes
        FROM playing_xi p
        LEFT JOIN deliveries d ON p.match_id=d.match_id AND p.player_name=d.batsman AND d.is_wide=0
        GROUP BY p.player_name
    """))
    bowl_stats = pd.DataFrame(run_query("""
        SELECT bowler AS player,
               SUM(CASE WHEN dismissal_kind IS NOT NULL AND
                   dismissal_kind NOT IN ('run out','retired hurt','retired out','obstructing the field')
                   THEN 1 ELSE 0 END) AS wickets,
               SUM(is_wide) AS wides, SUM(is_no_ball) AS noballs
        FROM deliveries GROUP BY bowler
    """))
    field_stats = pd.DataFrame(run_query("""
        SELECT fielder_name AS player,
               SUM(CASE WHEN wicket_kind IN ('caught','caught and bowled') THEN 1 ELSE 0 END) AS catches,
               SUM(CASE WHEN wicket_kind='stumped' THEN 1 ELSE 0 END) AS stumpings
        FROM wicket_fielders GROUP BY fielder_name
    """))
    result = (bat_stats
              .merge(bowl_stats, on='player', how='left')
              .merge(field_stats, on='player', how='left')
              .fillna(0))
    result = result[result.matches >= min_matches].copy()
    result['raw_per'] = (
        result.runs + result.fours * 0.5 + result.sixes * 1.5 +
        result.wickets * 20 + result.catches * 8 + result.stumpings * 12 -
        result.wides * 2 - result.noballs
    ) / result.matches
    mean_raw = result.raw_per.mean()
    result['cricket_per'] = (result.raw_per / mean_raw * 15).round(1) if mean_raw > 0 else result.raw_per.round(1)
    return result[['player','matches','cricket_per','runs','wickets','catches']].sort_values('cricket_per', ascending=False).reset_index(drop=True)


# ── 6. Clutch Index ────────────────────────────────
@st.cache_data(ttl=3600)
def score_clutch(min_clutch_balls: int = 50) -> pd.DataFrame:
    """
    Clutch situation: batting in 2nd innings, overs 16–19 (maximum pressure).
    Clutch Index = clutch_SR / overall_SR
    > 1.0 → performs BETTER under extreme pressure (true finisher).
    = 1.0 → same as career form.
    < 1.0 → drops off under pressure.
    Inspired by: NFL ESPN Clutch Factor / NBA Clutch Game Performance Index.
    """
    clutch = pd.DataFrame(run_query("""
        SELECT batsman AS player,
               SUM(batsman_runs) AS clutch_runs,
               COUNT(*) AS clutch_balls
        FROM deliveries
        WHERE inning=2 AND over>=16 AND is_wide=0
        GROUP BY batsman
    """))
    overall = pd.DataFrame(run_query("""
        SELECT batsman AS player,
               SUM(batsman_runs) AS runs, COUNT(*) AS balls
        FROM deliveries WHERE is_wide=0 GROUP BY batsman
    """))
    if clutch.empty or overall.empty:
        return pd.DataFrame()
    result = clutch[clutch.clutch_balls >= min_clutch_balls].merge(overall, on='player', how='left')
    result['clutch_sr']  = (result.clutch_runs / result.clutch_balls * 100).round(1)
    result['overall_sr'] = (result.runs / result.balls * 100).round(1)
    result['clutch_index'] = (result.clutch_sr / result.overall_sr.clip(lower=1.0)).round(3)
    return result[['player','clutch_balls','clutch_sr','overall_sr','clutch_index']].sort_values('clutch_index', ascending=False).reset_index(drop=True)


# ── 7. Dynamic Elo Rating ──────────────────────────
@st.cache_data(ttl=3600)
def score_elo(min_matches: int = 10, K: int = 32, start: int = 1500) -> pd.DataFrame:
    """
    Elo rating (Arpad Elo, 1960) applied to IPL batting performance.
    After each match: if pts > league_average → treated as 'win' → rating rises.
    Expected win probability uses current rating vs. fixed baseline (1500).
    Players who consistently outperform the field converge toward 1700+.
    K=32 (same as FIDE for active players).
    Inspired by: Chess Elo (FIDE) / ATP/WTA Tennis rankings.
    """
    match_pts = pd.DataFrame(run_query("""
        SELECT d.batsman AS player, d.match_id, m.date,
               SUM(d.batsman_runs) + SUM(CASE WHEN d.batsman_runs=6 THEN 2
                   WHEN d.batsman_runs=4 THEN 1 ELSE 0 END) AS pts
        FROM deliveries d JOIN matches m ON d.match_id=m.match_id
        WHERE d.is_wide=0
        GROUP BY d.batsman, d.match_id, m.date
    """))
    if match_pts.empty:
        return pd.DataFrame()

    league_mean = float(match_pts.pts.mean())
    match_pts = match_pts.sort_values(['date','match_id'])

    ratings: dict = {}
    counts:  dict = {}
    for row in match_pts.itertuples(index=False):
        p = row.player
        if p not in ratings:
            ratings[p] = float(start)
            counts[p] = 0
        r = ratings[p]
        expected = 1.0 / (1.0 + 10.0 ** ((start - r) / 400.0))
        actual   = 1.0 if row.pts > league_mean else 0.0
        ratings[p] = r + K * (actual - expected)
        counts[p] += 1

    result = pd.DataFrame([
        {'player': p, 'elo': round(ratings[p], 0), 'matches': counts[p]}
        for p in ratings
    ])
    return result[result.matches >= min_matches].sort_values('elo', ascending=False).reset_index(drop=True)


# ── 8. Consistency Score ───────────────────────────
@st.cache_data(ttl=3600)
def score_consistency(min_matches: int = 15) -> pd.DataFrame:
    """
    Reliability metric inspired by tennis first-serve percentage.
    CS = 1 / (1 + CV), where CV = std(match_pts) / mean(match_pts).
    Range 0–1: closer to 1.0 = delivers the same output every game.
    Captures the difference between a Kohli (high CS, floor never drops)
    and an explosive one-match wonder (low CS, huge variance).
    Inspired by: Tennis consistency stats (Borg/Djokovic first-serve reliability).
    """
    match_pts = pd.DataFrame(run_query("""
        SELECT batsman AS player, match_id,
               SUM(batsman_runs) + SUM(CASE WHEN batsman_runs=6 THEN 2 WHEN batsman_runs=4 THEN 1 ELSE 0 END) AS pts
        FROM deliveries WHERE is_wide=0
        GROUP BY batsman, match_id
    """))
    if match_pts.empty:
        return pd.DataFrame()
    agg = match_pts.groupby('player').agg(
        matches=('match_id','nunique'), mean_pts=('pts','mean'), std_pts=('pts','std')
    ).reset_index()
    agg = agg[agg.matches >= min_matches].copy()
    agg['cv'] = agg.std_pts / agg.mean_pts.clip(lower=0.1)
    agg['consistency'] = (1 / (1 + agg.cv)).round(3)
    agg['mean_pts'] = agg.mean_pts.round(1)
    agg['std_pts']  = agg.std_pts.round(1)
    return agg[['player','matches','consistency','mean_pts','std_pts']].sort_values('consistency', ascending=False).reset_index(drop=True)


# ── Score Registry ──────────────────────────────────
SCORE_REGISTRY = {
    "Dream11 Pts/Match": {
        "func": score_dream11, "key": "avg_pts",
        "cols": ["player","matches","avg_pts","best"],
        "desc": "Official Fantasy XI point system: batting runs + boundaries + wickets + fielding. Average points per match over career.",
        "origin": "🏏 Fantasy Cricket (Dream11)",
        "formula": "bat(run=1, 4=1, 6=2, 30+=4, 50+=8, 100+=16, duck=-2) + bowl(wkt=25, maiden=8, bowled/lbw+8, 3W+=4) + field(catch=8, stump=12, run-out=6)",
    },
    "Batting Pressure Index (BPI)": {
        "func": score_bpi, "key": "bpi",
        "cols": ["player","matches","balls","raw_sr","bpi","bpi_premium"],
        "desc": "Context-weighted strike rate. Chasing = 1.5×, death overs = 2×, powerplay = 1.3×. BPI Premium = how much pressure lifts a batter's output.",
        "origin": "📊 Cricket Analytics (CricViz-style)",
        "formula": "BPI = Σ(runs × chase_weight × phase_weight) / balls × 100",
    },
    "Player Impact (PI)": {
        "func": score_player_impact, "key": "pi_per_match",
        "cols": ["player","matches","bat_balls","pi_bat_pm","pi_bowl_pm","pi_per_match"],
        "desc": "Actual runs / savings vs historical phase baseline. Positive = above average. Combines batting surplus and bowling savings per match.",
        "origin": "📺 ESPNcricinfo Player Impact",
        "formula": "PI_bat = Σ(actual_runs − expected_rpb × balls). PI_bowl = Σ(expected_rpb × balls − runs_conceded). Phase baselines: PP=1.25, MID=1.27, DEATH=1.61 rpb.",
    },
    "Value Above Replacement (VAR)": {
        "func": score_var, "key": "total_var",
        "cols": ["player","matches","var_bat","var_bowl","total_var","var_pm"],
        "desc": "Runs produced/saved above a replacement-level IPL player. The cricket equivalent of baseball WAR. Higher = more irreplaceable.",
        "origin": "⚾ Baseball WAR (Bill James / FanGraphs)",
        "formula": "VAR_bat = (player_RPB − 25th_pct_RPB) × career_balls. VAR_bowl = (repl_economy − player_economy) × balls + wicket_premium.",
    },
    "Cricket PER": {
        "func": score_cricket_per, "key": "cricket_per",
        "cols": ["player","matches","cricket_per","runs","wickets","catches"],
        "desc": "All-round efficiency per match, normalized to league mean = 15. Allrounders score highest. Inspired by Hollinger's NBA PER.",
        "origin": "🏀 NBA PER (John Hollinger, 2003)",
        "formula": "Raw = (runs + 4s×0.5 + 6s×1.5 + wkts×20 + catches×8 − wides×2 − no-balls) / matches. Normalized: raw / mean × 15.",
    },
    "Clutch Index": {
        "func": score_clutch, "key": "clutch_index",
        "cols": ["player","clutch_balls","clutch_sr","overall_sr","clutch_index"],
        "desc": "Death-over chase SR vs career SR. >1.0 means the batter is a better finisher under pressure than their average suggests.",
        "origin": "🏈 NFL/NBA Clutch Factor (ESPN)",
        "formula": "Clutch SR = SR in overs 16-19, inning 2. Clutch Index = Clutch SR / Overall SR.",
    },
    "Dynamic Elo Rating": {
        "func": score_elo, "key": "elo",
        "cols": ["player","matches","elo"],
        "desc": "Rating updated after every match. Outperform the league average → rating rises. Sustained dominance → 1700+. Single purple patch won't fool it.",
        "origin": "♟️ Chess Elo (Arpad Elo, 1960) / 🎾 ATP Rankings",
        "formula": "new_elo = old_elo + K × (actual − expected). K=32. Expected = 1/(1+10^((1500−elo)/400)).",
    },
    "Consistency Score": {
        "func": score_consistency, "key": "consistency",
        "cols": ["player","matches","consistency","mean_pts","std_pts"],
        "desc": "1/(1+CV) where CV = std/mean of per-match fantasy points. Closer to 1.0 = same output every single game. Distinguish reliable engines from one-match wonders.",
        "origin": "🎾 Tennis Reliability (serve %, Borg-style)",
        "formula": "CS = 1 / (1 + std(match_pts) / mean(match_pts)). Range 0–1.",
    },
}


# ══════════════════════════════════════════════════
#  STREAMLIT UI
# ══════════════════════════════════════════════════

st.markdown("## 🏏 IPL AI Analyst")
st.caption("18 seasons · 278K deliveries · 1,169 matches · 8 scoring algorithms · Powered by Claude")

tab_ask, tab_scout, tab_fantasy, tab_quiz, tab_scores = st.tabs([
    "💬 Ask", "🔍 Scout", "⚡ Fantasy XI", "🧠 Quiz", "📊 Scores"
])


# ── Tab 1: Ask ─────────────────────────────────────
with tab_ask:
    st.markdown("Ask any question about IPL history in plain English.")
    question = st.text_input("Your question",
        placeholder="Who has the highest strike rate in death overs with 500+ balls faced?",
        key="ask_input")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Top 10 six hitters", key="ex1"):
            question = "Who has hit the most sixes in IPL history? Top 10."
    with col2:
        if st.button("Dhoni in finals", key="ex2"):
            question = "What is MS Dhoni's batting average and strike rate in IPL finals?"
    with col3:
        if st.button("Best economy death overs", key="ex3"):
            question = "Best economy rate in death overs (overs 16-19), minimum 500 balls bowled?"
    if question:
        with st.spinner("Generating SQL and querying..."):
            result = ask_ipl(question)
        st.markdown("#### Answer")
        st.markdown(result["answer"])
        with st.expander(f"SQL generated ({result.get('row_count', 0)} rows returned)"):
            st.code(result["sql"], language="sql")
        if result["data"]:
            st.dataframe(result["data"], use_container_width=True)


# ── Tab 2: Scout ───────────────────────────────────
with tab_scout:
    st.markdown("Generate scouting reports or compare two players head-to-head.")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        p1 = st.text_input("Player 1", placeholder="V Kohli", key="scout_p1")
    with col_p2:
        p2 = st.text_input("Player 2 (optional)", placeholder="RG Sharma", key="scout_p2")
    st.caption("Use short scorecard names. Not sure? Search below.")
    search_q = st.text_input("Search player name", placeholder="Type to search...", key="player_search")
    if search_q and len(search_q) >= 2:
        results = run_query(
            "SELECT player_name, player_full_name, bat_style, bowl_style FROM players "
            "WHERE LOWER(player_name) LIKE LOWER(?) ORDER BY player_name LIMIT 10",
            (f"%{search_q}%",))
        if results:
            st.dataframe(results, use_container_width=True, hide_index=True)
        else:
            st.info("No players found.")
    if st.button("Generate Report", key="scout_btn", disabled=not p1):
        with st.spinner("Pulling stats and generating report..."):
            result = scout_player(p1, p2 if p2 else None)
        if "error" in result:
            st.error(result["error"])
        else:
            st.markdown("#### Scouting Report")
            st.markdown(result["report"])
            with st.expander("Raw stats — " + p1):
                if result["stats1"].get("batting_by_phase"):
                    st.markdown("**Batting by phase**")
                    st.dataframe(result["stats1"]["batting_by_phase"], use_container_width=True, hide_index=True)
                if result["stats1"].get("batting_vs_type"):
                    st.markdown("**Batting vs pace/spin**")
                    st.dataframe(result["stats1"]["batting_vs_type"], use_container_width=True, hide_index=True)
            if p2 and result.get("stats2"):
                with st.expander("Raw stats — " + p2):
                    if result["stats2"].get("batting_by_phase"):
                        st.markdown("**Batting by phase**")
                        st.dataframe(result["stats2"]["batting_by_phase"], use_container_width=True, hide_index=True)


# ── Tab 3: Fantasy ─────────────────────────────────
with tab_fantasy:
    st.markdown("Get a Fantasy XI recommendation for any IPL matchup.")
    teams = [r["team_name"] for r in run_query("SELECT team_name FROM teams ORDER BY team_name")]
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        t1 = st.selectbox("Team 1", teams, index=0, key="fan_t1")
    with col_t2:
        t2 = st.selectbox("Team 2", teams, index=8, key="fan_t2")
    venue = st.text_input("Venue (optional)", placeholder="Wankhede", key="fan_venue")
    if st.button("Get Fantasy XI", key="fan_btn"):
        if t1 == t2:
            st.error("Pick two different teams.")
        else:
            with st.spinner("Analyzing form, matchups, and venue data..."):
                result = fantasy_advice(t1, t2, venue if venue else None)
            st.markdown("#### Recommendation")
            st.markdown(result["recommendation"])
            with st.expander("Head-to-head record"):
                if result["data"].get("h2h"):
                    st.dataframe(result["data"]["h2h"], use_container_width=True, hide_index=True)
            with st.expander(f"Recent form — {t1}"):
                if result["data"].get("team1_form"):
                    st.dataframe(result["data"]["team1_form"][:10], use_container_width=True, hide_index=True)
            with st.expander(f"Recent form — {t2}"):
                if result["data"].get("team2_form"):
                    st.dataframe(result["data"]["team2_form"][:10], use_container_width=True, hide_index=True)


# ── Tab 4: Quiz ────────────────────────────────────
with tab_quiz:
    st.markdown("Each question is a story. Read the clues, deduce the answer, type your guess.")

    # Session state
    for _k, _v in [
        ("quiz_score", 0), ("quiz_total", 0), ("quiz_q", None),
        ("quiz_revealed", False), ("quiz_guess", ""), ("quiz_correct", None),
    ]:
        if _k not in st.session_state:
            st.session_state[_k] = _v

    # Score bar
    if st.session_state.quiz_total > 0:
        pct = round(st.session_state.quiz_score * 100 / st.session_state.quiz_total)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Score", f"{st.session_state.quiz_score}/{st.session_state.quiz_total}")
        with c2: st.metric("Accuracy", f"{pct}%")
        with c3:
            if st.button("Reset", key="quiz_reset"):
                for _k in ["quiz_score","quiz_total","quiz_q","quiz_revealed","quiz_guess","quiz_correct"]:
                    st.session_state[_k] = 0 if _k in ["quiz_score","quiz_total"] else None if _k == "quiz_q" else False if _k == "quiz_revealed" else "" if _k == "quiz_guess" else None
                st.rerun()

    # New question button
    btn_label = "🎲 Next question" if st.session_state.quiz_q else "🎲 Start quiz"
    if st.button(btn_label, key="quiz_new", type="primary"):
        st.session_state.quiz_revealed = False
        st.session_state.quiz_guess    = ""
        st.session_state.quiz_correct  = None
        with st.spinner("Crafting your question..."):
            st.session_state.quiz_q = generate_quiz_question()
        st.rerun()

    q = st.session_state.quiz_q
    if q:
        # Question card
        st.markdown("---")
        st.markdown(f"#### {q['question']}")
        st.markdown("---")

        if not st.session_state.quiz_revealed:
            # Guess input
            guess = st.text_input(
                "Your answer",
                placeholder="Type your guess...",
                key="quiz_guess_input",
                label_visibility="collapsed",
            )
            col_reveal, col_hint = st.columns([2, 1])
            with col_reveal:
                if st.button("Reveal answer", key="quiz_reveal", disabled=not guess, type="primary"):
                    answer = q["answer"].strip().lower()
                    entered = guess.strip().lower()
                    # Accept if answer words appear in guess (handles partial names like "Gayle" for "CH Gayle")
                    answer_parts = answer.split()
                    is_correct = (
                        entered == answer or
                        any(part in entered for part in answer_parts if len(part) > 2) or
                        answer in entered
                    )
                    st.session_state.quiz_revealed = True
                    st.session_state.quiz_guess    = guess
                    st.session_state.quiz_correct  = is_correct
                    st.session_state.quiz_total   += 1
                    if is_correct:
                        st.session_state.quiz_score += 1
                    st.rerun()
            with col_hint:
                if st.button("Skip / give up", key="quiz_skip"):
                    st.session_state.quiz_revealed = True
                    st.session_state.quiz_guess    = ""
                    st.session_state.quiz_correct  = False
                    st.session_state.quiz_total   += 1
                    st.rerun()

        else:
            # Reveal state
            is_correct = st.session_state.quiz_correct
            guess      = st.session_state.quiz_guess

            if is_correct:
                st.success(f"✅  Correct! The answer is **{q['answer']}**")
                st.balloons()
            else:
                if guess:
                    st.error(f"❌  You said **{guess}**. The answer is **{q['answer']}**.")
                else:
                    st.warning(f"The answer is **{q['answer']}**.")

            st.markdown(f"*{q.get('punchline', '')}*")
            st.info(f"💡 {q.get('fun_fact', '')}")
            st.caption(f"Stat: {q.get('stat', '')}")



# ── Tab 5: Scores ──────────────────────────────────
with tab_scores:
    st.markdown("8 scoring algorithms — from cricket analytics, baseball, basketball, chess, and tennis.")

    lb_tab, player_tab, guide_tab = st.tabs(["🏆 Leaderboard", "🔬 Player Deep-Dive", "📖 Score Guide"])

    # ── Leaderboard ────────────────────────────────
    with lb_tab:
        sel = st.selectbox("Score system", list(SCORE_REGISTRY.keys()), key="lb_sel")
        info = SCORE_REGISTRY[sel]

        col_desc, col_origin = st.columns([3, 1])
        with col_desc:
            st.caption(info["desc"])
        with col_origin:
            st.caption(f"*{info['origin']}*")

        with st.spinner(f"Computing {sel}…"):
            try:
                df = info["func"]()
            except Exception as e:
                st.error(f"Error computing {sel}: {e}")
                df = pd.DataFrame()

        if not df.empty:
            top_n = st.slider("Show top N players", 10, 50, 20, key="lb_n")
            cols_to_show = [c for c in info["cols"] if c in df.columns]
            st.dataframe(df[cols_to_show].head(top_n), use_container_width=True, hide_index=True)

            if st.button("🤖 Claude's take on this leaderboard", key="lb_claude"):
                with st.spinner("Analysing…"):
                    top10 = df[cols_to_show].head(10).to_dict(orient='records')
                    analysis = call_claude(
                        system=(
                            "You are an elite cricket analyst. Interpret this scoring leaderboard. "
                            "Highlight what's surprising, who's missing from the top, and what the ranking "
                            "reveals about T20 cricket that raw averages hide. Under 200 words."
                        ),
                        user_msg=f"Score: {sel}\n{info['desc']}\n\nTop 10:\n{json.dumps(top10, indent=2, default=str)}",
                        max_tokens=500,
                    )
                    st.markdown(analysis)
        else:
            st.info("No data returned for the current threshold. Try lowering min_matches.")

    # ── Player Deep-Dive ───────────────────────────
    with player_tab:
        st.markdown("Load all 8 scores for any player and get a multi-metric analysis.")
        search = st.text_input("Player name", placeholder="V Kohli", key="scores_search")

        # Initialise session state buckets
        if "deep_dive_scores" not in st.session_state:
            st.session_state.deep_dive_scores = {}
        if "deep_dive_player" not in st.session_state:
            st.session_state.deep_dive_player = ""
        if "deep_dive_report" not in st.session_state:
            st.session_state.deep_dive_report = ""

        if search and st.button("Load All 8 Scores", key="scores_load"):
            # Clear stale report when loading a new player
            st.session_state.deep_dive_report = ""
            all_scores: dict = {}
            errors: list = []
            with st.spinner("Computing all 8 scores — this takes a few seconds on first load…"):
                for score_name, s_info in SCORE_REGISTRY.items():
                    try:
                        sdf = s_info["func"]()
                        if not sdf.empty and 'player' in sdf.columns:
                            mask = sdf.player.str.contains(search, case=False, na=False)
                            if mask.any():
                                idx   = int(sdf[mask].index[0])
                                row   = sdf.loc[idx]
                                total = len(sdf)
                                all_scores[score_name] = {
                                    "value":       round(float(row[s_info["key"]]), 2),
                                    "rank":        idx + 1,
                                    "total":       total,
                                    "percentile":  round((1 - (idx + 1) / total) * 100, 0),
                                    "origin":      s_info["origin"],
                                    "desc":        s_info["desc"],
                                }
                    except Exception as e:
                        errors.append(f"{score_name}: {e}")

            if errors:
                with st.expander("⚠️ Errors on some scores"):
                    for err in errors:
                        st.caption(err)

            if all_scores:
                st.session_state.deep_dive_scores = all_scores
                st.session_state.deep_dive_player = search
            else:
                st.session_state.deep_dive_scores = {}
                st.warning(f"No scoring data found for **{search}**. Check spelling — use short scorecard names like 'V Kohli' or 'MS Dhoni'.")

        # Render score card (persists across reruns)
        all_scores = st.session_state.deep_dive_scores
        player_loaded = st.session_state.deep_dive_player

        if all_scores:
            st.markdown(f"### {player_loaded} — Score Card")
            st.caption(f"Found in {len(all_scores)}/8 scoring systems")

            cols = st.columns(4)
            for i, (name, data) in enumerate(all_scores.items()):
                with cols[i % 4]:
                    pct = int(data['percentile'])
                    delta_str = f"Top {100 - pct}%" if pct >= 0 else None
                    st.metric(
                        label=name.split("(")[0].strip()[:22],
                        value=data["value"],
                        delta=delta_str,
                        help=f"{data['origin']}\nRank #{data['rank']} / {data['total']} players",
                    )

            if st.button("🤖 Multi-metric Claude analysis", key="deep_claude"):
                with st.spinner("Generating…"):
                    st.session_state.deep_dive_report = call_claude(
                        system=(
                            "You are an elite cricket analyst. Given 8 different scoring metrics for a player, "
                            "provide deep insight: which metrics show clear strengths or weaknesses, "
                            "what the cross-metric pattern reveals about their playing style and value, "
                            "and how they rank among IPL all-time greats on each dimension. "
                            "Be specific with numbers. Under 300 words."
                        ),
                        user_msg=f"Player: {player_loaded}\n\nScores:\n{json.dumps(all_scores, indent=2, default=str)}",
                        max_tokens=700,
                    )

            if st.session_state.deep_dive_report:
                st.markdown(st.session_state.deep_dive_report)

    # ── Score Guide ────────────────────────────────
    with guide_tab:
        st.markdown("### 8 Ways to Measure a Cricket Player")
        st.caption("Click any card to find out how it works and what it reveals.")

        import streamlit.components.v1 as components
        components.html("""<!DOCTYPE html>
<html><head><style>
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:transparent;padding:4px;}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;}
.flip-wrap{height:190px;perspective:1000px;cursor:pointer;}
.flip-inner{position:relative;width:100%;height:100%;transform-style:preserve-3d;transition:transform 0.5s cubic-bezier(0.4,0.2,0.2,1);border-radius:11px;}
.flip-wrap.flipped .flip-inner{transform:rotateY(180deg);}
.face{position:absolute;width:100%;height:100%;backface-visibility:hidden;-webkit-backface-visibility:hidden;border-radius:11px;padding:13px;display:flex;flex-direction:column;justify-content:space-between;}
.front{background:#1a1a2e;border:1px solid rgba(255,255,255,0.08);}
.front::before{content:"";position:absolute;top:0;left:0;width:4px;height:100%;border-radius:11px 0 0 11px;background:var(--accent);}
.sport-tag{font-size:9.5px;font-weight:600;letter-spacing:.05em;text-transform:uppercase;color:rgba(255,255,255,.4);}
.score-name{font-size:14px;font-weight:700;color:#fff;margin-top:4px;line-height:1.25;}
.tagline{font-size:11.5px;font-style:italic;color:#a0b4cc;margin-top:7px;line-height:1.45;flex-grow:1;}
.hint{font-size:9.5px;color:rgba(255,255,255,.85);text-align:right;}
.back{background:#0f3460;border:1px solid rgba(255,255,255,.1);transform:rotateY(180deg);}
.back::before{content:"";position:absolute;top:0;left:0;width:4px;height:100%;border-radius:11px 0 0 11px;background:var(--accent);}
.back-title{font-size:11.5px;font-weight:700;color:#fff;margin-bottom:5px;}
.explanation{font-size:11px;color:#c8d8e8;line-height:1.55;flex-grow:1;}
.example-box{margin-top:7px;background:rgba(255,255,255,.07);border-left:3px solid var(--accent);border-radius:0 5px 5px 0;padding:5px 8px;font-size:10.5px;color:#ddeeff;line-height:1.45;}
.example-box strong{color:#fff;}
</style></head><body><div class="grid">

<div class="flip-wrap" onclick="this.classList.toggle('flipped')">
  <div class="flip-inner">
    <div class="face front" style="--accent:#f7c948;">
      <div><div class="sport-tag">&#127987; Fantasy Cricket</div>
      <div class="score-name">Dream11 Pts/Match</div>
      <div class="tagline">Your fantasy manager&#39;s eye, averaged across a career.</div></div>
      <div class="hint">tap to flip &#x21A9;</div>
    </div>
    <div class="face back" style="--accent:#f7c948;">
      <div class="back-title">Dream11 Pts/Match</div>
      <div class="explanation">Runs, sixes, wickets, catches all earn points. Duck loses them. Added up and averaged across every match ever played.</div>
      <div class="example-box"><strong>KL Rahul</strong> ~49 pts/match across 135 games. Among the safest fantasy picks every season.</div>
      <div class="hint">tap to flip back &#x21A9;</div>
    </div>
  </div>
</div>

<div class="flip-wrap" onclick="this.classList.toggle('flipped')">
  <div class="flip-inner">
    <div class="face front" style="--accent:#e94560;">
      <div><div class="sport-tag">&#128202; Cricket Analytics</div>
      <div class="score-name">Batting Pressure Index</div>
      <div class="tagline">Who shows up when it matters, not just on flat tracks.</div></div>
      <div class="hint">tap to flip &#x21A9;</div>
    </div>
    <div class="face back" style="--accent:#e94560;">
      <div class="back-title">Batting Pressure Index</div>
      <div class="explanation">Weights every run by the pressure of the moment. A death-over chase run counts more than a first-innings powerplay run.</div>
      <div class="example-box"><strong>Rashid Khan</strong> BPI Premium +221. His clutch SR is nearly double his career average.</div>
      <div class="hint">tap to flip back &#x21A9;</div>
    </div>
  </div>
</div>

<div class="flip-wrap" onclick="this.classList.toggle('flipped')">
  <div class="flip-inner">
    <div class="face front" style="--accent:#00b4d8;">
      <div><div class="sport-tag">&#128250; ESPNcricinfo-inspired</div>
      <div class="score-name">Player Impact (PI)</div>
      <div class="tagline">Did you beat what an average IPL batter would have done?</div></div>
      <div class="hint">tap to flip &#x21A9;</div>
    </div>
    <div class="face back" style="--accent:#00b4d8;">
      <div class="back-title">Player Impact (PI)</div>
      <div class="explanation">Compares your runs to what an average IPL batter would have scored off those exact same balls, by phase. Positive = you added real value.</div>
      <div class="example-box"><strong>Travis Head</strong> +7.5 PI/match. Scores ~8 more runs per game than an average batter would.</div>
      <div class="hint">tap to flip back &#x21A9;</div>
    </div>
  </div>
</div>

<div class="flip-wrap" onclick="this.classList.toggle('flipped')">
  <div class="flip-inner">
    <div class="face front" style="--accent:#52b788;">
      <div><div class="sport-tag">&#9918; Baseball WAR</div>
      <div class="score-name">Value Above Replacement</div>
      <div class="tagline">How much worse would the team be without you?</div></div>
      <div class="hint">tap to flip &#x21A9;</div>
    </div>
    <div class="face back" style="--accent:#52b788;">
      <div class="back-title">Value Above Replacement</div>
      <div class="explanation">Runs scored above a fringe IPL replacement across your full career. Rewards both quality and longevity equally.</div>
      <div class="example-box"><strong>AB de Villiers</strong> 1,056 runs above replacement. The highest of any batter in IPL history.</div>
      <div class="hint">tap to flip back &#x21A9;</div>
    </div>
  </div>
</div>

<div class="flip-wrap" onclick="this.classList.toggle('flipped')">
  <div class="flip-inner">
    <div class="face front" style="--accent:#e07a5f;">
      <div><div class="sport-tag">&#127936; NBA PER (Hollinger)</div>
      <div class="score-name">Cricket PER</div>
      <div class="tagline">All-round efficiency in one number. League average is 15.</div></div>
      <div class="hint">tap to flip &#x21A9;</div>
    </div>
    <div class="face back" style="--accent:#e07a5f;">
      <div class="back-title">Cricket PER</div>
      <div class="explanation">Runs, wickets, catches per match combined and normalised. League average always = 15. Allrounders shine. Pure specialists get penalised.</div>
      <div class="example-box">Above 15 = better than average. Below 15 = worse. Simple as that.</div>
      <div class="hint">tap to flip back &#x21A9;</div>
    </div>
  </div>
</div>

<div class="flip-wrap" onclick="this.classList.toggle('flipped')">
  <div class="flip-inner">
    <div class="face front" style="--accent:#ff6b6b;">
      <div><div class="sport-tag">&#127944; NFL / NBA Clutch</div>
      <div class="score-name">Clutch Index</div>
      <div class="tagline">Does the big moment bring out the best or the worst?</div></div>
      <div class="hint">tap to flip &#x21A9;</div>
    </div>
    <div class="face back" style="--accent:#ff6b6b;">
      <div class="back-title">Clutch Index</div>
      <div class="explanation">Strike rate in overs 16-19 while chasing, divided by career strike rate. Above 1.0 means pressure brings out your best.</div>
      <div class="example-box">Score of <strong>1.2</strong> = 20% better under maximum pressure than in normal play. That is a finisher.</div>
      <div class="hint">tap to flip back &#x21A9;</div>
    </div>
  </div>
</div>

<div class="flip-wrap" onclick="this.classList.toggle('flipped')">
  <div class="flip-inner">
    <div class="face front" style="--accent:#b388ff;">
      <div><div class="sport-tag">&#9822; Chess Elo / ATP Tennis</div>
      <div class="score-name">Dynamic Elo Rating</div>
      <div class="tagline">A live ranking that moves after every single match.</div></div>
      <div class="hint">tap to flip &#x21A9;</div>
    </div>
    <div class="face back" style="--accent:#b388ff;">
      <div class="back-title">Dynamic Elo Rating</div>
      <div class="explanation">Beat the day&#39;s average and your rating rises. Fall below it and it drops. One great season will not take you to the top.</div>
      <div class="example-box"><strong>Virat Kohli</strong> Elo 1688 across 259 matches. 17 seasons of dominance keeps him in the top 3.</div>
      <div class="hint">tap to flip back &#x21A9;</div>
    </div>
  </div>
</div>

<div class="flip-wrap" onclick="this.classList.toggle('flipped')">
  <div class="flip-inner">
    <div class="face front" style="--accent:#4fc3f7;">
      <div><div class="sport-tag">&#127934; Tennis Reliability</div>
      <div class="score-name">Consistency Score</div>
      <div class="tagline">Can you set your watch by this player every match?</div></div>
      <div class="hint">tap to flip &#x21A9;</div>
    </div>
    <div class="face back" style="--accent:#4fc3f7;">
      <div class="back-title">Consistency Score</div>
      <div class="explanation">Measures how similar output is from match to match. Same average but wild swings = low score. Runs 0 to 1. Closer to 1 means always showing up.</div>
      <div class="example-box"><strong>B Sai Sudharsan</strong> 0.62. Barely any match-to-match fluctuation. A banker pick every week.</div>
      <div class="hint">tap to flip back &#x21A9;</div>
    </div>
  </div>
</div>

</div></body></html>""", height=440, scrolling=False)


# ── Footer ─────────────────────────────────────────
st.divider()
st.caption("Data: IPL Ball-by-Ball Database 2008–2025 · AI: Claude Sonnet · Scoring: 8 algorithms across 5 sports · Quiz: 15 question templates · Built with Streamlit")
