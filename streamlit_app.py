"""
streamlit_app.py — IPL AI Analyst
Single-file Streamlit app: SQLite + Claude API. No separate backend needed.

Run locally:  streamlit run streamlit_app.py
Deploy:       Push to GitHub → connect on share.streamlit.io
"""

import streamlit as st
import json
import sqlite3
import re
import random
from pathlib import Path

# ── Page config ────────────────────────────────────
st.set_page_config(
    page_title="IPL AI Analyst",
    page_icon="🏏",
    layout="wide",
)

# ── Custom styling ─────────────────────────────────
st.markdown("""
<style>
    .stApp { max-width: 1100px; margin: 0 auto; }
    div[data-testid="stExpander"] details summary span p {
        font-size: 0.9rem;
        color: #888;
    }
    .stat-card {
        background: var(--secondary-background-color);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 8px;
    }
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

5. teams (16 rows) — Canonical franchise names.
   Columns: team_id INT PK, team_name TEXT.

6. team_aliases (46 rows) — Historical name mappings.
   Columns: alias_id INT, team_id INT, alias_name TEXT.

7. wicket_fielders (10,540 rows) — Fielders involved in each dismissal.
   Columns: match_id INT, inning INT, over INT, ball INT,
   wicket_kind TEXT, player_out TEXT, fielder_name TEXT.

8. drs_reviews (872 rows), 9. replacements (473 rows).

JOIN KEYS: deliveries.match_id → matches.match_id.
Players join on player_name (short form like 'V Kohli').
SQLite syntax: use LOWER() + LIKE, not ILIKE.
For year filtering, use matches.year (INT).
Exclude wides when counting balls faced.
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
    """Lazy-init Anthropic client with the key from secrets."""
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
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
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

    stats["pom_count"] = run_query(
        "SELECT COUNT(*) as awards FROM matches WHERE player_of_match=?", (pn,)
    )[0]["awards"]

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
        user_msg=prompt,
        max_tokens=1500,
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


# ── /quiz logic ─────────────────────────────────────

QUIZ_TEMPLATES = [
    {
        "id": "most_sixes_career",
        "sql": """SELECT batsman as answer, SUM(CASE WHEN batsman_runs=6 THEN 1 ELSE 0 END) as val
                  FROM deliveries GROUP BY batsman ORDER BY val DESC LIMIT 4""",
        "template": "Who has hit the most sixes in IPL history with {val} maximums?",
        "answer_type": "player",
    },
    {
        "id": "most_runs_career",
        "sql": """SELECT batsman as answer, SUM(batsman_runs) as val
                  FROM deliveries GROUP BY batsman ORDER BY val DESC LIMIT 4""",
        "template": "Who is the all-time leading run scorer in IPL with {val} runs?",
        "answer_type": "player",
    },
    {
        "id": "most_wickets_career",
        "sql": """SELECT bowler as answer, SUM(CASE WHEN dismissal_kind IS NOT NULL
                  AND dismissal_kind NOT IN ('run out','retired hurt','retired out','obstructing the field')
                  THEN 1 ELSE 0 END) as val
                  FROM deliveries GROUP BY bowler ORDER BY val DESC LIMIT 4""",
        "template": "Which bowler has taken the most wickets in IPL history ({val})?",
        "answer_type": "player",
    },
    {
        "id": "most_pom",
        "sql": """SELECT player_of_match as answer, COUNT(*) as val FROM matches
                  WHERE player_of_match IS NOT NULL
                  GROUP BY player_of_match ORDER BY val DESC LIMIT 4""",
        "template": "Who has won the most Player of the Match awards in IPL ({val})?",
        "answer_type": "player",
    },
    {
        "id": "most_catches",
        "sql": """SELECT fielder_name as answer, COUNT(*) as val FROM wicket_fielders
                  WHERE wicket_kind='caught' GROUP BY fielder_name ORDER BY val DESC LIMIT 4""",
        "template": "Which player has taken the most catches in IPL history ({val})?",
        "answer_type": "player",
    },
    {
        "id": "highest_individual_score",
        "sql": """SELECT batsman as answer, SUM(batsman_runs) as val
                  FROM deliveries GROUP BY batsman, match_id ORDER BY val DESC LIMIT 4""",
        "template": "Who holds the record for the highest individual score in an IPL innings ({val})?",
        "answer_type": "player",
    },
    {
        "id": "most_team_wins",
        "sql": """SELECT winner as answer, COUNT(*) as val FROM matches
                  WHERE winner IS NOT NULL GROUP BY winner ORDER BY val DESC LIMIT 4""",
        "template": "Which franchise has the most wins in IPL history ({val})?",
        "answer_type": "team",
    },
    {
        "id": "most_titles",
        "sql": """SELECT winner as answer, COUNT(*) as val FROM matches
                  WHERE match_stage='Final' AND winner IS NOT NULL
                  GROUP BY winner ORDER BY val DESC LIMIT 4""",
        "template": "Which team has won the most IPL titles ({val})?",
        "answer_type": "team",
    },
    {
        "id": "best_economy_career",
        "sql": """SELECT bowler as answer,
                  ROUND(SUM(batsman_runs+extras)*6.0/NULLIF(SUM(CASE WHEN is_wide=0 AND is_no_ball=0 THEN 1 ELSE 0 END),0),2) as val
                  FROM deliveries GROUP BY bowler
                  HAVING SUM(CASE WHEN is_wide=0 AND is_no_ball=0 THEN 1 ELSE 0 END) >= 1000
                  ORDER BY val LIMIT 4""",
        "template": "Among bowlers with 1000+ balls, who has the best career economy rate ({val})?",
        "answer_type": "player",
    },
    {
        "id": "most_fours_career",
        "sql": """SELECT batsman as answer, SUM(CASE WHEN batsman_runs=4 THEN 1 ELSE 0 END) as val
                  FROM deliveries GROUP BY batsman ORDER BY val DESC LIMIT 4""",
        "template": "Who has hit the most fours in IPL history ({val})?",
        "answer_type": "player",
    },
    {
        "id": "highest_sr_career",
        "sql": """SELECT batsman as answer,
                  ROUND(SUM(batsman_runs)*100.0/NULLIF(SUM(CASE WHEN is_wide=0 THEN 1 ELSE 0 END),0),1) as val
                  FROM deliveries GROUP BY batsman
                  HAVING SUM(batsman_runs) >= 2000
                  ORDER BY val DESC LIMIT 4""",
        "template": "Among batters with 2000+ runs, who has the highest career strike rate ({val})?",
        "answer_type": "player",
    },
    {
        "id": "most_ducks",
        "sql": """WITH innings AS (
                    SELECT batsman, match_id, inning,
                           SUM(batsman_runs) as runs,
                           MAX(CASE WHEN player_dismissed=batsman THEN 1 ELSE 0 END) as got_out
                    FROM deliveries WHERE is_wide=0
                    GROUP BY batsman, match_id, inning
                  )
                  SELECT batsman as answer, SUM(CASE WHEN runs=0 AND got_out=1 THEN 1 ELSE 0 END) as val
                  FROM innings GROUP BY batsman ORDER BY val DESC LIMIT 4""",
        "template": "Which player has been dismissed for a duck (zero) the most times in IPL ({val})?",
        "answer_type": "player",
    },
    {
        "id": "most_stumpings",
        "sql": """SELECT fielder_name as answer, COUNT(*) as val FROM wicket_fielders
                  WHERE wicket_kind='stumped' GROUP BY fielder_name ORDER BY val DESC LIMIT 4""",
        "template": "Which wicketkeeper has the most stumpings in IPL history ({val})?",
        "answer_type": "player",
    },
    {
        "id": "final_winner_year",
        "sql": """SELECT winner as answer, year as val FROM matches
                  WHERE match_stage='Final' AND winner IS NOT NULL
                  ORDER BY RANDOM() LIMIT 1""",
        "template": "Which team won the IPL title in {val}?",
        "answer_type": "team",
        "needs_distractors": True,
    },
    {
        "id": "most_runs_season",
        "sql": """SELECT d.batsman as answer, SUM(d.batsman_runs) as val, m.year as extra
                  FROM deliveries d JOIN matches m ON d.match_id=m.match_id
                  GROUP BY d.batsman, m.year ORDER BY val DESC LIMIT 1""",
        "template": "Who scored the most runs in a single IPL season ({val} in {extra})?",
        "answer_type": "player",
        "needs_distractors": True,
    },
]


def generate_quiz_question() -> dict:
    """Generate a single quiz question from a random template."""
    template = random.choice(QUIZ_TEMPLATES)
    rows = run_query(template["sql"])

    if not rows:
        return None

    correct = rows[0]
    answer_text = correct["answer"]
    val = correct["val"]
    extra = correct.get("extra", "")

    # Build question text
    q_text = template["template"].format(val=val, extra=extra)

    # Build options
    if template.get("needs_distractors"):
        # Need to fetch plausible wrong answers
        if template["answer_type"] == "team":
            all_teams = run_query("SELECT team_name FROM teams ORDER BY RANDOM() LIMIT 10")
            distractors = [t["team_name"] for t in all_teams if t["team_name"] != answer_text][:3]
        else:
            # Get other top players
            distractors_q = run_query("""
                SELECT DISTINCT batsman as name FROM deliveries
                WHERE batsman != ? GROUP BY batsman
                ORDER BY SUM(batsman_runs) DESC LIMIT 10
            """, (answer_text,))
            distractors = [d["name"] for d in distractors_q][:3]
    else:
        distractors = [r["answer"] for r in rows[1:4]]

    # Pad if we don't have enough distractors
    if len(distractors) < 3:
        filler = run_query("SELECT player_name FROM players ORDER BY RANDOM() LIMIT 5")
        for f in filler:
            if f["player_name"] != answer_text and f["player_name"] not in distractors:
                distractors.append(f["player_name"])
            if len(distractors) >= 3:
                break

    options = [answer_text] + distractors[:3]
    random.shuffle(options)
    correct_idx = options.index(answer_text)

    # Claude polishes the question and adds a fun fact
    raw_payload = json.dumps({
        "question": q_text,
        "options": options,
        "correct_index": correct_idx,
        "correct_answer": answer_text,
        "stat_value": val,
    })

    try:
        polished = call_claude(
            system=(
                "You are an IPL cricket quiz host. Given a raw question and options, "
                "return a JSON object with:\n"
                '- "question": a polished, engaging version of the question\n'
                '- "options": the same 4 options in the same order (do NOT reorder)\n'
                '- "correct_index": the same correct index (do NOT change)\n'
                '- "fun_fact": a 1-2 sentence interesting fact about the answer\n\n'
                "Return ONLY valid JSON. No markdown, no backticks, no preamble."
            ),
            user_msg=raw_payload,
            max_tokens=300,
        )
        polished = re.sub(r'^```\w*\n?', '', polished)
        polished = re.sub(r'\n?```$', '', polished).strip()
        result = json.loads(polished)
        result["correct_index"] = correct_idx  # enforce original
        result["options"] = options  # enforce original
    except Exception:
        # Fallback: use raw question without Claude polish
        result = {
            "question": q_text,
            "options": options,
            "correct_index": correct_idx,
            "fun_fact": f"The answer is {answer_text} with a stat of {val}.",
        }

    return result


# ═══════════════════════════════════════════════════
#  STREAMLIT UI
# ═══════════════════════════════════════════════════

# Header
st.markdown("## 🏏 IPL AI Analyst")
st.caption("18 seasons · 278K deliveries · 1,169 matches · Powered by Claude")

tab_ask, tab_scout, tab_fantasy, tab_quiz = st.tabs(["💬 Ask", "🔍 Scout", "⚡ Fantasy XI", "🧠 Quiz"])

# ── Tab 1: Ask ─────────────────────────────────────
with tab_ask:
    st.markdown("Ask any question about IPL history in plain English.")

    question = st.text_input(
        "Your question",
        placeholder="Who has the highest strike rate in death overs with 500+ balls faced?",
        key="ask_input",
    )

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
            (f"%{search_q}%",),
        )
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


# ── Tab 4: Quiz ───────────────────────────────────
with tab_quiz:
    st.markdown("Test your IPL knowledge. Each question is generated from real stats.")

    # Session state for quiz
    if "quiz_score" not in st.session_state:
        st.session_state.quiz_score = 0
    if "quiz_total" not in st.session_state:
        st.session_state.quiz_total = 0
    if "quiz_q" not in st.session_state:
        st.session_state.quiz_q = None
    if "quiz_answered" not in st.session_state:
        st.session_state.quiz_answered = False
    if "quiz_selected" not in st.session_state:
        st.session_state.quiz_selected = None

    # Score display
    if st.session_state.quiz_total > 0:
        pct = round(st.session_state.quiz_score * 100 / st.session_state.quiz_total)
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Score", f"{st.session_state.quiz_score}/{st.session_state.quiz_total}")
        with col_s2:
            st.metric("Accuracy", f"{pct}%")
        with col_s3:
            if st.button("Reset score", key="quiz_reset"):
                st.session_state.quiz_score = 0
                st.session_state.quiz_total = 0
                st.session_state.quiz_q = None
                st.session_state.quiz_answered = False
                st.rerun()

    # Generate new question
    if st.button("🎲 New question" if st.session_state.quiz_q else "🎲 Start quiz", key="quiz_new", type="primary"):
        st.session_state.quiz_answered = False
        st.session_state.quiz_selected = None
        with st.spinner("Generating question..."):
            st.session_state.quiz_q = generate_quiz_question()
        st.rerun()

    # Display current question
    q = st.session_state.quiz_q
    if q:
        st.markdown(f"### {q['question']}")

        if not st.session_state.quiz_answered:
            # Show clickable options
            for i, option in enumerate(q["options"]):
                if st.button(option, key=f"quiz_opt_{i}", use_container_width=True):
                    st.session_state.quiz_answered = True
                    st.session_state.quiz_selected = i
                    st.session_state.quiz_total += 1
                    if i == q["correct_index"]:
                        st.session_state.quiz_score += 1
                    st.rerun()
        else:
            # Show results
            correct_idx = q["correct_index"]
            selected = st.session_state.quiz_selected
            is_correct = selected == correct_idx

            for i, option in enumerate(q["options"]):
                if i == correct_idx:
                    st.success(f"✅  {option}")
                elif i == selected and not is_correct:
                    st.error(f"❌  {option}")
                else:
                    st.markdown(f"⬜  {option}")

            if is_correct:
                st.balloons()
                st.markdown("**Correct!** 🎉")
            else:
                st.markdown(f"**Wrong.** The answer was **{q['options'][correct_idx]}**.")

            # Fun fact
            st.info(f"💡 {q.get('fun_fact', '')}")

            st.markdown("")
            st.button("Next question →", key="quiz_next_hint", on_click=lambda: None, disabled=True, help="Click '🎲 New question' above")


# ── Footer ─────────────────────────────────────────
st.divider()
st.caption("Data: IPL Ball-by-Ball Database 2008–2025 · AI: Claude Sonnet · Built with Streamlit")
