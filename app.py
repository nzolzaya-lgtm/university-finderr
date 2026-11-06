import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="University Finder", page_icon="🎓", layout="wide")

METRICS = {
    "ar_score": "Academic reputation",
    "er_score": "Employer reputation",
    "fsr_score": "Faculty/student ratio",
    "cpf_score": "Citations per faculty",
    "ifr_score": "International faculty",
    "isr_score": "International students",
    "irn_score": "Intl research network",
    "ger_score": "Employment outcomes",
    "sus_score": "Sustainability",
}
METRIC_COLS = list(METRICS)

PRESETS = {
    "Balanced": dict(ar_score=5, er_score=5, fsr_score=5, cpf_score=5, ifr_score=5,
                     isr_score=5, irn_score=5, ger_score=5, sus_score=5),
    "I want a good job after graduation": dict(ar_score=5, er_score=10, fsr_score=3, cpf_score=2,
                                              ifr_score=2, isr_score=3, irn_score=2, ger_score=10, sus_score=1),
    "I want to become a researcher": dict(ar_score=9, er_score=2, fsr_score=5, cpf_score=10,
                                          ifr_score=4, isr_score=3, irn_score=8, ger_score=2, sus_score=1),
    "Small classes and international life": dict(ar_score=4, er_score=4, fsr_score=10, cpf_score=2,
                                                 ifr_score=8, isr_score=9, irn_score=5, ger_score=4, sus_score=3),
}

RADAR_LABELS = [METRICS[c] for c in METRIC_COLS]


# ---------------------------------------------------------------
# 1. LOAD AND CLEAN DATA
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    path = "qs_2024.xlsx" if os.path.exists("qs_2024.xlsx") else "data/qs_2024.xlsx"
    cols = ["rank_2024", "rank_2023", "institution", "country_code", "country",
            "size", "focus", "research", "status",
            "ar_score", "ar_rank", "er_score", "er_rank", "fsr_score", "fsr_rank",
            "cpf_score", "cpf_rank", "ifr_score", "ifr_rank", "isr_score", "isr_rank",
            "irn_score", "irn_rank", "ger_score", "ger_rank", "sus_score", "sus_rank",
            "overall_score"]
    df = pd.read_excel(path, header=None, skiprows=4, names=cols)

    df["institution"] = df["institution"].astype(str).str.strip()
    df["country"] = df["country"].astype(str).str.strip()
    df = df.drop_duplicates(subset="institution")

    # ranks are text ("=76", "1001-1200", "1401+"): keep the label, make a number
    df["rank_label"] = df["rank_2024"].astype(str)
    df["rank_num"] = df["rank_label"].str.extract(r"(\d+)")[0].astype(int)

    # scores use "-" for missing: turn into numbers / NaN
    for c in [c for c in df.columns if c.endswith("_score")]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["size_label"] = df["size"].map({"XL": "Extra large", "L": "Large", "M": "Medium", "S": "Small"})
    df["focus_label"] = df["focus"].map({"FC": "Full comprehensive", "CO": "Comprehensive",
                                         "FO": "Focused", "SP": "Specialist"})
    df["research_label"] = df["research"].map({"VH": "Very high", "HI": "High",
                                               "MD": "Medium", "LO": "Low"})

    keep = (["institution", "country", "rank_label", "rank_num",
             "size_label", "focus_label", "research_label"]
            + METRIC_COLS + ["overall_score"])
    return df[keep].reset_index(drop=True)


@st.cache_data
def standardized_profiles(data):
    X = data[METRIC_COLS].copy()
    X = X.fillna(X.median())
    Z = ((X - X.mean()) / X.std()).fillna(0)
    return Z


def find_universities(data, weights, countries, sizes, focus, research, max_rank, min_coverage=0.6):
    pool = data[data["rank_num"] <= max_rank].copy()
    if countries:
        pool = pool[pool["country"].isin(countries)]
    if sizes:
        pool = pool[pool["size_label"].isin(sizes)]
    if focus:
        pool = pool[pool["focus_label"].isin(focus)]
    if research:
        pool = pool[pool["research_label"].isin(research)]
    if pool.empty:
        return pool

    w = pd.Series(weights, dtype=float)
    scores = pool[w.index]
    has = scores.notna()
    num = (scores.fillna(0) * w).sum(axis=1)
    den = (has * w).sum(axis=1)
    pool["coverage"] = den / w.sum()
    pool["match_score"] = (num / den.replace(0, np.nan)).round(1)
    pool = pool[pool["coverage"] >= min_coverage]
    return pool.sort_values("match_score", ascending=False)


def radar(rows, title):
    fig = go.Figure()
    for _, r in rows.iterrows():
        vals = r[METRIC_COLS].fillna(0).astype(float).tolist()
        fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=RADAR_LABELS + RADAR_LABELS[:1],
                                      fill="toself", opacity=0.45, name=r["institution"][:40]))
    fig.update_layout(title=title, polar=dict(radialaxis=dict(range=[0, 100])),
                      legend=dict(orientation="h", y=-0.15))
    return fig


df = load_data()
Z = standardized_profiles(df)

# ---------------------------------------------------------------
# 2. SIDEBAR
# ---------------------------------------------------------------
if "w_ar_score" not in st.session_state:
    for k, v in PRESETS["Balanced"].items():
        st.session_state[f"w_{k}"] = v


def apply_preset():
    for k, v in PRESETS[st.session_state["preset"]].items():
        st.session_state[f"w_{k}"] = v


st.sidebar.title("🎓 Your preferences")
st.sidebar.selectbox("Start from a student profile", list(PRESETS), key="preset", on_change=apply_preset)

st.sidebar.subheader("How important is each thing? (0-10)")
WEIGHTS = {}
for col, name in METRICS.items():
    WEIGHTS[col] = st.sidebar.slider(name, 0, 10, key=f"w_{col}")

st.sidebar.subheader("Filters")
countries = st.sidebar.multiselect("Country (empty = all)", sorted(df["country"].unique()))
sizes = st.sidebar.multiselect("University size", ["Small", "Medium", "Large", "Extra large"])
focus = st.sidebar.multiselect("Subject focus", ["Full comprehensive", "Comprehensive", "Focused", "Specialist"])
research = st.sidebar.multiselect("Research intensity", ["Very high", "High", "Medium", "Low"])
max_rank = st.sidebar.slider("Only universities ranked in the top", 50, 1500, 500, step=50)
top_n = st.sidebar.slider("How many results", 5, 30, 10)

# ---------------------------------------------------------------
# 3. HEADER
# ---------------------------------------------------------------
st.title("🎓 University Finder")
st.markdown("**Question:** Given what a student cares about and where they want to study, "
            "which universities are the best match, and which ones are similar to a university they already like?")

tab1, tab2, tab3, tab4 = st.tabs(["Find my university", "Find similar universities",
                                  "Explore the data", "About the data"])

# ---------------------------------------------------------------
# TAB 1: FINDER
# ---------------------------------------------------------------
with tab1:
    if sum(WEIGHTS.values()) == 0:
        st.warning("All weights are 0. Move at least one slider in the sidebar.")
    else:
        results = find_universities(df, WEIGHTS, countries, sizes, focus, research, max_rank)
        if results.empty:
            st.warning("No universities match these filters. Try a larger rank limit or fewer filters.")
        else:
            top = results.head(top_n)
            c1, c2, c3 = st.columns(3)
            c1.metric("Universities that match your filters", f"{len(results):,}")
            c2.metric("Best match", top.iloc[0]["institution"][:32])
            c3.metric("Countries in your top results", top["country"].nunique())

            left, right = st.columns([3, 2])
            fig = px.bar(top.sort_values("match_score"), x="match_score", y="institution",
                         orientation="h", color="match_score", color_continuous_scale="Greens",
                         hover_data=["country", "rank_label"],
                         title="Best matches for your preferences",
                         labels={"match_score": "Match score (0-100)", "institution": ""})
            fig.update_layout(coloraxis_showscale=False)
            left.plotly_chart(fig, width="stretch")
            left.caption("The match score is the average of the metrics you care about, weighted by your sliders.")

            if len(top) >= 3:
                right.plotly_chart(radar(top.head(3), "Your top 3: score on each metric"), width="stretch")
                right.caption("A larger shape means a higher score. Look for the metrics you care about most.")

            show = top[["institution", "country", "rank_label", "size_label",
                        "focus_label", "match_score"]].rename(columns={
                "institution": "University", "country": "Country", "rank_label": "QS rank",
                "size_label": "Size", "focus_label": "Focus", "match_score": "Match score"})
            st.dataframe(show, width="stretch", hide_index=True)

            st.subheader("Best countries for your preferences")
            by_country = (results.head(50).groupby("country")
                          .agg(universities=("institution", "count"), avg_match=("match_score", "mean"))
                          .sort_values("universities", ascending=False).head(10).reset_index())
            fig2 = px.bar(by_country, x="country", y="universities", color="avg_match",
                          color_continuous_scale="Blues", title="Countries among your 50 best matches",
                          labels={"universities": "Universities in top 50", "avg_match": "Average match"})
            st.plotly_chart(fig2, width="stretch")
            st.caption("Shows which countries have the most universities that fit your preferences.")

# ---------------------------------------------------------------
# TAB 2: SIMILAR
# ---------------------------------------------------------------
with tab2:
    st.markdown("Pick a university you like. The app compares the scores of all 9 metrics "
                "and finds the universities with the most similar profile (cosine similarity).")
    names = df.sort_values("rank_num")["institution"].tolist()
    default = names.index("Seoul National University") if "Seoul National University" in names else 0
    liked = st.selectbox("University you like", names, index=default)
    a, b = st.columns(2)
    sim_countries = a.multiselect("Only show universities in these countries (empty = anywhere)",
                                  sorted(df["country"].unique()), key="sim_countries")
    sim_n = b.slider("How many similar universities", 3, 15, 8)

    idx = df.index[df["institution"] == liked][0]
    v = Z.loc[idx].values
    sims = Z.values @ v / (np.linalg.norm(Z.values, axis=1) * np.linalg.norm(v) + 1e-9)
    out = df.copy()
    out["similarity"] = sims.round(3)
    out = out.drop(index=idx)
    if sim_countries:
        out = out[out["country"].isin(sim_countries)]
    out = out.sort_values("similarity", ascending=False).head(sim_n)

    row = df.loc[idx]
    st.info(f"**{row['institution']}**  |  {row['country']}  |  QS rank {row['rank_label']}")

    if out.empty:
        st.warning("No universities found with these filters.")
    else:
        l, r = st.columns(2)
        figs = px.bar(out.sort_values("similarity"), x="similarity", y="institution", orientation="h",
                      hover_data=["country", "rank_label"], title="Most similar universities",
                      labels={"similarity": "Similarity (1 = identical profile)", "institution": ""},
                      color="similarity", color_continuous_scale="Purples")
        figs.update_layout(coloraxis_showscale=False)
        l.plotly_chart(figs, width="stretch")
        l.caption("Similarity compares the shape of the scores, not the level, so a lower-ranked "
                  "university can still be a close match.")
        compare = pd.concat([df.loc[[idx]], out.head(2)])
        r.plotly_chart(radar(compare, "Your university vs its 2 closest matches"), width="stretch")
        r.caption("Shapes that overlap closely show universities with a similar profile.")
        st.dataframe(out[["institution", "country", "rank_label", "size_label", "similarity"]].rename(
            columns={"institution": "University", "country": "Country", "rank_label": "QS rank",
                     "size_label": "Size", "similarity": "Similarity"}),
            width="stretch", hide_index=True)

# ---------------------------------------------------------------
# TAB 3: EXPLORE
# ---------------------------------------------------------------
with tab3:
    st.subheader("Countries with the most universities in the top N")
    n_top = st.slider("Top N universities", 50, 1000, 200, step=50, key="n_top")
    counts = df[df["rank_num"] <= n_top]["country"].value_counts().head(15).reset_index()
    counts.columns = ["country", "universities"]
    fig = px.bar(counts.sort_values("universities"), x="universities", y="country", orientation="h",
                 title=f"Top 15 countries in the QS top {n_top}", labels={"country": ""})
    st.plotly_chart(fig, width="stretch")
    st.caption("A few countries (the United States, the United Kingdom, China) hold a large part of the top places.")

    l, r = st.columns(2)
    sub = df.dropna(subset=["ar_score", "er_score", "size_label"])
    fig = px.scatter(sub, x="ar_score", y="er_score", color="size_label", hover_name="institution",
                     hover_data=["country", "rank_label"], opacity=0.65,
                     title="Academic vs employer reputation",
                     labels={"ar_score": "Academic reputation", "er_score": "Employer reputation",
                             "size_label": "Size"})
    l.plotly_chart(fig, width="stretch")
    l.caption("Each dot is a university. Hover to see its name. The two reputations move together.")

    ranked = df.dropna(subset=["overall_score", "size_label"])
    fig = px.box(ranked, x="size_label", y="overall_score",
                 category_orders={"size_label": ["Small", "Medium", "Large", "Extra large"]},
                 title="Overall score by university size (ranked universities only)",
                 labels={"size_label": "Size", "overall_score": "Overall score"})
    r.plotly_chart(fig, width="stretch")
    r.caption("Size alone does not decide the score: small and extra large universities both reach the top.")

    corr = df[METRIC_COLS + ["overall_score"]].corr()
    names_c = [METRICS.get(c, "Overall score") for c in corr.columns]
    fig = px.imshow(corr.values, x=names_c, y=names_c, text_auto=".1f", zmin=-1, zmax=1,
                    color_continuous_scale="RdYlGn", title="Which scores move together?", aspect="auto")
    st.plotly_chart(fig, width="stretch")
    st.caption("Green means two scores rise together. Red means one is high when the other is low.")

    st.subheader("Does a better rank mean smaller classes or more international students?")
    metric_choice = st.radio("Metric", ["Faculty/student ratio", "International students"], horizontal=True)
    mcol = "fsr_score" if metric_choice.startswith("Faculty") else "isr_score"
    sub = df[df["rank_num"] <= 500].dropna(subset=[mcol])
    fig = px.scatter(sub, x="rank_num", y=mcol, hover_name="institution", opacity=0.6,
                     labels={"rank_num": "QS rank (1 = best)", mcol: metric_choice + " score"},
                     title=f"QS rank vs {metric_choice.lower()} score")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Correlation between rank number and this score: {sub['rank_num'].corr(sub[mcol]):.2f}. "
               "A negative number means better-ranked universities score higher.")

# ---------------------------------------------------------------
# TAB 4: ABOUT
# ---------------------------------------------------------------
with tab4:
    st.markdown("""
### About the data
- **Source:** QS World University Rankings 2024, official results table (Excel), Quacquarelli Symonds.
- **Link:** https://www.topuniversities.com/world-university-rankings
- **Size:** 1,498 universities in 104 countries, with 9 indicator scores and an overall score.

### The 9 scores (0-100)
Academic reputation, employer reputation, faculty/student ratio, citations per faculty,
international faculty, international students, international research network,
employment outcomes, and sustainability.

### Cleaning steps
1. Skipped the 4 header rows in the Excel file and gave clear column names.
2. Checked for duplicate university names (none found) and removed extra spaces.
3. Turned text ranks (`=76`, `1001-1200`, `1401+`) into a number, and kept the original text for display.
4. Turned `-` into missing values so scores are real numbers.
5. Replaced codes (XL, FC, VH) with readable labels.
6. Kept only the useful columns.

### How the match score works
Match score = sum(weight x score) / sum(weights of the metrics the university has).
A university needs scores for at least 60% of your total weight to appear, so it is not ranked on very little data.

### Limits
- Only about 600 universities have an overall score, so the finder uses the indicator scores.
- QS measures reputation and research more than teaching quality.
- Mongolian universities are not in this table.
""")
    st.dataframe(df.head(50), width="stretch")
