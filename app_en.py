import json
import os
import requests as _requests
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx

from fetch import fetch_all, AREA_KEYWORD_IDS, TIME_PERIODS
from graph import (
    build_all_graphs,
    compute_keyword_centrality, top_keywords_by_metric,
    search_keyword, compare_two_keywords,
    compare_keyword_trends, keyword_appearance_timeline,
    discipline_keyword_trends,
    top_journals_across_periods, institution_journal_profile,
    compute_inst_centrality, compare_inst_periods,
)
from models import KeywordCooccurrenceGraph, InstitutionCollaborationGraph

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="U.S. Area Studies: Concept Maps",
    page_icon="🗺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading data from OpenAlex…")
def load_data():
    return build_all_graphs(fetch_all())

all_graphs = load_data()
AREAS   = list(AREA_KEYWORD_IDS.keys())
PERIODS = list(TIME_PERIODS.keys())


# ── LLM helper ────────────────────────────────────────────────────────────────
def _get_api_key() -> str:
    """
    Retrieve the Anthropic API key from Streamlit secrets or environment variable.
    Priority: st.secrets > environment variable ANTHROPIC_API_KEY.
    """
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY", "")


def call_claude(system_prompt: str, user_prompt: str,
                max_tokens: int = 400) -> str:
    """
    Call the Anthropic API and return the assistant's text response.

    Requires an API key via one of:
      - .streamlit/secrets.toml  →  ANTHROPIC_API_KEY = "sk-ant-..."
      - Environment variable     →  ANTHROPIC_API_KEY=sk-ant-... streamlit run app_en.py

    Returns an error string if the key is missing or the call fails.
    """
    api_key = _get_api_key()
    if not api_key:
        return (
            "**AI analysis is not configured.**\n\n"
            "To enable this feature, provide an Anthropic API key using one of "
            "the two methods described in the README — either via "
            "`.streamlit/secrets.toml` or an environment variable. "
            "See the README for step-by-step instructions."
        )
    try:
        resp = _requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
                "x-api-key": api_key,
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return "".join(
            block["text"] for block in data.get("content", [])
            if block.get("type") == "text"
        )
    except Exception as e:
        return f"[AI analysis unavailable: {e}]"


# ── Drawing utilities ─────────────────────────────────────────────────────────
def draw_cooc_network(kg: KeywordCooccurrenceGraph,
                      highlight_kw: str | None = None,
                      max_nodes: int = 50):
    G = kg.graph
    if G.number_of_nodes() == 0:
        st.info("Not enough keyword co-occurrence data for this subset.")
        return
    top_nodes = {n for n, _ in
                 sorted(G.degree(), key=lambda x: x[1], reverse=True)[:max_nodes]}
    if highlight_kw and highlight_kw in G:
        top_nodes.add(highlight_kw)
    subG = G.subgraph(top_nodes)
    pos  = nx.spring_layout(subG, weight="weight", seed=42, k=1.2)
    centrality = compute_keyword_centrality(kg)
    ex, ey = [], []
    for u, v in subG.edges():
        x0,y0=pos[u]; x1,y1=pos[v]
        ex+=[x0,x1,None]; ey+=[y0,y1,None]
    nx_l,ny_l,texts,hovers,sizes,colors=[],[],[],[],[],[]
    for node in subG.nodes():
        x,y=pos[node]; nx_l.append(x); ny_l.append(y)
        texts.append(node)
        s=centrality.get(node,{}); ac=s.get("article_count",0); wd=s.get("weighted_degree",0)
        ti=" · ".join(i for i,_ in s.get("top_institutions",[])[:2])
        hovers.append(f"<b>{node}</b><br>Articles: {ac} | Weight: {wd}"
                      f"<br>Institutions: {ti or 'N/A'}")
        sizes.append(max(8,min(32,ac**0.6*5)))
        colors.append("#e67e22" if node==highlight_kw else wd)
    fig=go.Figure(data=[
        go.Scatter(x=ex,y=ey,mode="lines",
                   line=dict(width=0.5,color="#cccccc"),hoverinfo="none"),
        go.Scatter(x=nx_l,y=ny_l,mode="markers+text",text=texts,
                   textposition="top center",textfont=dict(size=9),
                   hovertext=hovers,hoverinfo="text",
                   marker=dict(size=sizes,color=colors,colorscale="Blues",
                               showscale=False,line=dict(width=1,color="white"))),
    ],layout=go.Layout(
        showlegend=False,hovermode="closest",
        margin=dict(b=5,l=5,r=5,t=5),height=460,
        xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
        yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
        plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig,use_container_width=True)


def draw_inst_network(ig: InstitutionCollaborationGraph, max_nodes: int = 40):
    G = ig.graph
    if G.number_of_nodes() == 0:
        st.info("No multi-institution collaboration data for this subset.")
        return
    top_nodes={n for n,_ in sorted(G.degree(),key=lambda x:x[1],reverse=True)[:max_nodes]}
    subG=G.subgraph(top_nodes)
    pos=nx.spring_layout(subG,weight="weight",seed=42,k=1.5)
    ex,ey=[],[]
    for u,v in subG.edges():
        x0,y0=pos[u];x1,y1=pos[v];ex+=[x0,x1,None];ey+=[y0,y1,None]
    nx_l,ny_l,texts,hovers,sizes=[],[],[],[],[]
    for node in subG.nodes():
        x,y=pos[node];nx_l.append(x);ny_l.append(y)
        short=node.replace("University of ","U. of ")
        texts.append(short[:22])
        ac=subG.nodes[node].get("article_count",0)
        hovers.append(f"<b>{node}</b><br>Articles: {ac} | Collaborators: {subG.degree(node)}")
        sizes.append(max(8,min(28,ac**0.6*4)))
    fig=go.Figure(data=[
        go.Scatter(x=ex,y=ey,mode="lines",
                   line=dict(width=1,color="#aac4e0"),hoverinfo="none"),
        go.Scatter(x=nx_l,y=ny_l,mode="markers+text",text=texts,
                   textposition="top center",textfont=dict(size=8),
                   hovertext=hovers,hoverinfo="text",
                   marker=dict(size=sizes,color="#2980b9",
                               line=dict(width=1,color="white"))),
    ],layout=go.Layout(
        showlegend=False,hovermode="closest",
        margin=dict(b=5,l=5,r=5,t=5),height=420,
        xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
        yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
        plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig,use_container_width=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🗺 U.S. Area Studies")
st.sidebar.caption("Concept Maps, 1960–2025")
st.sidebar.divider()
mode = st.sidebar.radio("Select a mode", [
    "🏠 Overview",
    "① Keyword Explorer",
    "② Period Comparison",
    "③ Keyword Comparison",
    "④ Publication Ecology",
    "🎯 Key Findings",
    "💡 Methodological Reflection",
], label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption("Data: [OpenAlex](https://openalex.org)  ·  US institutions  ·  Article type")


# ── Global session_state initialization ──────────────────────────────────────
# Must run on every rerender, before any mode block, so keys always exist.
for _key, _default in [
    ("explorer_analysis", None),   # AI output for Keyword Explorer
    ("cmp_result",        None),   # comparison result dict
    ("cmp_area",          None),
    ("cmp_period",        None),
    ("cmp_analysis",      None),   # AI output for Keyword Comparison
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default


# ════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if mode == "🏠 Overview":
    st.title("U.S. Area Studies: Concept Maps Across Time")
    st.subheader("Mapping the intellectual landscape of U.S. area studies research, 1960–2025")
    st.markdown("""
Area studies — the interdisciplinary academic project of understanding world regions —
has been central to U.S. university life for more than half a century.
This tool uses network analysis to visualize *how* that intellectual project has evolved:
which concepts dominated in each era, which institutions led the field,
and how the boundaries between research topics have shifted over time.
    """)
    st.divider()

    # 1. How the graphs are built
    st.header("1. How the graphs are built")
    st.markdown("""
This project constructs **three types of graphs**, all sourced from
[OpenAlex](https://openalex.org), a free and open scholarly metadata database
covering millions of academic articles.
    """)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("🔵 Primary: Keyword Co-occurrence Network")
        st.markdown("""
**Node** = a subject keyword (e.g., *China*, *colonialism*, *democracy*)
sourced from OpenAlex's keywords field, with discipline labels filtered out.

**Edge** = two keywords appear together in the same article.

**Edge weight** = number of articles where both keywords co-occur.

Each node also carries: which institutions most use this keyword,
which journals most publish it, and which discipline lenses accompany it.

*This is the primary analytical tool.
The structure of this network reveals the conceptual map of a research field —
something no table of articles can show.*
        """)
    with col2:
        st.subheader("🟢 Secondary: Journal–Institution Bipartite Graph")
        st.markdown("""
**Node type A** = academic journal (e.g., *Journal of Asian Studies*)
sourced from OpenAlex's primary_location.source field.

**Node type B** = US university or research institution
sourced from OpenAlex's authorships.institutions field.

**Edge** = an institution has published at least one article in that journal.

**Edge weight** = number of articles.

*This graph reveals the publication infrastructure of each field:
which universities dominate which journals,
and how that relationship changes over time.*
        """)
    with col3:
        st.subheader("🟡 Tertiary: Institution Collaboration Network")
        st.markdown("""
**Node** = US institution.

**Edge** = two institutions share an author on the same article (co-authorship).

**Edge weight** = number of co-authored articles.

**Note on sparsity:** Humanities and social-science area studies are
dominated by single-author publications. Collaboration edges are rare —
this graph documents an exception, not the norm.

*Used as a supplementary view in the Publication Ecology module.*
        """)

    # Data overview table — placed directly under graph structure explanation
    st.divider()
    st.subheader("Dataset at a glance")
    st.caption("All data sourced from OpenAlex. Filter: US institutions · article type · area studies keyword.")
    rows = []
    for area in AREAS:
        for period in PERIODS:
            cs  = all_graphs[area][period]["cooc"].summary()
            js  = all_graphs[area][period]["ji"].summary()
            ins = all_graphs[area][period]["inst"].summary()
            rows.append({
                "Area": area, "Period": period,
                "Articles Count": cs["num_articles"],
                "Keyword nodes": cs["num_keywords"],
                "Co-occurrence edges": cs["num_edges"],
                "Journals": js["num_journals"],
                "Institutions": js["num_institutions"],
                "Collab edges": ins["num_edges"],
            })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.divider()

    # 2. Time periods
    st.header("2. Why these three time periods?")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.subheader("1960–1975: Cold War Expansion and the Institutionalization of Area Studies")
        st.markdown("""
This period marked the rapid expansion of Area Studies in the United States.
During the Cold War, governments and universities invested heavily in the study of regions 
such as East Asia, the Soviet Union, the Middle East, Africa, and Latin America. 
The goal was not only academic understanding, but also political and strategic knowledge.
**Area Studies became institutionalized** through the creation of interdisciplinary centers 
that combined history, political science, anthropology, economics, and language training. 
At the same time, **decolonization in Asia and Africa** brought newly independent nations into global academic attention. 
Scholars increasingly focused on modernization, nationalism, revolution, and development.

This period captures the **institutionalization** of area studies —
when the field was defined by geopolitical urgency. 
It is important because it established the basic structure and methodology of modern Area Studies.
        """)
    with col_b:
        st.subheader("1990–2000: Globalization and the Crisis of Traditional Area Studies")
        st.markdown("""
After the end of the Cold War and the collapse of the Soviet Union, 
Area Studies entered a period of **reflection and transformation**. 
Many scholars questioned whether traditional region-based research was still necessary in an increasingly globalized world.
During the 1990s, concepts such as globalization, transnationalism, migration, and global flows became central in academia. 
Researchers argued that social, cultural, and economic processes could no longer be understood within fixed regional boundaries. 
At the same time, postcolonial scholars criticized Area Studies for reproducing Western perspectives and unequal power relations in knowledge production.
As a result, Area Studies began to move away from rigid geographic boundaries and toward more comparative, transnational, and theoretically critical approaches.

This period captures a **moment of crisis and redefinition**.
It is important because it challenged the intellectual foundations of traditional Area Studies and pushed the field to reinvent itself.
        """)
    with col_c:
        st.subheader("2015–2025: Digital Transformation and the Return of Geopolitics")
        st.markdown("""
In recent years, Area Studies has gained renewed importance 
due to major global crises and geopolitical tensions. 
Issues such as U.S.–China competition, the Russia–Ukraine war, climate change, 
pandemics, migration, and technological rivalry have made regional expertise valuable again. 
Governments and universities have renewed interest in language training, regional knowledge, and strategic studies.
At the same time, Area Studies has been transformed by digital methods and interdisciplinary approaches. 
Scholars increasingly use GIS, digital archives, network analysis, text mining, and data visualization to study regions. 
Research has also become more connected to global issues such as environmental justice, indigenous knowledge, and climate governance.

This period captures a **return of geopolitical demand**,
now centered on different actors and frameworks than the Cold War era.
        """)

    st.divider()

    # 3. Why these three fields
    st.header("3. Why Asian Studies, African Studies, and Latin American Studies?")
    st.markdown("""
This project chose to focus on Asian Studies, African Studies, and Latin American Studies 
because these three fields have played a central role in the historical development of Area Studies. 
Since the mid-twentieth century, they have attracted significant academic and political attention 
due to decolonization, Cold War geopolitics, globalization, and the growing importance of the Global South in international affairs.

European Studies was deliberately excluded from this analysis because the
United States *already had* deep institutional knowledge of Europe —
area studies funding targeted regions perceived as poorly understood.
Middle Eastern Studies, while important, has a distinct periodization
(strongly shaped by the 1973 oil crisis, the Iranian Revolution, and post-9/11
security imperatives) that would require separate treatment.

The three selected fields thus share a **common genealogy**
(third-world Cold War strategy) while showing divergent trajectories
in the post-Cold War and contemporary periods — making them ideal
for comparative analysis.
    """)


# ════════════════════════════════════════════════════════════════════════════
# MODE 1 — Keyword Explorer
# ════════════════════════════════════════════════════════════════════════════
elif mode == "① Keyword Explorer":
    st.title("① Keyword Explorer")
    st.markdown("""
Search for any research topic and see how it is positioned in the academic
concept network for a specific area studies field and time period.

In a keyword co-occurrence network, two words are **connected** when they appear
together in the same article. A keyword's *neighbors* reveal the intellectual
frame through which scholars approach that topic.
A keyword's *centrality* tells you how central or bridging that concept is
across the entire field.

**How to read the network graph:**
- **Node size** = number of articles containing this keyword
- **Node color** (blue intensity) = co-occurrence weight
- **Orange node** = your search result, highlighted
- **Edges** = the two keywords appear together in at least one article
    """)
    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1: sel_area   = st.selectbox("Area studies field", AREAS)
    with c2: sel_period = st.selectbox("Time period", PERIODS)
    with c3: query = st.text_input("Search keyword", placeholder="e.g. China")

    st.caption("""
💡 **Suggested keywords to explore:**
*Asian Studies* — China, Japan, Korea, Vietnam, Cold War, Belt and Road, democracy, nationalism ·
*African Studies* — Nigeria, Ghana, Kenya, South Africa, China, colonialism, development, oil, diaspora ·
*Latin American Studies* — Mexico, Brazil, Cuba, migration, revolution, indigenous, trade, poverty
    """)
    st.button("Search", type="primary")

    if query:
        kg      = all_graphs[sel_area][sel_period]["cooc"]
        matches = search_keyword(kg, query)
        if not matches:
            st.warning(f"No keyword matching '{query}' found. "
                       f"Try a broader term or check the suggestions above.")
        else:
            for m in matches[:3]:
                with st.expander(
                    f"**{m['keyword']}** — {m['article_count']} article(s)",
                    expanded=(matches.index(m) == 0)
                ):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Articles", m["article_count"])
                    col2.metric("Co-occ weight", m["weighted_degree"])
                    col3.metric("Betweenness", f"{m['betweenness_centrality']:.3f}")

                    st.markdown("""
**Metric guide:**
- **Co-occurrence weight** — total number of co-occurrence links
  (how "active" this keyword is in the network)
- **Betweenness centrality** — how often this keyword lies on the shortest path
  between two others; high values indicate a *bridge concept* connecting otherwise
  separate research clusters
                    """)

                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.subheader("Strongest co-occurrers")
                        st.caption("Keywords most often discussed alongside this one")
                        if m["neighbors"]:
                            st.dataframe(
                                pd.DataFrame(m["neighbors"],
                                             columns=["Keyword","Co-occurrence count"]),
                                hide_index=True, use_container_width=True)
                    with col_b:
                        st.subheader("Top institutions")
                        st.caption("US universities most active on this topic")
                        if m["top_institutions"]:
                            st.dataframe(
                                pd.DataFrame(m["top_institutions"],
                                             columns=["Institution","Articles"]),
                                hide_index=True, use_container_width=True)
                    with col_c:
                        st.subheader("Top journals")
                        st.caption("Journals most publishing on this topic")
                        if m["top_journals"]:
                            st.dataframe(
                                pd.DataFrame(m["top_journals"],
                                             columns=["Journal","Articles"]),
                                hide_index=True, use_container_width=True)

                    if m["discipline_co"]:
                        st.subheader("Research lenses applied")
                        st.caption(
                            "Discipline keywords (e.g. *Sociology*, *History*, *Law*) "
                            "most frequently accompanying this topic — "
                            "revealing *how* scholars approach it"
                        )
                        st.bar_chart(
                            pd.DataFrame(m["discipline_co"],
                                         columns=["Discipline","Co-occurrences"]
                                         ).set_index("Discipline"),
                            height=200)

            # ── LLM analysis for Keyword Explorer ────────────────────────────
            st.divider()
            st.subheader("🤖 AI Analysis")
            st.caption(
                "An AI reads the network data above and provides contextual "
                "interpretation, highlights what is surprising, and suggests "
                "further directions to explore."
            )
            if st.button("Generate AI analysis", key="llm_explorer"):
                m = matches[0]
                neighbors_str = ", ".join(
                    f"{kw} ({cnt})" for kw, cnt in m["neighbors"][:6])
                institutions_str = ", ".join(
                    inst for inst, _ in m["top_institutions"][:4])
                discipline_str = ", ".join(
                    d for d, _ in m["discipline_co"][:4])

                system = (
                    "You are an expert in U.S. area studies and academic network analysis. "
                    "Given data from a keyword co-occurrence network, provide a concise, "
                    "insightful interpretation in 3–4 short paragraphs. "
                    "Focus on: (1) what the co-occurrence pattern reveals about how this "
                    "topic is framed in the literature, (2) anything surprising or "
                    "counterintuitive, (3) one specific follow-up question worth exploring. "
                    "Be direct and analytical. Do not repeat the numbers back verbatim."
                )
                user = (
                    f"Keyword: '{m['keyword']}'\n"
                    f"Field: {sel_area}, Period: {sel_period}\n"
                    f"Article count: {m['article_count']}, "
                    f"Co-occurrence weight: {m['weighted_degree']}, "
                    f"Betweenness centrality: {m['betweenness_centrality']:.3f}\n"
                    f"Strongest co-occurring keywords: {neighbors_str}\n"
                    f"Top institutions using this keyword: {institutions_str}\n"
                    f"Discipline lenses applied: {discipline_str}"
                )
                with st.spinner("Analyzing…"):
                    st.session_state.explorer_analysis = call_claude(system, user, max_tokens=450)

            # Always display AI analysis from session_state if available
            if st.session_state.explorer_analysis:
                st.markdown(st.session_state.explorer_analysis)

            st.divider()
            st.subheader("Concept network")
            st.caption("Orange node = your search result")
            draw_cooc_network(kg, highlight_kw=matches[0]["keyword"])


# ════════════════════════════════════════════════════════════════════════════
# MODE 2 — Period Comparison
# ════════════════════════════════════════════════════════════════════════════
elif mode == "② Period Comparison":
    st.title("② Period Comparison")
    st.markdown("""
Compare the top research topics and discipline distributions across three
historical periods for one area studies field.
The graph structure changes over time — new keywords rise, old ones fade,
and the disciplinary lenses shift.
This view reveals the **intellectual evolution** of an entire research field
over 65 years.
    """)
    st.divider()

    c1, c2 = st.columns([2, 1])
    with c1: sel_area = st.selectbox("Area studies field", AREAS)
    with c2:
        metric_map = {
            "Co-occurrence weight":   "weighted_degree",
            "Article count":          "article_count",
            "Betweenness centrality": "betweenness_centrality",
        }
        metric_label = st.selectbox("Ranking metric", list(metric_map.keys()))
        metric = metric_map[metric_label]

    st.info("""
**Ranking metric guide:**
- **Co-occurrence weight** — the total number of co-occurrence links a keyword has
  (sum of all edge weights). Reflects how *actively connected* a keyword is in the network.
  Best for identifying the most "networked" concepts — words that appear together with
  many other concepts across many articles.
- **Article count** — the raw number of articles in which this keyword appears.
  Best for identifying the most *frequently studied* topics, regardless of how
  those articles connect to each other.
- **Betweenness centrality** — how often a keyword lies on the shortest path between
  two other keywords. High values indicate *bridge concepts* that link otherwise
  separate research clusters. Best for identifying interdisciplinary or connective
  topics that hold different sub-fields together.
    """)

    trends      = compare_keyword_trends(all_graphs, sel_area, metric=metric, n=12)
    disc_trends_base = discipline_keyword_trends(all_graphs, sel_area, n=10)

    tab_subj, tab_disc = st.tabs(["📍 Research Topics", "📚 Discipline Distribution"])

    with tab_subj:
        st.caption(
            "Subject keywords are *substantive* words describing what is studied — "
            "countries, regions, events, political concepts. "
            "Discipline words (sociology, history, computer science…) are filtered out."
        )

        # ── Concept networks FIRST ────────────────────────────────────────────
        st.subheader("Concept networks by period")
        st.markdown("""
Each network visualizes the keyword co-occurrence structure for one time period.

- **Nodes** = subject keywords (discipline words excluded)
- **Edges** = two keywords appear together in the same article
- **Node size** = article count · **Node color** (blue intensity) = co-occurrence weight
        """)
        net_cols = st.columns(3)
        for i, period in enumerate(PERIODS):
            with net_cols[i]:
                kg   = all_graphs[sel_area][period]["cooc"]
                summ = kg.summary()
                st.caption(
                    f"**{period}** — "
                    f"{summ['num_keywords']} keyword nodes · "
                    f"{summ['num_edges']} co-occurrence edges"
                )
                draw_cooc_network(kg, max_nodes=30)

        # ── Ranking tables ────────────────────────────────────────────────────
        st.divider()
        st.subheader("Top keywords by period")
        cols = st.columns(3)
        for i, period in enumerate(PERIODS):
            with cols[i]:
                st.subheader(period)
                kg   = all_graphs[sel_area][period]["cooc"]
                summ = kg.summary()
                st.caption(f"{summ['num_articles']} articles · "
                            f"{summ['num_keywords']} keywords · "
                            f"{summ['num_edges']} edges")
                top = trends.get(period, [])
                if top:
                    df = pd.DataFrame([
                        {"Keyword": t["keyword"],
                         metric_label: round(t.get(metric,0),3)}
                        for t in top
                    ])
                    st.dataframe(df, hide_index=True, use_container_width=True)

        # ── Keyword tracker ───────────────────────────────────────────────────
        st.divider()
        st.subheader("Track a keyword across periods")
        all_kws = set()
        for p in PERIODS:
            for d in trends.get(p, []):
                all_kws.add(d["keyword"])
        tracked = st.multiselect("Select keywords to track (max 6)",
                                 sorted(all_kws), max_selections=6)
        if tracked:
            rows = []
            for kw in tracked:
                tl = keyword_appearance_timeline(all_graphs, sel_area, kw)
                for p, cnt in tl.items():
                    rows.append({"Period":p,"Keyword":kw,"Articles":cnt})
            st.plotly_chart(
                px.line(pd.DataFrame(rows), x="Period", y="Articles",
                        color="Keyword", markers=True),
                use_container_width=True)

        # ── Bubble chart ──────────────────────────────────────────────────────
        st.divider()
        st.subheader("Keyword heat map")
        st.caption("Bubble size = article count. Larger = more research activity in that period.")
        bubble = []
        for p in PERIODS:
            for d in trends.get(p,[]):
                bubble.append({"Period":p,"Keyword":d["keyword"],
                                "Articles":d.get("article_count",0)})
        if bubble:
            bdf = pd.DataFrame(bubble)
            st.plotly_chart(
                px.scatter(bdf, x="Period", y="Keyword", size="Articles",
                           color="Articles", color_continuous_scale="Blues",
                           size_max=40,
                           height=max(400, bdf["Keyword"].nunique()*28)),
                use_container_width=True)

    with tab_disc:
        st.caption(
            "Discipline keywords describe *how* scholars approach a topic — "
            "the methodological and disciplinary lens (sociology, history, law…). "
            "Tracking these reveals whether a field is diversifying its methods "
            "or converging on a dominant approach."
        )
        top_n_disc = st.slider(
            "Number of discipline keywords to display", 5, 20, 10,
            help="Adjust to show more or fewer discipline keywords. "
                 "The bubble chart below updates automatically.")
        disc_trends = discipline_keyword_trends(all_graphs, sel_area, n=top_n_disc)

        cols2 = st.columns(3)
        for i, period in enumerate(PERIODS):
            with cols2[i]:
                st.subheader(period)
                kw_list = disc_trends.get(period, [])
                if kw_list:
                    st.bar_chart(
                        pd.DataFrame(kw_list, columns=["Discipline","Frequency"]
                                     ).set_index("Discipline"),
                        height=300)
                else:
                    st.info("No discipline keyword data.")

        st.divider()
        disc_bubble = []
        for p in PERIODS:
            for kw, cnt in disc_trends.get(p, []):
                disc_bubble.append({"Period":p,"Discipline":kw,"Frequency":cnt})
        if disc_bubble:
            dbdf = pd.DataFrame(disc_bubble)
            st.plotly_chart(
                px.scatter(dbdf, x="Period", y="Discipline",
                           size="Frequency", color="Frequency",
                           color_continuous_scale="Oranges", size_max=40,
                           title=f"{sel_area} — Discipline lens distribution across periods",
                           height=max(400, dbdf["Discipline"].nunique()*28)),
                use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# MODE 3 — Keyword Comparison
# ════════════════════════════════════════════════════════════════════════════
elif mode == "③ Keyword Comparison":
    st.title("③ Keyword Comparison")
    st.markdown("""
Compare two research topics by examining the neighborhoods of each
in the co-occurrence network.
    """)

    with st.expander("Why comparison instead of path-finding?", expanded=True):
        st.markdown("""
An earlier design of this tool used **shortest-path finding** (similar to the
Kevin Bacon number problem) to measure the "conceptual distance" between two keywords.

However, in a keyword co-occurrence network built from academic articles,
path-finding produces uninformative results: nearly all keyword pairs are only
1 or 2 hops apart. This is because every article creates direct edges between
*all* of its keywords — a single article with 5 keywords creates 10 direct
connections. As the dataset grows, the network becomes so dense that almost
every pair of keywords is directly connected.

**Neighborhood comparison is more informative** for this dataset because it asks
a richer question: not *how far apart* are two concepts, but *in what different
contexts* are they discussed?

Specifically, this tool reveals:
- **Shared neighbors** — concepts that co-occur with *both* keywords,
  indicating common intellectual ground (e.g., both *China* and *Nigeria*
  may appear alongside *development* and *investment*)
- **Unique to A** — concepts exclusively associated with keyword A,
  revealing its distinctive research context
- **Unique to B** — concepts exclusively associated with keyword B

This is a **graph-only insight**: no table of articles can show you
the relational context of a concept the way a co-occurrence network can.
        """)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        sel_area   = st.selectbox("Area studies field", AREAS)
        sel_period = st.selectbox("Time period", PERIODS)
    with c2:
        kw_a = st.text_input("Keyword A", placeholder="e.g. China")
        kw_b = st.text_input("Keyword B", placeholder="e.g. Nigeria")

    st.caption("""
💡 **Suggested comparisons:**
*African Studies* — China vs. Nigeria · colonialism vs. development · diaspora vs. migration ·
*Asian Studies* — China vs. Japan · Cold War vs. democracy · Korea vs. Vietnam ·
*Latin American Studies* — Mexico vs. Brazil · revolution vs. democracy · migration vs. trade
    """)

    if st.button("Compare", type="primary"):
        if not kw_a or not kw_b:
            st.warning("Please enter both keywords.")
        else:
            kg     = all_graphs[sel_area][sel_period]["cooc"]
            result = compare_two_keywords(kg, kw_a, kw_b)
            # Store result in session_state so it survives rerenders
            st.session_state.cmp_result   = result
            st.session_state.cmp_area     = sel_area
            st.session_state.cmp_period   = sel_period
            st.session_state.cmp_analysis = None   # reset previous AI output

    # Render results from session_state (persists across button clicks)
    result = st.session_state.cmp_result
    if result is not None:
        if "error" in result:
            st.error(result["error"])
        else:
            st.success(
                f"Comparing **{result['kw_a']}** and **{result['kw_b']}** "
                f"in {st.session_state.cmp_area}, {st.session_state.cmp_period}"
            )
            if result["direct_weight"] > 0:
                st.info(
                    f"These two keywords **directly co-occur** in "
                    f"{result['direct_weight']} article(s) — "
                    f"they are explicitly discussed together."
                )
                if result["direct_articles"]:
                    with st.expander("Sample articles where both appear"):
                        for t in result["direct_articles"]:
                            st.write(f"• {t}")
            else:
                st.info(
                    "These keywords do **not** directly co-occur — "
                    "they occupy separate regions of the concept map."
                )

            st.divider()
            col_a, col_s, col_b = st.columns(3)
            with col_a:
                st.subheader(f"Unique to **{result['kw_a']}**")
                st.caption("Concepts associated only with this keyword")
                for kw in result["unique_a"][:10]:
                    st.markdown(f"- {kw}")
                if not result["unique_a"]:
                    st.info("No unique neighbors.")
            with col_s:
                st.subheader("Shared concepts")
                st.caption("Common intellectual ground between both topics")
                if result["shared"]:
                    for kw in result["shared"][:10]:
                        st.markdown(f"- **{kw}**")
                else:
                    st.info("No shared neighbors — these topics are conceptually distant.")
            with col_b:
                st.subheader(f"Unique to **{result['kw_b']}**")
                st.caption("Concepts associated only with this keyword")
                for kw in result["unique_b"][:10]:
                    st.markdown(f"- {kw}")
                if not result["unique_b"]:
                    st.info("No unique neighbors.")

            st.divider()
            st.subheader("Co-occurrence strength")
            ch_a, ch_b = st.columns(2)
            with ch_a:
                st.caption(f"Top neighbors of **{result['kw_a']}**")
                if result["nbrs_a"]:
                    st.bar_chart(
                        pd.DataFrame(result["nbrs_a"],
                                     columns=["Keyword","Co-occurrences"]
                                     ).set_index("Keyword"), height=280)
            with ch_b:
                st.caption(f"Top neighbors of **{result['kw_b']}**")
                if result["nbrs_b"]:
                    st.bar_chart(
                        pd.DataFrame(result["nbrs_b"],
                                     columns=["Keyword","Co-occurrences"]
                                     ).set_index("Keyword"), height=280)

            # ── LLM analysis ──────────────────────────────────────────────────
            st.divider()
            st.subheader("🤖 AI Analysis")
            st.caption(
                "An AI interprets the comparison results: what the shared and "
                "unique neighbors reveal about how these two topics are framed "
                "differently in the literature, and what follow-up questions "
                "are worth pursuing."
            )
            if st.button("Generate AI analysis", key="llm_compare"):
                shared_str   = ", ".join(result["shared"][:8]) or "none"
                unique_a_str = ", ".join(result["unique_a"][:6]) or "none"
                unique_b_str = ", ".join(result["unique_b"][:6]) or "none"
                direct_info  = (
                    f"They directly co-occur in {result['direct_weight']} article(s)."
                    if result["direct_weight"] > 0
                    else "They do not directly co-occur."
                )
                system = (
                    "You are an expert in U.S. area studies and academic network analysis. "
                    "Given the results of a keyword neighborhood comparison, provide a "
                    "concise, insightful interpretation in 3–4 short paragraphs. "
                    "Focus on: (1) what the shared neighbors reveal about the common "
                    "intellectual framework linking these two topics, (2) what the unique "
                    "neighbors reveal about how each topic is distinctively framed, "
                    "(3) one specific follow-up comparison or question worth exploring. "
                    "Be analytical and specific. Do not simply list the keywords back."
                )
                user = (
                    f"Comparing '{result['kw_a']}' vs '{result['kw_b']}' "
                    f"in {st.session_state.cmp_area}, {st.session_state.cmp_period}.\n"
                    f"{direct_info}\n"
                    f"Shared neighbors: {shared_str}\n"
                    f"Unique to '{result['kw_a']}': {unique_a_str}\n"
                    f"Unique to '{result['kw_b']}': {unique_b_str}"
                )
                with st.spinner("Analyzing…"):
                    st.session_state.cmp_analysis = call_claude(system, user, max_tokens=450)

            # Always display AI analysis from session_state if available
            if st.session_state.cmp_analysis:
                st.markdown(st.session_state.cmp_analysis)


# ════════════════════════════════════════════════════════════════════════════
# MODE 4 — Publication Ecology
# ════════════════════════════════════════════════════════════════════════════
elif mode == "④ Publication Ecology":
    st.title("④ Publication Ecology")
    st.markdown("""
Explore the publication infrastructure of U.S. area studies:
which journals dominate each field, which institutions are most active,
how journals and institutions are connected, and how collaboration networks
look across time.
    """)
    st.divider()

    sel_area = st.selectbox("Area studies field", AREAS, key="pe_area")

    tab_jioverview, tab_detail, tab_collab = st.tabs([
        "📊 Journal–Institution Overview",
        "🔍 Journal / Institution Detail",
        "🏛 Institution Collaboration",
    ])

    with tab_jioverview:
        st.markdown("""
        Each column shows the most active journals for one time period.
        Comparing columns reveals which journals maintained dominance over decades
        and which institutions sustained long-term area studies programs.
        """)
        journal_trends = top_journals_across_periods(all_graphs, sel_area, n=10)
        st.subheader("Top journals by period")
        jcols = st.columns(3)
        for i, period in enumerate(PERIODS):
            with jcols[i]:
                st.caption(period)
                jlist = journal_trends.get(period, [])
                if jlist:
                    st.dataframe(
                        pd.DataFrame(jlist, columns=["Journal","Articles"]),
                        hide_index=True, use_container_width=True)

        st.divider()
        st.subheader("Journal heat across periods")
        st.caption("Bubble size = number of articles published in that journal in that period.")
        bj = []
        for p in PERIODS:
            for jname, cnt in journal_trends.get(p, []):
                bj.append({"Period":p,"Journal":jname,"Articles":cnt})
        if bj:
            bjdf = pd.DataFrame(bj)
            st.plotly_chart(
                px.scatter(bjdf, x="Period", y="Journal",
                           size="Articles", color="Articles",
                           color_continuous_scale="Teal", size_max=40,
                           height=max(400, bjdf["Journal"].nunique()*30)),
                use_container_width=True)

    with tab_detail:
        st.markdown("Drill into a specific journal or institution for one area + period.")
        c1, c2 = st.columns(2)
        with c1: period_d  = st.selectbox("Time period", PERIODS, key="det_period")
        with c2: view_type = st.radio("View by", ["Journal","Institution"],
                                      horizontal=True)
        jg = all_graphs[sel_area][period_d]["ji"]

        if view_type == "Journal":
            jnames = sorted([d["name"] for _,d in jg.graph.nodes(data=True)
                             if d.get("node_type")=="journal"])
            sel_j = st.selectbox("Select journal", jnames if jnames else ["(no data)"])
            if jnames:
                inst_list = jg.institutions_of_journal(sel_j)
                st.subheader(f"Institutions publishing in *{sel_j}*")
                if inst_list:
                    ct, cc = st.columns(2)
                    with ct:
                        st.dataframe(pd.DataFrame(inst_list,
                                      columns=["Institution","Articles"]),
                                      hide_index=True, use_container_width=True)
                    with cc:
                        fig = px.bar(pd.DataFrame(inst_list[:10],
                                      columns=["Institution","Articles"]),
                                      x="Articles", y="Institution", orientation="h")
                        fig.update_layout(yaxis={"autorange":"reversed"}, height=320)
                        st.plotly_chart(fig, use_container_width=True)
        else:
            inames = sorted([d["name"] for _,d in jg.graph.nodes(data=True)
                             if d.get("node_type")=="institution"])
            sel_i = st.selectbox("Select institution",
                                 inames if inames else ["(no data)"])
            if inames:
                profile = institution_journal_profile(
                    all_graphs, sel_area, period_d, sel_i)
                cj, ck = st.columns(2)
                with cj:
                    st.subheader("Journals")
                    if profile["journals"]:
                        st.dataframe(pd.DataFrame(profile["journals"],
                                      columns=["Journal","Articles"]),
                                      hide_index=True, use_container_width=True)
                with ck:
                    st.subheader("Research keyword profile")
                    st.caption(
                        "Subject keywords most associated with this "
                        "institution's publications in this field and period.")
                    if profile["top_keywords"]:
                        st.bar_chart(
                            pd.DataFrame(profile["top_keywords"],
                                         columns=["Keyword","Frequency"]
                                         ).set_index("Keyword"), height=260)

    with tab_collab:
        st.markdown("""
**Graph structure:**
- **Nodes** = US institutions that appear as co-authors on at least one article
- **Edges** = two institutions share an author on the same article (co-authorship)
- **Edge weight** = number of co-authored articles

**Note on sparsity:** Humanities and social-science area studies are historically
dominated by single-author publications. Multi-institution collaboration is the
exception, not the norm. A sparse graph here accurately reflects the publication
culture of the field — it is not a data problem.

The 2015–2025 period typically shows the most collaboration,
reflecting the broader trend toward team-based interdisciplinary research.
        """)
        inst_trends = compare_inst_periods(all_graphs, sel_area, n=10)
        st.subheader("Top collaborating institutions by period")
        icols = st.columns(3)
        for i, period in enumerate(PERIODS):
            with icols[i]:
                ig   = all_graphs[sel_area][period]["inst"]
                summ = ig.summary()
                st.caption(
                    f"**{period}** — "
                    f"{summ['num_institutions']} institution nodes · "
                    f"{summ['num_edges']} collaboration edges"
                )
                top = inst_trends.get(period, [])
                if top:
                    st.dataframe(pd.DataFrame([
                        {"Institution":t["name"],
                         "Articles":t["article_count"],
                         "Collaborators":len(t["collaborators"])}
                        for t in top]), hide_index=True, use_container_width=True)
                else:
                    st.info("No collaboration data.")

        st.divider()
        st.subheader("Collaboration networks by period")
        net_cols = st.columns(3)
        for i, period in enumerate(PERIODS):
            with net_cols[i]:
                ig   = all_graphs[sel_area][period]["inst"]
                summ = ig.summary()
                st.caption(
                    f"**{period}** — "
                    f"{summ['num_institutions']} institution nodes · "
                    f"{summ['num_edges']} collaboration edges"
                )
                draw_inst_network(ig, max_nodes=35)


# ════════════════════════════════════════════════════════════════════════════
# MODE 5 — Key Findings
# ════════════════════════════════════════════════════════════════════════════
elif mode == "🎯 Key Findings":
    st.title("🎯 Key Findings")
    st.markdown("""
The following findings emerge from analyzing keyword co-occurrence networks
across three area studies fields and three historical periods.
These patterns are derived from three complementary metrics —
**co-occurrence weight**, **betweenness centrality**, and **article count** —
and are visible in the graph structures in ways that no table of articles alone
could reveal.
    """)
    st.divider()

    tab_asian, tab_african, tab_latin, tab_inst = st.tabs([
        "🌏 Asian Studies",
        "🌍 African Studies",
        "🌎 Latin American Studies",
        "🏛 Institution Collaboration",
    ])

    with tab_asian:
        st.subheader("Asian Studies Concept Networks, 1960–2025")
        st.markdown("""
**From a single pole to a multi-hub network.** The most immediate insight from
the three network visualizations is structural. In 1960–1975, the graph is
organized around a single dominant node: *China*, with a co-occurrence weight of
504, dwarfs every other keyword. Most other nodes hang at the periphery,
connected to the center by one or two edges — a classic star topology. By
2015–2025, the graph has transformed into a dense multi-hub structure. *Ancient
history*, *Epistemology*, *Ethnic group*, and *Traditional medicine* have all
become visibly weighted nodes, and the center of the network is now a tightly
interconnected cluster rather than a single point. The number of co-occurrence
edges grew from 980 to 9,272 — nearly a tenfold increase — while articles only
grew 3.7-fold. The network is not just bigger; it is fundamentally differently
shaped.

**China's dominance is real but changing in kind.** China ranks first across all
three metrics in all three periods without exception. Yet the three metrics tell
different stories. Its co-occurrence weight grew sixfold (504 → 3,194) and its
article count more than tripled (201 → 683) — both suggesting an accelerating
centrality. But its betweenness centrality declined steadily: 0.645 → 0.617 →
0.434. China is studied more than ever, but it is no longer the sole conceptual
bridge through which other ideas connect. As the network has grown denser, other
keywords have built direct connections to each other, reducing their dependence
on China as an intermediary. The field has not moved away from China — it has
grown around it.

**Four keywords form the enduring intellectual core.** Across all three periods
and across all three metrics, only four keywords consistently appear in the top
rankings: *China*, *Ancient history*, *Development economics*, and *Ethnology*.
These represent the stable foundation of Asian Studies over sixty-five years —
the concepts that have never fallen out of the field's center of gravity
regardless of shifting funding, geopolitics, or disciplinary fashion.

**Epistemology's rise signals a field turning inward.** Of all the changes
visible in the data, the trajectory of *Epistemology* is the most analytically
significant. Its article count grew modestly across all three periods (9 → 25 →
61). Its co-occurrence weight was absent from the top rankings in 1960–1975,
then reached 99 in 1990–2000, then surged to 386 in 2015–2025. But the decisive
indicator is its betweenness centrality: it does not appear in the top rankings
in the first two periods, then jumps to third place (0.033) in 2015–2025. This
means *Epistemology* has become the primary bridge connecting otherwise separate
research clusters — linking *Traditional medicine*, *Asian Americans*, *South
Asia*, and *Ethnology* into a single conceptual network. Contemporary Asian
Studies scholars are not only studying Asia; they are increasingly interrogating
the frameworks through which Asia has been studied. This reflexive turn is
legible in the graph structure in a way that no list of articles could reveal.

Together, these findings demonstrate what the network structure makes visible:
not just which topics are studied, but how the intellectual architecture of an
entire field has been reorganized over time.
        """)

    with tab_african:
        st.subheader("African Studies Concept Networks, 1960–2025")
        st.markdown("""
**A network that grew denser as it shrank.** The most structurally significant
finding in African Studies is invisible without the graph. Between 1960–1975 and
1990–2000, article count *fell* from 224 to 151 — a 33% decline reflecting the
withdrawal of Cold War funding after the Soviet collapse. Yet co-occurrence edges
*increased* from 825 to 984. The network became more densely connected even as
fewer articles were being published. This means the scholars who remained in the
field during its contraction were producing more conceptually integrated work —
articles that ranged across multiple topics simultaneously rather than focusing on
a single subject. The graph makes this visible: the 1990–2000 network, despite
having fewer nodes than 1960–1975, shows a denser central cluster. By 2015–2025,
both articles (292) and edges (2,141) had recovered and surpassed the original
baseline, but the field that returned was structurally different from the one
that contracted.

**China is present but structurally peripheral.** China ranks first by
co-occurrence weight in all three periods (221 → 188 → 276), but its role in
African Studies is categorically different from its role in Asian Studies. In
Asian Studies, China's betweenness centrality reached 0.645 in 1960–1975 —
meaning it was the dominant conceptual bridge across the entire network. In
African Studies, China's betweenness never exceeded 0.323 and has declined
steadily to 0.272. China appears in African Studies primarily as an external
reference point — a geopolitical actor that African Studies scholars must
acknowledge — rather than as the organizing center around which the field's
intellectual architecture is built. The field is not about China; it is about
Africa, with China appearing at the edges of that conversation.

**Colonialism has moved from margin to structural center.** The trajectory of
*Colonialism* across three metrics tells the clearest story of intellectual
transformation in this dataset. In 1960–1975, it does not appear in the top
rankings by co-occurrence weight or article count at all, with a betweenness
centrality of just 0.028. By 1990–2000, it had entered the top rankings across
all three metrics, with betweenness rising to 0.076. In 2015–2025, its
co-occurrence weight had more than doubled (70 → 159), its article count reached
32, and its betweenness (0.083) made it the second-ranked bridge concept in the
network after China. The graph confirms this visually: in the rightmost network,
*Colonialism* has become one of the largest and most deeply colored nodes,
positioned at the center of a dense cluster that also includes *Islam*,
*Epistemology*, and *Art history*. African Studies has reorganized itself around
a postcolonial critical framework — and the network graph shows exactly when and
how that reorganization happened.

**Religion as an emerging organizing framework.** A second transformation is
visible in the 2015–2025 network: *Religious studies*, *Islam*, *History of
religions*, and *Anthropology of religion* have all become prominent nodes,
forming a distinct religious studies cluster. *Islam* in particular moved from
absent in 1960–1975 to a betweenness centrality of 0.077 in 2015–2025 — ranking
third among all bridge concepts. The study of African religion, Islam in
sub-Saharan Africa, and religious identity has become one of the field's primary
organizing frameworks alongside postcolonial critique.
        """)

    with tab_latin:
        st.subheader("Latin American Studies Concept Networks, 1960–2025")
        st.markdown("""
**A single keyword anchors the entire field across sixty-five years.** Latin
American Studies has the most concentrated intellectual core of the three fields:
*Economic history* is the only keyword that appears in the top rankings across
all three time periods and all three metrics simultaneously. Its dominance is
structural, not merely frequent — its betweenness centrality peaked at 0.442 in
1990–2000, meaning it served as the bridge through which nearly half of all
conceptual connections in the network passed. No other keyword in any of the
three fields reaches this level of network centrality. Latin American Studies is,
in a measurable sense, organized around a single conceptual spine in a way that
Asian Studies and African Studies are not. The graph makes this visible: in all
three periods, one large dark node sits at the center, and everything else
radiates outward from it.

**The Cold War political vocabulary has disappeared entirely.** The three-metric
data reveals a complete generational turnover in political concepts. In
1960–1975, *Communism*, *Modernization theory*, and *Public administration* all
appear in the top rankings — concepts that directly reflect the Cold War framing
of Latin America as a theater of ideological competition and U.S. development
policy. By 1990–2000, these had vanished, replaced by *Dictatorship* and
*Democracy* — the transitional justice vocabulary of the post-authoritarian
moment. By 2015–2025, both had receded, and *Indigenous*, *Colonialism*, and
*Diaspora* have emerged. Each period's political vocabulary is entirely
non-overlapping with the others. The network graph confirms this: the 1960–1975
graph shows *Democracy* and *Left-wing politics* as prominent nodes; the
2015–2025 graph shows neither, replaced by a cluster of postcolonial and
cultural concepts.

**A humanities turn is visible and measurable.** The shift from social science
to humanities frameworks is legible across all three metrics. In 1960–1975, the
top keywords are almost entirely social-scientific: *Economic history*,
*Modernization theory*, *Development economics*, *Public administration*. By
2015–2025, *Art history*, *Aesthetics*, *Narrative*, *Modernism (music)*, and
*Poetry* have all entered the network's core. The 2015–2025 graph is visually
the most distributed of the three — nodes spread across the full frame rather
than concentrated at a single center, reflecting a field that has diversified
its methodological toolkit across literary, artistic, and cultural studies.

**Portuguese as a discipline-defining anomaly.** The most striking single data
point in the Latin American Studies dataset is *Portuguese*, which records an
article count of 40 in 2015–2025 — more than three times the article count of
*Economic history* (11) in the same period and more than any other keyword in
any period across all three fields by relative weight. This reflects the growing
presence of Brazilian scholarship and Portuguese-language area studies within
U.S. academic publishing, signaling a geographic and linguistic rebalancing of
who produces Latin American Studies knowledge and from where.
        """)

    with tab_inst:
        st.subheader("Institution Collaboration Networks: 1960-2025")
        st.markdown("""
**Collaboration is structurally sparse, and that sparsity is a finding.** Across
all three fields and all three time periods, the collaboration graphs are far
less dense than the keyword co-occurrence networks. In Asian Studies, the field
with the most data, the collaboration network grew from 24 edges in 1960–1975 to
176 edges in 2015–2025. In African Studies and Latin American Studies, the
numbers are considerably smaller. This is not a data quality problem. It
accurately reflects a publication culture in which individual scholars produce
single-authored work — a norm that distinguishes humanities and interpretive
social-science area studies from natural science or quantitative social science
fields, where multi-author, multi-institution papers are standard.

**The network has transformed from geographically distributed to
elite-concentrated.** In 1960–1975, the collaboration graph for Asian Studies
shows a loosely connected set of institutions spread across the country —
including the University of Hawaii System, Honolulu University, and several
historically Black colleges and universities. This geographic distribution
reflects the Cold War mandate: federal funding through the National Defense
Education Act was explicitly designed to build area studies capacity across U.S.
institutions, not just at elite research universities. By 2015–2025, the
collaboration network has reorganized around a dense core of East Coast research
universities — Columbia, University of Pennsylvania, Johns Hopkins, Harvard,
Yale, and the University of Michigan. The number of edges grew nearly fivefold
(24 to 176), but much of that growth reflects deepening ties within an
already-connected cluster rather than the inclusion of new geographic or
institutional constituencies. The network grew inward, not outward.

**The growth in edges outpaces the growth in nodes.** Between 1960–1975 and
2015–2025, institution nodes in Asian Studies grew from 37 to 132 —
approximately 3.6-fold. Collaboration edges grew from 24 to 176 —
approximately 7.3-fold. This divergence means that the average institution in
2015–2025 has more collaboration partners than its 1960–1975 counterpart, and
that the network's density has increased even after controlling for size. The
field is not just larger; it is more interconnected within its active core.

**Latin American Studies remains a non-collaborative field throughout.** Across
all three periods, Latin American Studies produces almost no multi-institution
co-authored articles. This is the most extreme expression of the single-author
publication norm across the three fields examined. It distinguishes Latin
American Studies not only from natural sciences but also from Asian Studies and
African Studies, which at least show some growth in collaboration over time.
Whether this reflects the field's disciplinary traditions, its funding
structures, or the nature of its primary research methods — archival,
ethnographic, and literary — remains an open question that the network structure
itself raises but cannot answer.
        """)


# ════════════════════════════════════════════════════════════════════════════
# METHODOLOGICAL REFLECTION
# ════════════════════════════════════════════════════════════════════════════
elif mode == "💡 Methodological Reflection":
    st.title("💡 Methodological Reflection")
    st.markdown("""
This page documents the design choices made during this project — including
the approaches that were attempted and abandoned, and why.
Understanding what did not work is as important as understanding what did.
    """)
    st.divider()

    st.markdown("""
### **Revising the Original Project Design**
The original design of this project focused on institution collaboration networks. 
My initial idea was that if I wanted to understand 
how knowledge in U.S. Area Studies is produced, 
it would make sense to examine which universities and institutions collaborate with each other. 
Co-authorship networks are commonly used in bibliometric and network analysis research, 
so this seemed like a reasonable starting point.

This approach also appeared technically simple and clear. 
In the graph, nodes represented universities or institutions, 
while edges represented co-authored publications between them. 
OpenAlex provides reliable affiliation metadata, 
so building the collaboration networks was relatively straightforward.

However, after constructing the graphs, 
I realized that the collaboration networks were too sparse to provide meaningful analysis. 
In Latin American Studies, there were very few multi-institution publications. 
African Studies showed similar problems, 
with too few connections to produce stable network patterns. 
Even in Asian Studies, many articles were written by a single author or a single institution, 
meaning that a large portion of the dataset did not contribute to the collaboration graph. 
As a result, the networks contained many isolated nodes and very limited structural information.

More importantly, I realized that collaboration networks did not fully address the central question of my project. 
Even if two universities collaborated frequently, 
this only showed who worked together, not what kinds of ideas, topics, or intellectual connections shaped the field itself. 
My project was ultimately more interested in the development of research themes and conceptual relationships across different historical periods.

Because of this, 
I shifted the main analysis from institution collaboration networks 
to keyword co-occurrence networks. 
This solved several problems at once. 
Every article contributes multiple keyword relationships regardless of the number of authors or institutions, 
which creates denser and more informative graphs. 
More importantly, keyword networks better represent the intellectual structure of a field 
by showing which concepts are commonly discussed together and how research themes change over time.

At the same time, I decided not to completely remove the institution collaboration networks. 
Instead, I kept them as a supplementary layer of analysis. 
Interestingly, the sparsity of these networks itself became a meaningful finding, 
suggesting that much of Area Studies research in the United States 
is still produced by individual scholars or small institutional groups 
rather than large collaborative teams.


### **Revising the Interaction Design**

An earlier version of the project also included a shortest-path finding function within the keyword network. 
The idea was similar to the “Kevin Bacon number,” 
measuring the conceptual distance between two keywords.

Although the feature worked technically, it was not very meaningful analytically. 
Because keyword co-occurrence networks are highly dense, 
most keywords were already connected within one or two steps. 
As the dataset grew larger, shortest-path distance stopped being a useful measurement.

I replaced this feature with a neighborhood comparison approach. 
Instead of asking how far apart two concepts are, 
the new method examines the different intellectual contexts surrounding each keyword. 
This provided more interesting comparisons 
between research themes and better matched the interpretive goals of Area Studies research.


### **Keyword Filtering Decisions**

Another important methodological decision involved keyword filtering. 
OpenAlex includes both subject-related keywords (such as China, colonialism, or migration) 
and disciplinary labels (such as History, Sociology, or Political Science). 
If all of these keywords were included together, 
the network would mainly reflect disciplinary classifications 
rather than actual thematic relationships.

To address this issue, I created a custom blacklist of disciplinary and methodological terms 
before building the co-occurrence networks. 
This allowed the graph to focus more clearly on substantive research topics. 
At the same time, discipline-related keywords were still preserved separately for comparison purposes, 
making it possible to analyze both what scholars study and how they approach those topics methodologically.
    """)
