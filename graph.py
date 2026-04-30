"""
graph.py
Graph construction and all analysis algorithms.

Builds three graph types per (area × period):
  - KeywordCooccurrenceGraph     : primary   — concept map
  - JournalInstitutionGraph      : secondary — publication ecology
  - InstitutionCollaborationGraph: tertiary  — collaboration network (sparse)

Analysis functions:
  - keyword centrality
  - keyword comparison (replaces path-finding)
  - discipline keyword trends
  - journal/institution queries
  - institution collaboration queries
"""

from __future__ import annotations
import networkx as nx
from collections import Counter
from models import (
    Article,
    KeywordCooccurrenceGraph,
    JournalInstitutionGraph,
    InstitutionCollaborationGraph,
    DISCIPLINE_KEYWORDS,
)
from fetch import AREA_KEYWORD_IDS, TIME_PERIODS


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_cooccurrence_graph(area: str, period: str,
                             articles: list[Article]) -> KeywordCooccurrenceGraph:
    kg = KeywordCooccurrenceGraph(area=area, period=period)
    for art in articles:
        kg.add_article(art)
    s = kg.summary()
    print(f"  [cooc] {area}|{period}: {s['num_keywords']} nodes, {s['num_edges']} edges")
    return kg


def build_journal_institution_graph(area: str, period: str,
                                    articles: list[Article]) -> JournalInstitutionGraph:
    jg = JournalInstitutionGraph(area=area, period=period)
    for art in articles:
        jg.add_article(art)
    s = jg.summary()
    print(f"  [ji]   {area}|{period}: {s['num_journals']} journals, "
          f"{s['num_institutions']} institutions")
    return jg


def build_institution_collab_graph(area: str, period: str,
                                   articles: list[Article]
                                   ) -> InstitutionCollaborationGraph:
    ig = InstitutionCollaborationGraph(area=area, period=period)
    for art in articles:
        ig.add_article(art)
    s = ig.summary()
    print(f"  [inst] {area}|{period}: {s['num_institutions']} nodes, "
          f"{s['num_edges']} edges")
    return ig


def build_all_graphs(all_data: dict) -> dict:
    """
    Build all three graph types for every area × period combination.

    Returns
    -------
    { area: { period: { "cooc": ..., "ji": ..., "inst": ... } } }
    """
    all_graphs: dict = {}
    for area, period_dict in all_data.items():
        all_graphs[area] = {}
        for period, articles in period_dict.items():
            all_graphs[area][period] = {
                "cooc": build_cooccurrence_graph(area, period, articles),
                "ji":   build_journal_institution_graph(area, period, articles),
                "inst": build_institution_collab_graph(area, period, articles),
            }
    return all_graphs


# ---------------------------------------------------------------------------
# Keyword centrality
# ---------------------------------------------------------------------------

def compute_keyword_centrality(kg: KeywordCooccurrenceGraph) -> dict[str, dict]:
    """
    Compute centrality metrics for every node in the co-occurrence graph.

    Returns { keyword: { article_count, degree_centrality,
                         betweenness_centrality, weighted_degree,
                         top_institutions, top_journals, neighbors } }
    """
    G = kg.graph
    if G.number_of_nodes() == 0:
        return {}

    deg  = nx.degree_centrality(G)
    btw  = nx.betweenness_centrality(G, weight="weight", normalized=True)

    result = {}
    for kw in G.nodes():
        nd  = G.nodes[kw]
        wd  = sum(d["weight"] for _, _, d in G.edges(kw, data=True))
        result[kw] = {
            "keyword":               kw,
            "article_count":         nd.get("article_count", 0),
            "degree_centrality":     round(deg.get(kw, 0), 4),
            "betweenness_centrality":round(btw.get(kw, 0), 4),
            "weighted_degree":       wd,
            "top_institutions":      sorted(nd.get("institutions", {}).items(),
                                            key=lambda x: x[1], reverse=True)[:5],
            "top_journals":          sorted(nd.get("journals", {}).items(),
                                            key=lambda x: x[1], reverse=True)[:3],
            "discipline_co":         sorted(nd.get("discipline_co", {}).items(),
                                            key=lambda x: x[1], reverse=True)[:5],
            "neighbors":             kg.neighbors_of(kw)[:8],
        }
    return result


def top_keywords_by_metric(kg: KeywordCooccurrenceGraph,
                           metric: str = "weighted_degree",
                           n: int = 15) -> list[dict]:
    centrality = compute_keyword_centrality(kg)
    if not centrality:
        return []
    return sorted(centrality.values(),
                  key=lambda x: x.get(metric, 0), reverse=True)[:n]


# ---------------------------------------------------------------------------
# Keyword search
# ---------------------------------------------------------------------------

def search_keyword(kg: KeywordCooccurrenceGraph, query: str) -> list[dict]:
    """Fuzzy case-insensitive keyword search; returns list of matching dicts."""
    q = query.lower().strip()
    centrality = compute_keyword_centrality(kg)
    matches = [v for k, v in centrality.items() if q in k.lower()]
    return sorted(matches, key=lambda x: x["article_count"], reverse=True)


# ---------------------------------------------------------------------------
# Keyword comparison  (replaces path-finding in Mode 3)
# ---------------------------------------------------------------------------

def compare_two_keywords(kg: KeywordCooccurrenceGraph,
                         kw_a: str, kw_b: str) -> dict:
    """
    Compare two keywords by their co-occurrence neighborhoods.

    For each keyword, retrieve its top neighbors (co-occurring words).
    Then compute:
      - shared neighbors  : concepts discussed alongside BOTH keywords
      - unique to A       : concepts specific to keyword A
      - unique to B       : concepts specific to keyword B

    This reveals how two research topics are framed differently —
    or what common intellectual ground they share.

    Parameters
    ----------
    kw_a, kw_b : keyword strings (case-insensitive fuzzy match)

    Returns dict with keys: kw_a, kw_b, nbrs_a, nbrs_b,
                             shared, unique_a, unique_b, error(optional)
    """
    G = kg.graph

    def fuzzy(query: str) -> str | None:
        q = query.lower().strip()
        for node in G.nodes():
            if node.lower() == q:
                return node
        for node in G.nodes():
            if q in node.lower():
                return node
        return None

    node_a = fuzzy(kw_a)
    node_b = fuzzy(kw_b)

    if node_a is None:
        return {"error": f"Keyword not found: '{kw_a}'"}
    if node_b is None:
        return {"error": f"Keyword not found: '{kw_b}'"}
    if node_a == node_b:
        return {"error": "Both inputs match the same keyword."}

    # top-20 neighbors for each keyword, as sets
    nbrs_a: dict[str, int] = {nb: kg.graph[node_a][nb]["weight"]
                               for nb in G.neighbors(node_a)}
    nbrs_b: dict[str, int] = {nb: kg.graph[node_b][nb]["weight"]
                               for nb in G.neighbors(node_b)}

    set_a = set(nbrs_a.keys())
    set_b = set(nbrs_b.keys())

    shared   = sorted(set_a & set_b,
                       key=lambda x: nbrs_a[x] + nbrs_b[x], reverse=True)
    unique_a = sorted(set_a - set_b, key=lambda x: nbrs_a[x], reverse=True)
    unique_b = sorted(set_b - set_a, key=lambda x: nbrs_b[x], reverse=True)

    # also check direct connection between the two
    direct_weight = G[node_a][node_b]["weight"] if G.has_edge(node_a, node_b) else 0
    direct_articles = (G[node_a][node_b].get("articles", [])
                       if G.has_edge(node_a, node_b) else [])

    return {
        "kw_a":           node_a,
        "kw_b":           node_b,
        "nbrs_a":         sorted(nbrs_a.items(), key=lambda x: x[1], reverse=True)[:15],
        "nbrs_b":         sorted(nbrs_b.items(), key=lambda x: x[1], reverse=True)[:15],
        "shared":         shared[:12],
        "unique_a":       unique_a[:12],
        "unique_b":       unique_b[:12],
        "direct_weight":  direct_weight,
        "direct_articles":direct_articles[:3],
    }


# ---------------------------------------------------------------------------
# Cross-period comparison
# ---------------------------------------------------------------------------

def compare_keyword_trends(graphs: dict, area: str,
                           metric: str = "weighted_degree",
                           n: int = 12) -> dict[str, list[dict]]:
    """Compare top-n keyword rankings across all periods for one area."""
    return {period: top_keywords_by_metric(graphs[area][period]["cooc"],
                                           metric=metric, n=n)
            for period in TIME_PERIODS}


def keyword_appearance_timeline(graphs: dict, area: str,
                                keyword: str) -> dict[str, int]:
    """Track a keyword's article count across all time periods."""
    timeline = {}
    for period in TIME_PERIODS:
        G = graphs[area][period]["cooc"].graph
        count = 0
        for node in G.nodes():
            if node.lower() == keyword.lower():
                count = G.nodes[node].get("article_count", 0)
                break
        timeline[period] = count
    return timeline


def discipline_keyword_trends(graphs: dict, area: str,
                               n: int = 10) -> dict[str, list[tuple]]:
    """
    Return top-n discipline keywords per period for one area.
    Shows WHICH research lenses (sociology, history, law…) dominate over time.
    """
    return {period: graphs[area][period]["cooc"].top_discipline_keywords(n=n)
            for period in TIME_PERIODS}


# ---------------------------------------------------------------------------
# Journal–institution analysis
# ---------------------------------------------------------------------------

def top_journals_across_periods(graphs: dict, area: str,
                                n: int = 10) -> dict[str, list[tuple]]:
    return {period: graphs[area][period]["ji"].top_journals(n=n)
            for period in TIME_PERIODS}


def institution_journal_profile(graphs: dict, area: str,
                                period: str, inst_name: str) -> dict:
    """Full publication profile for one institution in one area+period."""
    jg = graphs[area][period]["ji"]
    kg = graphs[area][period]["cooc"]

    journals = jg.journals_of_institution(inst_name)

    kw_profile: Counter = Counter()
    for kw, data in kg.graph.nodes(data=True):
        cnt = data.get("institutions", {}).get(inst_name, 0)
        if cnt:
            kw_profile[kw] += cnt

    return {
        "institution": inst_name,
        "area":        area,
        "period":      period,
        "journals":    journals,
        "top_keywords":kw_profile.most_common(8),
    }


# ---------------------------------------------------------------------------
# Institution collaboration analysis
# ---------------------------------------------------------------------------

def compute_inst_centrality(ig: InstitutionCollaborationGraph) -> dict[str, dict]:
    """
    Compute centrality for the institution collaboration graph.
    Returns {} for empty or single-node graphs.
    """
    G = ig.graph
    if G.number_of_nodes() < 2:
        return {}

    deg = nx.degree_centrality(G)
    btw = nx.betweenness_centrality(G, weight="weight", normalized=True)

    result = {}
    for nd in G.nodes():
        data = G.nodes[nd]
        wd   = sum(d["weight"] for _, _, d in G.edges(nd, data=True))
        top_kw = sorted(data.get("keywords", {}).items(),
                        key=lambda x: x[1], reverse=True)[:5]
        result[nd] = {
            "name":                  nd,
            "article_count":         data.get("article_count", 0),
            "degree_centrality":     round(deg.get(nd, 0), 4),
            "betweenness_centrality":round(btw.get(nd, 0), 4),
            "weighted_degree":       wd,
            "collaborators":         list(G.neighbors(nd)),
            "top_keywords":          top_kw,
        }
    return result


def compare_inst_periods(graphs: dict, area: str,
                         n: int = 10) -> dict[str, list[dict]]:
    """Top-n institutions by weighted_degree across all periods."""
    result = {}
    for period in TIME_PERIODS:
        centrality = compute_inst_centrality(graphs[area][period]["inst"])
        result[period] = sorted(centrality.values(),
                                key=lambda x: x["weighted_degree"],
                                reverse=True)[:n]
    return result
