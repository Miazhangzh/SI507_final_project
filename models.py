"""
models.py
Core data models: Article, KeywordCooccurrenceGraph,
JournalInstitutionGraph, InstitutionCollaborationGraph
"""

from __future__ import annotations
from dataclasses import dataclass, field
from itertools import combinations
import networkx as nx


# ---------------------------------------------------------------------------
# Discipline keyword blacklist
# These words describe the research LENS, not the research SUBJECT.
# Filtered out from subject-keyword analysis; kept for discipline analysis.
# ---------------------------------------------------------------------------
DISCIPLINE_KEYWORDS: frozenset[str] = frozenset({
    "asian studies", "african studies", "latin american studies",
    "middle eastern studies", "european studies", "area studies",
    "latin americans", "africans", "asians",
    "political science", "sociology", "anthropology", "history",
    "economics", "geography", "archaeology", "philosophy",
    "humanities", "law", "politics", "literature", "linguistics",
    "cultural studies", "social sciences", "international relations",
    "public policy", "education", "psychology", "medicine",
    "art", "arts", "science", "religion", "theology",
    "gender studies", "women's studies", "ethnic studies",
    "development studies", "postcolonial studies",
    "library science", "information science",
    "library and information science",
    "computer science", "state (computer science)", "index (typography)", 
    "social science", "neuroscience",
    "data science", "cognitive science", "political economy",
    "communication", "journalism", "media studies",
    "urban studies", "environmental studies", "public health",
    "demography", "statistics", "biology", "physics",
    "qualitative research", "quantitative research",
    "ethnography", "survey", "discourse analysis",
    "mixed methods", "systematic review", "meta-analysis",
    "Power (physics)"
})


# ---------------------------------------------------------------------------
# Article
# ---------------------------------------------------------------------------
@dataclass
class Article:
    """
    Represents a single academic article with all fields needed for analysis.
    Sourced from the OpenAlex works API.
    """
    openalex_id:       str
    title:             str
    year:              int
    cited_by_count:    int            = 0
    keywords:          list[str]      = field(default_factory=list)
    topics:            list[str]      = field(default_factory=list)
    journal:           str            = ""
    journal_id:        str            = ""
    institution_names: list[str]      = field(default_factory=list)
    institution_ids:   list[str]      = field(default_factory=list)

    def subject_keywords(self) -> list[str]:
        """
        Return keywords with discipline words filtered out.
        e.g. ["China", "Cold War", "Asian studies"] -> ["China", "Cold War"]
        """
        return [kw for kw in self.keywords
                if kw.lower() not in DISCIPLINE_KEYWORDS]

    def discipline_keywords(self) -> list[str]:
        """
        Return only discipline/methodology keywords.
        e.g. ["China", "Sociology", "History"] -> ["Sociology", "History"]
        """
        return [kw for kw in self.keywords
                if kw.lower() in DISCIPLINE_KEYWORDS]

    def keyword_pairs(self) -> list[tuple[str, str]]:
        """
        Return all sorted pairs of subject keywords (used to build co-occurrence edges).
        Sorting guarantees (A,B) == (B,A) so no duplicate edges are created.
        """
        subj = self.subject_keywords()
        return [tuple(sorted(pair)) for pair in combinations(subj, 2)]

    def us_institutions(self) -> list[str]:
        """Return US institution names (already filtered during fetch)."""
        return self.institution_names


# ---------------------------------------------------------------------------
# KeywordCooccurrenceGraph  (primary graph)
# ---------------------------------------------------------------------------
class KeywordCooccurrenceGraph:
    """
    Primary graph: keyword co-occurrence network.

    Nodes = subject keywords (discipline words excluded)
    Edges = two keywords appear together in the same article
    Edge weight = number of articles where both keywords appear

    Each node also stores (Direction 1 integration):
      article_count  : how many articles contain this keyword
      institutions   : {institution_name: count}
      journals       : {journal_name: count}
      discipline_co  : discipline words that co-occur with this subject keyword
    """

    def __init__(self, area: str, period: str):
        self.area    = area
        self.period  = period
        self.graph: nx.Graph     = nx.Graph()
        self.articles: list[Article] = []

    def add_article(self, article: Article) -> None:
        """Add one article's keyword co-occurrence relationships to the graph."""
        self.articles.append(article)
        subj_kws = article.subject_keywords()
        disc_kws = article.discipline_keywords()

        # -- nodes --
        for kw in subj_kws:
            if not self.graph.has_node(kw):
                self.graph.add_node(kw,
                    article_count=0,
                    institutions={},
                    journals={},
                    discipline_co={},
                )
            nd = self.graph.nodes[kw]
            nd["article_count"] += 1

            for inst in article.institution_names:
                nd["institutions"][inst] = nd["institutions"].get(inst, 0) + 1

            if article.journal:
                nd["journals"][article.journal] = \
                    nd["journals"].get(article.journal, 0) + 1

            # accumulate discipline words that co-occur with this subject keyword
            for dkw in disc_kws:
                nd["discipline_co"][dkw] = nd["discipline_co"].get(dkw, 0) + 1

        # -- edges --
        for kw_a, kw_b in article.keyword_pairs():
            if self.graph.has_edge(kw_a, kw_b):
                self.graph[kw_a][kw_b]["weight"]   += 1
                self.graph[kw_a][kw_b]["articles"].append(article.title)
            else:
                self.graph.add_edge(kw_a, kw_b,
                    weight=1,
                    articles=[article.title])

    def top_keywords(self, n: int = 15) -> list[tuple[str, int]]:
        """Return top-n keywords by article count."""
        counts = [(kw, self.graph.nodes[kw]["article_count"])
                  for kw in self.graph.nodes]
        return sorted(counts, key=lambda x: x[1], reverse=True)[:n]

    def top_discipline_keywords(self, n: int = 15) -> list[tuple[str, int]]:
        """
        Return top-n discipline keywords across all articles in this graph.
        Aggregated from per-node discipline_co attributes.
        """
        freq: dict[str, int] = {}
        for kw in self.graph.nodes:
            for dkw, cnt in self.graph.nodes[kw]["discipline_co"].items():
                freq[dkw] = freq.get(dkw, 0) + cnt
        # de-duplicate by halving (each pair counted twice)
        return sorted(
            [(k, v // 2 or 1) for k, v in freq.items()],
            key=lambda x: x[1], reverse=True
        )[:n]

    def neighbors_of(self, keyword: str) -> list[tuple[str, int]]:
        """Return co-occurring keywords sorted by weight descending."""
        if keyword not in self.graph:
            return []
        return sorted(
            [(nbr, self.graph[keyword][nbr]["weight"])
             for nbr in self.graph.neighbors(keyword)],
            key=lambda x: x[1], reverse=True
        )

    def summary(self) -> dict:
        return {
            "area": self.area, "period": self.period,
            "num_articles": len(self.articles),
            "num_keywords": self.graph.number_of_nodes(),
            "num_edges":    self.graph.number_of_edges(),
            "density":      round(nx.density(self.graph), 4),
            "top_keywords": self.top_keywords(5),
        }


# ---------------------------------------------------------------------------
# JournalInstitutionGraph  (secondary graph)
# ---------------------------------------------------------------------------
class JournalInstitutionGraph:
    """
    Secondary graph: journal–institution bipartite graph.

    Node type "journal"      = academic journal
    Node type "institution"  = university / research center
    Edge weight = number of articles that institution published in that journal
    """

    def __init__(self, area: str, period: str):
        self.area    = area
        self.period  = period
        self.graph: nx.Graph     = nx.Graph()
        self.articles: list[Article] = []

    def _jid(self, name: str) -> str: return f"J::{name}"
    def _iid(self, name: str) -> str: return f"I::{name}"

    def add_article(self, article: Article) -> None:
        if not article.journal or not article.institution_names:
            return
        self.articles.append(article)
        j_id = self._jid(article.journal)

        if not self.graph.has_node(j_id):
            self.graph.add_node(j_id, node_type="journal",
                                name=article.journal, article_count=0)
        self.graph.nodes[j_id]["article_count"] += 1

        for inst in article.institution_names:
            i_id = self._iid(inst)
            if not self.graph.has_node(i_id):
                self.graph.add_node(i_id, node_type="institution",
                                    name=inst, article_count=0)
            self.graph.nodes[i_id]["article_count"] += 1
            if self.graph.has_edge(j_id, i_id):
                self.graph[j_id][i_id]["weight"] += 1
            else:
                self.graph.add_edge(j_id, i_id, weight=1)

    def top_journals(self, n: int = 10) -> list[tuple[str, int]]:
        return sorted(
            [(d["name"], d["article_count"])
             for _, d in self.graph.nodes(data=True)
             if d.get("node_type") == "journal"],
            key=lambda x: x[1], reverse=True
        )[:n]

    def top_institutions(self, n: int = 10) -> list[tuple[str, int]]:
        return sorted(
            [(d["name"], d["article_count"])
             for _, d in self.graph.nodes(data=True)
             if d.get("node_type") == "institution"],
            key=lambda x: x[1], reverse=True
        )[:n]

    def journals_of_institution(self, inst: str) -> list[tuple[str, int]]:
        i_id = self._iid(inst)
        if i_id not in self.graph:
            return []
        return sorted(
            [(self.graph.nodes[nb]["name"], self.graph[i_id][nb]["weight"])
             for nb in self.graph.neighbors(i_id)],
            key=lambda x: x[1], reverse=True
        )

    def institutions_of_journal(self, journal: str) -> list[tuple[str, int]]:
        j_id = self._jid(journal)
        if j_id not in self.graph:
            return []
        return sorted(
            [(self.graph.nodes[nb]["name"], self.graph[j_id][nb]["weight"])
             for nb in self.graph.neighbors(j_id)],
            key=lambda x: x[1], reverse=True
        )

    def summary(self) -> dict:
        journals = [d for _, d in self.graph.nodes(data=True)
                    if d.get("node_type") == "journal"]
        insts    = [d for _, d in self.graph.nodes(data=True)
                    if d.get("node_type") == "institution"]
        return {
            "area": self.area, "period": self.period,
            "num_articles": len(self.articles),
            "num_journals": len(journals),
            "num_institutions": len(insts),
            "num_edges": self.graph.number_of_edges(),
        }


# ---------------------------------------------------------------------------
# InstitutionCollaborationGraph  (tertiary graph, restored from v1)
# ---------------------------------------------------------------------------
class InstitutionCollaborationGraph:
    """
    Tertiary graph: institution collaboration network (restored from v1).

    Nodes = US institutions
    Edges = two institutions share at least one author on the same article
    Edge weight = number of co-authored articles

    Note: sparse by nature in humanities/social-science area studies,
    since single-author publications dominate. Used as supplementary view.
    """

    def __init__(self, area: str, period: str):
        self.area    = area
        self.period  = period
        self.graph: nx.Graph     = nx.Graph()
        self.articles: list[Article] = []

    def add_article(self, article: Article) -> None:
        """
        Add collaboration edges only if the article has >= 2 US institutions.
        Single-institution articles are recorded but produce no edges.
        """
        self.articles.append(article)
        insts = article.institution_names
        if len(insts) < 2:
            return   # no edge possible / 单机构无法产生边

        for inst in insts:
            if not self.graph.has_node(inst):
                self.graph.add_node(inst, article_count=0,
                                    keywords={})
            self.graph.nodes[inst]["article_count"] += 1
            for kw in article.subject_keywords():
                kd = self.graph.nodes[inst]["keywords"]
                kd[kw] = kd.get(kw, 0) + 1

        for a, b in combinations(insts, 2):
            if self.graph.has_edge(a, b):
                self.graph[a][b]["weight"] += 1
            else:
                self.graph.add_edge(a, b, weight=1)

    def top_institutions(self, n: int = 10) -> list[tuple[str, int]]:
        return sorted(
            [(nd, self.graph.nodes[nd]["article_count"])
             for nd in self.graph.nodes],
            key=lambda x: x[1], reverse=True
        )[:n]

    def summary(self) -> dict:
        return {
            "area": self.area, "period": self.period,
            "num_articles": len(self.articles),
            "num_institutions": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "density": round(nx.density(self.graph), 4),
        }
