"""
tests/test_graph.py
Test suite: verifies core behaviors and edge cases across all three graph types.

Reading these tests tells you:
- How Article filters subject vs discipline keywords
- How co-occurrence edges and node attributes are built
- How keyword comparison works (shared vs unique neighbors)
- How discipline keyword trends are extracted
- How the journal-institution bipartite graph is built
- How institution collaboration edges accumulate
- How centrality behaves on hub nodes vs isolated nodes
"""

import pytest
from models import (
    Article, KeywordCooccurrenceGraph,
    JournalInstitutionGraph, InstitutionCollaborationGraph,
    DISCIPLINE_KEYWORDS,
)
from graph import (
    compute_keyword_centrality, top_keywords_by_metric,
    search_keyword, compare_two_keywords,
    compute_inst_centrality,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_article(id_suffix, keywords, institutions=None,
                 journal="Test Journal", year=1970, cited=10):
    insts = institutions or []
    return Article(
        openalex_id=f"W{id_suffix}", title=f"Article {id_suffix}",
        year=year, cited_by_count=cited,
        keywords=keywords, journal=journal,
        institution_names=insts,
        institution_ids=[f"I{i}" for i in range(len(insts))],
    )


# ---------------------------------------------------------------------------
# TestArticle
# ---------------------------------------------------------------------------

class TestArticle:
    """Tests Article keyword filtering and pair generation."""

    def test_subject_keywords_removes_discipline_words(self):
        """subject_keywords() filters sociology, history, law, etc."""
        art = make_article("001", ["China", "Sociology", "Cold War", "History"])
        subj = art.subject_keywords()
        assert "China" in subj and "Cold War" in subj
        assert "Sociology" not in subj and "History" not in subj

    def test_discipline_keywords_returns_only_discipline_words(self):
        """discipline_keywords() returns only the filtered-out words."""
        art = make_article("002", ["China", "Sociology", "Law"])
        disc = art.discipline_keywords()
        assert "Sociology" in disc and "Law" in disc
        assert "China" not in disc

    def test_subject_keywords_case_insensitive(self):
        """Filter is case-insensitive: 'SOCIOLOGY' also removed."""
        art = make_article("003", ["China", "SOCIOLOGY"])
        assert "SOCIOLOGY" not in art.subject_keywords()

    def test_subject_keywords_removes_area_studies_labels(self):
        """'Asian studies' itself is a discipline label and should be removed."""
        art = make_article("004", ["Asian studies", "China"])
        assert "Asian studies" not in art.subject_keywords()
        assert "China" in art.subject_keywords()

    def test_subject_keywords_removes_library_science(self):
        """'Library science' added to blacklist — should be filtered."""
        art = make_article("005", ["Library science", "China"])
        assert "Library science" not in art.subject_keywords()

    def test_keyword_pairs_all_combinations(self):
        """Three subject keywords → 3 sorted pairs."""
        art = make_article("006", ["China", "Cold War", "Japan"])
        pairs = art.keyword_pairs()
        assert len(pairs) == 3
        assert ("China", "Cold War") in pairs

    def test_keyword_pairs_sorted_within_tuple(self):
        """Pairs are internally sorted to canonicalize edge direction."""
        art = make_article("007", ["Zebra", "Apple"])
        assert art.keyword_pairs() == [("Apple", "Zebra")]

    def test_keyword_pairs_empty_after_filtering(self):
        """If filtering leaves < 2 subject keywords, no pairs are produced."""
        art = make_article("008", ["China", "Sociology"])
        assert art.keyword_pairs() == []


# ---------------------------------------------------------------------------
# TestKeywordCooccurrenceGraph
# ---------------------------------------------------------------------------

class TestKeywordCooccurrenceGraph:
    """Tests co-occurrence graph construction and node attribute aggregation."""

    def test_two_subject_keywords_create_one_edge(self):
        kg = KeywordCooccurrenceGraph("Asian Studies", "1960–1975")
        kg.add_article(make_article("001", ["China", "Cold War"]))
        assert kg.graph.number_of_edges() == 1
        assert kg.graph["China"]["Cold War"]["weight"] == 1

    def test_repeated_cooccurrence_increases_weight(self):
        kg = KeywordCooccurrenceGraph("Asian Studies", "1960–1975")
        for i in range(3):
            kg.add_article(make_article(str(i), ["China", "Cold War"]))
        assert kg.graph["China"]["Cold War"]["weight"] == 3

    def test_three_keywords_produce_three_edges(self):
        kg = KeywordCooccurrenceGraph("Asian Studies", "1960–1975")
        kg.add_article(make_article("001", ["China", "Japan", "Korea"]))
        assert kg.graph.number_of_edges() == 3

    def test_discipline_words_not_added_as_nodes(self):
        kg = KeywordCooccurrenceGraph("Asian Studies", "1960–1975")
        kg.add_article(make_article("001", ["China", "Sociology", "Cold War"]))
        assert "Sociology" not in kg.graph.nodes()
        assert "China" in kg.graph.nodes()

    def test_institution_info_aggregates_on_node(self):
        kg = KeywordCooccurrenceGraph("Asian Studies", "1960–1975")
        kg.add_article(make_article("001", ["China"], institutions=["Harvard"]))
        kg.add_article(make_article("002", ["China"], institutions=["Harvard"]))
        kg.add_article(make_article("003", ["China"], institutions=["MIT"]))
        inst = kg.graph.nodes["China"]["institutions"]
        assert inst["Harvard"] == 2 and inst["MIT"] == 1

    def test_journal_info_aggregates_on_node(self):
        kg = KeywordCooccurrenceGraph("Asian Studies", "1960–1975")
        kg.add_article(make_article("001", ["China"], journal="JAS"))
        kg.add_article(make_article("002", ["China"], journal="JAS"))
        kg.add_article(make_article("003", ["China"], journal="PA"))
        jnl = kg.graph.nodes["China"]["journals"]
        assert jnl["JAS"] == 2 and jnl["PA"] == 1

    def test_discipline_cooccurrence_recorded_on_subject_node(self):
        """Discipline words that co-occur with a subject word are tracked."""
        kg = KeywordCooccurrenceGraph("Asian Studies", "1960–1975")
        kg.add_article(make_article("001", ["China", "Sociology"]))
        disc_co = kg.graph.nodes["China"]["discipline_co"]
        assert "Sociology" in disc_co

    def test_article_count_increments(self):
        kg = KeywordCooccurrenceGraph("African Studies", "1990–2000")
        kg.add_article(make_article("001", ["Nigeria", "Democracy"]))
        kg.add_article(make_article("002", ["Nigeria", "Economy"]))
        assert kg.graph.nodes["Nigeria"]["article_count"] == 2

    def test_empty_graph_has_no_nodes_or_edges(self):
        kg = KeywordCooccurrenceGraph("Latin American Studies", "2015–2025")
        assert kg.graph.number_of_nodes() == 0
        assert kg.graph.number_of_edges() == 0

    def test_top_discipline_keywords_returns_sorted_list(self):
        """top_discipline_keywords() returns discipline words sorted by frequency."""
        kg = KeywordCooccurrenceGraph("Asian Studies", "1960–1975")
        kg.add_article(make_article("001", ["China", "Sociology", "History"]))
        kg.add_article(make_article("002", ["Japan", "Sociology"]))
        top = kg.top_discipline_keywords(n=5)
        top_words = [t[0] for t in top]
        assert "Sociology" in top_words   # appears in both articles

    def test_neighbors_of_sorted_by_weight(self):
        kg = KeywordCooccurrenceGraph("Asian Studies", "1960–1975")
        for _ in range(3):
            kg.add_article(make_article("x", ["China", "Japan"]))
        kg.add_article(make_article("y", ["China", "Korea"]))
        nbrs = kg.neighbors_of("China")
        assert nbrs[0][0] == "Japan"   # weight 3 > 1


# ---------------------------------------------------------------------------
# TestKeywordComparison  (Mode 3 replacement)
# ---------------------------------------------------------------------------

class TestKeywordComparison:
    """
    Tests compare_two_keywords() — the replacement for path-finding.
    Verifies shared/unique neighbor logic and edge cases.
    """

    def _build_kg(self) -> KeywordCooccurrenceGraph:
        """
        Graph structure:
          China — Cold War, Japan, Korea (China is hub)
          Nigeria — Cold War, Ghana, Kenya (Nigeria is hub)
          Cold War — China, Nigeria (bridge between regions)
        """
        kg = KeywordCooccurrenceGraph("Mixed", "2015–2025")
        kg.add_article(make_article("001", ["China", "Cold War"]))
        kg.add_article(make_article("002", ["China", "Japan"]))
        kg.add_article(make_article("003", ["China", "Korea"]))
        kg.add_article(make_article("004", ["Nigeria", "Cold War"]))
        kg.add_article(make_article("005", ["Nigeria", "Ghana"]))
        kg.add_article(make_article("006", ["Nigeria", "Kenya"]))
        return kg

    def test_shared_neighbors_identified(self):
        """
        'Cold War' co-occurs with both China and Nigeria
        → should appear in shared neighbors.
        """
        kg = self._build_kg()
        result = compare_two_keywords(kg, "China", "Nigeria")
        assert "error" not in result
        assert "Cold War" in result["shared"]

    def test_unique_to_a_excludes_b_neighbors(self):
        """Japan and Korea are unique to China's neighborhood."""
        kg = self._build_kg()
        result = compare_two_keywords(kg, "China", "Nigeria")
        assert "Japan" in result["unique_a"]
        assert "Japan" not in result["unique_b"]

    def test_unique_to_b_excludes_a_neighbors(self):
        """Ghana and Kenya are unique to Nigeria's neighborhood."""
        kg = self._build_kg()
        result = compare_two_keywords(kg, "China", "Nigeria")
        assert "Ghana" in result["unique_b"]
        assert "Ghana" not in result["unique_a"]

    def test_direct_connection_detected(self):
        """If two keywords directly co-occur, direct_weight > 0."""
        kg = self._build_kg()
        result = compare_two_keywords(kg, "China", "Japan")
        assert result["direct_weight"] == 1

    def test_no_direct_connection_returns_zero(self):
        """If two keywords never co-occur directly, direct_weight == 0."""
        kg = self._build_kg()
        result = compare_two_keywords(kg, "Japan", "Nigeria")
        assert result["direct_weight"] == 0

    def test_unknown_keyword_returns_error(self):
        """Querying a nonexistent keyword returns error dict, no exception."""
        kg = self._build_kg()
        result = compare_two_keywords(kg, "China", "Nonexistent XYZ")
        assert "error" in result

    def test_same_keyword_returns_error(self):
        """Same keyword on both inputs returns error."""
        kg = self._build_kg()
        result = compare_two_keywords(kg, "China", "China")
        assert "error" in result

    def test_case_insensitive_lookup(self):
        """Keyword lookup is case-insensitive."""
        kg = self._build_kg()
        result = compare_two_keywords(kg, "china", "NIGERIA")
        assert "error" not in result

    def test_result_contains_expected_keys(self):
        """Result dict contains all expected keys."""
        kg = self._build_kg()
        result = compare_two_keywords(kg, "China", "Nigeria")
        for key in ["kw_a", "kw_b", "nbrs_a", "nbrs_b",
                    "shared", "unique_a", "unique_b",
                    "direct_weight", "direct_articles"]:
            assert key in result


# ---------------------------------------------------------------------------
# TestJournalInstitutionGraph
# ---------------------------------------------------------------------------

class TestJournalInstitutionGraph:
    """Tests journal-institution bipartite graph construction and queries."""

    def _build_jg(self) -> JournalInstitutionGraph:
        jg = JournalInstitutionGraph("Asian Studies", "1990–2000")
        jg.add_article(make_article("001", ["China"],
                        institutions=["Harvard"], journal="JAS"))
        jg.add_article(make_article("002", ["Japan"],
                        institutions=["Harvard", "MIT"], journal="PA"))
        jg.add_article(make_article("003", ["Korea"],
                        institutions=["Yale"], journal="JAS"))
        return jg

    def test_journal_nodes_added(self):
        jg = self._build_jg()
        names = [d["name"] for _, d in jg.graph.nodes(data=True)
                 if d.get("node_type") == "journal"]
        assert "JAS" in names and "PA" in names

    def test_institution_nodes_added(self):
        jg = self._build_jg()
        names = [d["name"] for _, d in jg.graph.nodes(data=True)
                 if d.get("node_type") == "institution"]
        assert "Harvard" in names and "MIT" in names and "Yale" in names

    def test_edge_weight_accumulates(self):
        jg = JournalInstitutionGraph("Asian Studies", "1960–1975")
        for i in range(3):
            jg.add_article(make_article(str(i), ["China"],
                            institutions=["Harvard"], journal="JAS"))
        j_id = jg._jid("JAS")
        i_id = jg._iid("Harvard")
        assert jg.graph[j_id][i_id]["weight"] == 3

    def test_journals_of_institution(self):
        jg = self._build_jg()
        journals = [j for j, _ in jg.journals_of_institution("Harvard")]
        assert "JAS" in journals and "PA" in journals

    def test_institutions_of_journal(self):
        jg = self._build_jg()
        insts = [i for i, _ in jg.institutions_of_journal("JAS")]
        assert "Harvard" in insts and "Yale" in insts
        assert "MIT" not in insts   # MIT published in PA, not JAS

    def test_article_without_journal_skipped(self):
        jg = JournalInstitutionGraph("Asian Studies", "1960–1975")
        jg.add_article(make_article("001", ["China"],
                        institutions=["Harvard"], journal=""))
        assert jg.graph.number_of_nodes() == 0

    def test_article_without_institution_skipped(self):
        jg = JournalInstitutionGraph("Asian Studies", "1960–1975")
        jg.add_article(make_article("001", ["China"],
                        institutions=[], journal="JAS"))
        assert jg.graph.number_of_nodes() == 0

    def test_top_journals_sorted(self):
        jg = JournalInstitutionGraph("Asian Studies", "1960–1975")
        for i in range(5):
            jg.add_article(make_article(str(i), ["China"],
                            institutions=["Harvard"], journal="JAS"))
        jg.add_article(make_article("x", ["Japan"],
                        institutions=["MIT"], journal="PA"))
        assert jg.top_journals(2)[0][0] == "JAS"


# ---------------------------------------------------------------------------
# TestInstitutionCollaborationGraph
# ---------------------------------------------------------------------------

class TestInstitutionCollaborationGraph:
    """Tests institution collaboration graph (tertiary, sparse by nature)."""

    def test_two_institutions_same_article_create_edge(self):
        ig = InstitutionCollaborationGraph("Asian Studies", "1960–1975")
        ig.add_article(make_article("001", ["China"],
                        institutions=["Harvard", "MIT"]))
        assert ig.graph.number_of_edges() == 1
        assert ig.graph["Harvard"]["MIT"]["weight"] == 1

    def test_repeated_collaboration_increases_weight(self):
        ig = InstitutionCollaborationGraph("Asian Studies", "1960–1975")
        for i in range(3):
            ig.add_article(make_article(str(i), ["China"],
                            institutions=["Harvard", "MIT"]))
        assert ig.graph["Harvard"]["MIT"]["weight"] == 3

    def test_single_institution_article_produces_no_edge(self):
        """Single-author/single-institution article should not create edges."""
        ig = InstitutionCollaborationGraph("Asian Studies", "1960–1975")
        ig.add_article(make_article("001", ["China"],
                        institutions=["Harvard"]))
        assert ig.graph.number_of_edges() == 0

    def test_three_institutions_create_three_edges(self):
        ig = InstitutionCollaborationGraph("Asian Studies", "1960–1975")
        ig.add_article(make_article("001", ["China"],
                        institutions=["Harvard", "MIT", "Yale"]))
        assert ig.graph.number_of_edges() == 3

    def test_keywords_accumulate_on_institution_node(self):
        """Subject keywords of articles accumulate on institution nodes."""
        ig = InstitutionCollaborationGraph("Asian Studies", "1960–1975")
        ig.add_article(make_article("001", ["China", "Cold War"],
                        institutions=["Harvard", "MIT"]))
        kw = ig.graph.nodes["Harvard"]["keywords"]
        assert "China" in kw and "Cold War" in kw

    def test_empty_graph_returns_empty_centrality(self):
        ig = InstitutionCollaborationGraph("Asian Studies", "1960–1975")
        assert compute_inst_centrality(ig) == {}

    def test_hub_institution_has_highest_weighted_degree(self):
        """Institution connected to all others should have highest weighted degree."""
        ig = InstitutionCollaborationGraph("Asian Studies", "1960–1975")
        ig.add_article(make_article("001", ["China"],
                        institutions=["Harvard", "MIT"]))
        ig.add_article(make_article("002", ["China"],
                        institutions=["Harvard", "Yale"]))
        ig.add_article(make_article("003", ["China"],
                        institutions=["Harvard", "Columbia"]))
        centrality = compute_inst_centrality(ig)
        harvard_wd = centrality["Harvard"]["weighted_degree"]
        mit_wd     = centrality["MIT"]["weighted_degree"]
        assert harvard_wd > mit_wd


# ---------------------------------------------------------------------------
# TestKeywordCentrality
# ---------------------------------------------------------------------------

class TestKeywordCentrality:
    """Tests centrality computation on co-occurrence graphs."""

    def test_hub_keyword_has_highest_weighted_degree(self):
        kg = KeywordCooccurrenceGraph("Asian Studies", "1960–1975")
        kg.add_article(make_article("001", ["China", "Japan"]))
        kg.add_article(make_article("002", ["China", "Korea"]))
        kg.add_article(make_article("003", ["China", "Vietnam"]))
        kg.add_article(make_article("004", ["Japan", "Korea"]))
        centrality = compute_keyword_centrality(kg)
        assert (centrality["China"]["weighted_degree"] >
                centrality["Japan"]["weighted_degree"])

    def test_empty_graph_returns_empty_dict(self):
        kg = KeywordCooccurrenceGraph("Asian Studies", "1960–1975")
        assert compute_keyword_centrality(kg) == {}

    def test_centrality_dict_has_all_expected_keys(self):
        kg = KeywordCooccurrenceGraph("Asian Studies", "1960–1975")
        kg.add_article(make_article("001", ["China", "Japan"]))
        c = compute_keyword_centrality(kg)
        for kw in ["China", "Japan"]:
            for key in ["article_count", "weighted_degree", "degree_centrality",
                        "betweenness_centrality", "top_institutions",
                        "top_journals", "discipline_co", "neighbors"]:
                assert key in c[kw], f"Missing key '{key}' for '{kw}'"


# ---------------------------------------------------------------------------
# TestSearchKeyword
# ---------------------------------------------------------------------------

class TestSearchKeyword:
    """Tests fuzzy keyword search."""

    def _build_kg(self) -> KeywordCooccurrenceGraph:
        kg = KeywordCooccurrenceGraph("Asian Studies", "2015–2025")
        kg.add_article(make_article("001", ["China", "Cold War", "Taiwan"]))
        kg.add_article(make_article("002", ["China", "Belt and Road"]))
        return kg

    def test_exact_match(self):
        assert len(search_keyword(self._build_kg(), "China")) >= 1

    def test_partial_match(self):
        results = search_keyword(self._build_kg(), "belt")
        assert any("Belt" in r["keyword"] for r in results)

    def test_case_insensitive(self):
        kg = self._build_kg()
        assert len(search_keyword(kg, "china")) == len(search_keyword(kg, "CHINA"))

    def test_no_match_returns_empty(self):
        assert search_keyword(self._build_kg(), "Nonexistent XYZ") == []

    def test_results_sorted_by_article_count(self):
        kg = self._build_kg()
        results = search_keyword(kg, "")
        counts = [r["article_count"] for r in results]
        assert counts == sorted(counts, reverse=True)
