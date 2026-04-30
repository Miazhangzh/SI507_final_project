# U.S. Area Studies: Concept Maps Across Time

A network analysis tool that maps the intellectual evolution of U.S. university
area studies research across three fields and three historical periods,
using data from the [OpenAlex](https://openalex.org) scholarly metadata API.

**Fields covered:** Asian Studies · African Studies · Latin American Studies  
**Time periods:** 1960–1975 · 1990–2000 · 2015–2025  
**Data source:** OpenAlex (free, open, no API key required)

---

## File Structure

All Python files are in the project root (flat structure):

```
final_project/
├── README.md           # Project documentation
├── app_en.py           # Streamlit application (submission version, English)
├── models.py           # Core data models and graph classes
├── fetch.py            # OpenAlex API data fetching and local caching
├── graph.py            # Graph construction and analysis algorithms
├── test_graph.py       # Test suite — 51 tests across 7 test classes
├── requirements.txt    # Python dependencies
├── data/
    └── cache/          # Auto-created on first run; stores API responses as JSON

```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app_en.py

# 3. Run tests
python3 -m pytest test_graph.py -v
```

On first launch, the app fetches data from OpenAlex (approximately 1–2 minutes
for all 9 area × period combinations) and caches it to `data/cache/`.
All subsequent runs load from cache and start in seconds.

---

## Application Modes

The app has seven navigation modes accessible from the left sidebar:

| Mode | Description |
|---|---|
| 🏠 **Overview** | Project background, graph structure explained, dataset table, rationale for time periods and field selection |
| ① **Keyword Explorer** | Search any topic keyword; see co-occurring keywords, institutions, journals, discipline lenses, and optional AI interpretation |
| ② **Period Comparison** | Compare top keywords and discipline distributions across three periods; concept network visualizations, keyword tracker, bubble charts |
| ③ **Keyword Comparison** | Enter two keywords to compare their co-occurrence neighborhoods; reveals shared intellectual ground and distinctive research contexts; optional AI interpretation |
| ④ **Publication Ecology** | Journal–institution bipartite graph; drill into specific journals or institutions; institution collaboration network |
| 🎯 **Key Findings** | Written analytical findings for Asian Studies, African Studies, Latin American Studies, and Institution Collaboration networks |
| 💡 **Methodological Reflection** | Documents the original design, why it was revised, and the key methodological decisions made during the project |

---

## File Descriptions

### `models.py`

Defines four data model classes and the discipline keyword blacklist.

**`DISCIPLINE_KEYWORDS`** — a `frozenset` of ~45 discipline and methodology
labels (e.g., *sociology*, *history*, *computer science*) that are filtered
out from subject-keyword analysis. Subject keywords and discipline keywords
are tracked separately throughout the project.

**`Article`** — represents one academic article from OpenAlex.
Stores title, year, citation count, raw keywords, journal, and institution names.

Key methods:
- `subject_keywords()` — returns keywords with discipline labels removed
- `discipline_keywords()` — returns only discipline/methodology labels
- `keyword_pairs()` — returns all sorted pairs of subject keywords;
  each pair becomes one co-occurrence edge in the graph

**`KeywordCooccurrenceGraph`** — the primary graph structure.
- **Nodes** = subject keywords
- **Edges** = two keywords appear together in the same article
- **Edge weight** = number of articles where both keywords co-occur
- Each node stores which institutions and journals most use that keyword,
  and which discipline words co-occur with it (secondary graph integration)

**`JournalInstitutionGraph`** — the secondary graph structure (bipartite).
- **Node type A** = academic journal
- **Node type B** = US institution
- **Edge** = institution has published in that journal
- **Edge weight** = number of articles

**`InstitutionCollaborationGraph`** — the tertiary graph structure.
- **Nodes** = US institutions
- **Edges** = two institutions share an author on the same article
- **Edge weight** = number of co-authored articles
- Intentionally sparse, reflecting the single-author publication culture
  of humanities area studies

---

### `fetch.py`

Handles all OpenAlex API communication and local caching.

**Configuration:**
- `AREA_KEYWORD_IDS` — maps field names to OpenAlex keyword IDs
- `TIME_PERIODS` — maps period labels to API date range strings
- `PAGE_SIZE = 200` — maximum results per API page
- `MAX_PAGES = 25` — cap per fetch to prevent runaway requests

**API filters applied per request:**
```
keywords.id          = <area keyword>
authorships.institutions.country_code = US
type                 = article
publication_year     = <period range>
```

**Public functions:**
- `fetch_articles(area, period_label)` — fetches one area × period combination,
  reading from local cache if available, calling the API otherwise
- `fetch_all()` — fetches all 9 combinations; called once at app startup

**Private helpers** (prefixed `_`):
- `_cache_path()` — generates the cache file path for a given area × period
- `_load_cache()` / `_save_cache()` — read and write JSON cache files
- `_fetch_raw()` — makes paginated API requests and returns raw JSON results
- `_parse_article()` — converts one raw OpenAlex JSON object into an `Article`

---

### `graph.py`

Builds all three graph types and provides all analysis functions.

**Graph construction:**
- `build_cooccurrence_graph(area, period, articles)` → `KeywordCooccurrenceGraph`
- `build_journal_institution_graph(area, period, articles)` → `JournalInstitutionGraph`
- `build_institution_collab_graph(area, period, articles)` → `InstitutionCollaborationGraph`
- `build_all_graphs(all_data)` — orchestrates all three for all areas × periods;
  returns `{ area: { period: { "cooc": ..., "ji": ..., "inst": ... } } }`

**Keyword analysis:**
- `compute_keyword_centrality(kg)` — computes degree centrality,
  betweenness centrality, and weighted degree for every node
- `top_keywords_by_metric(kg, metric, n)` — ranks keywords by specified metric
- `search_keyword(kg, query)` — fuzzy, case-insensitive keyword search
- `compare_two_keywords(kg, kw_a, kw_b)` — compares two keywords by their
  co-occurrence neighborhoods; returns shared neighbors, unique-to-A,
  unique-to-B, and direct co-occurrence details

**Cross-period analysis:**
- `compare_keyword_trends(graphs, area, metric, n)` — top-N keywords per period
- `keyword_appearance_timeline(graphs, area, keyword)` — article count across periods
- `discipline_keyword_trends(graphs, area, n)` — top discipline keywords per period

**Journal–institution analysis:**
- `top_journals_across_periods(graphs, area, n)` — most active journals per period
- `institution_journal_profile(graphs, area, period, inst_name)` — one
  institution's journals and keyword profile for a given area + period

**Institution collaboration analysis:**
- `compute_inst_centrality(ig)` — centrality metrics for collaboration graph nodes
- `compare_inst_periods(graphs, area, n)` — top collaborating institutions per period

---

### `app_en.py`

The main Streamlit application (submission version, English only).

In addition to rendering the seven navigation modes, this file contains:

**`_get_api_key()`** — retrieves the Anthropic API key from
`st.secrets` (`.streamlit/secrets.toml`) or the `ANTHROPIC_API_KEY`
environment variable. Returns an empty string if neither is set.

**`call_claude(system_prompt, user_prompt, max_tokens)`** — calls the
Anthropic API (`claude-sonnet-4-20250514`) and returns the response text.
If no API key is configured, returns a plain-text message explaining
how to enable the feature. Used in two places:
- **Keyword Explorer** — interprets a keyword's network position,
  co-occurrence pattern, and institutional usage
- **Keyword Comparison** — interprets the shared and unique neighbors
  of two compared keywords

**`st.session_state` usage** — comparison results and AI outputs are stored
in session state (`cmp_result`, `cmp_analysis`, `explorer_analysis`) so they
persist when the user clicks "Generate AI analysis" without resetting the page.

**`draw_cooc_network(kg, highlight_kw, max_nodes)`** — renders the keyword
co-occurrence network using Plotly. Node size encodes article count;
node color encodes co-occurrence weight; highlighted node shown in orange.

**`draw_inst_network(ig, max_nodes)`** — renders the institution
collaboration network using Plotly.

---

### `test_graph.py`

51 tests organized into 7 test classes. Reading the test suite provides
a complete behavioral specification of the project.

| Test class | What it verifies |
|---|---|
| `TestArticle` | Keyword filtering logic, pair generation, case sensitivity, area studies label removal |
| `TestKeywordCooccurrenceGraph` | Node/edge construction, weight accumulation, institution and journal attribute aggregation, discipline co-occurrence tracking |
| `TestKeywordComparison` | Shared/unique neighbor identification, direct co-occurrence detection, case-insensitive lookup, error handling for unknown keywords and identical inputs |
| `TestJournalInstitutionGraph` | Bipartite graph construction, edge weight accumulation, journal and institution query methods, skipping articles missing journal or institution data |
| `TestInstitutionCollaborationGraph` | Collaboration edge creation, single-institution article filtering, keyword accumulation on institution nodes, hub node centrality |
| `TestKeywordCentrality` | Hub node detection, empty graph handling, presence of all expected output keys |
| `TestSearchKeyword` | Exact and partial matching, case insensitivity, empty result handling, result ordering by article count |

Run all tests:
```bash
python3 -m pytest test_graph.py -v
```

---

## Enabling AI Analysis

The Keyword Explorer and Keyword Comparison modes include an optional AI
analysis feature powered by Claude (`claude-sonnet-4-20250514`).
The rest of the app works fully without it.

**To enable, provide an Anthropic API key by one of these methods:**

**Method 1 — Streamlit secrets (recommended):**

Create a file at `.streamlit/secrets.toml` in the project root:
```toml
ANTHROPIC_API_KEY = "sk-ant-your-key-here"
```
Then run normally: `streamlit run app_en.py`

**Method 2 — Environment variable:**
```bash
ANTHROPIC_API_KEY=sk-ant-your-key-here streamlit run app_en.py
```

API keys can be obtained at [console.anthropic.com](https://console.anthropic.com).

**Without a key:** clicking "Generate AI analysis" displays a plain-text
message explaining how to configure the feature. No error is thrown.

---

## .gitignore

Add the following to `.gitignore` before pushing to GitHub:

```
# API key — never commit this
.streamlit/secrets.toml

# OpenAlex cache — large JSON files, not needed in the repo
data/cache/

# Python artifacts
__pycache__/
*.pyc
.DS_Store
```

---

## Graph Design Rationale

This project originally used institution collaboration networks as the primary
graph structure. That design was abandoned because humanities area studies is
dominated by single-author publications — most articles produce no collaboration
edges, making the graphs too sparse to be analytically useful.

Keyword co-occurrence networks were adopted instead: every article, regardless
of authorship, contributes edges between its subject keywords. This makes the
graphs substantially denser and — more importantly — answers the right question
for this domain: not *who works together*, but *what ideas are being connected*.

The institution collaboration graphs are retained as a supplementary view.
Their sparsity, reinterpreted, became a substantive finding: that area studies
in the United States is produced primarily by individual scholars working alone,
consistently across six decades and three fields.

A full account of these methodological decisions is available in the
**💡 Methodological Reflection** mode within the app.

---

## Dependencies

```
requests>=2.31.0       # OpenAlex API and Anthropic API calls
networkx>=3.2          # Graph construction and analysis
streamlit>=1.32.0      # Web interface
plotly>=5.18.0         # Interactive network visualizations and charts
pandas>=2.0.0          # Data tables
pytest>=7.4.0          # Test runner
```

Install all at once:
```bash
pip install -r requirements.txt
```
