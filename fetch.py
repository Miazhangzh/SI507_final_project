"""
fetch.py
--------

Fetch data from OpenAlex API with local caching

Changes from institution-collaboration version:
- 不再过滤"至少两个机构"，单人发表文章也保留（因为每篇都能贡献关键词共现边）
  No longer filters for ≥2 institutions; solo papers kept (each contributes keyword edges)
- 额外提取期刊信息（source字段）
  Additionally extracts journal info (from source field)
- 分页拉取全量，无数量上限
  Paginated full fetch, no article count cap
"""

import json
import time
import requests
from pathlib import Path
from models import Article

# ── 配置 / Configuration ──────────────────────────────────────────────────────

BASE_URL  = "https://api.openalex.org/works"
CACHE_DIR = Path("data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

AREA_KEYWORD_IDS: dict[str, str] = {
    "Asian Studies":          "asian-studies",
    "African Studies":        "african-studies",
    "Latin American Studies": "latin-american-studies",
}

TIME_PERIODS: dict[str, str] = {
    "1960–1975": "1960-1975",
    "1990–2000": "1990-2000",
    "2015–2025": "2015-2025",
}

PAGE_SIZE = 200    # OpenAlex 单页最大条数 / max results per page
MAX_PAGES = 25     # 防止超大数据集 / guard against very large datasets


# ── 缓存 / Cache ──────────────────────────────────────────────────────────────

def _cache_path(area: str, period: str) -> Path:
    key = f"{area}_{period}_kw".replace(" ", "_").replace("–", "-")
    return CACHE_DIR / f"{key}.json"

def _load_cache(area: str, period: str) -> list[dict] | None:
    p = _cache_path(area, period)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None

def _save_cache(area: str, period: str, data: list[dict]) -> None:
    p = _cache_path(area, period)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  [缓存 / Cached] {p} ({len(data)} 条)")


# ── API 分页拉取 / Paginated API fetch ───────────────────────────────────────

def _fetch_raw(area: str, period_range: str) -> list[dict]:
    """
    分页拉取全量原始数据（不限制文章数量，不按引用量排序）
    Fetch all raw results with pagination (no article cap, no citation sort)

    筛选条件 / filters:
      keywords.id = <area keyword>
      authorships.institutions.country_code = US
      type = article
      publication_year = <range>
    """
    keyword_id = AREA_KEYWORD_IDS[area]
    filters = ",".join([
        f"keywords.id:{keyword_id}",
        "authorships.institutions.country_code:US",
        "type:article",
        f"publication_year:{period_range}",
    ])
    # 请求所需字段（含 primary_location 获取期刊信息）
    # Request needed fields (including primary_location for journal info)
    select_fields = ",".join([
        "id", "title", "publication_year", "cited_by_count",
        "keywords", "topics", "authorships", "primary_location",
    ])

    all_results: list[dict] = []
    print(f"  [拉取 / Fetching] {area} | {period_range}")

    for page in range(1, MAX_PAGES + 1):
        resp = requests.get(BASE_URL, params={
            "filter":   filters,
            "select":   select_fields,
            "per_page": PAGE_SIZE,
            "page":     page,
        }, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        all_results.extend(results)

        if page == 1:
            total = data["meta"]["count"]
            pages_needed = min((total + PAGE_SIZE - 1) // PAGE_SIZE, MAX_PAGES)
            print(f"  [总量 / Total] {total} 篇，拉取 {pages_needed} 页")

        if len(results) < PAGE_SIZE:
            break   # 最后一页 / last page reached
        time.sleep(0.3)

    print(f"  [完成 / Done] 共 {len(all_results)} 条原始数据")
    return all_results


# ── 解析 / Parsing ────────────────────────────────────────────────────────────

def _parse_article(raw: dict) -> Article | None:
    """
    把 OpenAlex 原始 JSON 解析成 Article 对象
    Parse OpenAlex raw JSON into an Article object

    保留所有文章（单人发表也保留，因为关键词共现不需要多机构）
    Keep all articles (solo papers included; keyword co-occurrence doesn't need multi-inst)
    """
    # 提取关键词 / extract keywords
    keywords = [kw["display_name"]
                for kw in raw.get("keywords", [])
                if kw.get("display_name")]

    # 没有任何关键词的文章对共现网络无贡献，跳过
    # Articles with no keywords contribute nothing to co-occurrence; skip
    if not keywords:
        return None

    # 提取主题 / extract topics
    topics = [t["display_name"]
              for t in raw.get("topics", [])
              if t.get("display_name")]

    # 提取期刊信息（来自 primary_location.source）
    # Extract journal info from primary_location.source
    journal = ""
    journal_id = ""
    loc = raw.get("primary_location") or {}
    source = loc.get("source") or {}
    if source:
        journal    = source.get("display_name", "")
        journal_id = source.get("id", "")

    # 提取机构信息（去重，只保留US机构名）
    # Extract institution info (deduplicated, US only)
    seen_inst_ids: set[str] = set()
    institution_names: list[str] = []
    institution_ids: list[str] = []

    for authorship in raw.get("authorships", []):
        for inst in authorship.get("institutions", []):
            inst_id = inst.get("id", "")
            if (inst.get("country_code") == "US"
                    and inst_id
                    and inst_id not in seen_inst_ids):
                seen_inst_ids.add(inst_id)
                institution_names.append(inst.get("display_name", ""))
                institution_ids.append(inst_id)

    return Article(
        openalex_id      = raw.get("id", ""),
        title            = raw.get("title", "Untitled"),
        year             = raw.get("publication_year", 0),
        cited_by_count   = raw.get("cited_by_count", 0),
        keywords         = keywords,
        topics           = topics,
        journal          = journal,
        journal_id       = journal_id,
        institution_names= institution_names,
        institution_ids  = institution_ids,
    )


# ── 主入口 / Main entry ───────────────────────────────────────────────────────

def fetch_articles(area: str, period_label: str) -> list[Article]:
    """
    获取某方向+时间段的全量文章（优先读缓存）
    Fetch all articles for given area+period (cache-first)
    """
    period_range = TIME_PERIODS[period_label]

    cached = _load_cache(area, period_label)
    if cached is not None:
        print(f"  [缓存读取 / Cache hit] {area} | {period_label} ({len(cached)} 条)")
        raw_list = cached
    else:
        raw_list = _fetch_raw(area, period_range)
        _save_cache(area, period_label, raw_list)

    articles, skipped = [], 0
    for raw in raw_list:
        art = _parse_article(raw)
        if art:
            articles.append(art)
        else:
            skipped += 1

    print(f"  [解析 / Parsed] {area} | {period_label}: "
          f"{len(articles)} 篇有效，{skipped} 篇无关键词跳过")
    return articles


def fetch_all() -> dict[str, dict[str, list[Article]]]:
    """
    获取所有方向 × 时间段的数据
    Fetch data for all areas × time periods
    Returns { area: { period_label: [Article, ...] } }
    """
    all_data: dict = {}
    for area in AREA_KEYWORD_IDS:
        all_data[area] = {}
        for period_label in TIME_PERIODS:
            print(f"\n── {area} | {period_label} ──")
            all_data[area][period_label] = fetch_articles(area, period_label)
    return all_data
