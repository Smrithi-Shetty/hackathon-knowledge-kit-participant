"""
Knowledge Hackathon submission - Version 1

Approach
--------
1. Keep every snapshot instead of overwriting old pages.
2. Split pages into searchable sections/blocks.
3. Select the correct snapshot for current/as-of questions.
4. Rank blocks using lexical relevance, qualifiers and source authority.
5. Prefer authoritative rate/product/contact pages over stale blogs,
   newsroom pages and calculators.
6. Abstain when evidence is too weak.
7. Track likely contradictions across snapshots.

No external APIs or third-party packages are required.
"""

from __future__ import annotations

import math
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Import the official harness contract.
# This keeps the submission runnable regardless of the current working folder.
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve()

for parent in HERE.parents:
    contract_path = parent / "harness" / "contract.py"
    if contract_path.exists():
        sys.path.insert(0, str(parent / "harness"))
        break

from contract import Answer, Conflict, Snapshot, Source  # noqa: E402


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.$%'-][a-z0-9]+)*", re.I)

NUMBER_RE = re.compile(
    r"""
    (?:
        \$\s?\d[\d,]*(?:\.\d+)?
        |
        \d+(?:\.\d+)?\s?%
        |
        \d[\d,]*(?:\.\d+)?
    )
    """,
    re.X | re.I,
)

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "can", "could",
    "do", "does", "for", "from", "give", "have", "how", "i",
    "in", "is", "it", "know", "me", "my", "of", "on", "please",
    "provide", "rate", "rates", "tell", "the", "their", "there",
    "they", "to", "today", "us", "what", "whats", "which", "with",
    "would", "you", "your",
}


SYNONYMS = {
    "cd": "certificate",
    "cds": "certificate",
    "certificates": "certificate",
    "autos": "auto",
    "automobile": "auto",
    "vehicle": "auto",
    "vehicles": "auto",
    "mortgages": "mortgage",
    "branches": "branch",
    "locations": "branch",
    "apy": "apy",
    "apr": "apr",
    "fees": "fee",
    "cards": "card",
    "checking": "checking",
    "savings": "savings",
}


def normalize_token(token: str) -> str:
    token = token.lower().strip()
    return SYNONYMS.get(token, token)


def tokens(text: str) -> list[str]:
    values = []

    for token in TOKEN_RE.findall(text.lower()):
        token = normalize_token(token)

        if token and token not in STOPWORDS:
            values.append(token)

    return values


def normalized_text(text: str) -> str:
    return " ".join(tokens(text))


def extract_numbers(text: str) -> set[str]:
    result = set()

    for value in NUMBER_RE.findall(text):
        cleaned = re.sub(r"\s+", "", value.lower())
        cleaned = cleaned.replace(",", "")
        result.add(cleaned)

    return result


def clean_markdown(text: str) -> str:
    """
    Light cleanup only.

    We intentionally DO NOT aggressively strip Markdown because:
    - URLs are valuable evidence.
    - table rows are valuable evidence.
    - headings help establish local context.
    """

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove most image syntax while keeping alternate text.
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)

    # Turn ordinary links into "label URL" so both remain searchable.
    text = re.sub(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        r"\1 \2",
        text,
    )

    return text


def iso_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None

    value = value[:10]

    if not DATE_RE.match(value):
        return None

    try:
        year, month, day = map(int, value.split("-"))
        return date(year, month, day)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------

class Block:
    def __init__(
        self,
        snapshot_id,
        snapshot_date,
        url,
        title,
        text,
        normalized,
        token_counts,
        numbers,
        authority,
    ):
        self.snapshot_id = snapshot_id
        self.snapshot_date = snapshot_date
        self.url = url
        self.title = title
        self.text = text
        self.normalized = normalized
        self.token_counts = token_counts
        self.numbers = numbers
        self.authority = authority


class SnapshotIndex:
    def __init__(
        self,
        snapshot_id,
        taken_on,
        blocks,
    ):
        self.snapshot_id = snapshot_id
        self.taken_on = taken_on
        self.blocks = blocks


# ---------------------------------------------------------------------------
# Knowledge system
# ---------------------------------------------------------------------------

class KnowledgeSystem:
    def __init__(self):
        self.snapshots: list[SnapshotIndex] = []

        # all historical versions by canonical URL
        self.url_history: dict[str, list[Block]] = defaultdict(list)

        # calculated on demand
        self._conflicts_cache: Optional[list[Conflict]] = None


    # -----------------------------------------------------------------------
    # Ingestion
    # -----------------------------------------------------------------------

    def ingest(self, snapshot: Snapshot) -> None:
        snap_date = iso_date(snapshot.taken_on)

        if snap_date is None:
            return

        indexed_blocks: list[Block] = []

        for page in snapshot.pages:
            authority = self._authority(page.url, page.title)

            for block_text in self._split_page(page.title, page.markdown):
                block_text = block_text.strip()

                if not block_text:
                    continue

                block_tokens = tokens(block_text)

                if not block_tokens:
                    continue

                block = Block(
                    snapshot_id=snapshot.snapshot_id,
                    snapshot_date=snap_date,
                    url=page.url,
                    title=page.title or "",
                    text=block_text,
                    normalized=normalized_text(block_text),
                    token_counts=Counter(block_tokens),
                    numbers=extract_numbers(block_text),
                    authority=authority,
                )

                indexed_blocks.append(block)
                self.url_history[self._canonical_url(page.url)].append(block)

        self.snapshots.append(
            SnapshotIndex(
                snapshot_id=snapshot.snapshot_id,
                taken_on=snap_date,
                blocks=indexed_blocks,
            )
        )

        self.snapshots.sort(key=lambda s: s.taken_on)
        self._conflicts_cache = None


    def _split_page(self, title: str, markdown: str) -> list[str]:
        """
        Split pages into moderately-sized evidence blocks.

        Important design choice:
        Keep adjacent Markdown/table rows together where possible. Many rates are
        expressed as table rows and become meaningless if every line is stored
        individually.
        """

        text = clean_markdown(markdown)

        lines = text.splitlines()

        blocks: list[str] = []
        current: list[str] = []

        heading_context = title.strip()

        def flush():
            nonlocal current

            if not current:
                return

            body = "\n".join(current).strip()

            if body:
                prefix = heading_context.strip()

                if prefix and prefix.lower() not in body[:150].lower():
                    body = f"{prefix}\n{body}"

                blocks.append(body)

            current = []

        for raw_line in lines:
            line = raw_line.strip()

            if not line:
                flush()
                continue

            # Heading
            if line.startswith("#"):
                flush()

                heading = re.sub(r"^#+\s*", "", line).strip()

                if heading:
                    heading_context = (
                        f"{title} | {heading}"
                        if title and heading.lower() != title.lower()
                        else heading
                    )

                continue

            current.append(line)

            # Avoid extremely large blocks.
            joined_len = sum(len(x) for x in current)

            if joined_len >= 1400:
                flush()

        flush()

        # Also include the title as retrieval context.
        if title:
            blocks.append(title)

        return blocks


    # -----------------------------------------------------------------------
    # Snapshot selection
    # -----------------------------------------------------------------------

    def _select_snapshot(
        self,
        as_of: Optional[str],
    ) -> Optional[SnapshotIndex]:

        if not self.snapshots:
            return None

        if as_of is None:
            return self.snapshots[-1]

        target = iso_date(as_of)

        if target is None:
            return self.snapshots[-1]

        eligible = [
            snapshot
            for snapshot in self.snapshots
            if snapshot.taken_on <= target
        ]

        if eligible:
            return eligible[-1]

        # There was no crawl yet at the requested historical date.
        return None


    # -----------------------------------------------------------------------
    # Source authority
    # -----------------------------------------------------------------------

    def _authority(self, url: str, title: str) -> float:
        """
        Heuristic authority ordering.

        Current official product/rate/contact pages should generally outrank
        calculators, news stories and old educational content.
        """

        u = (url or "").lower()
        t = (title or "").lower()

        score = 1.0

        # High-authority current informational areas.
        if "/rates" in u or u.endswith("/rates"):
            score += 7.0

        if "#rates" in u:
            score += 7.0

        if any(
            part in u
            for part in (
                "/personal/loans/",
                "/personal/savings",
                "/personal/checking",
                "/personal/credit-cards",
                "/business/credit-cards",
                "/business/cash-management",
                "/contact",
            )
        ):
            score += 5.0

        if "rate" in t:
            score += 3.0

        if any(
            word in t
            for word in (
                "certificate",
                "checking",
                "savings",
                "mortgage",
                "auto",
                "credit card",
                "contact",
            )
        ):
            score += 2.0

        # Lower-authority contexts for CURRENT operational values.
        if "/calculator" in u or "calculator" in u or "calculator" in t:
            score -= 4.0

        if "/newsroom/" in u or "newsroom" in t:
            score -= 3.0

        if "/learn/blog/" in u or "/blog/" in u:
            score -= 3.0

        if "/resources/" in u:
            score -= 1.0

        # Obvious stale/archival terminology.
        if any(
            marker in u or marker in t
            for marker in (
                "archive",
                "historical",
                "press-release",
            )
        ):
            score -= 2.0

        return max(score, 0.1)


    # -----------------------------------------------------------------------
    # Query understanding
    # -----------------------------------------------------------------------

    def _query_profile(self, question: str) -> dict:
        q_lower = question.lower()

        q_tokens = tokens(question)
        q_numbers = extract_numbers(question)

        profile = {
            "raw": question,
            "lower": q_lower,
            "tokens": q_tokens,
            "token_set": set(q_tokens),
            "numbers": q_numbers,
            "wants_url": any(
                phrase in q_lower
                for phrase in (
                    "what link",
                    "what url",
                    "url for",
                    "link should",
                    "website for",
                    "where can i find",
                )
            ),
            "wants_all": any(
                phrase in q_lower
                for phrase in (
                    "what are",
                    "all ",
                    "rates for",
                    "credit cards",
                    "certificate of accounts",
                    "certificate rates",
                )
            ),
        }

        # Semantic qualifiers.
        profile["metric_apr"] = "apr" in q_lower
        profile["metric_apy"] = "apy" in q_lower
        profile["jumbo"] = "jumbo" in q_lower
        profile["business"] = "business" in q_lower
        profile["personal"] = "personal" in q_lower

        return profile


    # -----------------------------------------------------------------------
    # Ranking
    # -----------------------------------------------------------------------

    def _idf(self, snapshot: SnapshotIndex) -> dict[str, float]:
        """
        Small in-memory IDF map for the selected snapshot.

        With ~700 pages this is easily fast enough after ingestion.
        """

        doc_freq = Counter()

        for block in snapshot.blocks:
            for term in block.token_counts:
                doc_freq[term] += 1

        total = max(len(snapshot.blocks), 1)

        return {
            term: math.log((total + 1) / (freq + 1)) + 1.0
            for term, freq in doc_freq.items()
        }


    def _score_block(
        self,
        block: Block,
        profile: dict,
        idf: dict[str, float],
    ) -> float:

        q_tokens = profile["tokens"]

        if not q_tokens:
            return 0.0

        score = 0.0

        # Weighted lexical overlap.
        for term in q_tokens:
            frequency = block.token_counts.get(term, 0)

            if frequency:
                score += (1.0 + math.log(frequency)) * idf.get(term, 1.0)

        block_lower = block.text.lower()
        url_lower = block.url.lower()
        title_lower = block.title.lower()

        # Exact phrase / important entity bonuses.
        important_phrases = (
            "auto loan",
            "personal loan",
            "mortgage",
            "certificate",
            "credit card",
            "share savings",
            "bright savings",
            "routing number",
            "lobby hours",
            "drive-up",
            "mobile wallet",
            "student loan",
            "home$tart",
            "hy-vee",
        )

        for phrase in important_phrases:
            if phrase in profile["lower"] and (
                phrase in block_lower
                or phrase in title_lower
                or phrase.replace(" ", "-") in url_lower
            ):
                score += 6.0

        # Numeric qualifiers are extremely important for terms, balances,
        # years and loan durations.
        if profile["numbers"]:
            matches = profile["numbers"] & block.numbers

            score += 5.0 * len(matches)

        # APR/APY separation.
        if profile["metric_apr"]:
            if "apr" in block_lower:
                score += 5.0

            if "apy" in block_lower and "apr" not in block_lower:
                score -= 4.0

        if profile["metric_apy"]:
            if "apy" in block_lower:
                score += 5.0

            if "apr" in block_lower and "apy" not in block_lower:
                score -= 4.0

        # Jumbo / standard qualifier.
        if profile["jumbo"]:
            if "jumbo" in block_lower:
                score += 5.0
            else:
                score -= 2.0

        # Business / personal qualifier.
        if profile["business"]:
            if "/business/" in url_lower or "business" in block_lower:
                score += 4.0
            if "/personal/" in url_lower:
                score -= 2.0

        if profile["personal"]:
            if "/personal/" in url_lower:
                score += 3.0
            if "/business/" in url_lower:
                score -= 2.0

        # URL questions strongly reward a relevant canonical URL.
        if profile["wants_url"]:
            if "https://" in block.text or "http://" in block.text:
                score += 3.0

        # Authority is important, but only after relevance.
        overlap_count = len(profile["token_set"] & set(block.token_counts))

        if overlap_count:
            score += min(block.authority, 10.0) * 0.70

        return score


    # -----------------------------------------------------------------------
    # Answer generation
    # -----------------------------------------------------------------------

    def answer(
        self,
        question: str,
        as_of: Optional[str] = None,
    ) -> Answer:

        snapshot = self._select_snapshot(as_of)

        if snapshot is None:
            return Answer(
                text="I don't know. The available website snapshots do not cover that date.",
                sources=[],
                confident=False,
            )

        profile = self._query_profile(question)
        idf = self._idf(snapshot)

        ranked: list[tuple[float, Block]] = []

        for block in snapshot.blocks:
            score = self._score_block(block, profile, idf)

            if score > 0:
                ranked.append((score, block))

        ranked.sort(
            key=lambda item: (
                item[0],
                item[1].authority,
                -len(item[1].text),
            ),
            reverse=True,
        )

        if not ranked:
            return Answer(
                text="I don't know. I couldn't find that information in the website snapshot.",
                sources=[],
                confident=False,
            )

        best_score, best = ranked[0]

        # Require at least some meaningful relevance.
        relevant_terms = (
            profile["token_set"] & set(best.token_counts)
        )

        if best_score < 4.0 or not relevant_terms:
            return Answer(
                text="I don't know. I couldn't find enough support for that answer in the website snapshot.",
                sources=[],
                confident=False,
            )

        answer_text = self._compose_answer(
            question=question,
            profile=profile,
            ranked=ranked,
        )

        if answer_text is None:
            return Answer(
                text="I don't know. The website doesn't clearly list that information.",
                sources=[],
                confident=False,
            )

        selected_blocks = self._select_sources(profile, ranked)

        sources = []

        seen_urls = set()

        for block in selected_blocks:
            canonical = self._canonical_url(block.url)

            if canonical in seen_urls:
                continue

            seen_urls.add(canonical)

            quote = self._short_quote(block.text)

            sources.append(
                Source(
                    url=block.url,
                    snapshot_id=block.snapshot_id,
                    quote=quote,
                )
            )

            if len(sources) >= 3:
                break

        return Answer(
            text=answer_text,
            sources=sources,
            confident=True,
        )


    def _compose_answer(
        self,
        question: str,
        profile: dict,
        ranked: list[tuple[float, Block]],
    ) -> Optional[str]:

        _, best = ranked[0]

        q = question.lower()

        # For URL/link questions return the relevant canonical URL when possible.
        if profile["wants_url"]:
            url = self._best_url_from_block(best)

            if url:
                return url

            if best.url:
                return best.url

        # Some questions explicitly request lists.
        # Combine only strongly-related top evidence blocks, avoiding number
        # dumping from unrelated pages.
        if profile["wants_all"]:
            selected = self._strong_related_blocks(profile, ranked)

            if selected:
                text = "\n".join(
                    self._compact_block(block.text)
                    for block in selected[:4]
                )

                return text[:1800].strip()

        # General fact question.
        compact = self._compact_block(best.text)

        if not compact:
            return None

        # Increase useful context for rate/table questions.
        if any(
            key in q
            for key in (
                "rate",
                "apr",
                "apy",
                "hours",
                "address",
                "routing",
                "how many",
                "when",
            )
        ):
            return compact[:1000]

        return compact[:850]


    def _strong_related_blocks(
        self,
        profile: dict,
        ranked: list[tuple[float, Block]],
    ) -> list[Block]:

        if not ranked:
            return []

        best_score = ranked[0][0]

        result = []

        for score, block in ranked:
            if score < max(best_score * 0.72, best_score - 8.0):
                break

            # Require meaningful overlap.
            overlap = profile["token_set"] & set(block.token_counts)

            if len(overlap) >= 1:
                result.append(block)

            if len(result) >= 5:
                break

        return result


    def _select_sources(
        self,
        profile: dict,
        ranked: list[tuple[float, Block]],
    ) -> list[Block]:

        if not ranked:
            return []

        best_score = ranked[0][0]

        result = []

        for score, block in ranked:
            if score < best_score * 0.65:
                break

            result.append(block)

            if len(result) >= 4:
                break

        return result


    def _best_url_from_block(self, block: Block) -> Optional[str]:
        urls = re.findall(
            r"https?://[^\s)\]>]+",
            block.text,
            flags=re.I,
        )

        if urls:
            dupaco = [
                url.rstrip(".,")
                for url in urls
                if "dupaco.com" in url.lower()
            ]

            if dupaco:
                return dupaco[0]

            return urls[0].rstrip(".,")

        return block.url or None



    def _compact_block(self, text: str) -> str:
        lines = []

        for line in text.splitlines():
            line = line.strip()

            if not line:
                continue

            # Remove large amounts ofolding.
            line = re.sub(r"^[-*]\s+", "", line)
            line = line.replace("\\", "")

            lines.append(line)

        text = " ".join(lines)

        text = re.sub(r"\s+", " ", text).strip()

        return text


    def _short_quote(self, text: str) -> Optional[str]:
        compact = self._compact_block(text)

        if not compact:
            return None

        return compact[:300]


    # -----------------------------------------------------------------------
    # Conflict detection
    # -----------------------------------------------------------------------

    def conflicts(self) -> list[Conflict]:
        """
        Conservative conflict detector.

        We intentionally avoid declaring every changed number a contradiction:
        rate tables contain legitimate terms/tiers, and historical changes alone
        are not necessarily contradictory.

        Version 1 focuses on same-URL facts that change through time while
        retaining enough context to identify likely real contradictions.
        """

        if self._conflicts_cache is not None:
            return self._conflicts_cache

        conflicts: list[Conflict] = []

        # Signature -> observations.
        observations = defaultdict(list)

        for canonical_url, history in self.url_history.items():
            for block in history:
                text_lower = block.text.lower()

                # Only numeric/rate-like blocks are useful here.
                values = sorted(
                    value
                    for value in block.numbers
                    if "%" in value or value.startswith("$")
                )

                if not values:
                    continue

                # Build a semantic signature while dropping numeric values.
                signature_tokens = [
                    token
                    for token in tokens(text_lower)
                    if not re.search(r"\d", token)
                ]

                signature_tokens = signature_tokens[:18]

                if len(signature_tokens) < 2:
                    continue

                signature = (
                    canonical_url,
                    tuple(signature_tokens),
                )

                observations[signature].append(
                    (block, tuple(values))
                )

        seen_topics = set()

        for _, items in observations.items():
            distinct_values = {
                values
                for _, values in items
            }

            if len(distinct_values) < 2:
                continue

            # Latest observation wins for current value.
            items.sort(
                key=lambda x: x[0].snapshot_date
            )

            latest_block, latest_values = items[-1]

            all_values = []

            source_list = []

            for block, values in items:
                for value in values:
                    if value not in all_values:
                        all_values.append(value)

                source_list.append(
                    Source(
                        url=block.url,
                        snapshot_id=block.snapshot_id,
                        quote=self._short_quote(block.text),
                    )
                )

            if len(all_values) < 2:
                continue

            topic = self._topic_from_block(latest_block)

            topic_key = topic.lower()

            if topic_key in seen_topics:
                continue

            seen_topics.add(topic_key)

            resolution = (
                ", ".join(latest_values)
                if latest_values
                else None
            )

            conflicts.append(
                Conflict(
                    topic=topic,
                    values=all_values[:8],
                    sources=source_list[-5:],
                    resolution=resolution,
                    reason=(
                        "The value changed across website snapshots; "
                        "the latest applicable snapshot is used for the current answer."
                    ),
                )
            )

            # Avoid uncontrolled output.
            if len(conflicts) >= 100:
                break

        self._conflicts_cache = conflicts
        return conflicts


    def _topic_from_block(self, block: Block) -> str:
        compact = self._compact_block(block.text)

        if len(compact) > 120:
            compact = compact[:117] + "..."

        if compact:
            return compact

        return block.title or block.url


    # -----------------------------------------------------------------------
    # URL helpers
    # -----------------------------------------------------------------------

    def _canonical_url(self, url: str) -> str:
        if not url:
            return ""

        url = url.strip().lower()

        # Ignore fragment/query variation for history grouping.
        url = url.split("#", 1)[0]
        url = url.split("?", 1)[0]

        return url.rstrip("/")


# ---------------------------------------------------------------------------
# Required entry point
# ---------------------------------------------------------------------------

def build():
    return KnowledgeSystem()