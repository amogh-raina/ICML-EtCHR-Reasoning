#!/usr/bin/env python3
"""
Parser for ECHR case HTML files.
Extracts: case number, case name, date, section, facts, and court assessments
for Articles 3, 9, 10, and 11 of the Convention.

Court assessments are split by subsection (Admissibility / Merits), with
citations and paragraph cross-references extracted per subsection.
"""

import os
import json
import re
from html.parser import HTMLParser

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
HTML_DIRS = [
    os.path.join(DATA_DIR, 'html'),
    os.path.join(DATA_DIR, 'art3', 'html'),
]
OUTPUT_PATH = os.path.join(DATA_DIR, 'parsed_cases_all(3,9,10,11).json')


# ---------------------------------------------------------------------------
# Step 1: HTML -> flat list of text nodes
# ---------------------------------------------------------------------------

class HTMLTextExtractor(HTMLParser):
    """Strip HTML to a flat list of text nodes, skipping <style> blocks."""

    def __init__(self):
        super().__init__()
        self.in_style = False
        self.texts = []
        self.current = ""

    def handle_starttag(self, tag, attrs):
        if tag == 'style':
            self.in_style = True
        # Block-level tags flush the current buffer
        if tag in ('br', 'p', 'div', 'tr', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'table'):
            self._flush()

    def handle_endtag(self, tag):
        if tag == 'style':
            self.in_style = False
        if tag in ('p', 'div', 'tr', 'td', 'th', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'table'):
            self._flush()

    def handle_data(self, data):
        if not self.in_style:
            self.current += data

    def _flush(self):
        t = self.current.strip()
        if t:
            self.texts.append(t)
        self.current = ""


def extract_texts(html_path):
    parser = HTMLTextExtractor()
    with open(html_path, 'r', encoding='utf-8') as f:
        parser.feed(f.read())
    return parser.texts


# ---------------------------------------------------------------------------
# Step 2: Header extraction
# ---------------------------------------------------------------------------

def extract_header(texts):
    """Return (section, case_name, app_no, date) from the first ~40 nodes."""
    section = texts[0].strip() if texts else None
    case_name = None
    app_no = None
    date = None

    for t in texts[:40]:
        s = t.strip()

        if s.upper().startswith('CASE OF') and case_name is None:
            case_name = s

        if s.startswith('(Application') and app_no is None:
            app_no = s

        if re.match(r'^\d{1,2}\s+\w+\s+\d{4}$', s) and date is None:
            date = s

    return section, case_name, app_no, date


# ---------------------------------------------------------------------------
# Step 3: Major section boundaries (THE FACTS / THE LAW / FOR THESE REASONS)
# ---------------------------------------------------------------------------

def find_section_boundaries(texts):
    """
    Returns (facts_idx, law_idx, reasons_idx).
    Uses the *last* standalone occurrence of each marker, working backwards
    from FOR THESE REASONS so that Table-of-Contents entries in Grand Chamber
    cases are skipped.
    """
    reasons_idx = None
    for i in range(len(texts) - 1, -1, -1):
        if texts[i].strip().upper().startswith('FOR THESE REASONS'):
            reasons_idx = i
            break

    law_idx = None
    upper_bound = reasons_idx if reasons_idx is not None else len(texts)
    for i in range(upper_bound - 1, -1, -1):
        if texts[i].strip() == 'THE LAW':
            law_idx = i
            break

    facts_idx = None
    upper_bound = law_idx if law_idx is not None else len(texts)
    for i in range(upper_bound - 1, -1, -1):
        if texts[i].strip() == 'THE FACTS':
            facts_idx = i
            break

    return facts_idx, law_idx, reasons_idx


# ---------------------------------------------------------------------------
# Step 4: Extract THE FACTS paragraphs
# ---------------------------------------------------------------------------

def extract_facts(texts, facts_idx, law_idx):
    if facts_idx is None or law_idx is None:
        return []
    return [t.strip() for t in texts[facts_idx + 1 : law_idx] if t.strip()]


# ---------------------------------------------------------------------------
# Step 5: Identify article sections inside THE LAW
# ---------------------------------------------------------------------------

def is_article_section_header(text):
    """
    True if *text* opens a new article-level section inside THE LAW.
    Used to delimit spans — catches both target-article sections and
    non-target sections (Article 41, Other violations, etc.) so that
    target-article spans are correctly bounded.

    Patterns matched:
        ALLEGED VIOLATION OF ARTICLE 10 ...
        COMPLAINTS UNDER ARTICLE 11 ...
        APPLICATION OF ARTICLE 41 OF THE CONVENTION
        OTHER ALLEGED VIOLATIONS OF THE CONVENTION
        VI. COMPLAINTS UNDER ARTICLE 10 ...
        SUBSTANTIVE LIMB OF ARTICLE 3 OF THE CONVENTION
        PROCEDURAL LIMB OF ARTICLE 3 OF THE CONVENTION
    """
    upper = text.strip().upper()
    # Strip leading Roman-numeral prefix (e.g. "XVII. ")
    upper = re.sub(r'^[IVXLC]+\.\s*', '', upper)
    return bool(re.match(
        r'(ALLEGED|COMPLAINTS?|APPLICATION\s+OF\s+ARTICLE|OTHER\s+ALLEGED|'
        r'SUBSTANTIVE\s+LIMB\s+OF|PROCEDURAL\s+LIMB\s+OF)\s',
        upper,
    ))


_TARGET_ARTICLES = {3, 9, 10, 11}


def is_target_article(text):
    """
    True if the section header is *primarily* about Article 3, 9, 10, or 11.
    "Primarily" means the first article number after the VIOLATION/COMPLAINT
    keyword is one of the targets.  This avoids false positives like
    "ARTICLE 18 ... TAKEN IN CONJUNCTION WITH ARTICLE 10".
    """
    upper = text.strip().upper()
    match = re.search(
        r'(?:VIOLATION|COMPLAINTS?)\s+(?:OF|UNDER|IN\s+VIOLATION\s+OF)\s+ARTICLES?\s+(\d+)',
        upper,
    )
    if not match:
        # Fallback: try without the preposition
        match = re.search(r'ARTICLES?\s+(\d+)', upper)
    if match:
        return int(match.group(1)) in _TARGET_ARTICLES
    return False


def get_target_articles(text):
    """Return sorted list of target article numbers (3/9/10/11) mentioned.

    Handles patterns like:
      "ARTICLE 10"
      "ARTICLES 10 AND 11"
      "ARTICLES 3, 9, 10 AND 11"
    by greedily extracting all digit tokens that follow the ARTICLE keyword.
    """
    upper = text.strip().upper()
    nums = set()
    for m in re.finditer(r'ARTICLES?\s+([\d\s,ANOD]+)', upper):
        for num_m in re.finditer(r'\d+', m.group(1)):
            nums.add(int(num_m.group()))
    return sorted(nums & _TARGET_ARTICLES)


# ---------------------------------------------------------------------------
# Step 6: Find "The Court's assessment" inside an article section
# ---------------------------------------------------------------------------

_ASSESSMENT_RE = re.compile(
    r"(?:court|chamber|grand\s+chamber)['\u2019]s\s+assessment",
    re.IGNORECASE,
)


def is_assessment_header(text):
    """
    True if this text node is a Court's assessment *header*, not a paragraph
    that happens to mention "court's assessment" in passing.
    The phrase must appear within the first ~80 characters (real headers are
    short or start with the phrase; body paragraphs bury it deep in text).
    """
    prefix = text.strip()[:80]
    return bool(_ASSESSMENT_RE.search(prefix))


# Patterns that signal we have left the Court's assessment and entered a
# party-submission block.  Only these hard-stop the collection loop.
# "Merits" / "Admissibility" labels are NOT stops — they are skipped over so
# that cases where the merits section has no separate "Court's assessment"
# header still get collected.
_SUBMISSION_RE = re.compile(
    r"""^
    (?:the\s+)?
    (?:applicant|government|parties|third[\s\-]party|intervener)
    """,
    re.IGNORECASE | re.VERBOSE,
)


def is_submission_boundary(text):
    """
    True if *text* is a standalone party-submission header
    ("The applicant's submissions", "The Government", "The parties'
    submissions", etc.) — the only case where we stop collecting the
    current Court's assessment paragraphs.

    Numbered body paragraphs, lettered/Greek sub-headings inside the
    assessment, and structural labels like "Merits" / "Admissibility"
    are all excluded and will be skipped or collected normally.
    """
    t = text.strip()
    if not t:
        return False
    # Numbered paragraphs — always body content
    if re.match(r'^\d+[\.\s]', t):
        return False
    # Sub-headings inside the assessment: (a), (b), (i), (α), (β) …
    if re.match(r'^\([a-zA-Z\u03b1-\u03c9ivxIVX\d]+\)', t):
        return False
    return bool(_SUBMISSION_RE.match(t))


_SUBSECTION_RE = re.compile(
    r'^(admissibility|merits)\b', re.IGNORECASE
)


_SUBSECTION_PREFIX_RE = re.compile(
    r'^(?:[A-Z]\.?\s+|[IVXLC]+\.?\s+|\d+\.?\s+)*',
    re.IGNORECASE,
)


def _strip_subsection_prefix(text):
    """Strip leading letter/Roman/number prefixes like 'B.   ' or 'A.'."""
    return _SUBSECTION_PREFIX_RE.sub('', text.strip()).strip()


def _find_subsection_boundaries(law_texts, span_start, span_end):
    """
    Locate 'Admissibility' and 'Merits' subsection boundaries within an
    article span.  Returns (admiss_start, merits_start) where each is the
    index of the subsection header or None if not found.
    Handles prefixed labels like 'A.  Admissibility', 'B.   Merits'.
    """
    admiss_start = None
    merits_start = None
    for i in range(span_start, span_end):
        t = law_texts[i].strip()
        if not t:
            continue
        stripped = _strip_subsection_prefix(t).lower()
        if stripped.startswith('admissibility') and admiss_start is None:
            admiss_start = i
        elif stripped.startswith('merits') and merits_start is None:
            merits_start = i
    return admiss_start, merits_start


def _collect_assessment_paragraphs(law_texts, region_start, region_end):
    """
    Collect Court's assessment paragraphs from a region (Admissibility or
    Merits subsection).  Looks for 'Court's assessment' headers; if none
    found, collects all numbered paragraphs directly (skipping party
    submissions).
    """
    assess_starts = [
        i for i in range(region_start, region_end)
        if is_assessment_header(law_texts[i])
    ]

    paragraphs = []

    if assess_starts:
        for a_pos, a_idx in enumerate(assess_starts):
            a_end = (assess_starts[a_pos + 1]
                     if a_pos + 1 < len(assess_starts) else region_end)
            for j in range(a_idx + 1, a_end):
                t = law_texts[j].strip()
                if not t:
                    continue
                if is_submission_boundary(t):
                    break
                if _SUBSECTION_RE.match(t) and len(t) < 60:
                    continue
                paragraphs.append(t)
    else:
        # No explicit "Court's assessment" header — collect paragraphs
        # after party submissions end, or all numbered paragraphs if there
        # are no party headers at all.
        has_parties = any(
            is_submission_boundary(law_texts[i].strip())
            for i in range(region_start, region_end)
        )
        collecting = not has_parties
        for j in range(region_start + 1, region_end):
            t = law_texts[j].strip()
            if not t:
                continue
            if is_submission_boundary(t):
                collecting = False
                continue
            if not collecting:
                # After a submission boundary, the next numbered paragraph
                # without a submission header means we've entered the
                # Court's own text (many cases omit the header).
                if re.match(r'^\d+[\.\s]', t) and not is_submission_boundary(t):
                    collecting = True
            if collecting:
                if _SUBSECTION_RE.match(t) and len(t) < 60:
                    continue
                paragraphs.append(t)

    return paragraphs


def _build_subsection(subsection_title, paragraphs):
    """Build a subsection dict with ref_paragraphs and citations extracted
    from its court assessment paragraphs."""
    return {
        'subsection_title': subsection_title,
        'paragraphs': paragraphs,
        'ref_paragraphs': extract_ref_paragraphs(paragraphs),
        'citations': extract_citations(paragraphs),
    }


def extract_court_assessments(texts, law_idx, reasons_idx):
    """
    Walk THE LAW section, find article sections for Art 3/9/10/11,
    then extract court assessments split by Admissibility / Merits
    subsections within each article section.
    """
    law_texts = texts[law_idx : reasons_idx]

    # Locate all article-section headers (any article, not just targets)
    header_indices = [
        i for i, t in enumerate(law_texts) if is_article_section_header(t)
    ]

    # Build (start, end, title) spans for target articles only
    target_spans = []
    for pos, idx in enumerate(header_indices):
        if is_target_article(law_texts[idx]):
            end = header_indices[pos + 1] if pos + 1 < len(header_indices) else len(law_texts)
            target_spans.append((idx, end, law_texts[idx].strip()))

    results = []
    for span_start, span_end, section_title in target_spans:
        articles = get_target_articles(section_title)

        admiss_start, merits_start = _find_subsection_boundaries(
            law_texts, span_start, span_end
        )

        subsections = []

        if admiss_start is not None:
            # Admissibility region ends at Merits or span_end
            admiss_end = merits_start if merits_start is not None else span_end
            admiss_paras = _collect_assessment_paragraphs(
                law_texts, admiss_start, admiss_end
            )
            subsections.append(
                _build_subsection('Admissibility', admiss_paras)
            )

        if merits_start is not None:
            merits_paras = _collect_assessment_paragraphs(
                law_texts, merits_start, span_end
            )
            subsections.append(
                _build_subsection('Merits', merits_paras)
            )

        if not subsections:
            # No Admissibility/Merits headers — treat entire span as a
            # single block (some simpler cases lack these subsections).
            all_paras = _collect_assessment_paragraphs(
                law_texts, span_start, span_end
            )
            subsections.append(
                _build_subsection('General', all_paras)
            )

        results.append({
            'section_title': section_title,
            'articles': articles,
            'subsections': subsections,
        })

    return results


# ---------------------------------------------------------------------------
# Step 7: Paragraph cross-references and ECHR citations from facts
# ---------------------------------------------------------------------------

# Hyphen / dash variants that can appear in ECHR text between range bounds:
#   -  U+002D HYPHEN-MINUS
#   ‐  U+2010 HYPHEN
#   ‑  U+2011 NON-BREAKING HYPHEN   (appears in e.g. "23‑24")
#   ‒  U+2012 FIGURE DASH
#   –  U+2013 EN DASH
#   —  U+2014 EM DASH
#   ―  U+2015 HORIZONTAL BAR
#   −  U+2212 MINUS SIGN
_DASH_CLASS = r'[-\u2010\u2011\u2012\u2013\u2014\u2015\u2212]'
_DASH_CHARS = '\u2010\u2011\u2012\u2013\u2014\u2015\u2212'


# Matches "paragraph(s) X, Y and Z-W above/below".
# Requiring "above" or "below" avoids false positives from citation paragraph
# numbers written as "§§ 16-18" in cited cases.  Supports both hyphen and
# "to" as range separators (e.g. "paragraphs 23 to 25 above").
_PARA_REF_RE = re.compile(
    r'\bparagraphs?\s+'
    r'((?:\d+(?:\s*(?:' + _DASH_CLASS + r'|\bto\b)\s*\d+)?'
    r'\s*(?:,\s*|\s+and\s+))*'
    r'\d+(?:\s*(?:' + _DASH_CLASS + r'|\bto\b)\s*\d+)?)'
    r'\s+(?:above|below)',
    re.IGNORECASE,
)


def _expand_para_list(s):
    """'47, 51 and 54-65' → sorted list of ints [47, 51, 54, ..., 65]."""
    nums = set()
    s = re.sub(r'\band\b', ',', s, flags=re.IGNORECASE)
    s = re.sub(r'\s+to\s+', '-', s, flags=re.IGNORECASE)
    # Normalise every dash variant to a plain hyphen-minus.
    for ch in _DASH_CHARS:
        s = s.replace(ch, '-')
    s = s.replace('\xa0', ' ')
    # Collapse whitespace around hyphens so "23 - 25" becomes "23-25".
    s = re.sub(r'\s*-\s*', '-', s)
    for token in re.split(r'[,\s]+', s):
        token = token.strip()
        if not token:
            continue
        m = re.match(r'^(\d+)-(\d+)$', token)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a < b < a + 500:   # sanity-check range size
                nums.update(range(a, b + 1))
        elif re.match(r'^\d+$', token):
            nums.add(int(token))
    return sorted(nums)


def extract_ref_paragraphs(facts):
    """Return sorted, deduplicated list of paragraph numbers cross-referenced
    in the facts section (patterns like 'see paragraphs 47 and 51 above')."""
    nums = set()
    for para in facts:
        text = para.replace('\xa0', ' ')
        for m in _PARA_REF_RE.finditer(text):
            nums.update(_expand_para_list(m.group(1)))
    return sorted(nums)


# Matches "no./nos. <block-of-app-numbers>".
# App-number format: 4-6 digits / 2 digits (e.g. 28396/95, 1257/21).
_NOS_BLOCK_RE = re.compile(
    r'\bnos?\.\s*'
    r'((?:\d{4,6}/\d{2}\s*(?:,\s*|\s+and\s+))*\d{4,6}/\d{2})',
    re.IGNORECASE,
)
_APP_NO_RE = re.compile(r'\d{4,6}/\d{2}')

_LEAD_NOISE_RE = re.compile(
    r'^(?:see\b|also\b|in\b|and\b|or\b|cf\.?,?|e\.g\.?,?|i\.e\.?,?|'
    r'notably,?|particularly,?|namely\b|compare\b|'
    r'cited\s+in\b|'
    r'it\s+referred\s+to\b|the\s+court\s+referred\s+to\b|'
    r'the\s+(?:\w+\s+)?applicants?\s+in\b|'
    r'for\s+(?:more\s+)?(?:information|details?|examples?|reference|background|'
    r'illustrative\s+purposes?|a\s+\w+(?:\s+\w+)?)[^,;]*?[,;]\s*(?:see\s+)?|'
    r'among\s+(?:many\s+)?(?:other\s+)?(?:authorities,?\s*)?)\s*',
    re.IGNORECASE,
)


def _extract_case_title(full_text, nos_start):
    """Look backwards from nos_start position to find the case title."""
    window = full_text[max(0, nos_start - 250): nos_start]

    # Strip trailing preamble that precedes "no./nos.":
    #   "Case Name ([GC], "  →  "Case Name"
    #   "Case Name [GC], "   →  "Case Name"
    #   "Case Name ("        →  "Case Name"
    #   "Case Name, "        →  "Case Name"
    window = re.sub(r'\s*\(\s*\[?GC\]?\s*,?\s*$', '', window)
    window = re.sub(r'\s*\[GC\]\s*,?\s*$', '', window)
    window = re.sub(r'\s*\(\s*$', '', window)
    window = window.rstrip(', ').strip()

    # Find the start of the title: rightmost hard boundary in the window.
    # Note: '. ' is NOT used — it appears inside "v. Belgium" and would split titles.
    start = 0
    for marker in ['(', ';', '\n', '\u2013 ', '– ']:
        pos = window.rfind(marker)
        if pos != -1 and pos + len(marker) > start:
            start = pos + len(marker)

    title = window[start:].strip()

    # If title starts with "no. XXXXX/XX ... ) ," it came from inside a
    # previous citation's parenthetical — strip it to get the real case name.
    title = re.sub(
        r'^nos?\.\s*\d{4,6}/\d{2}.*?\)\s*,?\s*',
        '', title, flags=re.IGNORECASE,
    )

    title = _LEAD_NOISE_RE.sub('', title).strip().strip(',').strip()

    # Handle "... noise ..., see Case Name" — take everything after last "see"
    inner_see = re.search(r'(?:,\s*|\bsee\s+)(?:also\s+)?([A-Z\u00C0-\u00FF].+)', title)
    if inner_see and 'v.' in inner_see.group(1):
        title = inner_see.group(1).strip()

    title = title.strip().strip(',').strip()

    if 'v.' in title and 3 < len(title) < 200:
        return title
    return None


def extract_citations(facts):
    """
    Extract ECHR application-number citations from facts paragraphs.

    Returns a flat, deduplicated list of application number strings
    (e.g. ['28114/95', '12365/03']), in order of first appearance.
    """
    full_text = ' '.join(p.replace('\xa0', ' ') for p in facts)
    seen = set()
    result = []

    for m in _NOS_BLOCK_RE.finditer(full_text):
        for app_no in _APP_NO_RE.findall(m.group(1)):
            if app_no not in seen:
                seen.add(app_no)
                result.append(app_no)

    return result


# ---------------------------------------------------------------------------
# Metadata enrichment: violation and conclusion fields
# ---------------------------------------------------------------------------

JSON_DIR = os.path.join(DATA_DIR, 'json')

_TARGET = _TARGET_ARTICLES   # {3, 9, 10, 11}


def parse_violation_field(raw):
    """
    Filter the semicolon-separated violation codes to those that belong to
    Article 9, 10, or 11.  Keeps both base codes ('10') and sub-codes ('10-1').
    """
    if not raw:
        return []
    codes = [c.strip() for c in raw.split(';') if c.strip()]
    result = []
    for code in codes:
        base = code.split('-')[0].split('+')[0].strip()
        try:
            if int(base) in _TARGET:
                result.append(code)
        except ValueError:
            pass
    return result


def parse_conclusion_field(raw):
    """
    Extract conclusion clauses that discuss Article 9, 10, or 11.

    The conclusion field is a single string of semicolon-separated chunks.
    Each top-level clause begins with a keyword like "Violation of Article X",
    "No violation of Article X", or "Remainder inadmissible".  Sub-clauses
    (in parentheses) belong to the preceding top-level clause.

    Strategy: split by ';', walk the chunks, detect clause starts, and keep
    only those clauses that are about a target article.
    """
    if not raw:
        return []

    _CLAUSE_START_RE = re.compile(
        r'^(violation|no\s+violation|remainder|pecuniary|non-pecuniary|'
        r'just\s+satisfaction|preliminary|strike|relinquishment|friendly|'
        r'not\s+necessary)',
        re.IGNORECASE,
    )
    _ARTICLE_NUM_RE = re.compile(r'article\s+(\d+)', re.IGNORECASE)

    chunks = [c.strip() for c in raw.split(';')]

    clauses = []      # list of (is_target, list_of_chunks)
    current_chunks = []
    current_target = False

    for chunk in chunks:
        if not chunk:
            continue
        if _CLAUSE_START_RE.match(chunk):
            # Save previous clause
            if current_chunks:
                clauses.append((current_target, current_chunks))
            # Start new clause
            nums = {int(m.group(1)) for m in _ARTICLE_NUM_RE.finditer(chunk)}
            current_target = bool(nums & _TARGET)
            current_chunks = [chunk]
        else:
            current_chunks.append(chunk)

    if current_chunks:
        clauses.append((current_target, current_chunks))

    result = []
    for is_target, parts in clauses:
        if is_target:
            result.append(';'.join(parts))
    return result


def load_metadata(item_id):
    """Return (violation_list, conclusion_list) for item_id, or ([], [])."""
    json_path = os.path.join(JSON_DIR, f'{item_id}.json')
    if not os.path.exists(json_path):
        return [], []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        cols = data['results'][0]['columns']
        violation = parse_violation_field(cols.get('violation', ''))
        conclusion = parse_conclusion_field(cols.get('conclusion', ''))
        return violation, conclusion
    except Exception:
        return [], []


# ---------------------------------------------------------------------------
# Main: parse one case
# ---------------------------------------------------------------------------

def parse_case(html_path):
    texts = extract_texts(html_path)

    if not texts:
        return {
            'case_number': None,
            'case_name': None,
            'date': None,
            'section': None,
            'facts': [],
            'court_assessments': [],
            'empty_file': True,
        }

    section, case_name, app_no, date = extract_header(texts)
    facts_idx, law_idx, reasons_idx = find_section_boundaries(texts)

    facts = extract_facts(texts, facts_idx, law_idx)
    court_assessments = extract_court_assessments(texts, law_idx, reasons_idx) \
        if law_idx is not None and reasons_idx is not None else []

    return {
        'case_number': app_no,
        'case_name': case_name,
        'date': date,
        'section': section,
        'facts': facts,
        'court_assessments': court_assessments,
    }


# ---------------------------------------------------------------------------
# Run over all files
# ---------------------------------------------------------------------------

def main():
    results = {}  # keyed by item_id to deduplicate across directories
    for html_dir in HTML_DIRS:
        if not os.path.isdir(html_dir):
            continue
        print(f'\n--- Processing {html_dir} ---')
        for fname in sorted(os.listdir(html_dir)):
            if not fname.endswith('.html'):
                continue
            item_id = fname.replace('.html', '')
            if item_id in results:
                continue  # already processed from another directory
            fpath = os.path.join(html_dir, fname)
            print(f'Parsing {item_id} ...', end=' ')
            try:
                parsed = parse_case(fpath)
                violation, conclusion = load_metadata(item_id)
                n_facts = len(parsed['facts'])
                n_assess = sum(
                    len(p)
                    for ca in parsed['court_assessments']
                    for sub in ca['subsections']
                    for p in [sub['paragraphs']]
                )
                result = {
                    'item_id': item_id,
                    'case_number': parsed['case_number'],
                    'case_name': parsed['case_name'],
                    'date': parsed['date'],
                    'section': parsed['section'],
                    'facts': parsed['facts'],
                    'court_assessments': parsed['court_assessments'],
                    'violation': violation,
                    'conclusion': conclusion,
                }
                if parsed.get('empty_file'):
                    result['empty_file'] = True
                print(f'OK  (facts: {n_facts}, assessment paragraphs: {n_assess})')
                results[item_id] = result
            except Exception as e:
                print(f'ERROR: {e}')
                results[item_id] = {'item_id': item_id, 'error': str(e)}

    output = sorted(results.values(), key=lambda r: r['item_id'])
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f'\nWrote {len(output)} cases -> {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
