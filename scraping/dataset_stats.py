#!/usr/bin/env python3
"""
Statistics for the parsed ECHR cases dataset.
Run from the project root: python3 -m scraping.dataset_stats
"""

import json
import os
from collections import defaultdict
from datetime import datetime

DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'data',
    'parsed_cases_all(3,9,10,11).json'
)


def load_data():
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def word_count(text):
    return len(text.split())


def words_in_paragraphs(paragraphs):
    return sum(word_count(p) for p in paragraphs)


def parse_date(date_str):
    """Parse dates like '3 April 2025'."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, '%d %B %Y')
    except ValueError:
        return None


def print_separator(char='=', width=70):
    print(char * width)


def print_header(title):
    print()
    print_separator()
    print(f'  {title}')
    print_separator()


# ---------------------------------------------------------------------------
# 1. Case-level statistics by article
# ---------------------------------------------------------------------------

def case_level_stats(data):
    """Compute per-article statistics at the case level."""
    print_header('CASE-LEVEL STATISTICS BY ARTICLE')

    # Map each case to its articles (from court_assessments)
    article_cases = defaultdict(list)  # article -> list of case dicts
    multi_article_cases = []

    valid_cases = [c for c in data if not c.get('error') and not c.get('empty_file')]

    for case in valid_cases:
        case_articles = set()
        for ca in case.get('court_assessments', []):
            for a in ca.get('articles', []):
                case_articles.add(a)
                article_cases[a].append(case)
        if len(case_articles) > 1:
            multi_article_cases.append((case, case_articles))

    # Overall counts
    print(f'\nTotal entries in dataset: {len(data)}')
    print(f'Valid cases (non-empty, no errors): {len(valid_cases)}')
    print(f'Empty files: {sum(1 for c in data if c.get("empty_file"))}')
    print(f'Cases with court assessments: {sum(1 for c in valid_cases if c.get("court_assessments"))}')
    print(f'Cases referencing multiple target articles: {len(multi_article_cases)}')

    if multi_article_cases:
        print(f'  Examples:')
        for case, arts in multi_article_cases[:5]:
            print(f'    {case["item_id"]} ({case.get("case_name", "?")}): Articles {sorted(arts)}')

    # Per-article breakdown
    print(f'\n{"Article":<10} {"Cases":<8} {"Avg Facts Words":<18} {"Avg Facts Paras":<18} {"Date Range"}')
    print('-' * 85)

    for art in sorted(article_cases.keys()):
        cases = article_cases[art]
        # Deduplicate (a case can appear multiple times if it has multiple sections for same article)
        seen_ids = set()
        unique_cases = []
        for c in cases:
            if c['item_id'] not in seen_ids:
                seen_ids.add(c['item_id'])
                unique_cases.append(c)

        n = len(unique_cases)
        facts_words = [words_in_paragraphs(c.get('facts', [])) for c in unique_cases]
        facts_paras = [len(c.get('facts', [])) for c in unique_cases]
        avg_facts_words = sum(facts_words) / n if n else 0
        avg_facts_paras = sum(facts_paras) / n if n else 0

        dates = [parse_date(c.get('date')) for c in unique_cases]
        dates = [d for d in dates if d is not None]
        if dates:
            date_range = f'{min(dates).strftime("%d %b %Y")} - {max(dates).strftime("%d %b %Y")}'
        else:
            date_range = 'N/A'

        print(f'Art {art:<6} {n:<8} {avg_facts_words:<18.0f} {avg_facts_paras:<18.1f} {date_range}')

    return article_cases


# ---------------------------------------------------------------------------
# 2. Court assessment statistics by article and subsection
# ---------------------------------------------------------------------------

def assessment_stats(data):
    """Per-article, per-subsection statistics for court assessments."""
    print_header('COURT ASSESSMENT STATISTICS BY ARTICLE & SUBSECTION')

    # Collect: article -> subsection_title -> list of subsection dicts
    article_subsections = defaultdict(lambda: defaultdict(list))

    valid_cases = [c for c in data if not c.get('error') and not c.get('empty_file')]

    for case in valid_cases:
        for ca in case.get('court_assessments', []):
            for art in ca.get('articles', []):
                for sub in ca.get('subsections', []):
                    article_subsections[art][sub['subsection_title']].append(sub)

    # Print table
    print(f'\n{"Article":<10} {"Subsection":<16} {"Count":<7} '
          f'{"Avg Words":<12} {"Avg Paras":<12} '
          f'{"Avg Citations":<15} {"Avg Ref Paras":<14}')
    print('-' * 95)

    for art in sorted(article_subsections.keys()):
        for sub_title in ['Admissibility', 'Merits', 'General']:
            subs = article_subsections[art].get(sub_title, [])
            if not subs:
                continue
            n = len(subs)
            avg_words = sum(words_in_paragraphs(s['paragraphs']) for s in subs) / n
            avg_paras = sum(len(s['paragraphs']) for s in subs) / n
            avg_citations = sum(len(s.get('citations', [])) for s in subs) / n
            avg_refs = sum(len(s.get('ref_paragraphs', [])) for s in subs) / n

            print(f'Art {art:<6} {sub_title:<16} {n:<7} '
                  f'{avg_words:<12.0f} {avg_paras:<12.1f} '
                  f'{avg_citations:<15.1f} {avg_refs:<14.1f}')
        print()


# ---------------------------------------------------------------------------
# 3. Violation / Conclusion statistics
# ---------------------------------------------------------------------------

def violation_stats(data):
    """Statistics on violation outcomes."""
    print_header('VIOLATION & CONCLUSION STATISTICS')

    valid_cases = [c for c in data if not c.get('error') and not c.get('empty_file')]

    # Per-article violation counts
    article_violations = defaultdict(lambda: {'total': 0, 'with_violation': 0, 'codes': defaultdict(int)})

    for case in valid_cases:
        case_articles = set()
        for ca in case.get('court_assessments', []):
            for a in ca.get('articles', []):
                case_articles.add(a)

        violations = case.get('violation', [])
        for art in case_articles:
            article_violations[art]['total'] += 1
            if violations:
                # Check if this article has a violation
                art_violated = any(
                    v.split('-')[0].split('+')[0].strip() == str(art)
                    for v in violations
                )
                if art_violated:
                    article_violations[art]['with_violation'] += 1
                    for v in violations:
                        base = v.split('-')[0].split('+')[0].strip()
                        if base == str(art):
                            article_violations[art]['codes'][v] += 1

    print(f'\n{"Article":<10} {"Total Cases":<14} {"With Violation":<16} {"Rate":<10} {"Violation Codes"}')
    print('-' * 80)
    for art in sorted(article_violations.keys()):
        info = article_violations[art]
        rate = info['with_violation'] / info['total'] * 100 if info['total'] else 0
        codes = ', '.join(f'{k}({v})' for k, v in sorted(info['codes'].items()))
        print(f'Art {art:<6} {info["total"]:<14} {info["with_violation"]:<16} {rate:<10.1f}% {codes}')

    # Conclusion patterns
    print(f'\n  Conclusion patterns (sample):')
    for case in valid_cases[:5]:
        if case.get('conclusion'):
            print(f'    {case["item_id"]}: {case["conclusion"][0][:80]}...')


# ---------------------------------------------------------------------------
# 4. Citation network statistics
# ---------------------------------------------------------------------------

def citation_stats(data):
    """Statistics on cross-references and ECHR citations."""
    print_header('CITATION & CROSS-REFERENCE STATISTICS')

    valid_cases = [c for c in data if not c.get('error') and not c.get('empty_file')]

    # Gather citation counts per case
    all_citations = []
    all_refs = []
    citation_freq = defaultdict(int)  # app_no -> how many cases cite it

    for case in valid_cases:
        case_citations = set()
        case_refs = set()
        for ca in case.get('court_assessments', []):
            for sub in ca.get('subsections', []):
                for c in sub.get('citations', []):
                    case_citations.add(c)
                    citation_freq[c] += 1
                for r in sub.get('ref_paragraphs', []):
                    case_refs.add(r)
        all_citations.append(len(case_citations))
        all_refs.append(len(case_refs))

    cases_with_cites = sum(1 for c in all_citations if c > 0)
    cases_with_refs = sum(1 for r in all_refs if r > 0)

    print(f'\n  Cases with citations: {cases_with_cites}/{len(valid_cases)}')
    print(f'  Cases with ref_paragraphs: {cases_with_refs}/{len(valid_cases)}')
    print(f'\n  Citations per case:')
    print(f'    Mean: {sum(all_citations)/len(all_citations):.1f}')
    print(f'    Max:  {max(all_citations)}')
    print(f'    Min (among non-zero): {min(c for c in all_citations if c > 0) if cases_with_cites else "N/A"}')
    print(f'\n  Ref paragraphs per case:')
    print(f'    Mean: {sum(all_refs)/len(all_refs):.1f}')
    print(f'    Max:  {max(all_refs)}')

    # Most-cited application numbers
    print(f'\n  Most-cited application numbers (top 10):')
    for app_no, count in sorted(citation_freq.items(), key=lambda x: -x[1])[:10]:
        print(f'    {app_no}: cited in {count} cases')


# ---------------------------------------------------------------------------
# 5. Content completeness & quality check
# ---------------------------------------------------------------------------

def completeness_stats(data):
    """Check for potential data quality issues."""
    print_header('DATA COMPLETENESS & QUALITY')

    valid_cases = [c for c in data if not c.get('error') and not c.get('empty_file')]

    issues = {
        'no_case_name': [],
        'no_date': [],
        'no_facts': [],
        'no_court_assessments': [],
        'admissibility_empty': [],
        'merits_empty': [],
    }

    for case in valid_cases:
        cid = case['item_id']
        if not case.get('case_name'):
            issues['no_case_name'].append(cid)
        if not case.get('date'):
            issues['no_date'].append(cid)
        if not case.get('facts'):
            issues['no_facts'].append(cid)
        if not case.get('court_assessments'):
            issues['no_court_assessments'].append(cid)
        for ca in case.get('court_assessments', []):
            for sub in ca.get('subsections', []):
                if sub['subsection_title'] == 'Admissibility' and not sub['paragraphs']:
                    issues['admissibility_empty'].append(cid)
                if sub['subsection_title'] == 'Merits' and not sub['paragraphs']:
                    issues['merits_empty'].append(cid)

    print(f'\n  Valid cases: {len(valid_cases)}')
    for issue, ids in issues.items():
        status = 'OK' if not ids else f'{len(ids)} cases'
        print(f'  {issue:<30} {status}')
        if ids and len(ids) <= 5:
            for cid in ids:
                print(f'    - {cid}')


# ---------------------------------------------------------------------------
# 6. Word-length distribution (detailed)
# ---------------------------------------------------------------------------

def word_length_distribution(data):
    """Detailed word length stats with percentiles."""
    print_header('WORD LENGTH DISTRIBUTION (COURT ASSESSMENTS)')

    article_subsections = defaultdict(lambda: defaultdict(list))
    valid_cases = [c for c in data if not c.get('error') and not c.get('empty_file')]

    for case in valid_cases:
        for ca in case.get('court_assessments', []):
            for art in ca.get('articles', []):
                for sub in ca.get('subsections', []):
                    wc = words_in_paragraphs(sub['paragraphs'])
                    article_subsections[art][sub['subsection_title']].append(wc)

    print(f'\n{"Article":<10} {"Subsection":<16} {"N":<6} '
          f'{"Min":<8} {"25th":<8} {"Median":<8} {"75th":<8} {"Max":<8} {"Total Words"}')
    print('-' * 100)

    for art in sorted(article_subsections.keys()):
        for sub_title in ['Admissibility', 'Merits', 'General']:
            wcs = article_subsections[art].get(sub_title, [])
            if not wcs:
                continue
            wcs_sorted = sorted(wcs)
            n = len(wcs_sorted)

            def percentile(data, p):
                idx = int(p / 100 * (len(data) - 1))
                return data[idx]

            print(f'Art {art:<6} {sub_title:<16} {n:<6} '
                  f'{min(wcs_sorted):<8} {percentile(wcs_sorted, 25):<8} '
                  f'{percentile(wcs_sorted, 50):<8} {percentile(wcs_sorted, 75):<8} '
                  f'{max(wcs_sorted):<8} {sum(wcs_sorted)}')
        print()


# ---------------------------------------------------------------------------
# Suggestions
# ---------------------------------------------------------------------------

def print_suggestions():
    print_header('SUGGESTED ADDITIONAL ANALYSES')
    suggestions = [
        '1. Temporal trends: violation rates over time (by year/quarter)',
        '2. Complexity correlation: do longer facts correlate with longer assessments?',
        '3. Citation overlap: which cases share the most cited precedents?',
        '4. Country analysis: extract respondent state from case_name (X v. COUNTRY)',
        '5. Section/Chamber distribution: which sections handle which articles?',
        '6. Assessment-to-facts ratio: how much of the judgment is reasoning vs. background?',
        '7. Self-citation depth: do ref_paragraphs mostly point to facts or other law sections?',
        '8. Admissibility filtering: what fraction of cases have substantive admissibility discussion vs. boilerplate?',
        '9. Multi-article cases: are certain article combinations more common (e.g. Art 3+13)?',
        '10. Violation predictors: word count / citation density differences between violation vs. no-violation outcomes',
    ]
    print()
    for s in suggestions:
        print(f'  {s}')
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data = load_data()
    case_level_stats(data)
    assessment_stats(data)
    violation_stats(data)
    citation_stats(data)
    completeness_stats(data)
    word_length_distribution(data)
    print_suggestions()


if __name__ == '__main__':
    main()
