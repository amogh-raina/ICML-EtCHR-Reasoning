#!/usr/bin/env python3
"""
Convert Excel human annotation sheets to the JSON shape used by
Scripts/plot_human_scores.py.

The Excel sheets do not include A_category/B_category columns, so this script
uses combined_30_seed42_og.csv as the row-order/category key.

Outputs by default:
    data/annotation/human_1_annotations.json
    data/annotation/human_2_annotations.json
    data/annotation/human_3_annotations.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


BASE_DIR = Path(__file__).parent.parent
ANNOTATION_DIR = BASE_DIR / "data" / "annotation"
METADATA_CSV = ANNOTATION_DIR / "combined_30_seed42_og.csv"

CATEGORIES = ["non_curated", "instruction", "gpt_assisted"]
STEPS = ["step_1", "step_2", "step_3", "step_4"]

XLSX_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def empty_annotations() -> dict:
    return {
        category: {
            **{
                step: {"occurrence": [], "comprehensiveness": []}
                for step in STEPS
            },
            "overall_conciseness": [],
        }
        for category in CATEGORIES
    }


def column_index(cell_ref: str) -> int:
    match = re.match(r"([A-Z]+)", cell_ref)
    if not match:
        raise ValueError(f"Bad Excel cell reference: {cell_ref}")
    value = 0
    for char in match.group(1):
        value = value * 26 + ord(char) - ord("A") + 1
    return value - 1


def read_xlsx_sheet(path: Path) -> list[list[str]]:
    """Read the first worksheet from an .xlsx file using only stdlib XML."""
    with zipfile.ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", XLSX_NS):
                shared_strings.append(
                    "".join(text.text or "" for text in item.findall(".//a:t", XLSX_NS))
                )

        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows: list[list[str]] = []
        for row in sheet.findall(".//a:sheetData/a:row", XLSX_NS):
            values: list[str] = []
            for cell in row.findall("a:c", XLSX_NS):
                idx = column_index(cell.attrib["r"])
                while len(values) <= idx:
                    values.append("")

                cell_type = cell.attrib.get("t")
                raw_value = cell.find("a:v", XLSX_NS)
                inline_string = cell.find("a:is", XLSX_NS)
                if cell_type == "s" and raw_value is not None:
                    value = shared_strings[int(raw_value.text or "0")]
                elif cell_type == "inlineStr" and inline_string is not None:
                    value = "".join(
                        text.text or ""
                        for text in inline_string.findall(".//a:t", XLSX_NS)
                    )
                elif raw_value is not None:
                    value = raw_value.text or ""
                else:
                    value = ""
                values[idx] = value.strip()
            rows.append(values)

    max_width = max(len(row) for row in rows)
    for row in rows:
        row.extend([""] * (max_width - len(row)))
    return rows


def parse_occurrence(value: str) -> int | None:
    text = value.strip().lower()
    if not text:
        return None
    if text.startswith("y"):
        return 1
    if text == "tes":
        return 1
    if text.startswith("n"):
        return 0
    if text in {"1", "1.0"}:
        return 1
    if text in {"0", "0.0"}:
        return 0
    raise ValueError(f"Cannot parse occurrence value: {value!r}")


def parse_score(value: str) -> int | None:
    text = value.strip().replace("'", "")
    if not text:
        return None
    match = re.search(r"\d+(?:\.\d+)?", text)
    if not match:
        raise ValueError(f"Cannot parse score value: {value!r}")
    score = int(round(float(match.group())))
    if not 0 <= score <= 5:
        raise ValueError(f"Score outside 0-5 range: {value!r}")
    return score


def load_metadata(path: Path) -> list[dict[str, str]]:
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def find_header(rows: list[list[str]]) -> tuple[int, dict[str, int]]:
    required = {"item_id", "A_Step_1 Occurrence", "B_Step_1 Occurrence"}
    for idx, row in enumerate(rows):
        present = {value for value in row if value}
        if required.issubset(present):
            return idx, {name: col for col, name in enumerate(row) if name}
    raise ValueError("Could not find annotation header row")


def convert_sheet(path: Path, metadata_rows: list[dict[str, str]]) -> tuple[dict, list[str]]:
    rows = read_xlsx_sheet(path)
    if len(rows) < 2:
        raise ValueError(f"{path} does not look like an annotation sheet")

    header_idx, col = find_header(rows)
    data_rows = rows[header_idx + 1:]
    if len(data_rows) != len(metadata_rows):
        raise ValueError(
            f"{path} has {len(data_rows)} data rows, but metadata has "
            f"{len(metadata_rows)} rows"
        )

    annotations = empty_annotations()
    warnings: list[str] = []

    for row_num, (row, meta) in enumerate(zip(data_rows, metadata_rows), start=1):
        item_id = row[col["item_id"]]
        if item_id != meta["item_id"]:
            warnings.append(
                f"row {row_num}: item_id mismatch {item_id!r} != {meta['item_id']!r}"
            )

        for side in ("A", "B"):
            category = meta[f"{side}_category"]
            for step_num, step_key in enumerate(STEPS, start=1):
                occurrence_col = f"{side}_Step_{step_num} Occurrence"
                comprehensiveness_col = f"{side}_Step_{step_num} Comprehensiveness"

                occurrence = parse_occurrence(row[col[occurrence_col]])
                comprehensiveness = parse_score(row[col[comprehensiveness_col]])

                if occurrence is None:
                    if comprehensiveness is not None:
                        warnings.append(
                            f"row {row_num} side {side} {step_key}: "
                            "score present without occurrence"
                        )
                    continue

                if comprehensiveness is None and occurrence == 0:
                    comprehensiveness = 0

                annotations[category][step_key]["occurrence"].append(occurrence)
                if comprehensiveness is not None:
                    annotations[category][step_key]["comprehensiveness"].append(
                        comprehensiveness
                    )

            conciseness = parse_score(row[col[f"{side}_Overall Conciseness"]])
            if conciseness is not None:
                annotations[category]["overall_conciseness"].append(conciseness)

    return annotations, warnings


def merge_annotations(paths: list[Path]) -> dict:
    merged = empty_annotations()
    for path in paths:
        current = json.loads(path.read_text(encoding="utf-8"))
        for category in CATEGORIES:
            for step in STEPS:
                for metric in ("occurrence", "comprehensiveness"):
                    merged[category][step][metric].extend(
                        current[category][step][metric]
                    )
            merged[category]["overall_conciseness"].extend(
                current[category]["overall_conciseness"]
            )
    return merged


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")


def print_counts(label: str, data: dict) -> None:
    print(label)
    for category in CATEGORIES:
        parts = []
        for step in STEPS:
            step_data = data[category][step]
            parts.append(
                f"{step}: occ={len(step_data['occurrence'])}, "
                f"comp={len(step_data['comprehensiveness'])}"
            )
        parts.append(f"conc={len(data[category]['overall_conciseness'])}")
        print(f"  {category}: " + "; ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ECHR Excel annotation sheets to human annotation JSON."
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=METADATA_CSV,
        help="Combined annotation CSV used for row-order category metadata.",
    )
    parser.add_argument(
        "--sheet",
        action="append",
        nargs=2,
        metavar=("XLSX", "OUT_JSON"),
        default=[
            (
                str(ANNOTATION_DIR / "Copy of ECHR annotation sheet.xlsx"),
                str(ANNOTATION_DIR / "human_1_annotations.json"),
            ),
            (
                str(ANNOTATION_DIR / "ECHR annotation sheet.xlsx"),
                str(ANNOTATION_DIR / "human_2_annotations.json"),
            ),
            (
                str(ANNOTATION_DIR / "ECHR annotation sheet completed.xlsx"),
                str(ANNOTATION_DIR / "human_3_annotations.json"),
            ),
        ],
        help="Input workbook and output JSON. Repeat for multiple annotators.",
    )
    parser.add_argument(
        "--merge-output",
        type=Path,
        default=None,
        help="Optional combined JSON output. Omit to write only individual annotator files.",
    )
    args = parser.parse_args()

    metadata_rows = load_metadata(args.metadata_csv)
    output_paths: list[Path] = []

    for xlsx_raw, out_raw in args.sheet:
        xlsx_path = Path(xlsx_raw)
        out_path = Path(out_raw)
        data, warnings = convert_sheet(xlsx_path, metadata_rows)
        write_json(out_path, data)
        output_paths.append(out_path)
        print_counts(f"Wrote {out_path}", data)
        for warning in warnings:
            print(f"  warning: {warning}")

    if args.merge_output is not None:
        merged = merge_annotations(output_paths)
        write_json(args.merge_output, merged)
        print_counts(f"Wrote {args.merge_output}", merged)


if __name__ == "__main__":
    main()
