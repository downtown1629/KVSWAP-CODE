#!/usr/bin/env python3
"""Export KVSwap NVTX ranges from an Nsight Systems SQLite database."""

import argparse
import csv
import re
import sqlite3
from collections import defaultdict
from pathlib import Path


PREFIXES = (
    "KVSWAP_TOKEN",
    "KVSWAP_LAYER",
    "KVSWAP_STAGE",
    "KVSWAP_ATTENTION_STAGE",
    "KVSWAP_MOE_STAGE",
    "KVSWAP_ATTENTION_CHUNK",
    "KVSWAP_PREFILL_CHUNK",
)
COLORS = {
    "in": "#9e9e9e",
    "out": "#616161",
    "attention_swa": "#42a5f5",
    "attention_global": "#1565c0",
    "moe": "#7e57c2",
    "load_weight": "#fb8c00",
    "load_cache": "#ffb74d",
    "load_hidden": "#90a4ae",
    "attention_mask": "#cfd8dc",
    "rope": "#fdd835",
    "sync_kv": "#8d6e63",
    "compute": "#ef5350",
    "store_hidden": "#78909c",
    "store_cache": "#ffa726",
    "prefetch_cache": "#f57c00",
    "prefetch_sync": "#795548",
    "prefetch_wait": "#5d4037",
    "kv_concat": "#ab47bc",
    "speculate": "#26c6da",
    "norm": "#fbc02d",
    "qkv_projection": "#26a69a",
    "attention": "#1e88e5",
    "attention_output": "#1e88e5",
    "output_projection": "#3949ab",
    "cache_pack": "#ff8f00",
    "router": "#d4e157",
    "materialize": "#ff7043",
    "dispatch_compute": "#66bb6a",
    "residual": "#43a047",
}


def parse_fields(name):
    return dict(re.findall(r"([A-Za-z_]+)=([^ ]+)", name))


def range_kind(name):
    return name.split(" ", 1)[0]


def short_name(name):
    fields = parse_fields(name)
    if name.startswith("KVSWAP_LAYER"):
        return fields.get("kind", "layer")
    if "_STAGE" in name:
        return fields.get("name", "stage")
    if "_CHUNK" in name:
        return f"chunk {fields.get('start', '?')}:{fields.get('end', '?')}"
    return name


def load_ranges(database):
    connection = sqlite3.connect(database)
    query = """
        SELECT n.start, n.end, COALESCE(n.text, s.value), n.globalTid
        FROM NVTX_EVENTS AS n
        LEFT JOIN StringIds AS s ON s.id = n.textId
        WHERE n.end IS NOT NULL AND COALESCE(n.text, s.value) LIKE 'KVSWAP_%'
        ORDER BY n.start, n.end DESC
    """
    try:
        rows = connection.execute(query).fetchall()
    finally:
        connection.close()
    return [
        {"start": row[0], "end": row[1], "name": row[2], "tid": row[3]}
        for row in rows if row[2] and row[2].startswith(PREFIXES)
    ]


def annotate_hierarchy(ranges):
    tokens = [item for item in ranges if range_kind(item["name"]) == "KVSWAP_TOKEN"]
    for item in ranges:
        item["token"] = next((
            token for token in tokens
            if token["tid"] == item["tid"]
            and token["start"] <= item["start"]
            and item["end"] <= token["end"]
        ), None)
    for item in ranges:
        token = item["token"]
        if not token or range_kind(item["name"]) in ("KVSWAP_TOKEN", "KVSWAP_LAYER"):
            item["layer"] = None
            continue
        item["layer"] = next((
            layer for layer in ranges
            if range_kind(layer["name"]) == "KVSWAP_LAYER"
            and layer["tid"] == item["tid"]
            and layer["start"] <= item["start"]
            and item["end"] <= layer["end"]
        ), None)
    return tokens


def svg_escape(value):
    return (str(value).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def write_svg(path, title, rows, origin, finish):
    label_width = 210
    plot_width = 1400
    row_height = 24
    legend_names = sorted({short_name(item["name"]) for _, bars in rows for item in bars})
    legend_columns = 5
    legend_rows = max((len(legend_names) + legend_columns - 1) // legend_columns, 1)
    legend_row_height = 20
    top = 62 + legend_rows * legend_row_height
    width = label_width + plot_width + 30
    height = top + row_height * len(rows) + 45
    span = max(finish - origin, 1)

    def x(timestamp):
        return label_width + (timestamp - origin) * plot_width / span

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:monospace;font-size:11px}.title{font-size:15px;font-weight:bold}.axis{fill:#555}.bar{stroke:#333;stroke-width:.35}</style>',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="8" y="22" class="title">{svg_escape(title)}</text>',
        '<text x="8" y="41" class="axis">CPU NVTX wall time; nested hierarchy is not additive. Use .nsys-rep for GPU correlation.</text>',
    ]
    legend_width = width / legend_columns
    for index, name in enumerate(legend_names):
        column = index % legend_columns
        row = index // legend_columns
        xpos = 8 + column * legend_width
        ypos = 51 + row * legend_row_height
        color = COLORS.get(name, "#b0bec5")
        lines.append(f'<rect x="{xpos:.1f}" y="{ypos:.1f}" width="13" height="13" fill="{color}" stroke="#333" stroke-width=".35"/>')
        lines.append(f'<text x="{xpos + 18:.1f}" y="{ypos + 11:.1f}">{svg_escape(name)}</text>')
    for fraction in (0, .25, .5, .75, 1):
        xpos = label_width + fraction * plot_width
        elapsed_ms = span * fraction / 1e6
        lines.append(f'<line x1="{xpos:.1f}" y1="48" x2="{xpos:.1f}" y2="{height - 25}" stroke="#ddd"/>')
        lines.append(f'<text x="{xpos:.1f}" y="{height - 8}" text-anchor="middle" class="axis">{elapsed_ms:.1f} ms</text>')
    for index, (label, bars) in enumerate(rows):
        ypos = top + index * row_height
        lines.append(f'<text x="{label_width - 6}" y="{ypos + 15}" text-anchor="end">{svg_escape(label)}</text>')
        for item in bars:
            name = short_name(item["name"])
            xpos = x(item["start"])
            bar_width = max(x(item["end"]) - xpos, .75)
            duration_ms = (item["end"] - item["start"]) / 1e6
            color = COLORS.get(name, "#b0bec5")
            tip = f"{item['name']} | {duration_ms:.3f} ms"
            lines.append(
                f'<rect class="bar" x="{xpos:.2f}" y="{ypos + 3}" width="{bar_width:.2f}" height="16" fill="{color}">'
                f'<title>{svg_escape(tip)}</title></rect>'
            )
            if bar_width >= max(54, len(name) * 7):
                lines.append(
                    f'<text x="{xpos + bar_width / 2:.2f}" y="{ypos + 15}" '
                    f'text-anchor="middle" fill="#111">{svg_escape(name)}</text>'
                )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n")


def write_outputs(database, output_prefix, layers_per_page=8):
    if layers_per_page < 1:
        raise ValueError("layers_per_page must be positive")
    ranges = load_ranges(database)
    tokens = annotate_hierarchy(ranges)
    if not tokens:
        raise RuntimeError("no KVSWAP_TOKEN ranges found")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    raw_csv = output_prefix.with_suffix(".nvtx.csv")
    with raw_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("phase", "step", "level", "model_layer", "name", "start_ms", "duration_ms"))
        for item in ranges:
            token_fields = parse_fields(item["token"]["name"]) if item["token"] else {}
            layer_fields = parse_fields(item["layer"]["name"]) if item.get("layer") else {}
            origin = item["token"]["start"] if item["token"] else ranges[0]["start"]
            writer.writerow((
                token_fields.get("phase", ""), token_fields.get("step", ""),
                range_kind(item["name"]), layer_fields.get("model_layer", ""),
                item["name"], f"{(item['start'] - origin) / 1e6:.6f}",
                f"{(item['end'] - item['start']) / 1e6:.6f}",
            ))

    summary = defaultdict(lambda: [0, 0])
    for item in ranges:
        if not item["token"] or range_kind(item["name"]) not in (
            "KVSWAP_STAGE", "KVSWAP_ATTENTION_STAGE", "KVSWAP_MOE_STAGE"
        ):
            continue
        token_fields = parse_fields(item["token"]["name"])
        key = (token_fields["phase"], token_fields["step"], range_kind(item["name"]), short_name(item["name"]))
        summary[key][0] += 1
        summary[key][1] += item["end"] - item["start"]
    summary_csv = output_prefix.with_suffix(".stage-summary.csv")
    with summary_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("phase", "step", "level", "stage", "count", "total_ms", "mean_ms"))
        for key, (count, duration) in sorted(summary.items()):
            writer.writerow((*key, count, f"{duration / 1e6:.6f}", f"{duration / count / 1e6:.6f}"))

    overview_rows = []
    for token in tokens:
        fields = parse_fields(token["name"])
        layers = [item for item in ranges if item["token"] is token and range_kind(item["name"]) == "KVSWAP_LAYER"]
        overview_rows.append((f"{fields['phase']} step {fields['step']}", layers))
    write_svg(output_prefix.with_suffix(".token-gantt.svg"), "KVSwap token and layer timeline",
              overview_rows, tokens[0]["start"], tokens[-1]["end"])

    detail_paths = []
    for token in tokens:
        fields = parse_fields(token["name"])
        token_ranges = [item for item in ranges if item["token"] is token]
        layers = [item for item in token_ranges if range_kind(item["name"]) == "KVSWAP_LAYER"]
        model_layer_ids = sorted({
            int(parse_fields(layer["name"])["model_layer"])
            for layer in layers if int(parse_fields(layer["name"])["model_layer"]) >= 0
        })
        pages = [
            model_layer_ids[start:start + layers_per_page]
            for start in range(0, len(model_layer_ids), layers_per_page)
        ] or [[]]
        for page_ids in pages:
            rows = []
            selected_layers = [
                layer for layer in layers
                if int(parse_fields(layer["name"])["model_layer"]) in page_ids
            ]
            for layer in selected_layers:
                layer_fields = parse_fields(layer["name"])
                label = f"L{layer_fields.get('model_layer', '?')} {layer_fields.get('kind', 'layer')}"
                top = [item for item in token_ranges if item.get("layer") is layer and range_kind(item["name"]) == "KVSWAP_STAGE"]
                detail = [item for item in token_ranges if item.get("layer") is layer and range_kind(item["name"]) in ("KVSWAP_ATTENTION_STAGE", "KVSWAP_MOE_STAGE")]
                rows.append((label, top))
                if detail:
                    rows.append(("  sub-stage", detail))
            suffix = "none" if not page_ids else f"{page_ids[0]:02d}-{page_ids[-1]:02d}"
            path = output_prefix.parent / (
                f"{output_prefix.name}.step-{int(fields['step']):03d}-{fields['phase']}"
                f".layers-{suffix}.gantt.svg"
            )
            write_svg(
                path,
                f"KVSwap {fields['phase']} step {fields['step']} | model layers {suffix}",
                rows, token["start"], token["end"],
            )
            detail_paths.append(path)
    return [raw_csv, summary_csv, output_prefix.with_suffix(".token-gantt.svg"), *detail_paths]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path, help="SQLite exported by nsys stats/export")
    parser.add_argument("--output-prefix", type=Path, help="output prefix (default: input without .sqlite)")
    parser.add_argument("--layers-per-page", type=int, default=8,
                        help="model layers in each detailed SVG (default: 8)")
    args = parser.parse_args()
    prefix = args.output_prefix or args.database.with_suffix("")
    for path in write_outputs(args.database, prefix, args.layers_per_page):
        print(path)


if __name__ == "__main__":
    main()
