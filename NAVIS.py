#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════ ══════╗
║                     NAVIS.py                                           ║
║                                                                        ║
║                                                                        ║
║   NAVIS - NAnopore Visualization & Interactive Statistiques            ║    
║                                             ║
║  Multi-file statistical analysis of Nanopore sequencing data.                    ║
║  Generates an interactive HTML report including:                                   ║
║    - Cumulative statistics table (Raw / Filtered)                   ║
║    - Summary histograms (horizontal bars, 4 panels)                                      ║
║    - Distribution curves (length, bases, quality)                     ║
║    - Stacked length vs quality heatmaps                                  ║
║    - TXT exports: raw and filtered statistics                ║
║                                                                             ║
║  Changes:                                                       ║
║    - Global "Hide Outliers" button (prominent, top of page)          ║
║      applied to heatmaps AND distribution curves                   ║
║    - Length outlier detection via configurable percentile         ║
║      via --outlier_percentile (default: 99.5)                               ║
║    - 3 display modes: Raw / Filtered / No Outliers                     ║
║                                                                             ║
║  Usage :                                                                    ║
║    python NAVIS.py                                                     ║
║        -i sample1.txt sample2.txt ...   # input files (required)   ║
║        -o rapport.html                  # output HTML file            ║
║        -t 4                             # number of CPU threads             ║
║        -b 1000                          # bin size (bp) for heatmaps ║
║        --min_len 500                    # minimum read length (bp)  ║
║        --max_len 30000                  # maximum read length (bp)  ║
║        --min_qual 8                     # minimum mean quality          ║
║        --max_qual 25                    # maximum mean quality          ║
║        --outlier_percentile 99.5          # outlier percentile threshold (default)║
║                                                                             ║
║  Input file format (TSV, with header):                          ║
║    read_id   length   mean_quality                                          ║
║    read_001  5432     14.3                                                  ║
║    ...                                                                      ║
╚═════════════════════════════════════════════════════════════════════════════╝

Version history:

"""

import argparse
import gzip
import re
import shutil
import subprocess
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from multiprocessing import Pool, cpu_count
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os


# ==============================================================================
# SECTION 1 — COMMAND-LINE ARGUMENTS
# ==============================================================================

def parse_args():
    """
    Defines and parses all command-line arguments.

    Available arguments:
      -i / --input             : one or more TXT input files (required)
      -o / --output            : output HTML filename
                                 (default: NAVIS_summary.html)
      -t / --threads           : number of CPU threads (default: 1)
      -b / --bin_size          : bin size in bp for heatmaps (default: 1000)
                                 → increase for very long reads,
                                     decrease for finer resolution
      --min_len                : minimum read length filter in bp (e.g. 500)
      --max_len                : maximum read length filter in bp (e.g. 50000)
      --min_qual               : minimum quality filter (e.g. 8.0)
      --max_qual               : maximum quality filter (e.g. 25.0)
      --outlier_percentile     : length percentile above which a read
                                 is considered an outlier (default: 99.5)
                                 → e.g. 99 = exclude the longest 1% of reads
                                 → e.g. 95 = exclude the longest 5% of reads
                                 → applied independently per file
    """
    parser = argparse.ArgumentParser(description="NAVIS summary statistics V13.0")
    parser.add_argument("-i", "--input", nargs='+', required=True,
                        help="TXT input files with columns: read_id, length, mean_quality")
    parser.add_argument("-o", "--output", default="NAVIS_summary.html",
                        help="Output HTML file")
    parser.add_argument("-t", "--threads", type=int, default=0,
                        help="Number of CPU threads (default: 0 = use all available CPUs)")
    parser.add_argument("-b", "--bin_size", type=int, default=1000,
                        help="Bin size in bp for heatmaps (default 1000)")
    parser.add_argument("--min_len", type=int, default=None,
                        help="Minimum read length filter (bp)")
    parser.add_argument("--max_len", type=int, default=None,
                        help="Maximum read length filter (bp)")
    parser.add_argument("--min_qual", type=float, default=None,
                        help="Minimum mean quality filter")
    parser.add_argument("--max_qual", type=float, default=None,
                        help="Maximum mean quality filter")
    parser.add_argument("--outlier_percentile", type=float, default=99.5,
                        help="Percentile threshold for hiding length outliers (default: 99.5). "
                             "Reads with length > Nth percentile are excluded from "
                             "heatmaps and distribution curves in 'No Outliers' mode.")
    parser.add_argument("--low_memory", action="store_true", default=False,
                        help="Enable low-memory mode: files read sequentially (1 thread), "
                             "raw read data freed after per-file processing, "
                             "figures built and written to HTML one at a time. "
                             "Slower but avoids RAM crashes on large datasets.")
    parser.add_argument("--light_html", action="store_true", default=False,
                        help="Reduce heatmap Y-axis bins from 50 (default) to 25 "
                             "for a lighter HTML output file. Useful for very large "
                             "datasets where file size matters more than resolution.")
    return parser.parse_args()


# ==============================================================================
# SECTION 2 — UTILITY FUNCTIONS
# ==============================================================================

def parse_line(line):
    """
    Parses one line of a TSV input file.
    Expected columns: read_id (col 0), length (col 1), mean_quality (col 2).
    Returns a (length, mean_quality) tuple, or None if the line is invalid.
    Malformed lines are silently skipped.
    """
    try:
        parts = line.rstrip().split("\t")
        length = int(parts[1])
        mean_q = float(parts[2])
        return length, mean_q
    except:
        return None


def fmt_pct(value):
    """
    Formats a percentile value for display, preserving decimals when present.
    Examples: 99.5 -> "99.5", 99 -> "99", 99.0 -> "99", 95.25 -> "95.25".
    Avoids the rounding issue where ':.0f' turns 99.5 into '100'.
    """
    return f"{value:g}"


def n50(lengths):
    """
    Computes the N50 of a list of read lengths.
    N50 is the length L such that 50% of all sequenced bases
    are contained in reads of length >= L.
    Returns 0 if the list is empty.
    """
    if not lengths:
        return 0
    sorted_lengths = sorted(lengths, reverse=True)
    total = sum(sorted_lengths)
    running = 0
    for l in sorted_lengths:
        running += l
        if running >= total / 2:
            return l
    return 0


def compute_stats(filename, lengths, quals):
    """
    Computes all descriptive statistics for a given file.
    Uses numpy arrays throughout for speed on large datasets (>1M reads).

    Filename handling:
      - Strips the directory path (keeps only the basename)
      - Removes known extensions: .fastq.gz, .fq.gz, .fastq, .fq, .txt, .tsv, .gz
      → To support additional extensions, add them to the 'ext' list below.
    """
    stats   = {}
    n_reads = len(lengths)

    # --- Strip path and known file extensions
    base = os.path.basename(filename)
    for ext in ('.fastq.gz', '.fq.gz', '.fastq', '.fq', '.txt', '.tsv', '.gz'):
        if base.endswith(ext):
            base = base[:-len(ext)]
            break
    stats['file'] = base

    if n_reads == 0:
        # Empty file: return zeros for all stats
        for key in ('n_reads', 'total_bases', 'avg_length', 'min_length', 'max_length',
                    'median_length', 'perc75_length', 'n50',
                    'avg_quality', 'median_quality', 'perc25_quality',
                    'perc75_quality', 'min_quality', 'max_quality'):
            stats[key] = 0
        return stats

    # Convert to numpy arrays once — all operations below are vectorized
    arr_len  = np.asarray(lengths,  dtype=np.int64)
    arr_qual = np.asarray(quals,    dtype=np.float64)

    # --- Length statistics (fully vectorized)
    stats['n_reads']       = n_reads
    stats['total_bases']   = int(arr_len.sum())
    stats['avg_length']    = float(arr_len.mean())
    stats['min_length']    = int(arr_len.min())
    stats['max_length']    = int(arr_len.max())
    stats['median_length'] = float(np.median(arr_len))
    stats['perc75_length'] = float(np.percentile(arr_len, 75))
    stats['n50']           = n50(lengths)

    # --- Quality statistics (fully vectorized)
    stats['avg_quality']    = float(arr_qual.mean())
    stats['median_quality'] = float(np.median(arr_qual))
    stats['perc25_quality'] = float(np.percentile(arr_qual, 25))
    stats['perc75_quality'] = float(np.percentile(arr_qual, 75))
    stats['min_quality']    = float(arr_qual.min())
    stats['max_quality']    = float(arr_qual.max())

    return stats


def open_input_file(filename, n_threads=1):
    """
    Opens an input file for reading, with automatic decompression for .gz files.

    Strategy for .gz files:
      1. pigz  — parallel gzip decompressor (uses n_threads), much faster on
                 large files if installed. Detected via shutil.which('pigz').
      2. gzip  — Python standard library fallback if pigz is not available.
         A warning is printed recommending pigz installation.

    For plain .txt files: opened directly with no decompression.

    Returns a file-like object (text mode, utf-8) ready for line iteration.
    n_threads is passed to pigz via -p (ignored for plain files and gzip fallback).
    """
    if filename.endswith('.gz'):
        if shutil.which('pigz'):
            # pigz: decompress to stdout, pipe into Python as a text stream
            proc = subprocess.Popen(
                ['pigz', '-d', '-c', '-p', str(max(1, n_threads)), filename],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
            return io.TextIOWrapper(proc.stdout, encoding='utf-8', errors='replace')
        else:
            return gzip.open(filename, 'rt', encoding='utf-8', errors='replace')
    else:
        return open(filename, 'r', encoding='utf-8', errors='replace')


def process_file(filename, n_threads=1):
    """
    Reads and processes a complete input file.

    Supported formats (auto-detected by extension):
      .txt      — plain TSV, read directly
      .txt.gz   — gzip-compressed TSV, decompressed via pigz (parallel) or
                  Python gzip (fallback with warning if pigz not installed)

    Reading strategy (fastest to slowest, with automatic fallback):
      1. pandas read_csv  — fastest for large files (>1M rows), uses C parser
      2. np.loadtxt       — fallback if pandas fails (does not support streams)
      3. line-by-line     — final fallback for irregular/malformed files

    Only columns 1 (length) and 2 (mean_quality) are read; read_id is skipped.

    Returns a tuple: (filename, lengths, quals, stats).
    Called in parallel via multiprocessing.Pool or sequentially in low_memory mode.
    n_threads is forwarded to pigz for parallel decompression of .gz files.
    """
    is_gz = filename.endswith('.gz')
    try:
        if is_gz:
            # For .gz files: decompress via open_input_file, read into StringIO
            # so pandas can use its fast C parser on the in-memory buffer.
            with open_input_file(filename, n_threads=n_threads) as fh:
                raw = fh.read()
            buf = io.StringIO(raw)
            df = pd.read_csv(buf, sep='\t', usecols=[1, 2],
                             header=0, dtype={1: 'int32', 2: 'float32'},
                             engine='c', na_filter=False)
        else:
            # Plain .txt: pandas reads directly from disk (fastest path)
            df = pd.read_csv(filename, sep='\t', usecols=[1, 2],
                             header=0, dtype={1: 'int32', 2: 'float32'},
                             engine='c', na_filter=False)
        lengths = df.iloc[:, 0].tolist()
        quals   = df.iloc[:, 1].tolist()
    except Exception:
        try:
            # Second fallback: numpy (plain files only — no stream support)
            if not is_gz:
                data = np.loadtxt(filename, dtype=float, usecols=(1, 2), comments='read_id')
                if data.ndim == 1:
                    data = data.reshape(1, 2)
                lengths = data[:, 0].astype(int).tolist()
                quals   = data[:, 1].tolist()
            else:
                raise ValueError("numpy fallback not supported for .gz files")
        except Exception:
            # Final fallback: line-by-line parser (works for both plain and .gz)
            lengths, quals = [], []
            with open_input_file(filename, n_threads=n_threads) as fh:
                for line in fh:
                    if line.startswith("read_id"):
                        continue
                    res = parse_line(line)
                    if res:
                        l, q = res
                        lengths.append(l)
                        quals.append(q)
    stats = compute_stats(filename, lengths, quals)
    return (filename, lengths, quals, stats)


def apply_filters(lengths, quals, min_len, max_len, min_qual, max_qual):
    """
    Filters reads based on length and quality thresholds.
    Uses numpy boolean masking — much faster than a Python loop on large lists.
    If a threshold is None, it is not applied.
    Returns two filtered lists: (filtered_lengths, filtered_quals).
    """
    if not lengths:
        return [], []
    arr_l = np.asarray(lengths, dtype=np.int64)
    arr_q = np.asarray(quals,   dtype=np.float64)
    mask  = np.ones(len(arr_l), dtype=bool)
    if min_len  is not None: mask &= arr_l >= min_len
    if max_len  is not None: mask &= arr_l <= max_len
    if min_qual is not None: mask &= arr_q >= min_qual
    if max_qual is not None: mask &= arr_q <= max_qual
    return arr_l[mask].tolist(), arr_q[mask].tolist()


def apply_outlier_filter(lengths, quals, percentile):
    """
    Filters length outliers by percentile threshold.

    Method: computes the Nth percentile of the length distribution
    (across all reads in the file), then excludes reads whose length
    exceeds that threshold.

    Why use a percentile for Nanopore data?
      - Nanopore length distributions are log-normal and highly
          skewed: a few ultra-long reads can crush the scale of
          heatmaps and curves, hiding the bulk of the data.
      - The percentile is robust, intuitive, and adapts to each dataset
          without assuming a distribution shape (unlike Z-score or IQR).
      - Example effects:
            percentile=99 → excludes the longest 1% of reads
            percentile=95 → excludes the longest 5% of reads

    Tunable parameter:
      percentile (float): passed via --outlier_percentile on the command line
                           Default: 99.5

    Returns (filtered_lengths, filtered_quals, threshold):
      - filtered_lengths / filtered_quals: retained reads
      - threshold: computed threshold value (in bp), printed in the report
    """
    if not lengths:
        return lengths, quals, 0

    # Compute length threshold at the Nth percentile
    threshold = np.percentile(lengths, percentile)

    arr_l = np.asarray(lengths, dtype=np.int64)
    arr_q = np.asarray(quals,   dtype=np.float64)
    mask  = arr_l <= threshold
    return arr_l[mask].tolist(), arr_q[mask].tolist(), threshold


def histogram_curve(data, nbins=200, x_range=None):
    """
    Computes a raw (non-normalized) histogram from a list of values.

    Tunable parameters:
      nbins:   number of histogram bins
                 → 200 for lengths (wide distribution)
                 → 100 for quality scores (narrower distribution)
                 → Increase for finer resolution, decrease to smooth.
      x_range: optional (xmin, xmax) tuple passed to np.histogram as range=.
                 When set, bins are computed exactly within [xmin, xmax],
                 so no data points appear at x=0 for filtered datasets.
                 This is the most reliable way to fix the axis start position.

    Returns (x, counts): bin centres and corresponding counts.
    """
    if not data:
        return np.array([0]), np.array([0])
    counts, bin_edges = np.histogram(data, bins=nbins, density=False,
                                     range=x_range if x_range else None)
    x = (bin_edges[:-1] + bin_edges[1:]) / 2  # bin centres
    return x, counts


def px_colors(n):
    """
    Generates a list of n distinct colours from the Plotly qualitative palette.
    The palette cycles automatically if n exceeds the number of available colours (10).

    To change the colour palette, replace 'Plotly' with another palette name,
    e.g.: 'D3', 'G10', 'T10', 'Alphabet', 'Dark24', 'Light24'.
    See: https://plotly.com/python/discrete-color/
    """
    import plotly.express as px
    palette = px.colors.qualitative.Plotly
    return [palette[i % len(palette)] for i in range(n)]


def fig_to_compact_scroll(fig, include_plotlyjs=False):
    """
    Like fig_to_compact(), but wraps the figure in a .plot-wrapper-scroll div
    that has a fixed max-height with a vertical scrollbar.
    Used for histograms when there are many samples.
    """
    _plotly_div_counter[0] += 1
    div_id = f"plotly-fig-{_plotly_div_counter[0]}"
    json_data = fig.to_json(pretty=False, remove_uids=True)
    cdn_tag = ('<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>\n'
               if include_plotlyjs else "")
    html = (
        f"{cdn_tag}"
        f"<div class='plot-wrapper-scroll'>"
        f"<div id='{div_id}'></div>"
        f"<script>"
        f"(function(){{var d={json_data};"
        f"Plotly.newPlot('{div_id}',d.data,d.layout,{{responsive:true}});"
        f"}})();"
        f"</script>"
        f"</div>"
    )
    return html


def build_density_data(lengths, quals, stats, bin_size, y_bins=50):
    """
    Computes the 2D density matrix (length × quality) for heatmaps.

    Tunable parameter:
      bin_size: bin size in bp on the X-axis (length)
                 → passed via the -b / --bin_size command-line argument
                 → default value: 1000 bp
                 → use 500 for short reads (<10 kb),
                     5000 for very long reads

      30 fixed bins on the Y-axis (quality): change the '30' in np.histogram2d
      to increase or decrease quality resolution.

    Returns (H.T, xedges, yedges): transposed matrix for Plotly + bin edges.
    """
    if lengths:
        min_len, max_len = min(lengths), max(lengths)
        if min_len == max_len:
            max_len = min_len + bin_size  # avoid an empty histogram
        x_bins = np.arange(min_len, max_len + bin_size, bin_size)
        if len(x_bins) < 2:
            x_bins = np.array([min_len, max_len])
        H, xedges, yedges = np.histogram2d(lengths, quals, bins=(x_bins, y_bins))  # ← Y bins (quality resolution)
    else:
        # Empty input: return a zero matrix as a safe fallback
        H = np.zeros((y_bins, 1))
        xedges = np.array([0, bin_size])
        yedges = np.linspace(0, 50, y_bins + 1)
    return H.T, xedges, yedges


# ==============================================================================
# SECTION 2b — HTML SIZE UTILITIES
# ==============================================================================

_plotly_div_counter = [0]  # mutable counter for unique div IDs

def fig_to_compact(fig, include_plotlyjs=False, xranges=None):
    """
    Converts a Plotly figure to a compact HTML fragment.

    Strategy (vs fig.to_html):
      - fig.to_json(pretty=False) produces raw JSON data only (~3-10x smaller
        than fig.to_html which embeds a full React component tree).
      - A single <script>Plotly.newPlot(...)</script> call renders the figure.
      - Plotly JS is included once via CDN (include_plotlyjs=True on first call).

    xranges (optional dict): axis ranges forced AFTER rendering via Plotly.relayout().
      e.g. {'xaxis': [500, 30000], 'xaxis2': [500, 30000], 'xaxis3': [8, 25]}
      More reliable than setting range in layout JSON, which Plotly.js may override
      during its autorange recalculation at render time.

    The result is wrapped in a resizable .plot-wrapper div.
    """
    import json as _json
    _plotly_div_counter[0] += 1
    div_id = f"plotly-fig-{_plotly_div_counter[0]}"
    json_data = fig.to_json(pretty=False, remove_uids=True)
    cdn_tag = ('<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>\n'
               if include_plotlyjs else "")
    if xranges:
        relayout_arg = {f"{ax}.range": rng for ax, rng in xranges.items()}
        relayout_js = f"Plotly.relayout('{div_id}',{_json.dumps(relayout_arg)});"
    else:
        relayout_js = ""
    html = (
        f"{cdn_tag}"
        f"<div class='plot-wrapper'>"
        f"<div id='{div_id}' style='width:100%;height:100%;'></div>"
        f"<script>"
        f"(function(){{var d={json_data};"
        f"Plotly.newPlot('{div_id}',d.data,d.layout,{{responsive:true}}).then(function(){{{relayout_js}}});"
        f"}})();"
        f"</script>"
        f"</div>"
    )
    return html


def minify_html(html):
    """
    Lightweight HTML minification (no external dependency).
    Removes: redundant whitespace between tags, HTML comments,
    leading/trailing spaces on lines.
    Does NOT touch content inside <script> or <style> blocks.
    Typical reduction: 5-15% on top of JSON compactness.
    """
    # Collapse runs of whitespace between tags (but not inside them)
    html = re.sub(r'>\s+<', '><', html)
    # Remove HTML comments (but not IE conditionals)
    html = re.sub(r'<!--(?!\[).*?-->', '', html, flags=re.DOTALL)
    # Strip leading/trailing whitespace on each line
    html = '\n'.join(line.strip() for line in html.splitlines() if line.strip())
    return html


# ==============================================================================
# SECTION 3 — FIGURE BUILDERS
# ==============================================================================

def build_overlay_figure(results_data, title_suffix="Raw", xmin_len=None, xmin_qual=None):
    """
    Builds the overlaid distribution curve figure (one curve per file).

    Layout (3 rows × 1 column):
      Row 1: Read length distribution
      Row 2: Cumulative bases per length bin (in Mb)
      Row 3: Quality distribution

    ┌─────────────────────────────────────────────────────────┐
    │  TUNABLE PARAMETERS                                 │
    │                                                         │
    │  In make_subplots():                                 │
    │    vertical_spacing = 0.10  → gap between panels   │
    │    vertical_spacing   = 0.18  → gap between panels       │
    │                                                         │
    │  In fig.update_layout():                             │
    │    height = 900  → total figure height in px  │
    │    width  = 1000 → total figure width in px     │
    │                                                         │
    │  In histogram_curve():                               │
    │    nbins=200 (length)  → histogram resolution  │
    │    nbins=100 (quality) → histogram resolution  │
    │                                                         │
    │  Curve thickness:                                │
    │    line=dict(width=2) → increase to thicken, decrease to thin  │
    └─────────────────────────────────────────────────────────┘
    """
    n = len(results_data)
    colors = px_colors(n)
    traces_len, traces_bases, traces_qual = [], [], []

    for idx, (fname, lengths, quals, stats) in enumerate(results_data):
        color = colors[idx]

        # --- Read length distribution curve
        # x_range fixes bin computation to [xmin_len, xmax_len] so no x=0 point
        # is generated, making the axis start exactly at min_len for filtered data.
        len_range  = (xmin_len,  float(max(lengths)))  if xmin_len  and lengths else None
        qual_range = (xmin_qual, float(max(quals)))     if xmin_qual and quals   else None
        x_len, y_len = histogram_curve(lengths, nbins=200, x_range=len_range)
        hover_len = [f"{y} reads<br>Length: {x:.0f}<br>File: {stats['file']}"
                     for x, y in zip(x_len, y_len)]
        traces_len.append(go.Scatter(
            x=x_len, y=y_len, mode="lines",
            line=dict(color=color, width=2),  # ← line width
            name=stats['file'],
            hovertext=hover_len, hoverinfo="text"
        ))

        # --- Cumulative bases curve (length × count, in Mb)
        y_bases = [x * y / 1e6 for x, y in zip(x_len, y_len)]
        hover_bases = [f"{val:.2f} Mb<br>Length: {x:.0f}<br>File: {stats['file']}"
                       for x, val in zip(x_len, y_bases)]
        traces_bases.append(go.Scatter(
            x=x_len, y=y_bases, mode="lines",
            line=dict(color=color, width=2),
            hovertext=hover_bases, hoverinfo="text",
            showlegend=False
        ))

        # --- Quality distribution curve
        x_qual, y_qual = histogram_curve(quals, nbins=100, x_range=qual_range)
        hover_qual = [f"{y} reads<br>Quality: {x:.2f}<br>File: {stats['file']}"
                      for x, y in zip(x_qual, y_qual)]
        traces_qual.append(go.Scatter(
            x=x_qual, y=y_qual, mode="lines",
            line=dict(color=color, width=2),
            hovertext=hover_qual, hoverinfo="text",
            showlegend=False
        ))

    # Compute axis ranges from actual data, skipping empty traces.
    # Files with 0 reads after filtering produce traces with x=[] which would
    # cause min([]) to crash or return 0, pulling the axis back to 0.
    all_x_len  = [x for t in traces_len  if t.x is not None and len(t.x) > 0 for x in t.x]
    all_x_qual = [x for t in traces_qual if t.x is not None and len(t.x) > 0 for x in t.x]
    # Fallback to [0, 1] if all traces are empty (edge case: all files filtered out)
    if all_x_len:
        xmin_len_data  = float(min(all_x_len))
        xmax_len_data  = float(max(all_x_len))
    else:
        xmin_len_data, xmax_len_data = 0, 1
    if all_x_qual:
        xmin_qual_data = float(min(all_x_qual))
        xmax_qual_data = float(max(all_x_qual))
    else:
        xmin_qual_data, xmax_qual_data = 0, 1

    # --- 3-row × 1-column vertical layout
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            "Read length distribution",
            "Cumulative bases per length bin",
            "Quality distribution",
        ],
        vertical_spacing=0.10,    # ← vertical gap between the 3 panels
    )

    for trace in traces_len:
        fig.add_trace(trace, row=1, col=1)
    for trace in traces_bases:
        fig.add_trace(trace, row=2, col=1)
    for trace in traces_qual:
        fig.add_trace(trace, row=3, col=1)

    # X/Y axis titles
    fig.update_xaxes(title_text="Length (bases)", row=1, col=1)
    fig.update_yaxes(title_text="Number of reads", row=1, col=1)
    fig.update_xaxes(title_text="Length (bases)", row=2, col=1)
    fig.update_yaxes(title_text="Bases (Mb)", row=2, col=1)
    fig.update_xaxes(title_text="Mean quality", row=3, col=1)
    fig.update_yaxes(title_text="Number of reads", row=3, col=1)

    # Ghost anchor traces: invisible points placed exactly at xmin force Plotly
    # to include xmin in the computed range. This is the only reliable method —
    # rangemode, autorange=False, and relayout all fail because Plotly.js
    # recomputes the range client-side and snaps to 0 by default.
    # opacity=0, hoverinfo='skip', showlegend=False → completely invisible.
    if xmin_len_data > 0:
        fig.add_trace(go.Scatter(
            x=[xmin_len_data, xmax_len_data], y=[0, 0],
            mode='markers', opacity=0,
            hoverinfo='skip', showlegend=False,
            marker=dict(size=0)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[xmin_len_data, xmax_len_data], y=[0, 0],
            mode='markers', opacity=0,
            hoverinfo='skip', showlegend=False,
            marker=dict(size=0)
        ), row=2, col=1)
    if xmin_qual_data > 0:
        fig.add_trace(go.Scatter(
            x=[xmin_qual_data, xmax_qual_data], y=[0, 0],
            mode='markers', opacity=0,
            hoverinfo='skip', showlegend=False,
            marker=dict(size=0)
        ), row=3, col=1)

    fig.update_layout(
        xaxis =dict(rangemode='normal'),
        xaxis2=dict(rangemode='normal'),
        xaxis3=dict(rangemode='normal'),
        yaxis =dict(rangemode='tozero'),
        yaxis2=dict(rangemode='tozero'),
        yaxis3=dict(rangemode='tozero'),
    )

    fig.update_layout(
        height=900,                              # ← total figure height in px (3 panels)
        width=1000,                              # ← total figure width in px
        title_text=f"Distributions – {title_suffix}",
        hovermode="closest"
    )
    return fig


def build_heatmap_figure(results_data, bin_size, title_suffix="Raw", xmin_len=None, y_bins=50):
    """
    Builds stacked length vs quality heatmaps, one panel per file.
    All panels share the same X-axis for easy visual comparison.

    ┌─────────────────────────────────────────────────────────┐
    │  TUNABLE PARAMETERS                                 │
    │                                                         │
    │  Panel dimensions:                                  │
    │    panel_h   = 90   → height of each heatmap panel in px  │
    │    spacing_h = 10   → whitespace between panels in px  │
    │                                                         │
    │  Horizontal filename label (annotation on the left):     │
    │    label_x = -0.05  → X position in paper coordinates  │
    │                        (negative = left of plot area) │
    │                        more negative = further left    │
    │    font size=12     → font size of the filename label  │
    │    "Quality" label  → small grey text below the filename,    │
    │                        same position, font size=9      │
    │                                                         │
    │  Left margin (space for annotations):          │
    │    margin=dict(l=320, ...) → increase if long filenames     │
    │    overflow out of the figure                   │
    │    Decrease to bring heatmaps closer to the left edge │
    │                                                         │
    │  Heatmap resolution:                             │
    │    bin_size (via -b) → X-axis resolution                │
    │    50 bins on Y-axis → change in build_density_data │
    │                                                         │
    │  Colour scale:                                  │
    │    colorscale="Viridis" → alternatives: "Plasma", "Inferno", │
    │    "Magma", "Hot", "YlOrRd", "Blues", "RdBu"           │
    │                                                         │
    │  Reference lines:                                     │
    │    N50         : color="white", dash="dash", width=1.5    │
    │    Median qual : color="white", dash="dot",  width=1.5    │
    │                                                         │
    │  Total width: width=1000                           │
    └─────────────────────────────────────────────────────────┘
    """
    results_data = list(results_data)  # ensure we can iterate multiple times

    # ┌──────────────────────────────────────────────────────────────────┐
    # │  TUNABLE PARAMETER                                               │
    # │    xmax_percentile = 99.5  → use 95 for stricter capping,         │
    # │                            100 to disable capping (absolute max) │
    # └──────────────────────────────────────────────────────────────────┘
    xmax_percentile = 99.5  # ← percentile cap applied BEFORE building density matrices

    # Step 1: compute global_xmax FIRST, before building density matrices.
    # This is critical: if we cap the X range AFTER building H, the reads
    # beyond the cap are already baked into H but invisible (outside the
    # displayed range), which does NOT fix the scale problem.
    # By capping lengths before histogram2d, the bins are built within the
    # visible range and all data is actually displayed.
    all_lengths_flat = []
    for (fname, lengths, quals, stats) in results_data:
        all_lengths_flat.extend(lengths)

    global_xmin = min(lengths[0] if lengths else 0
                      for (_, lengths, _, _) in results_data) if results_data else 0
    if all_lengths_flat:
        global_xmax = float(np.percentile(all_lengths_flat, xmax_percentile))
    else:
        global_xmax = 1

    # Step 2: build density matrices with lengths capped at global_xmax.
    # Reads beyond global_xmax are excluded from the histogram so their bins
    # don't exist and can't distort the colour scale.
    colors = px_colors(len(results_data))
    density_data = []

    for idx, (fname, lengths, quals, stats) in enumerate(results_data):
        # Cap lengths (and matching quals) to global_xmax before histogram
        capped = [(l, q) for l, q in zip(lengths, quals) if l <= global_xmax]
        cap_lengths = [l for l, q in capped]
        cap_quals   = [q for l, q in capped]
        H, xedges, yedges = build_density_data(cap_lengths, cap_quals, stats, bin_size, y_bins=y_bins)
        density_data.append((H, xedges, yedges, stats, colors[idx]))

    n_density = len(density_data)

    # Compute total figure height
    panel_h   = 90   # ← height of each heatmap panel in px
    spacing_h = 10   # ← whitespace between panels in px
    total_h   = n_density * panel_h + (n_density - 1) * spacing_h + 80

    fig = make_subplots(
        rows=n_density, cols=1,
        shared_xaxes=True,
        vertical_spacing=spacing_h / total_h,
        row_heights=[1.0] * n_density,
        subplot_titles=None
    )

    for idx, (H, xedges, yedges, stats, color) in enumerate(density_data):
        row_idx = 1 + idx
        xc = (xedges[:-1] + xedges[1:]) / 2
        yc = (yedges[:-1] + yedges[1:]) / 2

        # --- Main heatmap trace
        # Per-panel normalisation: each file uses its own colour scale,
        # preventing a heavily sequenced file (high counts) from visually
        # dominating all others (white backgrounds on low-count panels).
        # zmin=0 ensures empty bins stay black (bottom of the colour scale).
        z_nonzero = H[H > 0]
        zmax_panel = float(np.percentile(z_nonzero, 99.5)) if z_nonzero.size > 0 else 1
        fig.add_trace(go.Heatmap(
            x=xc, y=yc, z=H,
            colorscale="Viridis",  # ← colour scale
            showscale=False,
            zmin=0,               # ← empty bins = black (bottom of scale)
            zmax=zmax_panel,      # ← max at the 99.5th percentile of non-zero counts
                                  #   → each panel has its own scale
                                  #   → use zmax=H.max() for an absolute shared scale
            hovertemplate=(
                f"<b>{stats['file']}</b><br>"
                "Length: %{x:.0f} bp<br>"
                "Quality: %{y:.2f}<br>"
                "Count: %{z}<extra></extra>"
            )
        ), row=row_idx, col=1)

        # --- Vertical N50 reference line
        if stats['n50'] > 0:
            fig.add_trace(go.Scatter(
                x=[stats['n50'], stats['n50']], y=[yedges[0], yedges[-1]],
                mode="lines",
                line=dict(color="white", dash="dash", width=1.5),  # ← N50 line style
                showlegend=False, hoverinfo="skip"
            ), row=row_idx, col=1)

        # --- Horizontal median quality reference line
        if stats['median_quality'] > 0:
            fig.add_trace(go.Scatter(
                x=[global_xmin, global_xmax],
                y=[stats['median_quality'], stats['median_quality']],
                mode="lines",
                line=dict(color="white", dash="dot", width=1.5),  # ← median quality line style
                showlegend=False, hoverinfo="skip"
            ), row=row_idx, col=1)

        # --- N50 annotation label
        if stats['n50'] > 0:
            fig.add_annotation(
                x=stats['n50'], y=yedges[-1],
                text=f"N50:{stats['n50']:,}",
                showarrow=False,
                font=dict(color="white", size=10),
                bgcolor="rgba(0,0,0,0.4)",
                xanchor="left", yanchor="top",
                row=row_idx, col=1
            )

        # --- Median quality annotation label
        if stats['median_quality'] > 0:
            fig.add_annotation(
                x=global_xmax, y=stats['median_quality'],
                text=f"Q\u0303:{stats['median_quality']:.1f}",
                showarrow=False,
                font=dict(color="white", size=10),
                bgcolor="rgba(0,0,0,0.4)",
                xanchor="right", yanchor="bottom",
                row=row_idx, col=1
            )

        # --- Y-axis: short "Quality" label (filename is shown as an annotation)
        fig.update_yaxes(
            title_text="Quality",               # ← short Y-axis label
            title_font=dict(size=9, color="#888"),
            title_standoff=4,
            tickfont=dict(size=9),
            range=[yedges[0], yedges[-1]],
            row=row_idx, col=1
        )

        # --- Horizontal filename label to the left of each panel
        # Technique: annotation in "paper" X coordinates (0 = left edge of figure),
        # and "data" Y coordinates (centred on the current panel).
        #
        # To reference the correct Y-axis for each panel (row_idx):
        #   row 1 → yref="y",  row 2 → yref="y2",  row 3 → yref="y3",  etc.
        yref_str = "y" if row_idx == 1 else f"y{row_idx}"
        y_center = (yedges[0] + yedges[-1]) / 2   # vertical centre of the panel

        # Filename label (bold, coloured per file)
        n_reads_str = f"{stats['n_reads']:,}"
        fig.add_annotation(
            xref="paper", yref=yref_str,
            x=-0.05,           # ← X position: negative = left of the plot area
                               #   more negative (-0.05) = further left
                               #   less negative (-0.005) = close to plot edge
            y=y_center,        # vertically centred on the panel
            text=f"<b>{stats['file']}</b><br><span style='font-size:9px;color:#888'>{n_reads_str} reads</span>",
            showarrow=False,
            font=dict(size=12, color=color),   # ← label font size and colour
            xanchor="right",   # text ends at x (extends leftward)
            yanchor="middle",
            textangle=0,       # ← 0 = horizontal (change to -90 for vertical)
        )

        # Small grey "Quality" label below the filename
        fig.add_annotation(
            xref="paper", yref=yref_str,
            x=-0.01,           # same X position as the filename
            y=yedges[0],       # bottom of the panel
            text="<i style='color:#aaa;font-size:9px'>Quality</i>",
            showarrow=False,
            font=dict(size=9, color="#aaa"),
            xanchor="right",
            yanchor="bottom",
            textangle=0,
        )

    fig.update_xaxes(title_text="Read length (bp)", row=n_density, col=1)
    # Apply xmin_len if provided: start X-axis at filter threshold, not at 0
    effective_xmin = xmin_len if xmin_len is not None else global_xmin
    fig.update_xaxes(range=[effective_xmin, global_xmax])

    fig.update_layout(
        height=total_h,
        width=1000,              # ← total figure width in px
        title_text=f"Length vs Quality Heatmaps \u2013 {title_suffix}",
        hovermode="closest",
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=320, r=40, t=60, b=50),
        # ↑ l=320: left margin in px to accommodate filename annotations
        #   Increase if long filenames overflow (e.g. l=400)
        #   Decrease to bring heatmaps closer to the left edge (ex: l=150)
    )
    return fig


def _fmt_reads(n):
    """Format read count: 1.21K, 45.3K, 1.20M"""
    if n >= 1e6:   return f"{n/1e6:.2g}M"
    if n >= 1000:  return f"{n/1e3:.3g}K"
    return str(int(n))

def _fmt_bases_bp(bp):
    """Format median/N50 length in bp: 1.55Kb, 450bp"""
    if bp >= 1e6:  return f"{bp/1e6:.2f}Mb"
    if bp >= 1000: return f"{bp/1e3:.2f}Kb"
    return f"{int(bp)}bp"

def _fmt_bases_mb(mb):
    """Format total bases in Mb: 1.2Mb, 950Kb"""
    if mb >= 1000: return f"{mb/1e3:.1f}Gb"
    if mb >= 1:    return f"{mb:.1f}Mb"
    return f"{mb*1e3:.1f}Kb"

def _fmt_qual(q):
    """Format quality score: Q22.40"""
    return f"Q{q:.1f}"

def build_hist_figure(all_stats, title_suffix="Raw"):
    """
    Builds the 4 summary histogram panels (horizontal bars, 4 rows).

    Layout:
      Panel 1: Number of reads
      Panel 2: Total bases (Mb)
      Panel 3: Median length + N50 (overlaid)
      Panel 4: Median quality

    Note: the "Hide Outliers" button does NOT affect the summary histograms,
    because they display aggregated statistics (median, N50, total)
    that are not sensitive to individual extreme values.

    ┌─────────────────────────────────────────────────────────┐
    │  TUNABLE PARAMETERS                                 │
    │                                                         │
    │  Bar width:                                   │
    │    Scales automatically with file count                        │
    │    → Adjust bar_h for thicker/thinner bars            │
    │                                                         │
    │  N50 transparency:                                     │
    │    opacity=0.45 → 0.0 invisible, 1.0 opaque            │
    │                                                         │
    │  Spacing:                                           │
    │    vertical_spacing = auto (80px fixed gap)                            │
    │    vertical_spacing   = 0.18                            │
    │                                                         │
    │  Size: height=dynamic, width=1000                       │
    └─────────────────────────────────────────────────────────┘
    """
    files  = [s['file'] for s in all_stats]
    colors = px_colors(len(all_stats))
    n      = max(len(files), 1)

    # Height: scales with number of files so bars remain readable.
    # ┌──────────────────────────────────────────────────────────┐
    # │  bar_h     = 22   → px per bar (horizontal bars)        │
    # │  min_h     = 300  → minimum panel height                │
    # │  panel_h   = max(min_h, n * bar_h)                      │
    # │  Increase bar_h for thicker bars, decrease for denser   │
    # └──────────────────────────────────────────────────────────┘
    bar_h   = 22   # ← px per sample per panel
    min_h   = 300  # ← minimum panel height in px
    panel_h = max(min_h, n * bar_h)
    total_h = panel_h * 4 + 220  # 4 panels + titles/spacing

    # Left margin: accommodate longest filename
    max_name_len = max((len(s['file']) for s in all_stats), default=10)
    margin_left  = max(120, min(max_name_len * 7, 400))
    # ↑ ~7 px per character, capped at 400 px

    fig_hist = make_subplots(
        rows=4, cols=1,
        subplot_titles=[
            "Number of reads",
            "Total bases (Mb)",
            "Median length & N50 (bp)",
            "Median quality",
        ],
        vertical_spacing=max(0.02, 80 / total_h),  # ← fixed ~80px gap between panels
    )

    # Horizontal bars: y=files, x=values
    # Names appear on the Y-axis — always fully readable regardless of sample count.

    # Panel 1 — Number of reads
    fig_hist.add_trace(go.Bar(
        y=files, x=[s['n_reads'] for s in all_stats],
        orientation='h', marker_color=colors,
        showlegend=False, name="Reads",
        text=[_fmt_reads(s['n_reads']) for s in all_stats],
        textposition='inside', insidetextanchor='middle',
        textfont=dict(size=11, color='black'),
    ), row=1, col=1)

    # Panel 2 — Total bases (Mb)
    fig_hist.add_trace(go.Bar(
        y=files, x=[s['total_bases'] / 1e6 for s in all_stats],
        orientation='h', marker_color=colors,
        showlegend=False, name="Bases (Mb)",
        text=[_fmt_bases_mb(s['total_bases'] / 1e6) for s in all_stats],
        textposition='inside', insidetextanchor='middle',
        textfont=dict(size=11, color='black'),
    ), row=2, col=1)

    # Panel 3 — Median length (solid) + N50 (transparent, overlaid)
    fig_hist.add_trace(go.Bar(
        y=files, x=[s['median_length'] for s in all_stats],
        orientation='h', marker_color=colors,
        showlegend=False, name="Median length",
        text=[_fmt_bases_bp(s['median_length']) for s in all_stats],
        textposition='inside', insidetextanchor='middle',
        textfont=dict(size=11, color='black'),
    ), row=3, col=1)
    fig_hist.add_trace(go.Bar(
        y=files, x=[s['n50'] for s in all_stats],
        orientation='h', marker_color=colors,
        opacity=0.45,  # ← N50 bar transparency
        showlegend=False, name="N50",
        text=[_fmt_bases_bp(s['n50']) for s in all_stats],
        textposition='inside', insidetextanchor='middle',
        textfont=dict(size=11, color='black'),
    ), row=3, col=1)

    # Panel 4 — Median quality
    fig_hist.add_trace(go.Bar(
        y=files, x=[s['median_quality'] for s in all_stats],
        orientation='h', marker_color=colors,
        showlegend=False, name="Median quality",
        text=[_fmt_qual(s['median_quality']) for s in all_stats],
        textposition='inside', insidetextanchor='middle',
        textfont=dict(size=11, color='black'),
    ), row=4, col=1)

    # Show X-axis on both top and bottom of every panel for readability
    # with many samples (horizontal bars can be very tall).
    for row in range(1, 5):
        fig_hist.update_xaxes(side="bottom", showticklabels=True, row=row, col=1)
        fig_hist.update_xaxes(mirror=True, row=row, col=1)
    # Mirror X-axis to top and bottom of each panel for readability
    # when there are many samples and the figure is tall/scrollable.
    for row, title in [(1, "Reads"), (2, "Mb"), (3, "Length (bp)"), (4, "Quality")]:
        fig_hist.update_xaxes(
            title_text=title,
            showticklabels=True,
            mirror=True,   # ← duplicate ticks/labels on the opposite side
            row=row, col=1
        )

    fig_hist.update_layout(
        height=total_h,         # ← scales with number of files
        width=1000,             # ← total width in px
        showlegend=False,
        title_text=f"Summary Histograms \u2013 {title_suffix}",
        barmode="overlay",      # N50 and median overlaid in panel 3
        margin=dict(l=margin_left, r=40, t=60, b=40),
        # ↑ left margin adapts to longest filename
    )
    return fig_hist


def build_table_html(all_stats, table_id):
    """
    Generates the HTML cumulative statistics table.

    ┌─────────────────────────────────────────────────────────┐
    │  TUNABLE PARAMETERS                                 │
    │                                                         │
    │  Header colour: background:#2c3e50             │
    │  Font size: font-size:13px                 │
    │  Row striping: #f9f9f9 / #ffffff              │
    └─────────────────────────────────────────────────────────┘
    """
    headers = [
        "File", "n_reads", "Total bases", "Avg length", "Min", "Max",
        "Median", "75th perc", "N50",
        "Avg qual", "Median qual", "25th perc", "75th perc", "Min qual", "Max qual"
    ]
    html = (f"<table id='{table_id}' border='1' "
            f"style='border-collapse:collapse;width:100%;font-size:13px;'><tr>")
    for h in headers:
        html += f"<th style='background:#2c3e50;color:white;padding:6px 10px;'>{h}</th>"
    html += "</tr>"
    for i, stats in enumerate(all_stats):
        bg = "#f9f9f9" if i % 2 == 0 else "#ffffff"
        row = [
            stats['file'], stats['n_reads'], stats['total_bases'],
            f"{stats['avg_length']:.2f}", stats['min_length'], stats['max_length'],
            stats['median_length'], stats['perc75_length'], stats['n50'],
            f"{stats['avg_quality']:.2f}", f"{stats['median_quality']:.2f}",
            f"{stats['perc25_quality']:.2f}", f"{stats['perc75_quality']:.2f}",
            f"{stats['min_quality']:.2f}", f"{stats['max_quality']:.2f}"
        ]
        html += (f"<tr style='background:{bg};'>"
                 + "".join(f"<td style='padding:5px 10px;'>{v}</td>" for v in row)
                 + "</tr>")
    html += "</table>"
    return html


# ==============================================================================
# SECTION 3b — MODULE-LEVEL HELPERS FOR MULTIPROCESSING
# (must be at module level, not inside main(), to be picklable by Pool)
# ==============================================================================

def _make_filtered(args_tuple):
    """Worker function: apply quality/length filters to one file's data."""
    fname, lengths, quals, _, min_len, max_len, min_qual, max_qual = args_tuple
    fl, fq = apply_filters(lengths, quals, min_len, max_len, min_qual, max_qual)
    return (fname, fl, fq, compute_stats(fname, fl, fq))


def _make_no_outliers(args_tuple):
    """Worker function: apply outlier filter to one file's data."""
    fname, lengths, quals, _, percentile = args_tuple
    fl, fq, threshold = apply_outlier_filter(lengths, quals, percentile)
    return (fname, fl, fq, compute_stats(fname, fl, fq), threshold)


# ==============================================================================
# SECTION 4 — MAIN FUNCTION
# ==============================================================================

def main():
    """
    Orchestrates the full pipeline:
      1. Parse command-line arguments
      2. Read input files (parallel or sequential)
      3. Apply quality/length filters → "Filtered" dataset
      4. Apply outlier filter (length percentile) → "No Outliers" dataset
      5. Build all Plotly figures
      6. Generate the HTML report with:
           - Per-section toggle buttons: Raw / Filtered / No Outliers
             (visible per section)
           - Toggle buttons for histograms/table: Raw ↔ Filtered
             (heatmaps and curves also have No Outliers mode)
      7. Export statistics to TXT files
    """
    args   = parse_args()
    # threads=0 means "use all available CPUs" (both normal and low_memory modes)
    n_cpus = cpu_count() if args.threads == 0 else min(args.threads, cpu_count())
    if args.low_memory:
        print(f"⚠️  Low-memory mode enabled: sequential file processing, "
              f"figures streamed to disk one at a time. "
              f"Using {n_cpus} thread(s).", flush=True)
    else:
        print(f"Using {n_cpus} CPU thread(s)", flush=True)

    filter_active = any(
        x is not None for x in [args.min_len, args.max_len, args.min_qual, args.max_qual]
    )

    if args.low_memory:
        # -----------------------------------------------------------------------
        # LOW-MEMORY PATH
        # Files are read and processed one at a time.
        # Raw read data (lengths, quals) is freed after each file is processed;
        # only stats dicts and density matrices are kept in RAM.
        # -----------------------------------------------------------------------
        results_raw         = []
        results_filtered    = []
        results_no_outliers = []
        outlier_thresholds  = {}

        for i, fname in enumerate(args.input):
            print(f"  [{i+1}/{len(args.input)}] Reading {fname} ...", flush=True)
            fname, lengths, quals, stats = process_file(fname, n_threads=n_cpus)

            # Raw entry (keep lengths/quals for figure building later)
            results_raw.append((fname, lengths, quals, stats))

            # Filtered
            fl, fq = apply_filters(lengths, quals,
                                   args.min_len, args.max_len,
                                   args.min_qual, args.max_qual)
            fstats = compute_stats(fname, fl, fq)
            results_filtered.append((fname, fl, fq, fstats))
            del fl, fq  # free filtered copy immediately

            # No Outliers
            nl, nq, threshold = apply_outlier_filter(lengths, quals,
                                                     args.outlier_percentile)
            nstats = compute_stats(fname, nl, nq)
            results_no_outliers.append((fname, nl, nq, nstats))
            outlier_thresholds[nstats['file']] = threshold
            del nl, nq  # free outlier-filtered copy immediately

            # Free raw read data — stats dict and density matrix are enough
            # from this point for figure building
            del lengths, quals

    else:
        # -----------------------------------------------------------------------
        # NORMAL PATH — parallel reading and processing
        # -----------------------------------------------------------------------
        from functools import partial
        _process = partial(process_file, n_threads=max(1, n_cpus // len(args.input)))
        with Pool(n_cpus) as pool:
            results = pool.map(_process, args.input)

        # Dataset 1: Raw
        results_raw = [(fname, lengths, quals, stats)
                       for fname, lengths, quals, stats in results]

        # Datasets 2 & 3: Filtered and No Outliers in parallel
        filt_args = [
            (fname, lengths, quals, stats,
             args.min_len, args.max_len, args.min_qual, args.max_qual)
            for fname, lengths, quals, stats in results
        ]
        nout_args = [
            (fname, lengths, quals, stats, args.outlier_percentile)
            for fname, lengths, quals, stats in results
        ]

        with Pool(n_cpus) as pool:
            filt_results = pool.map(_make_filtered,    filt_args)
            nout_results = pool.map(_make_no_outliers, nout_args)

        results_filtered    = [(f, l, q, s)     for f, l, q, s    in filt_results]
        results_no_outliers = [(f, l, q, s)     for f, l, q, s, _ in nout_results]
        outlier_thresholds  = {s['file']: thr   for _, _, _, s, thr in nout_results}

    all_stats_raw         = [s for _, _, _, s in results_raw]
    all_stats_filtered    = [s for _, _, _, s in results_filtered]
    all_stats_no_outliers = [s for _, _, _, s in results_no_outliers]

    # --- Export TXT statistics files
    base_out     = os.path.splitext(args.output)[0]
    table_header = [
        "File", "n_reads", "Total bases", "Avg length", "Min", "Max", "Median",
        "75th perc", "N50", "Avg qual", "Median qual", "25th perc", "75th perc",
        "Min qual", "Max qual"
    ]

    def write_txt(all_stats, suffix):
        """Writes one statistics dataset to a TSV file."""
        path = f"{base_out}_{suffix}.txt"
        with open(path, "w") as fh:
            fh.write("\t".join(table_header) + "\n")
            for stats in all_stats:
                row = [
                    stats['file'], stats['n_reads'], stats['total_bases'],
                    f"{stats['avg_length']:.2f}", stats['min_length'], stats['max_length'],
                    stats['median_length'], stats['perc75_length'], stats['n50'],
                    f"{stats['avg_quality']:.2f}", f"{stats['median_quality']:.2f}",
                    f"{stats['perc25_quality']:.2f}", f"{stats['perc75_quality']:.2f}",
                    f"{stats['min_quality']:.2f}", f"{stats['max_quality']:.2f}"
                ]
                fh.write("\t".join(str(v) for v in row) + "\n")
        return path

    txt_raw         = write_txt(all_stats_raw,         "cumulative_raw")
    txt_filtered    = write_txt(all_stats_filtered,    "cumulative_filtered")

    # --- Figure building
    # Y-axis bin count for heatmaps: 15 in --light_html mode, 30 otherwise
    y_bins = 25 if args.light_html else 50
    print("Building figures...", flush=True)

    if args.low_memory:
        # -------------------------------------------------------------------
        # LOW-MEMORY FIGURE PATH
        # Each figure is built, converted to HTML, then immediately deleted.
        # The HTML string is written directly to the output file — never more
        # than one Plotly figure object in RAM at a time.
        # -------------------------------------------------------------------
        def wrap_lm(fig, first=False):
            """Build compact HTML fragment and immediately free the figure object."""
            html = fig_to_compact(fig, include_plotlyjs=first)
            del fig
            return html

        # Build and convert each figure one at a time
        print("  building table_raw...",    flush=True); table_raw_html  = build_table_html(all_stats_raw,      "table_raw")
        print("  building table_filt...",   flush=True); table_filt_html = build_table_html(all_stats_filtered, "table_filtered")
        print("  building hist_raw...",     flush=True); hist_raw_html   = fig_to_compact_scroll(build_hist_figure(all_stats_raw,      "Raw"), include_plotlyjs=True)
        print("  building hist_filt...",    flush=True); hist_filt_html  = fig_to_compact_scroll(build_hist_figure(all_stats_filtered, "Filtered"))
        print("  building overlay_raw...",  flush=True)
        _fig, _xr = build_overlay_figure(results_raw,         "Raw")
        overlay_raw_html  = fig_to_compact(_fig, xranges=_xr); del _fig
        print("  building overlay_filt...", flush=True)
        _fig, _xr = build_overlay_figure(results_filtered,    "Filtered",     xmin_len=args.min_len, xmin_qual=args.min_qual)
        overlay_filt_html = fig_to_compact(_fig, xranges=_xr); del _fig
        print("  building overlay_nout...", flush=True)
        _fig, _xr = build_overlay_figure(results_no_outliers, "No Outliers")
        overlay_nout_html = fig_to_compact(_fig, xranges=_xr); del _fig
        print("  building heatmap_raw...",  flush=True); heatmap_raw_html  = wrap_lm(build_heatmap_figure(results_raw,         args.bin_size, "Raw",          y_bins=y_bins))
        print("  building heatmap_filt...", flush=True); heatmap_filt_html = wrap_lm(build_heatmap_figure(results_filtered,    args.bin_size, "Filtered",    xmin_len=args.min_len, y_bins=y_bins))
        print("  building heatmap_nout...", flush=True); heatmap_nout_html = wrap_lm(build_heatmap_figure(results_no_outliers, args.bin_size, "No Outliers", y_bins=y_bins))

    else:
        # -------------------------------------------------------------------
        # NORMAL FIGURE PATH — all figures built in parallel
        # -------------------------------------------------------------------
        figure_tasks = {
            "hist_raw":      (build_hist_figure,    [all_stats_raw,         "Raw"]),
            "hist_filt":     (build_hist_figure,    [all_stats_filtered,    "Filtered"]),
            "overlay_raw":   (build_overlay_figure, [results_raw,         "Raw"]),
            "overlay_filt":  (build_overlay_figure, [results_filtered,    "Filtered",   args.min_len, args.min_qual]),
            "overlay_nout":  (build_overlay_figure, [results_no_outliers, "No Outliers"]),
            "heatmap_raw":   (build_heatmap_figure, [results_raw,         args.bin_size, "Raw",          None,         y_bins]),
            "heatmap_filt":  (build_heatmap_figure, [results_filtered,    args.bin_size, "Filtered",   args.min_len, y_bins]),
            "heatmap_nout":  (build_heatmap_figure, [results_no_outliers, args.bin_size, "No Outliers", None,        y_bins]),
            "table_raw":     (build_table_html,     [all_stats_raw,         "table_raw"]),
            "table_filt":    (build_table_html,     [all_stats_filtered,    "table_filtered"]),
        }
        figures = {}
        fig_xranges = {}  # stores xranges for figures that return (fig, xranges)
        with ThreadPoolExecutor(max_workers=min(len(figure_tasks), n_cpus)) as ex:
            future_to_key = {
                ex.submit(fn, *fargs): key
                for key, (fn, fargs) in figure_tasks.items()
            }
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                result = future.result()
                # build_overlay_figure returns (fig, xranges); others return fig directly
                if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
                    figures[key], fig_xranges[key] = result
                else:
                    figures[key] = result
                print(f"  ✓ {key}", flush=True)

    # -------------------------------------------------------------------------
    # ACTIVE FILTER BANNER
    # -------------------------------------------------------------------------
    filter_info = []
    if args.min_len  is not None: filter_info.append(f"min_len = {args.min_len} bp")
    if args.max_len  is not None: filter_info.append(f"max_len = {args.max_len} bp")
    if args.min_qual is not None: filter_info.append(f"min_qual = {args.min_qual}")
    if args.max_qual is not None: filter_info.append(f"max_qual = {args.max_qual}")

    # Summary counts
    n_files       = len(args.input)
    total_raw     = sum(s['n_reads'] for s in all_stats_raw)
    total_nout    = sum(s['n_reads'] for s in all_stats_no_outliers)
    total_filt    = sum(s['n_reads'] for s in all_stats_filtered)

    filter_banner = (
        f"<div style='background:#e8f4fd;border-left:4px solid #3498db;"
        f"padding:10px 16px;margin:12px 0;border-radius:4px;font-size:14px;'>"
        f"<b>Files:</b> {n_files} &nbsp;|&nbsp; "
        f"<b>Reads — Raw:</b> {total_raw:,} &nbsp;|&nbsp; "
        f"<b>No Outliers (P{fmt_pct(args.outlier_percentile)}):</b> {total_nout:,} &nbsp;|&nbsp; "
        f"<b>Filtered:</b> {total_filt:,}"
        f"{'&nbsp;|&nbsp;<b>Active filters:</b> ' + ' | '.join(filter_info) if filter_info else ''}"
        f"</div>"
    )

    # =========================================================================
    # JAVASCRIPT — TOGGLE LOGIC
    # =========================================================================
    # Two toggle systems coexist:
    #
    #  1. PER-SECTION BUTTONS (Raw / No Outliers / Filtered)
    #     → Present on all sections (table, histograms, overlay, heatmaps)
    #     → For overlay and heatmaps: 3 modes (Raw / No Outliers / Filtered)
    #     → For table and histograms: 2 modes (Raw / Filtered only)
    #
    # How the systems interact:
    #     Clicking any section button switches that section to the chosen mode.
    #     The active button is highlighted; others are reset.
    # =========================================================================
    toggle_js = """
    <script>
    // -----------------------------------------------------------------------
    // toggleView: switch display mode for a section
    //   section: 'table', 'histograms', 'overlay', 'heatmaps'
    //   mode:    'raw', 'filtered', or 'no_outliers'
    //
    // Overlay and heatmap sections support 3 modes: raw / no_outliers / filtered
    // Table and histograms support 2 modes: raw / filtered
    // -----------------------------------------------------------------------
    function toggleView(section, mode) {
        // Get all possible divs for this section
        var rawEl  = document.getElementById(section + '_raw');
        var filtEl = document.getElementById(section + '_filtered');
        var noutEl = document.getElementById(section + '_no_outliers');

        // Get all possible buttons
        var btnRaw  = document.getElementById('btn_' + section + '_raw');
        var btnFilt = document.getElementById('btn_' + section + '_filtered');
        var btnNout = document.getElementById('btn_' + section + '_no_outliers');

        // Masquer toutes les versions
        if (rawEl)  rawEl.style.display  = 'none';
        if (filtEl) filtEl.style.display = 'none';
        if (noutEl) noutEl.style.display = 'none';

        // Désactiver tous les boutons
        if (btnRaw)  btnRaw.className  = 'toggle-btn';
        if (btnFilt) btnFilt.className = 'toggle-btn';
        if (btnNout) btnNout.className = 'toggle-btn-outlier';  // couleur verte inactive

        // Afficher la div et activer le bouton du mode sélectionné
        if (mode === 'raw' && rawEl) {
            rawEl.style.display = '';
            if (btnRaw) btnRaw.className = 'toggle-btn active';
        } else if (mode === 'filtered' && filtEl) {
            filtEl.style.display = '';
            if (btnFilt) btnFilt.className = 'toggle-btn active';
        } else if (mode === 'no_outliers' && noutEl) {
            noutEl.style.display = '';
            if (btnNout) btnNout.className = 'toggle-btn-outlier active';  // vert actif
        }
    }
    </script>
    """

    # =========================================================================
    # CSS GLOBAL DU RAPPORT HTML
    # =========================================================================
    # ┌─────────────────────────────────────────────────────────┐
    # │  TUNABLE PARAMETERS (CSS)                          │
    # │                                                         │
    # │  body max-width: 1100px → largeur max de la page       │
    # │                                                         │
    # │  .global-bar: top sticky bar for the global No Outliers button  │
    # │    background → bar background colour (#1a252f)                 │
    # │    padding    → inner spacing of the bar                        │
    # │    top        → stays at top on scroll (position: sticky)       │
    # │                                                                  │
    # │  .global-btn: style for the 3 global mode buttons               │
    # │    border-color / color → inactive button colour                │
    # │    .active: background → active button colour                   │
    # │    font-size: 15px → button text size                           │
    # │                                                                  │
    # │  .plot-wrapper: resizable figure container                       │
    # │    min-width / min-height → minimum dimensions                  │
    # └─────────────────────────────────────────────────────────┘
    toggle_css = """
    <style>
    body {
        font-family: Arial, sans-serif;
        max-width: 1100px;  /* page max-width — increase for wider layout */
        margin: auto;
        padding: 20px;
        background: #fafafa;
    }
    h1 { color: #2c3e50; }
    h2 { color: #34495e; margin-top: 40px; }

    /* ---- Boutons toggle par section ---- */
    .toggle-bar { margin: 12px 0; }
    .toggle-btn {
        display: inline-block;
        padding: 8px 22px;
        cursor: pointer;
        border: 2px solid #3498db;   /* ← section button colour (blue) */
        border-radius: 20px;
        font-size: 14px;
        font-weight: bold;
        color: #3498db;
        background: white;
        margin-right: 6px;
        transition: all 0.2s;
    }
    .toggle-btn.active {
        background: #3498db;
        color: white;
    }
    .toggle-btn:hover { background: #d6eaf8; }

    /* ---- Bouton No Outliers : couleur distincte verte ---- */
    /* ┌──────────────────────────────────────────────────────┐ */
    /* │  Pour changer la couleur du bouton No Outliers :    │ */
    /* │    border/color : couleur inactive                   │ */
    /* │    .active background : couleur active              │ */
    /* │  Valeurs actuelles : vert #27ae60                   │ */
    /* └──────────────────────────────────────────────────────┘ */
    .toggle-btn-outlier {
        display: inline-block;
        padding: 8px 22px;
        cursor: pointer;
        border: 2px solid #27ae60;   /* ← No Outliers button colour (green) */
        border-radius: 20px;
        font-size: 14px;
        font-weight: bold;
        color: #27ae60;
        background: white;
        margin-right: 6px;
        transition: all 0.2s;
    }
    .toggle-btn-outlier.active {
        background: #27ae60;         /* ← active No Outliers colour */
        color: white;
    }
    .toggle-btn-outlier:hover { background: #d5f5e3; }

    /* ---- Conteneur redimensionnable pour chaque figure ---- */
    .plot-wrapper {
        resize: both;           /* freely resizable */
        overflow: auto;
        min-width: 400px;       /* ← minimum width */
        min-height: 200px;      /* ← minimum height */
        width: 100%;
        border: 1px dashed #ccc;
        border-radius: 6px;
        padding: 4px;
        box-sizing: border-box;
        background: white;
    }
    .plot-wrapper .js-plotly-plot,
    .plot-wrapper .plotly-graph-div {
        width: 100% !important;
        height: 100% !important;
    }

    /* ---- Scrollable wrapper for histograms with many samples ---- */
    /* ┌──────────────────────────────────────────────────────────┐  */
    /* │  max-height: 600px → visible height before scrolling    │  */
    /* │  Increase for more visible rows, decrease to save space  │  */
    /* └──────────────────────────────────────────────────────────┘  */
    .plot-wrapper-scroll {
        overflow-y: auto;       /* vertical scrollbar when content exceeds max-height */
        overflow-x: auto;
        max-height: 600px;      /* ← visible window height in px */
        width: 100%;
        border: 1px dashed #ccc;
        border-radius: 6px;
        padding: 4px;
        box-sizing: border-box;
        background: white;
    }
    .plot-wrapper-scroll .js-plotly-plot,
    .plot-wrapper-scroll .plotly-graph-div {
        width: 100% !important;
    }
    </style>
    """

    # -------------------------------------------------------------------------
    # HTML HELPERS
    # -------------------------------------------------------------------------

    def toggle_bar_2(section):
        """
        Raw / Filtered button bar for a section (2 modes).
        Used for: statistics table and histograms.
        """
        return (
            f"<div class='toggle-bar'>"
            f"<span id='btn_{section}_raw' class='toggle-btn active' "
            f"onclick=\"toggleView('{section}','raw')\">&#128200; Raw</span>"
            f"<span id='btn_{section}_filtered' class='toggle-btn' "
            f"onclick=\"toggleView('{section}','filtered')\">&#128292; Filtered</span>"
            f"</div>"
        )

    def toggle_bar_3(section, percentile):
        """
        Button bar with 3 modes in order: Raw / No Outliers / Filtered.
        - Raw and Filtered: blue buttons (.toggle-btn)
        - No Outliers: green button (.toggle-btn-outlier)
        Used for the overlay and heatmap sections.

        Parameter:
          percentile: value shown on the button (e.g. .5 → 'No Outliers P99.5')
        """
        return (
            f"<div class='toggle-bar'>"
            # Bouton Raw (bleu)
            f"<span id='btn_{section}_raw' class='toggle-btn active' "
            f"onclick=\"toggleView('{section}','raw')\">&#128200; Raw</span>"
            # Bouton No Outliers (vert, couleur distincte)
            f"<span id='btn_{section}_no_outliers' class='toggle-btn-outlier' "
            f"onclick=\"toggleView('{section}','no_outliers')\">"
            f"&#128683; No Outliers <small>(P{fmt_pct(percentile)})</small></span>"
            # Bouton Filtered (bleu)
            f"<span id='btn_{section}_filtered' class='toggle-btn' "
            f"onclick=\"toggleView('{section}','filtered')\">&#128292; Filtered</span>"
            f"</div>"
        )

    def toggle_section_2(section, raw_html, filt_html):
        """Section with 2 modes: Raw (visible by default) and Filtered (hidden)."""
        return (
            toggle_bar_2(section)
            + f"<div id='{section}_raw'>{raw_html}</div>"
            + f"<div id='{section}_filtered' style='display:none;'>{filt_html}</div>"
        )

    def toggle_section_3(section, raw_html, filt_html, nout_html, percentile):
        """
        Section with 3 buttons in order: Raw / No Outliers / Filtered.
          - Raw         : visible by default
          - No Outliers : hidden, green button
          - Filtered    : hidden, blue button
        """
        return (
            toggle_bar_3(section, percentile)
            + f"<div id='{section}_raw'>{raw_html}</div>"
            + f"<div id='{section}_no_outliers' style='display:none;'>{nout_html}</div>"
            + f"<div id='{section}_filtered' style='display:none;'>{filt_html}</div>"
        )

    def wrap(fig, first=False, xranges=None):
        """
        Converts a Plotly figure to compact HTML using fig_to_compact().
        first=True: embeds the Plotly CDN script tag (only needed once per page).
        xranges: optional axis ranges forwarded to fig_to_compact for post-render relayout.
        Not used in --low_memory mode (figures are pre-converted and freed).
        """
        return fig_to_compact(fig, include_plotlyjs=first, xranges=xranges)

    # -------------------------------------------------------------------------
    # FINAL HTML ASSEMBLY
    # In --low_memory mode, HTML strings were already built above (figures freed).
    # In normal mode, figures are converted here via wrap().
    # -------------------------------------------------------------------------
    if args.low_memory:
        w_hist_raw      = hist_raw_html
        w_hist_filt     = hist_filt_html
        w_overlay_raw   = overlay_raw_html
        w_overlay_filt  = overlay_filt_html
        w_overlay_nout  = overlay_nout_html
        w_heatmap_raw   = heatmap_raw_html
        w_heatmap_filt  = heatmap_filt_html
        w_heatmap_nout  = heatmap_nout_html
    else:
        w_hist_raw      = fig_to_compact_scroll(figures["hist_raw"],  include_plotlyjs=True)
        w_hist_filt     = fig_to_compact_scroll(figures["hist_filt"])
        w_overlay_raw  = wrap(figures["overlay_raw"],  xranges=fig_xranges.get("overlay_raw"))
        w_overlay_filt = wrap(figures["overlay_filt"], xranges=fig_xranges.get("overlay_filt"))
        w_overlay_nout = wrap(figures["overlay_nout"], xranges=fig_xranges.get("overlay_nout"))
        w_heatmap_raw   = wrap(figures["heatmap_raw"])
        w_heatmap_filt  = wrap(figures["heatmap_filt"])
        w_heatmap_nout  = wrap(figures["heatmap_nout"])
        table_raw_html  = figures["table_raw"]
        table_filt_html = figures["table_filt"]

    print("Writing HTML report...", flush=True)
    html_full = (
        "<html><head><meta charset='utf-8'>"
        "<title>NAVIS - NAnopore Visualization & Interactive Statistiques</title>"
        + toggle_css
        + "</head><body>"

        + "<h1>NAVIS - NAnopore Visualization & Interactive Statistiques</h1>"
        + filter_banner

        # Section 1: Statistics table (2 modes: Raw / Filtered)
        + "<h2>&#128203; Cumulative Statistics Table</h2>"
        + toggle_section_2("table", table_raw_html, table_filt_html)

        # Section 2: Summary histograms (2 modes: Raw / Filtered)
        + "<h2>&#128202; Summary Histograms</h2>"
        + toggle_section_2("histograms", w_hist_raw, w_hist_filt)

        # Section 3: Distribution curves (3 modes: Raw / Filtered / No Outliers)
        + "<h2>&#128200; Read Length &amp; Quality Distributions</h2>"
        + toggle_section_3("overlay",
                           w_overlay_raw, w_overlay_filt, w_overlay_nout,
                           args.outlier_percentile)

        # Section 4: Heatmaps (3 modes: Raw / Filtered / No Outliers)
        + "<h2>&#128293; Length vs Quality Heatmaps</h2>"
        + toggle_section_3("heatmaps",
                           w_heatmap_raw, w_heatmap_filt, w_heatmap_nout,
                           args.outlier_percentile)

        + toggle_js
        + "</body></html>"
    )

    # --- Minification + écriture du fichier HTML final
    print("Minifying and writing HTML...", flush=True)
    html_final = minify_html(html_full)
    size_before = len(html_full.encode("utf-8"))
    size_after  = len(html_final.encode("utf-8"))
    print(f"  HTML size: {size_before/1e6:.1f} MB → {size_after/1e6:.1f} MB "
          f"(minified, -{100*(1-size_after/size_before):.0f}%)", flush=True)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html_final)

    # --- Gzip version (.html.gz) — opens natively in Firefox and Chrome
    gz_output = args.output + ".gz"
    with gzip.open(gz_output, "wt", encoding="utf-8", compresslevel=9) as f:
        f.write(html_final)
    size_gz = os.path.getsize(gz_output)
    print(f"  Gzip size: {size_gz/1e6:.1f} MB "
          f"(-{100*(1-size_gz/size_after):.0f}% vs minified HTML)", flush=True)

    # --- Confirmation messages
    print(f"✅ Interactive HTML report    : {args.output}")
    print(f"✅ Gzip HTML (smaller)        : {gz_output}")
    print(f"✅ Raw statistics (TXT)         : {txt_raw}")
    print(f"✅ Filtered statistics (TXT)       : {txt_filtered}")
    print(f"   Outlier percentile         : P{fmt_pct(args.outlier_percentile)}")
    for name, thr in outlier_thresholds.items():
        print(f"   Threshold [{name}] : ≤ {thr:,.0f} bp")
    if filter_active:
        print(f"   Active filters : {' | '.join(filter_info)}")
    else:
        print("   No quality/length filters specified.")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()