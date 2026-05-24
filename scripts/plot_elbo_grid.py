#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "altair>=6.1.0",
#     "vl-convert-python>=1.9.0.post1",
# ]
# ///
"""Plot diagnostics from a sweep of `eval_elbo_bound.py` runs.

Reads every `summary.json` under --input_root (default
`outputs/elbo_bound_grid/elf_b_owt`), and writes Altair plots to --output_dir
(default `<input_root>/plots`). Run with:

  uv run scripts/plot_elbo_grid.py

Plots produced as both `.html` and `.png`:
  1. nelbo_vs_posterior_sigma
  2. cross_proposal_agreement
  3. cross_proposal_delta
  4. importance_weight_tail
  5. ess_fraction
  6. latent_weight_tail
  7. latent_weight_ess_fraction
  8. nelbo_decomposition
"""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

try:
    import altair as alt
except ImportError as exc:
    raise SystemExit(
        "Missing Altair. Run this script with `uv run scripts/plot_elbo_grid.py` "
        "or install its inline uv script dependencies."
    ) from exc


alt.data_transformers.disable_max_rows()


def proposal_label(args):
    name = args["time_proposal"]
    if name == "sigmoid_normal":
        label = f"sigmoid_normal(loc={args['time_proposal_loc']:g}, scale={args['time_proposal_scale']:g})"
        if args.get("t_max", 1.0) < 1.0:
            label += f", t_max={args['t_max']:g}"
        return label
    if name == "beta":
        label = f"Beta({args['time_proposal_alpha']:g}, {args['time_proposal_beta']:g})"
        if args.get("t_max", 1.0) < 1.0:
            label += f", t_max={args['t_max']:g}"
        return label
    if name == "truncated_uniform":
        return f"trunc_uniform(t_min={args['t_min']:g}, t_max={args.get('t_max', 1.0):g})"
    return name


def load_summaries(input_root):
    """Return list of dicts with the per-run scalars we need."""
    rows = []
    for path in sorted(Path(input_root).glob("*/summary.json")):
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        summary = blob["summary"]
        rows.append({
            "run": path.parent.name,
            "proposal": proposal_label(blob),
            "proposal_kind": blob["time_proposal"],
            "posterior_sigma": blob["posterior_sigma"],
            "posterior_sigma_label": f"{blob['posterior_sigma']:g}",
            "mc_samples": blob["mc_samples"],
            "repeats": blob["repeats"],
            "weight_mode": blob.get("weight_mode", "vdm_xpred"),
            "t_max": blob.get("t_max", 1.0),
            "mean_nelbo": summary["mean_token_nelbo_per_token"],
            "std_nelbo": summary["std_token_nelbo_per_token"],
            "stderr_nelbo": summary["stderr_token_nelbo_per_token"],
            "repeat_estimates": summary.get("repeat_estimates", []),
            "max_iw": summary["max_integral_importance_weight"],
            "mean_iw": summary["mean_integral_importance_weight"],
            "std_iw": summary["std_integral_importance_weight"],
            "max_over_mean_iw": summary["max_over_mean_integral_importance_weight"],
            "ess_frac": summary["integral_importance_weight_ess_frac"],
            "mean_latent_weight": summary["mean_latent_weight"],
            "std_latent_weight": summary["std_latent_weight"],
            "latent_weight_ess_frac": summary["latent_weight_ess_frac"],
            "max_latent_weight": summary["max_latent_weight"],
            "max_over_mean_latent_weight": summary["max_over_mean_latent_weight"],
            "mean_t": summary["mean_time"],
            "max_t": summary["max_time"],
            "decomposition": _extract_decomposition(blob),
        })
    if not rows:
        raise FileNotFoundError(
            f"No summary.json files found under {input_root}/*/. Did the grid run?"
        )
    return rows


def _extract_decomposition(blob):
    """Mean per-token decomposition across repeats."""
    detail = blob.get("repeats_detail", [])
    if not detail:
        return None
    keys = (
        "latent_nelbo_per_token",
        "latent_endpoint_const_per_token",
        "decoder_nll_per_token",
        "posterior_logq_per_token",
    )
    out = {}
    for key in keys:
        values = [row[key] for row in detail if key in row]
        if values:
            out[key] = float(statistics.fmean(values))
    return out or None


def _group_by(rows, key):
    out = defaultdict(list)
    for row in rows:
        out[row[key]].append(row)
    return out


def unbiased_rows(rows):
    """Rows whose time proposal has full support and should estimate the same ELBO."""
    return [row for row in rows if row["proposal_kind"] != "truncated_uniform"]


def _ci_row(row):
    out = dict(row)
    width = 1.96 * row["stderr_nelbo"]
    out["ci_lower"] = row["mean_nelbo"] - width
    out["ci_upper"] = row["mean_nelbo"] + width
    return out


def _save_chart(chart, out_base):
    html_path = out_base.with_suffix(".html")
    png_path = out_base.with_suffix(".png")
    chart.save(str(html_path))
    try:
        chart.save(str(png_path), scale_factor=2)
    except Exception as exc:
        print(f"Skipping PNG export for {out_base.name}: {exc}")


def _chart(values):
    return alt.Chart(alt.Data(values=values))


# ---- plotters -------------------------------------------------------------


def plot_nelbo_vs_sigma(rows, out_base):
    rows = [_ci_row(row) for row in unbiased_rows(rows)]
    if not rows:
        print("Skipping nELBO-vs-sigma plot: no unbiased proposal rows.")
        return

    best = min(rows, key=lambda r: r["mean_nelbo"])
    best["best_label"] = (
        f"lowest plotted mean={best['mean_nelbo']:.3f}; "
        f"sigma_q={best['posterior_sigma']:g}; {best['proposal']}"
    )
    base = _chart(rows).encode(
        x=alt.X(
            "posterior_sigma:Q",
            scale=alt.Scale(type="log"),
            title="posterior sigma_q",
        ),
        color=alt.Color("proposal:N", title="proposal"),
        tooltip=[
            "proposal:N",
            alt.Tooltip("posterior_sigma:Q", format=".4g"),
            alt.Tooltip("mean_nelbo:Q", format=".4f"),
            alt.Tooltip("stderr_nelbo:Q", format=".4f"),
            alt.Tooltip("ess_frac:Q", format=".3f"),
            alt.Tooltip("latent_weight_ess_frac:Q", format=".3f"),
        ],
    )
    line = base.mark_line(point=True).encode(
        y=alt.Y("mean_nelbo:Q", title="nELBO per token (nats)")
    )
    error = base.mark_errorbar().encode(
        y=alt.Y("ci_lower:Q", title="nELBO per token (nats)"),
        y2="ci_upper:Q",
    )
    rule = _chart([best]).mark_rule(strokeDash=[3, 3], color="#555").encode(
        y="mean_nelbo:Q"
    )
    text = _chart([best]).mark_text(
        align="left",
        dx=8,
        dy=-8,
        fontSize=11,
        color="#333",
    ).encode(
        x="posterior_sigma:Q",
        y="mean_nelbo:Q",
        text="best_label:N",
    )
    chart = (error + line + rule + text).properties(
        width=760,
        height=430,
        title="Token nELBO vs posterior width",
    ).interactive()
    _save_chart(chart, out_base)


def plot_cross_proposal_agreement(rows, out_base):
    rows = [_ci_row(row) for row in unbiased_rows(rows)]
    if not rows:
        print("Skipping cross-proposal agreement plot: no unbiased proposal rows.")
        return
    base = _chart(rows).encode(
        x=alt.X("posterior_sigma_label:N", title="posterior sigma_q"),
        xOffset=alt.XOffset("proposal:N"),
        color=alt.Color("proposal:N", title="proposal"),
        tooltip=[
            "proposal:N",
            "posterior_sigma_label:N",
            alt.Tooltip("mean_nelbo:Q", format=".4f"),
            alt.Tooltip("stderr_nelbo:Q", format=".4f"),
        ],
    )
    bars = base.mark_bar().encode(
        y=alt.Y("mean_nelbo:Q", title="nELBO per token (nats)")
    )
    error = base.mark_errorbar().encode(y="ci_lower:Q", y2="ci_upper:Q")
    chart = (bars + error).properties(
        width=max(760, 80 * len(rows)),
        height=430,
        title="Cross-proposal agreement at each sigma_q",
    ).interactive()
    _save_chart(chart, out_base)


def plot_cross_proposal_delta(rows, out_base):
    rows = unbiased_rows(rows)
    if not rows:
        print("Skipping cross-proposal delta plot: no unbiased proposal rows.")
        return
    by_sigma = _group_by(rows, "posterior_sigma")
    data = []
    for sigma, sigma_rows in by_sigma.items():
        median = float(statistics.median([r["mean_nelbo"] for r in sigma_rows]))
        for row in sigma_rows:
            width = 1.96 * row["stderr_nelbo"]
            delta = row["mean_nelbo"] - median
            item = dict(row)
            item["delta"] = delta
            item["delta_lower"] = delta - width
            item["delta_upper"] = delta + width
            item["posterior_sigma_label"] = f"{sigma:g}"
            data.append(item)

    base = _chart(data).encode(
        x=alt.X("posterior_sigma_label:N", title="posterior sigma_q"),
        xOffset=alt.XOffset("proposal:N"),
        color=alt.Color("proposal:N", title="proposal"),
        tooltip=[
            "proposal:N",
            "posterior_sigma_label:N",
            alt.Tooltip("delta:Q", format=".4f"),
            alt.Tooltip("stderr_nelbo:Q", format=".4f"),
        ],
    )
    bars = base.mark_bar().encode(
        y=alt.Y("delta:Q", title="nELBO - median proposal nELBO (nats/token)")
    )
    error = base.mark_errorbar().encode(y="delta_lower:Q", y2="delta_upper:Q")
    zero = _chart([{"zero": 0.0}]).mark_rule(color="#444").encode(y="zero:Q")
    chart = (bars + error + zero).properties(
        width=max(760, 80 * len(data)),
        height=430,
        title="Cross-proposal disagreement at each sigma_q",
    ).interactive()
    _save_chart(chart, out_base)


def plot_weight_tail(rows, out_base, ratio_key, xlabel, title):
    by_proposal = _group_by(unbiased_rows(rows), "proposal")
    if not by_proposal:
        print(f"Skipping {out_base.name}: no unbiased proposal rows.")
        return
    data = []
    for proposal in sorted(by_proposal):
        ratios = [r[ratio_key] for r in by_proposal[proposal]]
        value = max(ratios)
        if math.isfinite(value) and value > 0:
            data.append({"proposal": proposal, "value": value, "label": f"{value:.1f}"})
    thresholds = [
        {"threshold": 10.0, "status": "healthy"},
        {"threshold": 100.0, "status": "worrying"},
        {"threshold": 1000.0, "status": "broken"},
    ]
    bars = _chart(data).mark_bar(color="#4C78A8").encode(
        y=alt.Y("proposal:N", sort="-x", title="proposal"),
        x=alt.X(
            "value:Q",
            scale=alt.Scale(type="log"),
            title=xlabel,
        ),
        tooltip=["proposal:N", alt.Tooltip("value:Q", format=".2f")],
    )
    labels = _chart(data).mark_text(align="left", dx=4, fontSize=11).encode(
        y=alt.Y("proposal:N", sort="-x"),
        x="value:Q",
        text="label:N",
    )
    rules = _chart(thresholds).mark_rule(strokeDash=[4, 3]).encode(
        x="threshold:Q",
        color=alt.Color(
            "status:N",
            scale=alt.Scale(
                domain=["healthy", "worrying", "broken"],
                range=["#2ca02c", "#ff7f0e", "#d62728"],
            ),
            title="reference",
        ),
        tooltip=["status:N", "threshold:Q"],
    )
    chart = (bars + labels + rules).properties(width=760, height=360, title=title)
    _save_chart(chart, out_base)


def plot_ess_fraction(rows, out_base, ess_key, xlabel, title):
    by_proposal = _group_by(unbiased_rows(rows), "proposal")
    if not by_proposal:
        print(f"Skipping {out_base.name}: no unbiased proposal rows.")
        return
    data = []
    for proposal in sorted(by_proposal):
        ess_fracs = [r[ess_key] for r in by_proposal[proposal]]
        value = min(ess_fracs)
        data.append({"proposal": proposal, "value": value, "label": f"{value:.2f}"})
    thresholds = [
        {"threshold": 0.1, "status": "warning"},
        {"threshold": 0.5, "status": "good"},
    ]
    bars = _chart(data).mark_bar(color="#59A14F").encode(
        y=alt.Y("proposal:N", sort="-x", title="proposal"),
        x=alt.X("value:Q", scale=alt.Scale(domain=[0, 1]), title=xlabel),
        tooltip=["proposal:N", alt.Tooltip("value:Q", format=".3f")],
    )
    labels = _chart(data).mark_text(align="left", dx=4, fontSize=11).encode(
        y=alt.Y("proposal:N", sort="-x"),
        x="value:Q",
        text="label:N",
    )
    rules = _chart(thresholds).mark_rule(strokeDash=[4, 3]).encode(
        x="threshold:Q",
        color=alt.Color(
            "status:N",
            scale=alt.Scale(domain=["warning", "good"], range=["#ff7f0e", "#2ca02c"]),
            title="reference",
        ),
        tooltip=["status:N", "threshold:Q"],
    )
    chart = (bars + labels + rules).properties(width=760, height=360, title=title)
    _save_chart(chart, out_base)


def plot_decomposition(rows, out_base):
    candidates = [
        row for row in unbiased_rows(rows)
        if row.get("ess_frac", 0.0) >= 0.1
        and row.get("max_over_mean_iw", math.inf) < 100
        and row.get("latent_weight_ess_frac", 0.0) >= 0.1
        and row.get("max_over_mean_latent_weight", math.inf) < 100
    ]
    candidate_label = "lowest healthy setting"
    if not candidates:
        candidates = unbiased_rows(rows)
        candidate_label = "lowest unbiased setting"
    if not candidates:
        print("Skipping decomposition plot: no unbiased proposal rows.")
        return
    best = min(candidates, key=lambda r: r["mean_nelbo"])
    if best["decomposition"] is None:
        print("Skipping decomposition plot: no repeats_detail in summaries.")
        return

    components = [
        ("latent_nelbo_per_token", "L_diff latent integrand"),
        ("latent_endpoint_const_per_token", "endpoint const"),
        ("decoder_nll_per_token", "decoder CE"),
        ("posterior_logq_per_token", "log q(z|s)"),
    ]
    data = []
    for key, label in components:
        value = best["decomposition"].get(key, 0.0)
        data.append({
            "component": label,
            "value": value,
            "label": f"{value:+.2f}",
        })
    total = sum(row["value"] for row in data)
    bars = _chart(data).mark_bar().encode(
        y=alt.Y(
            "component:N",
            sort=[label for _, label in components],
            title="component",
        ),
        x=alt.X("value:Q", title="contribution to nELBO (nats/token)"),
        color=alt.Color("component:N", legend=None),
        tooltip=[
            "component:N",
            alt.Tooltip("value:Q", format="+.3f"),
        ],
    )
    labels = _chart(data).mark_text(align="left", dx=4, fontSize=11).encode(
        y=alt.Y("component:N", sort=[label for _, label in components]),
        x="value:Q",
        text="label:N",
    )
    zero = _chart([{"zero": 0.0}]).mark_rule(color="#444").encode(x="zero:Q")
    title = (
        f"Decomposition at {candidate_label}: sigma_q={best['posterior_sigma']:g}, "
        f"{best['proposal']} | total nELBO/token={total:.3f}"
    )
    chart = (bars + labels + zero).properties(width=760, height=300, title=title)
    _save_chart(chart, out_base)


def write_table(rows, out_path):
    """Tidy CSV of the per-run scalars for downstream analysis."""
    fields = [
        "run", "proposal", "proposal_kind", "posterior_sigma", "mc_samples", "repeats",
        "weight_mode", "t_max",
        "mean_nelbo", "std_nelbo", "stderr_nelbo",
        "mean_iw", "std_iw", "max_iw", "max_over_mean_iw", "ess_frac",
        "mean_t", "max_t",
        "mean_latent_weight", "std_latent_weight", "max_latent_weight",
        "max_over_mean_latent_weight",
        "latent_weight_ess_frac",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["proposal"], r["posterior_sigma"])):
            writer.writerow({
                key: "" if isinstance(row.get(key), float) and not math.isfinite(row[key])
                else row.get(key, "")
                for key in fields
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_root", type=str,
        default="outputs/elbo_bound_grid/elf_b_owt",
        help="Directory containing one subdirectory per sweep run.",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir or input_root / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_summaries(input_root)
    print(f"Loaded {len(rows)} runs from {input_root}")

    plot_nelbo_vs_sigma(rows, output_dir / "nelbo_vs_posterior_sigma")
    plot_cross_proposal_agreement(rows, output_dir / "cross_proposal_agreement")
    plot_cross_proposal_delta(rows, output_dir / "cross_proposal_delta")
    plot_weight_tail(
        rows,
        output_dir / "importance_weight_tail",
        "max_over_mean_iw",
        "max integral IW / mean integral IW (worst across sigma_q)",
        "Integral importance-weight tail by proposal",
    )
    plot_ess_fraction(
        rows,
        output_dir / "ess_fraction",
        "ess_frac",
        "integral-weight ESS / mc_samples (worst across sigma_q)",
        "Integral importance-weight ESS by proposal",
    )
    plot_weight_tail(
        rows,
        output_dir / "latent_weight_tail",
        "max_over_mean_latent_weight",
        "max latent weight / mean latent weight (worst across sigma_q)",
        "Latent-weight tail by proposal",
    )
    plot_ess_fraction(
        rows,
        output_dir / "latent_weight_ess_fraction",
        "latent_weight_ess_frac",
        "latent-weight ESS / mc_samples (worst across sigma_q)",
        "Latent-weight ESS by proposal",
    )
    plot_decomposition(rows, output_dir / "nelbo_decomposition")
    write_table(rows, output_dir / "summary_table.csv")

    print(f"Wrote Altair plots and summary_table.csv to {output_dir}")


if __name__ == "__main__":
    main()
