"""Evaluate complete NEW job sets. Scientific failures are never replaced."""

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manifold_experiments.protocol import GEOMETRIES, SIGMAS, SIZES, digest
from manifold_experiments.worker import load


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as f:
        json.dump(value, f, indent=2, allow_nan=True)


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def run(script, *args):
    subprocess.run(
        [sys.executable, "-X", "utf8", "-B", str(ROOT / script), *map(str, args)],
        cwd=ROOT,
        check=True,
    )


def quantiles(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    q1, median, q3 = (
        np.quantile(values, [0.25, 0.5, 0.75]) if len(values) else [float("nan")] * 3
    )
    return dict(
        median=float(median),
        q1=float(q1),
        q3=float(q3),
        finite_repeat_count=len(values),
    )


def population_audit(records, exp, output, smoke=False):
    rows = []
    diagnostics = []
    errors = []
    data = {}
    for record in records:
        j, p = record["job"], record["result"]
        data[j["method"], j["geometry"], j["sigma"], j["resolution"]] = p
        if p.get("failure_count", 0) or p.get("single_failure_count", 0):
            errors.append(
                dict(
                    job=j,
                    failures=p.get("failures"),
                    reason="unresolved population queries",
                )
            )
    for (m, g, s, res), p in data.items():
        if res != "production":
            continue
        q = data[m, g, s, "reference"]
        if m == "MS-STSP":
            continue
        metrics = p["metrics"]
        rows.append(
            dict(
                method=m,
                geometry=g,
                sigma=s,
                query_count=p["query_count"],
                **{
                    key: metrics[key]
                    for key in (
                        "single_E_infinity",
                        "single_E_average",
                        "multi_E_infinity",
                        "multi_E_average",
                    )
                },
            )
        )
        metric_difference = max(
            abs(float(metrics[k]) - float(q["metrics"][k]))
            for k in metrics
            if k.endswith(("infinity", "average"))
        )
        difference = float("nan")
        correction = 0.0
        if m != "YY":
            difference = max(
                float(
                    np.max(np.linalg.norm(np.asarray(p[k]) - np.asarray(q[k]), axis=1))
                )
                for k in ("single_points", "multi_points")
            )
            for raw in (p, q):
                for query in raw["query_rows"]:
                    for fit in query.get("fits", []) + (
                        [query["single_fit"]] if query.get("single_fit") else []
                    ):
                        correction = max(
                            correction, float(fit["final"]["coordinate_correction"])
                        )
            weights = p["weights"]
            weight = max(map(abs, weights["constraint_residual"]))
        else:
            weight = float(p["weights"]["max_abs_residual"])
        diagnostics.append(
            dict(
                method=m,
                geometry=g,
                sigma=s,
                point_difference=difference,
                metric_difference=metric_difference,
                root_correction=correction,
                weight_residual=weight,
            )
        )
    for m in sorted({r["method"] for r in rows}):
        selected = [r for r in rows if r["method"] == m]
        dd = [d for d in diagnostics if d["method"] == m]
        minimum = min(float(r["multi_E_infinity"]) for r in selected)
        if m != "YY":
            minimum = min(minimum, min(float(r["multi_E_average"]) for r in selected))
        # Final manuscript six-manifold checks supersede the looser historical
        # per-package checks. Solver tolerances themselves remain unchanged.
        fraction = 0.01 if m == "YY" else 0.001
        for d in dd:
            diff = (
                d["metric_difference"]
                if m == "YY"
                else max(d["metric_difference"], d["point_difference"])
            )
            if not np.isfinite(diff) or diff > fraction * minimum:
                errors.append(
                    dict(
                        reason="population refinement gate",
                        **d,
                        budget=fraction * minimum,
                    )
                )
            if m != "YY" and d["root_correction"] > 0.001 * minimum:
                errors.append(
                    dict(
                        reason="population root-position gate",
                        **d,
                        budget=0.001 * minimum,
                    )
                )
            if d["weight_residual"] > 1e-11:
                errors.append(dict(reason="population weight gate", **d))
    csv_write(output / "VALUES.csv", rows)
    csv_write(output / "PRECISION.csv", diagnostics)
    report = dict(
        schema="FRESH_POPULATION_NUMERICAL_AUDIT_V1",
        status="FAIL" if errors else "PASS",
        evidence=(
            "reduced smoke only"
            if smoke
            else "fresh CPU reference or same-input gated GPU population computations"
        ),
        errors=errors,
        all_failures_retained=True,
        diagnostics_not_rigorous_error_bounds=True,
    )
    write(output / "AUDIT.json", report)
    if errors and not smoke:
        raise ArithmeticError(
            "Population precision/failed-query audit failed: "
            + str(output / "AUDIT.json")
        )
    return rows, data


def weight_only_control(data, out):
    analyze = load(
        Path("experiments/population_verification/weight_control/analyze.py"),
        "fresh_weight_analysis",
    )
    values = []
    precision = []
    minimum = float("inf")
    maxcorr = 0.0
    branchdiff = 0.0
    for g in GEOMETRIES:
        for s in SIGMAS:
            a = analyze.evaluate(
                g, s, "production", packet=data["STSP", g, s, "production"]
            )
            b = analyze.evaluate(
                g, s, "reference", packet=data["STSP", g, s, "reference"]
            )
            ms = data["MS-STSP", g, s, "production"]
            msref = data["MS-STSP", g, s, "reference"]
            if not ms["all_queries_successful"] or not msref["all_queries_successful"]:
                raise ArithmeticError("MS-STSP failures present")
            for x, y in zip(ms["query_rows"], msref["query_rows"]):
                if x["query_id"] != y["query_id"]:
                    raise ValueError("MS query set changed")
                branchdiff = max(
                    branchdiff,
                    float(
                        np.max(
                            np.linalg.norm(
                                np.asarray(x["points"]) - np.asarray(y["points"]),
                                axis=1,
                            )
                        )
                    ),
                    float(
                        np.linalg.norm(np.asarray(x["multi"]) - np.asarray(y["multi"]))
                    ),
                )
            for name in ("single", "naive", "proposed"):
                precision.append(
                    dict(
                        geometry=g,
                        sigma=s,
                        estimator=name,
                        point_difference=float(
                            np.max(
                                np.linalg.norm(
                                    a["points"][name] - b["points"][name], axis=1
                                )
                            )
                        ),
                        metric_difference=max(
                            abs(
                                a["metrics"][name + "_E_" + k]
                                - b["metrics"][name + "_E_" + k]
                            )
                            for k in ("infinity", "average")
                        ),
                    )
                )
            minimum = min(
                minimum,
                a["metrics"]["proposed_E_infinity"],
                a["metrics"]["proposed_E_average"],
            )
            maxcorr = max(
                maxcorr,
                a["diagnostics"]["max_final_root_correction"],
                b["diagnostics"]["max_final_root_correction"],
            )
            values.append(
                dict(
                    geometry=g,
                    sigma=s,
                    **a["metrics"],
                    ms_stsp_E_infinity=ms["E_infinity"],
                    ms_stsp_E_average=ms["E_average"],
                    proposed_w1=a["weights"][0],
                    proposed_w2=a["weights"][1],
                    naive_w1=4 / 3,
                    naive_w2=-1 / 3,
                    naive_over_sigma2=a["metrics"]["naive_E_infinity"] / s**2,
                    proposed_over_sigma4=a["metrics"]["proposed_E_infinity"] / s**4,
                )
            )
    errors = []
    for row in precision:
        if max(row["point_difference"], row["metric_difference"]) > 0.001 * minimum:
            errors.append(row)
    if branchdiff > min(0.001 * minimum, 7.156008674513714e-12):
        errors.append(dict(MS_branch_resolution_difference=branchdiff))
    if maxcorr > 0.1 * minimum:
        errors.append(dict(root_correction=maxcorr, budget=0.1 * minimum))
    report = dict(
        status="FAIL" if errors else "PASS",
        errors=errors,
        minimum_proposed_error=minimum,
        point_metric_difference_budget=0.001 * minimum,
        root_correction_budget=0.1 * minimum,
        maximum_MS_branch_resolution_difference=branchdiff,
        evidence="new shared fits; independent one-dimensional coefficient checks in analyze.evaluate",
    )
    csv_write(out / "VALUES.csv", values)
    csv_write(out / "PRECISION.csv", precision)
    write(out / "NUMERICAL_CHECK.json", report)
    # This new schema deliberately makes no historical independent-distance
    # or strict-circle rerun claim. The figure-only renderer reads its status.
    write(
        out / "INDEPENDENT_PRECISION.json",
        dict(
            status=report["status"],
            schema="FRESH_COEFFICIENT_AND_REFINEMENT_V1",
            independent_old_archive_checks_replayed=False,
        ),
    )
    slopes = []
    for g in GEOMETRIES:
        block = [r for r in values if r["geometry"] == g]
        for name in ("single", "naive", "proposed", "ms_stsp"):
            slopes.append(
                dict(
                    geometry=g,
                    method=name,
                    slope_all_7=float(
                        np.polyfit(
                            np.log(SIGMAS),
                            np.log([r[name + "_E_infinity"] for r in block]),
                            1,
                        )[0]
                    ),
                )
            )
    csv_write(out / "SLOPES.csv", slopes)
    # Independent one-dimensional high-precision theoretical constant; no
    # empirical errors enter its calculation or any extrapolation weights.
    import mpmath as mp

    with mp.workdps(70):

        def beta(k):
            lam = mp.mpf(1) / k

            def integral(j):
                return mp.quad(
                    lambda t: t ** (mp.mpf(".5") + j - 1)
                    * (1 - t) ** mp.mpf("6.5")
                    * mp.exp(-t / (2 * lam**2)),
                    [0, mp.mpf(".1"), mp.mpf(".5"), 1],
                )

            return (1 + k * k * (1 - integral(2) / integral(1)) / 15) / 2

        gamma = abs((4 * beta(3) - beta(6)) / 3)
    csv_write(
        out / "CIRCLE_CONSTANT.csv",
        [
            dict(
                sigma=r["sigma"],
                naive_over_sigma2=r["naive_over_sigma2"],
                Gamma_1_2=float(gamma),
                relative_difference_from_Gamma=abs(
                    r["naive_over_sigma2"] / float(gamma) - 1
                ),
                proposed_over_sigma4=r["proposed_over_sigma4"],
            )
            for r in values
            if r["geometry"] == GEOMETRIES[0]
        ],
    )
    if errors:
        raise ArithmeticError(
            "Fresh controlled-comparison numerical audit failed; see " + str(out)
        )
    return report


def finite_sample_summary(records, smoke=False):
    methods = sorted({p["job"]["method"] for p in records})
    rows = []
    lookup = {
        (r["job"]["method"], r["job"]["geometry"], r["job"]["repeat"]): r["result"]
        for r in records
    }
    geometries = GEOMETRIES[:1] if smoke else GEOMETRIES
    sizes = (2048, 4096) if smoke else SIZES
    repeats = 1 if smoke else 20
    for m in methods:
        groups = (
            ("Single", "Multi") if m == "YY" else ("Single_h1", "Single_h2", "Multi")
        )
        for g in geometries:
            parents = [lookup[m, g, r] for r in range(repeats)]
            nq = 2 if smoke else 144
            for p in parents:
                if p["N_q"] != nq or [c["n"] for c in p["results"]] != list(sizes):
                    raise ValueError("E2 changed query/prefix protocol")
            for i, n in enumerate(sizes):
                for estimator in groups:
                    failed = 0
                    complete = 0
                    taxonomy = Counter()
                    metrics = []
                    for parent in parents:
                        cell = parent["results"][i]
                        outcomes = cell["groups"][estimator]
                        if len(outcomes) != nq:
                            raise ValueError("E2 omitted query outcomes")
                        failure = [
                            str(q["reason"])
                            for q in outcomes
                            if q["status"] != "success"
                        ]
                        failed += len(failure)
                        complete += not failure
                        taxonomy.update(failure)
                        metrics.append(cell["summaries"][estimator])
                    rate = failed / (repeats * nq)
                    for metric in ("E_avg", "E_infty"):
                        rows.append(
                            dict(
                                method=m,
                                geometry=g,
                                scale=float(n),
                                parent_scale=float(n),
                                estimator=estimator,
                                metric=metric,
                                **quantiles([x[metric] for x in metrics]),
                                complete_repeats=complete,
                                query_count_per_repeat=nq,
                                query_failure_rate=rate,
                                reliable=(
                                    not smoke and complete >= 18 and rate <= 0.01
                                ),
                                failure_taxonomy=json.dumps(
                                    dict(taxonomy), sort_keys=True
                                ),
                            )
                        )
    return rows


def bandwidth_summary(records, smoke=False):
    support = load(
        Path("experiments/bandwidth_robustness/science_support.py"),
        "fresh_e3_summary_design",
    )
    lookup = {
        (
            r["job"]["method"],
            r["job"]["phase"],
            r["job"]["geometry"],
            r["job"]["repeat"],
        ): r["result"]
        for r in records
    }
    rows = []
    for m in support.GROUPS:
        for phase in ("main", "rev"):
            for g in (GEOMETRIES[:1] if smoke else GEOMETRIES):
                query, reference = support.assets(g, phase)
                if smoke:
                    query = query[:2]
                    reference = reference[:128]
                repeats = 1 if smoke or phase == "rev" else 20
                parents = [lookup[m, phase, g, r] for r in range(repeats)]
                scales = (0.2, 0.3) if smoke else support.GRID
                for p in parents:
                    if not np.array_equal(p["query_coordinates"], query) or p[
                        "reference_count"
                    ] != len(reference):
                        raise ValueError("E3 query/reference protocol changed")
                    if tuple(c["scale"] for c in p["cells"]) != tuple(scales):
                        raise ValueError("E3 scales missing")
                    if not smoke and p["n"] != support.N[g]:
                        raise ValueError("E3 sample size changed")
                for i, s in enumerate(scales):
                    for estimator in support.GROUPS[m]:
                        failed = 0
                        complete = 0
                        taxonomy = Counter()
                        metrics = []
                        for parent in parents:
                            cell = parent["cells"][i]
                            outcomes = cell["groups"][estimator]
                            if len(outcomes) != len(query):
                                raise ValueError("E3 missing success/failure outcomes")
                            failure = [
                                str(q["reason"])
                                for q in outcomes
                                if q["status"] != "success"
                            ]
                            failed += len(failure)
                            complete += not failure
                            taxonomy.update(failure)
                            if phase == "main":
                                metric = cell["summaries"][estimator]
                            else:
                                # One entire fitted cloud per scale/estimator. Never
                                # pool distances or errors from separate shards.
                                cloud = np.asarray(
                                    [
                                        q["point"]
                                        for q in outcomes
                                        if q["status"] == "success"
                                    ]
                                )
                                distances = (
                                    cKDTree(cloud).query(reference, k=1, workers=1)[0]
                                    if len(cloud)
                                    else np.array([np.nan])
                                )
                                metric = dict(
                                    E_reverse_mean=float(np.mean(distances)),
                                    E_reverse=float(np.max(distances)),
                                )
                            metrics.append(metric)
                        rate = failed / (len(query) * repeats)
                        for metric in (
                            ("E_avg", "E_infty")
                            if phase == "main"
                            else ("E_reverse_mean", "E_reverse")
                        ):
                            rows.append(
                                dict(
                                    method=m,
                                    phase=phase,
                                    geometry=g,
                                    estimator=estimator,
                                    scale=s,
                                    metric=metric,
                                    **quantiles([x[metric] for x in metrics]),
                                    complete_repeats=complete,
                                    total_repeats=repeats,
                                    query_failure_rate=rate,
                                    reliable=not smoke
                                    and rate <= 0.01
                                    and (phase == "rev" or complete >= 18),
                                    query_count=len(query),
                                    reference_count=len(reference),
                                    n=parents[0]["n"],
                                    sigma=0.015,
                                    query_failure_count=failed,
                                    failure_taxonomy_json=json.dumps(
                                        dict(taxonomy), sort_keys=True
                                    ),
                                )
                            )
    return rows


def correspondence_summary(records, out, smoke=False):
    records = [r["result"] for r in records if r["job"].get("kind") != "counterexample"]
    allrows = [r for p in records for r in p["summary"]]
    rows = []
    for method, count in sorted(
        {(r["method"], r["candidate_count"]) for r in allrows},
        key=lambda v: (v[0], v[1] or 0),
    ):
        selected = [
            r for r in allrows if (r["method"], r["candidate_count"]) == (method, count)
        ]
        if len(selected) != (1 if smoke else 10):
            raise ValueError("E4 repeat grid incomplete")
        row = dict(
            method=method,
            candidate_count=count,
            repeat_count=len(selected),
            complete_repeats=sum(r["complete"] for r in selected),
            total_query_failures=sum(r["failure_count"] for r in selected),
            total_candidate_failures=sum(
                r["candidate_failure_count"] for r in selected
            ),
        )
        for key in (
            "E_infinity",
            "E_average",
            "E_match",
            "E_base",
            "shared_seconds",
            "correspondence_seconds",
            "total_seconds",
            "failure_rate",
        ):
            values = [
                r[key]
                for r in selected
                if r["complete"]
                or key
                in (
                    "failure_rate",
                    "shared_seconds",
                    "correspondence_seconds",
                    "total_seconds",
                )
            ]
            qq = quantiles(values)
            for name in ("median", "q1", "q3"):
                row[key + "_" + name] = qq[name]
        rows.append(row)
    # Per-method reference coordinates must be identical, not just query IDs.
    for p in records:
        fixed = {
            q["query_id"]: q["reference"]
            for q in p["per_query"]
            if q["method"] == "ENF"
        }
        for q in p["per_query"]:
            if not np.array_equal(q["reference"], fixed[q["query_id"]]):
                raise ValueError("E4 unmatched common query")
        for c in p["method_diagnostics"]["CP"]:
            if c["status"] == "success" and (
                c["full_residual"] > 1e-9
                or c["projected_residual"] > 1e-11
                or c["normalized_stationarity"] > 1e-10
            ):
                raise ArithmeticError("E4 CP residual/stationarity gate")
    csv_write(out / "SUMMARY.csv", rows)
    write(
        out / "AUDIT.json",
        dict(
            status="PASS",
            schema="FRESH_E4_COMMON_QUERY_SUMMARY_V1",
            summary_sha256=hashlib.sha256(
                (out / "SUMMARY.csv").read_bytes()
            ).hexdigest(),
            all_failures_retained=True,
            geometry_assisted_common_queries=True,
            CP_local_empirical_sheet=True,
            timing="fresh serial CPU measurement, including common-reference construction; not an algorithmic universal cost",
        ),
    )
    return rows


def plot_smoke_diagnostics(runroot, tables):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = runroot / "smoke_figures"
    out.mkdir()
    for exp, rows in tables.items():
        fig, ax = plt.subplots(figsize=(8, 4))
        groups = {}
        for r in rows:
            if "median" in r:
                key = (
                    r.get("method", ""),
                    r.get("phase", ""),
                    r.get("estimator", ""),
                    r.get("metric", ""),
                )
                groups.setdefault(key, []).append(
                    (float(r["scale"]), float(r["median"]))
                )
            elif "single_E_infinity" in r:
                for name in ("single", "multi"):
                    groups.setdefault((r["method"], name), []).append(
                        (r["sigma"], float(r[name + "_E_infinity"]))
                    )
            elif "E_infinity_median" in r:
                points = (
                    [(r["candidate_count"], r["E_infinity_median"])]
                    if r["method"] == "NN"
                    else [(32, r["E_infinity_median"]), (128, r["E_infinity_median"])]
                )
                groups.setdefault((r["method"], "E_infinity"), []).extend(points)
        for key, pairs in groups.items():
            x, y = zip(*sorted(pairs))
            ax.plot(x, y, "o-", label="/".join(key))
        ax.set_yscale("log")
        ax.set_title(exp + ": reduced fresh end-to-end smoke (NOT paper evidence)")
        if groups:
            ax.legend(fontsize=6)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / (exp + "." + ext), dpi=150)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args()
    folder = a.run.resolve()
    plan = json.loads((folder / "PLAN.json").read_bytes())
    records = []
    from release_contract import run_source_digest

    if run_source_digest(verify=True) != plan["source_manifest_sha256"]:
        raise ValueError("Current source differs from run")
    for job in plan["jobs"]:
        with gzip.open(
            folder / "packets" / (job["id"] + ".json.gz"), "rt", encoding="utf8"
        ) as f:
            packet = json.load(f)
        if (
            packet.get("schema") != "PAPER_FRESH_TASK_V1"
            or packet["job"] != job
            or packet["source_manifest_sha256"] != plan["source_manifest_sha256"]
        ):
            raise ValueError("Incomplete/source-mismatched new run")
        records.append(packet)
    if {
        f.stem.removesuffix(".json") for f in (folder / "packets").glob("*.json.gz")
    } != {j["id"] for j in plan["jobs"]}:
        raise ValueError("Unexpected duplicate/foreign packets")
    smoke = plan["profile"] == "smoke"
    tables = {}
    render = folder / "figures"
    errors = []
    for exp in plan["experiments"]:
        selected = [r for r in records if r["job"]["experiment"] == exp]
        out = folder / "evaluation" / exp
        out.mkdir(parents=True, exist_ok=False)
        try:
            if exp in ("E1", "nonuniform_E1"):
                rows, data = population_audit(selected, exp, out, smoke)
                tables[exp] = rows
                if not smoke:
                    presentation = load(
                        Path("experiments/population_verification/presentation.py"),
                        "fresh_population_presentation",
                    )
                    for method in sorted({r["method"] for r in rows}):
                        for metric in ("average", "infinity"):
                            if (
                                exp == "E1"
                                and method == "STSP"
                                and metric == "infinity"
                            ):
                                continue
                            presentation.six_panel(
                                rows,
                                method,
                                metric,
                                render / exp,
                                nonuniform=exp != "E1",
                            )
                    if exp == "E1":
                        normalized = [
                            dict(
                                method=r["method"],
                                sigma=r["sigma"],
                                single_E_infinity_over_sigma2=r["single_E_infinity"]
                                / r["sigma"] ** 2,
                                multi_E_infinity_over_sigma4=r["multi_E_infinity"]
                                / r["sigma"] ** 4,
                            )
                            for r in rows
                            if r["geometry"] == GEOMETRIES[0]
                        ]
                        presentation.circle(render / exp, normalized)
                        weight_only_control(data, out / "controlled")
                        run(
                            Path(
                                "experiments/population_verification/weight_control/render_and_report.py"
                            ),
                            "--figure-only",
                            "--analysis",
                            out / "controlled",
                            "--output",
                            render / "controlled",
                        )
            elif exp in ("E2", "nonuniform_E2"):
                rows = finite_sample_summary(selected, smoke)
                tables[exp] = rows
                csv_write(out / "SUMMARY.csv", rows)
                if not smoke:
                    run(
                        (
                            Path("experiments/finite_sample_fixed_scales/reproduce.py")
                            if exp == "E2"
                            else Path(
                                "experiments/nonuniform/fixed_scales/reproduce.py"
                            )
                        ),
                        "--table",
                        out / "SUMMARY.csv",
                        "--output",
                        render / exp,
                    )
            elif exp == "E3":
                rows = bandwidth_summary(selected, smoke)
                tables[exp] = rows
                csv_write(out / "SUMMARY.csv", rows)
                if not smoke:
                    run(
                        Path("experiments/bandwidth_robustness/reproduce.py"),
                        "--table",
                        out / "SUMMARY.csv",
                        "--output",
                        render / exp,
                    )
            elif exp == "E4":
                tables[exp] = correspondence_summary(selected, out, smoke)
                if not smoke:
                    run(
                        Path(
                            "experiments/correspondence_accuracy/correspondence/plot_finite.py"
                        ),
                        "--summary",
                        out,
                        "--output",
                        render / exp,
                    )
            elif exp == "E5":
                sys.path.insert(
                    0, str(ROOT / "experiments/yy_directional_ablation/src")
                )
                from e56v2.experiment import audit, canonical_hash

                population = json.loads(
                    (
                        ROOT
                        / "experiments/yy_directional_ablation/config/population.json"
                    ).read_bytes()
                )
                finite = json.loads(
                    (
                        ROOT
                        / "experiments/yy_directional_ablation/config/finite_sample.json"
                    ).read_bytes()
                )
                if smoke:
                    tables[exp] = [
                        dict(
                            method=m,
                            phase=r["job"]["phase"],
                            scale=i + 1,
                            median=q["geometric_max"],
                            metric="geometric_max",
                        )
                        for i, r in enumerate(selected)
                        for m, q in r["result"]["summaries"].items()
                    ]
                else:
                    for item in selected:
                        if (
                            item["result"]["failure_count"]
                            or item["result"]["retained_count"]
                            != item["result"]["query_count"]
                        ):
                            raise ArithmeticError(
                                "E5 declared population/finite query set has failures; packets retained"
                            )
                        j = item["job"]
                        kind = j["phase"]
                        cfg = population if kind == "population" else finite
                        name = (
                            f"population_sigma_{j['sigma']:.6f}.json"
                            if kind == "population"
                            else f"finite_n{j['n']:07d}_r{j['repeat']:03d}.json"
                        )
                        write(
                            out
                            / (
                                "population"
                                if kind == "population"
                                else "finite_sample"
                            )
                            / "packets"
                            / name,
                            dict(
                                metadata=dict(
                                    config_sha256=canonical_hash(cfg),
                                    source_sha256=plan["source_manifest_sha256"],
                                    formal_evidence=item["result"]["formal_evidence"],
                                ),
                                result=item["result"],
                            ),
                        )
                    report = audit(
                        out, population, finite, plan["source_manifest_sha256"]
                    )
                    if report["status"] != "PASS":
                        raise ArithmeticError("Fresh E5 original audit failed")
                    run(
                        Path(
                            "experiments/yy_directional_ablation/scripts/plot_from_raw_packets.py"
                        ),
                        "--results",
                        out,
                    )
        except Exception as error:
            errors.append(dict(experiment=exp, error=repr(error)))
    if smoke:
        plot_smoke_diagnostics(folder, tables)
    report = dict(
        schema="PAPER_FRESH_EVALUATION_V1",
        status="FAIL" if errors else "PASS",
        profile=plan["profile"],
        paper_evidence=not smoke,
        source_manifest_sha256=plan["source_manifest_sha256"],
        tasks=len(records),
        errors=errors,
        outputs=(
            "smoke_figures"
            if smoke
            else "figures plus evaluation/E5/figures and counterexample"
        ),
        limitations="Numerical refinement diagnostics are not certified error bounds. Different hardware can change floating-point results/timing. Smoke is not evidence for the paper.",
    )
    write(folder / "EVALUATION.json", report)
    if errors:
        raise SystemExit(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
