from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = PROJECT_ROOT / "data" / "output_greedy_0422"
OUTPUT_DIR = INPUT_ROOT / "aplot"


def safe_float(value: Any, default: float = float("nan")) -> float:
	try:
		return float(value)
	except (TypeError, ValueError):
		return float(default)


def safe_int(value: Any, default: int = 0) -> int:
	try:
		return int(value)
	except (TypeError, ValueError):
		return int(default)


def mean_or_nan(values: list[float]) -> float:
	clean = [v for v in values if not math.isnan(v)]
	if not clean:
		return float("nan")
	return float(np.mean(clean))


def std_or_nan(values: list[float]) -> float:
	clean = [v for v in values if not math.isnan(v)]
	if not clean:
		return float("nan")
	return float(np.std(clean))


def ensure_output_dir() -> None:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def discover_result_files(root: Path) -> list[Path]:
	files: list[Path] = []
	for path in root.rglob("results.json"):
		parts = set(path.parts)
		if "aggregate" in parts or "aplot" in parts:
			continue
		files.append(path)
	return sorted(files)


def extract_record(path: Path) -> dict[str, Any] | None:
	try:
		payload = load_json(path)
	except (OSError, json.JSONDecodeError) as exc:
		print(f"[skip] failed to read {path}: {exc}")
		return None

	target = payload.get("target", {})
	reconstruction_metrics = payload.get("reconstruction_metrics", {})
	composite_scores = payload.get("composite_scores", {})
	greedy_metrics = payload.get("greedy_metrics", {})
	simulator_metrics = payload.get("simulator_metrics", {})
	timing_seconds = payload.get("timing_seconds", {})
	timing_normalized = payload.get("timing_normalized", {})

	original_dice = safe_float(reconstruction_metrics.get("DC"))
	original_yield = safe_float(reconstruction_metrics.get("Y"))
	original_hd = safe_float(reconstruction_metrics.get("HD"))
	original_score = safe_float(composite_scores.get("score"))
	original_loss = safe_float(composite_scores.get("loss"))

	selected_dice = safe_float(greedy_metrics.get("selected_dice"))
	selected_yield = safe_float(greedy_metrics.get("selected_grid_yield"))
	selected_hd = safe_float(greedy_metrics.get("selected_hell_d"))
	selected_loss = safe_float(greedy_metrics.get("selected_loss"))
	selected_score = selected_dice + (0.1 * selected_yield) - selected_hd

	wall_clock_time = safe_float(timing_seconds.get("wall_clock_time"))
	optimization_time = safe_float(timing_seconds.get("optimization_time"))
	recompute_time = safe_float(timing_seconds.get("recompute_time"))
	artifact_write_time = safe_float(timing_seconds.get("artifact_write_time"))
	model_forward_time = safe_float(timing_seconds.get("model_forward_time"))
	simulator_forward_time = safe_float(timing_seconds.get("simulator_forward_time"))
	metric_computation_time = safe_float(timing_seconds.get("metric_computation_time"))
	greedy_elapsed_time = safe_float(timing_seconds.get("greedy_elapsed_time"))

	normalized_wall = safe_float(timing_normalized.get("normalized_wall_clock_time"))
	normalized_greedy = safe_float(timing_normalized.get("normalized_greedy_elapsed_time"))
	evals_per_second = safe_float(timing_normalized.get("evals_per_second"))
	model_time_per_call = safe_float(timing_normalized.get("model_forward_time_per_call"))
	simulator_time_per_call = safe_float(timing_normalized.get("simulator_forward_time_per_call"))

	contact_count = safe_int(greedy_metrics.get("contact_count"))
	active_count = safe_int(greedy_metrics.get("active_count"))
	active_ratio = safe_float(greedy_metrics.get("active_ratio"))

	return {
		"path": str(path),
		"target_name": str(target.get("name", path.parent.parent.name)),
		"subject": str(payload.get("subject", "")),
		"hemisphere": str(payload.get("hemisphere", "")),
		"grid_valid": bool(payload.get("grid_valid", True)),
		"original_dice": original_dice,
		"original_yield": original_yield,
		"original_hd": original_hd,
		"original_score": original_score,
		"original_loss": original_loss,
		"selected_dice": selected_dice,
		"selected_yield": selected_yield,
		"selected_hd": selected_hd,
		"selected_score": selected_score,
		"selected_loss": selected_loss,
		"delta_dice": selected_dice - original_dice,
		"delta_yield": selected_yield - original_yield,
		"delta_hd": selected_hd - original_hd,
		"delta_score": selected_score - original_score,
		"delta_loss": selected_loss - original_loss,
		"loss_improvement": original_loss - selected_loss,
		"wall_clock_time": wall_clock_time,
		"optimization_time": optimization_time,
		"recompute_time": recompute_time,
		"artifact_write_time": artifact_write_time,
		"model_forward_time": model_forward_time,
		"simulator_forward_time": simulator_forward_time,
		"metric_computation_time": metric_computation_time,
		"greedy_elapsed_time": greedy_elapsed_time,
		"normalized_wall_clock_time": normalized_wall,
		"normalized_greedy_elapsed_time": normalized_greedy,
		"evals_per_second": evals_per_second,
		"model_forward_time_per_call": model_time_per_call,
		"simulator_forward_time_per_call": simulator_time_per_call,
		"simulator_forward_calls": safe_int(simulator_metrics.get("simulator_forward_calls")),
		"model_forward_calls": safe_int(simulator_metrics.get("model_forward_calls")),
		"simulator_calls_per_eval": safe_float(simulator_metrics.get("simulator_calls_per_eval")),
		"model_calls_per_eval": safe_float(simulator_metrics.get("model_calls_per_eval")),
		"contact_count": contact_count,
		"active_count": active_count,
		"active_ratio": active_ratio,
		"greedy_fraction_of_wall": greedy_elapsed_time / wall_clock_time if wall_clock_time > 0 else float("nan"),
	}


def save_line_identity(ax: Axes, values_x: np.ndarray, values_y: np.ndarray) -> None:
	finite_x = values_x[np.isfinite(values_x)]
	finite_y = values_y[np.isfinite(values_y)]
	if finite_x.size == 0 or finite_y.size == 0:
		return
	lo = float(min(finite_x.min(), finite_y.min()))
	hi = float(max(finite_x.max(), finite_y.max()))
	if lo == hi:
		lo -= 1.0
		hi += 1.0
	ax.plot([lo, hi], [lo, hi], color="black", linestyle="--", linewidth=1, alpha=0.6)
	ax.set_xlim(lo, hi)
	ax.set_ylim(lo, hi)


def save_scatter(x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, title: str, out_path: Path) -> None:
	fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
	ax.scatter(x, y, s=22, alpha=0.8)
	save_line_identity(ax, x, y)
	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_title(title)
	ax.grid(alpha=0.25)
	fig.tight_layout()
	fig.savefig(out_path)
	plt.close(fig)


def save_hist(values: np.ndarray, title: str, xlabel: str, out_path: Path, bins: int = 24) -> None:
	finite = values[np.isfinite(values)]
	if finite.size == 0:
		return
	fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
	ax.hist(finite, bins=bins, color="#3b82f6", alpha=0.85, edgecolor="white")
	ax.axvline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
	ax.set_title(title)
	ax.set_xlabel(xlabel)
	ax.set_ylabel("Count")
	fig.tight_layout()
	fig.savefig(out_path)
	plt.close(fig)


def save_bar(labels: list[str], values: list[float], title: str, ylabel: str, out_path: Path, rotation: int = 45) -> None:
	if not labels:
		return
	fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), 4.8), dpi=160)
	x = np.arange(len(labels))
	ax.bar(x, values, color="#0f766e", alpha=0.9)
	ax.axhline(0.0, color="black", linewidth=1, alpha=0.7)
	ax.set_xticks(x)
	ax.set_xticklabels(labels, rotation=rotation, ha="right")
	ax.set_title(title)
	ax.set_ylabel(ylabel)
	fig.tight_layout()
	fig.savefig(out_path)
	plt.close(fig)


def save_time_breakdown(records: list[dict[str, Any]], out_path: Path) -> None:
	if not records:
		return
	components = [
		("optimization_time", "Optimization"),
		("greedy_elapsed_time", "Greedy"),
		("recompute_time", "Recompute"),
		("metric_computation_time", "Metric"),
		("model_forward_time", "Model forward"),
		("simulator_forward_time", "Simulator forward"),
		("artifact_write_time", "Write"),
	]

	means = []
	names = []
	for key, label in components:
		values = [safe_float(r.get(key)) for r in records]
		means.append(mean_or_nan(values))
		names.append(label)

	fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
	# Plot only the mean composition so the figure stays readable.
	ax.bar(names, means, color=["#2563eb", "#7c3aed", "#06b6d4", "#f59e0b", "#10b981", "#ef4444", "#64748b"])
	ax.set_title("Mean Timing Breakdown")
	ax.set_ylabel("Seconds")
	ax.tick_params(axis="x", rotation=25)
	fig.tight_layout()
	fig.savefig(out_path)
	plt.close(fig)


def main() -> None:
	ensure_output_dir()

	result_files = discover_result_files(INPUT_ROOT)
	if not result_files:
		raise FileNotFoundError(f"no results.json files found under {INPUT_ROOT}")

	records: list[dict[str, Any]] = []
	for path in result_files:
		record = extract_record(path)
		if record is not None:
			records.append(record)

	if not records:
		raise RuntimeError("no valid result records could be loaded")

	with open(OUTPUT_DIR / "greedy_summary.json", "w", encoding="utf-8") as f:
		json.dump(records, f, indent=2)

	fieldnames = sorted(records[0].keys())
	with open(OUTPUT_DIR / "greedy_summary.csv", "w", encoding="utf-8", newline="") as f:
		import csv

		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(records)

	metric_pairs = [
		("original_dice", "selected_dice", "Original Dice", "Greedy Dice", "dice_before_after.png"),
		("original_yield", "selected_yield", "Original Yield", "Greedy Yield", "yield_before_after.png"),
		("original_hd", "selected_hd", "Original HD", "Greedy HD", "hd_before_after.png"),
		("original_loss", "selected_loss", "Original Loss", "Greedy Loss", "loss_before_after.png"),
		("original_score", "selected_score", "Original Score", "Greedy Score", "score_before_after.png"),
	]

	for orig_key, sel_key, xlab, ylab, filename in metric_pairs:
		x = np.array([safe_float(r.get(orig_key)) for r in records], dtype=np.float64)
		y = np.array([safe_float(r.get(sel_key)) for r in records], dtype=np.float64)
		save_scatter(x, y, xlab, ylab, f"{ylab} vs {xlab}", OUTPUT_DIR / filename)

	delta_map = [
		("delta_dice", "Delta Dice", "delta_dice_hist.png"),
		("delta_yield", "Delta Yield", "delta_yield_hist.png"),
		("delta_hd", "Delta HD", "delta_hd_hist.png"),
		("delta_loss", "Delta Loss", "delta_loss_hist.png"),
		("delta_score", "Delta Score", "delta_score_hist.png"),
		("loss_improvement", "Loss Improvement (Original - Greedy)", "loss_improvement_hist.png"),
	]

	for key, label, filename in delta_map:
		arr = np.array([safe_float(r.get(key)) for r in records], dtype=np.float64)
		save_hist(arr, f"{label} Distribution", label, OUTPUT_DIR / filename)

	time_keys = [
		("wall_clock_time", "Wall Clock Time", "wall_clock_time_hist.png"),
		("greedy_elapsed_time", "Greedy Elapsed Time", "greedy_elapsed_time_hist.png"),
		("optimization_time", "Optimization Time", "optimization_time_hist.png"),
		("recompute_time", "Recompute Time", "recompute_time_hist.png"),
		("metric_computation_time", "Metric Computation Time", "metric_computation_time_hist.png"),
		("model_forward_time", "Model Forward Time", "model_forward_time_hist.png"),
		("simulator_forward_time", "Simulator Forward Time", "simulator_forward_time_hist.png"),
		("artifact_write_time", "Artifact Write Time", "artifact_write_time_hist.png"),
		("normalized_wall_clock_time", "Normalized Wall Clock Time", "normalized_wall_time_hist.png"),
		("normalized_greedy_elapsed_time", "Normalized Greedy Elapsed Time", "normalized_greedy_time_hist.png"),
		("evals_per_second", "Evaluations per Second", "evals_per_second_hist.png"),
	]

	for key, label, filename in time_keys:
		arr = np.array([safe_float(r.get(key)) for r in records], dtype=np.float64)
		save_hist(arr, f"{label} Distribution", label, OUTPUT_DIR / filename)

	mean_labels = ["Dice", "Yield", "HD", "Loss", "Score"]
	mean_values = [
		mean_or_nan([safe_float(r.get("delta_dice")) for r in records]),
		mean_or_nan([safe_float(r.get("delta_yield")) for r in records]),
		mean_or_nan([safe_float(r.get("delta_hd")) for r in records]),
		mean_or_nan([safe_float(r.get("delta_loss")) for r in records]),
		mean_or_nan([safe_float(r.get("delta_score")) for r in records]),
	]
	save_bar(mean_labels, mean_values, "Mean Greedy Delta Across All Runs", "Delta (Greedy - Original)", OUTPUT_DIR / "mean_delta_summary.png", rotation=0)

	time_mean_labels = [
		"Wall",
		"Greedy",
		"Optimize",
		"Recompute",
		"Metric",
		"Model",
		"Simulator",
		"Write",
	]
	time_mean_values = [
		mean_or_nan([safe_float(r.get("wall_clock_time")) for r in records]),
		mean_or_nan([safe_float(r.get("greedy_elapsed_time")) for r in records]),
		mean_or_nan([safe_float(r.get("optimization_time")) for r in records]),
		mean_or_nan([safe_float(r.get("recompute_time")) for r in records]),
		mean_or_nan([safe_float(r.get("metric_computation_time")) for r in records]),
		mean_or_nan([safe_float(r.get("model_forward_time")) for r in records]),
		mean_or_nan([safe_float(r.get("simulator_forward_time")) for r in records]),
		mean_or_nan([safe_float(r.get("artifact_write_time")) for r in records]),
	]
	save_bar(time_mean_labels, time_mean_values, "Mean Timing Summary", "Seconds", OUTPUT_DIR / "mean_timing_summary.png", rotation=0)
	save_time_breakdown(records, OUTPUT_DIR / "mean_timing_breakdown.png")

	by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for record in records:
		by_target[str(record["target_name"])].append(record)

	target_names = sorted(by_target.keys())
	target_delta_loss = [mean_or_nan([safe_float(r.get("delta_loss")) for r in by_target[name]]) for name in target_names]
	target_delta_yield = [mean_or_nan([safe_float(r.get("delta_yield")) for r in by_target[name]]) for name in target_names]
	target_loss_improvement = [mean_or_nan([safe_float(r.get("loss_improvement")) for r in by_target[name]]) for name in target_names]
	target_greedy_time = [mean_or_nan([safe_float(r.get("greedy_elapsed_time")) for r in by_target[name]]) for name in target_names]

	save_bar(target_names, target_delta_loss, "Mean Delta Loss by Target", "Greedy - Original", OUTPUT_DIR / "delta_loss_by_target.png")
	save_bar(target_names, target_delta_yield, "Mean Delta Yield by Target", "Greedy - Original", OUTPUT_DIR / "delta_yield_by_target.png")
	save_bar(target_names, target_loss_improvement, "Mean Loss Improvement by Target", "Original - Greedy", OUTPUT_DIR / "loss_improvement_by_target.png")
	save_bar(target_names, target_greedy_time, "Mean Greedy Elapsed Time by Target", "Seconds", OUTPUT_DIR / "greedy_elapsed_by_target.png")

	valid_rate = [
		sum(1 for r in by_target[name] if bool(r.get("grid_valid"))) / max(1, len(by_target[name]))
		for name in target_names
	]
	save_bar(target_names, valid_rate, "Grid Valid Rate by Target", "Rate", OUTPUT_DIR / "grid_valid_rate_by_target.png")

	overall = {
		"n_records": int(len(records)),
		"mean_delta_dice": mean_or_nan([safe_float(r.get("delta_dice")) for r in records]),
		"mean_delta_yield": mean_or_nan([safe_float(r.get("delta_yield")) for r in records]),
		"mean_delta_hd": mean_or_nan([safe_float(r.get("delta_hd")) for r in records]),
		"mean_delta_loss": mean_or_nan([safe_float(r.get("delta_loss")) for r in records]),
		"mean_delta_score": mean_or_nan([safe_float(r.get("delta_score")) for r in records]),
		"mean_loss_improvement": mean_or_nan([safe_float(r.get("loss_improvement")) for r in records]),
		"mean_greedy_elapsed_time": mean_or_nan([safe_float(r.get("greedy_elapsed_time")) for r in records]),
		"mean_wall_clock_time": mean_or_nan([safe_float(r.get("wall_clock_time")) for r in records]),
		"mean_normalized_greedy_elapsed_time": mean_or_nan([safe_float(r.get("normalized_greedy_elapsed_time")) for r in records]),
		"mean_normalized_wall_clock_time": mean_or_nan([safe_float(r.get("normalized_wall_clock_time")) for r in records]),
		"fraction_improved_loss": float(np.mean([safe_float(r.get("loss_improvement")) > 0 for r in records])),
		"fraction_improved_yield": float(np.mean([safe_float(r.get("delta_yield")) > 0 for r in records])),
		"fraction_improved_dice": float(np.mean([safe_float(r.get("delta_dice")) > 0 for r in records])),
		"fraction_reduced_hd": float(np.mean([safe_float(r.get("delta_hd")) < 0 for r in records])),
	}
	with open(OUTPUT_DIR / "greedy_overall_summary.json", "w", encoding="utf-8") as f:
		json.dump(overall, f, indent=2)

	print(f"loaded records: {len(records)}")
	print(f"plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
	main()
