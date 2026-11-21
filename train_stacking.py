"""Literature-grouped cross-validation for critical coagulation concentration."""

from __future__ import annotations

import argparse
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


PROJECT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DEFAULT_DATA_PATH = PROJECT_DIR / "Data.xlsx"
DEFAULT_SHEET_NAME = "Standardized data"
TARGET_COLUMN = "CCC (mM)"
LITERATURE_ID_COLUMN = "Literature ID"


RANDOM_STATE = 2026
OUTER_SPLITS = 5
INNER_SPLITS = 3
META_SPLITS = 4
N_JOBS = max(1, min(8, (os.cpu_count() or 4) - 1))

GROUP_SPLIT_RESTARTS_OUTER = 140
GROUP_SPLIT_RESTARTS_INNER = 50
GROUP_SPLIT_Y_BINS = 8

AUTO_SELECT_PREPROCESSOR = True
KNN_NEIGHBORS = 7
PREPROCESSOR_CANDIDATES = [
    {"Name": "ScaledKNN_NoIF", "Imputer": "knn"},
    {"Name": "Median_NoIF", "Imputer": "median"},
]

# Average two seeded GBR fits when refitting on each outer training set.
GBR_BAG_SEEDS = (0, 101)

# Stacking candidate grids and acceptance thresholds.
STACK_L2_GRID = (0.0005, 0.0015, 0.005, 0.015, 0.050, 0.150)
STACK_META_REPEATS = 6
STACK_STAGE1_SHRINK_GRID = (0.0, 0.35, 0.65, 1.0)
STACK_MAX_PARTNERS = 5
STACK_PARTNER_R2_TOL = 0.16
STACK_ABSOLUTE_MIN_R2 = 0.50
STACK_FINAL_META_WEIGHT = 0.60
STACK_BIAS_BOUND = 0.080
STACK_BIAS_L2 = 0.12
STACK_GBR_MIN_WEIGHT = 0.74
STACK_GBR_MAX_WEIGHT = 1.08

STACK_CHAMPION_GUARD_ENABLED = True
STACK_GUARD_REQUIRE_GBR_TRAIN_CHAMPION = True
STACK_GUARD_GBR_CHAMPION_TOL = 0.0020

STACK_GUARD_HARD_MIN_GBR_WEIGHT = 0.820
STACK_GUARD_STANDARD_MIN_GBR_WEIGHT = 0.860
STACK_GUARD_MAX_POSITIVE_PARTNER_WEIGHT = 0.240
STACK_GUARD_MAX_ABS_PARTNER_WEIGHT = 0.340
STACK_GUARD_COMPLEX_MIN_GBR_WEIGHT = 0.900
STACK_GUARD_INTERCEPT_EDGE = 0.095
STACK_GUARD_MIN_GAIN_AT_INTERCEPT_EDGE = 0.006

STACK_GUARD_STRONG_GAIN = 0.0080
STACK_GUARD_STRONG_WORST_REPEAT_GAIN = 0.0030

STACK_GUARD_WIDE_BLEND_PARTNER_WEIGHT = 0.150
STACK_GUARD_WIDE_BLEND_MIN_WORST_REPEAT_GAIN = 0.0010

STACK_META2_CORRECTOR_CANDIDATES = [
    {"Name": "Ridge_a0.5", "Family": "ridge", "alpha": 0.5},
    {"Name": "Ridge_a2", "Family": "ridge", "alpha": 2.0},
    {"Name": "Ridge_a8", "Family": "ridge", "alpha": 8.0},
    {"Name": "Ridge_a25", "Family": "ridge", "alpha": 25.0},
    {
        "Name": "MetaGBR_Stump_Huber", "Family": "gbr",
        "n_estimators": 45, "learning_rate": 0.025, "max_depth": 1,
        "min_samples_leaf": 24, "subsample": 0.82, "loss": "huber",
    },
    {
        "Name": "MetaGBR_Stump_Smooth", "Family": "gbr",
        "n_estimators": 70, "learning_rate": 0.016, "max_depth": 1,
        "min_samples_leaf": 30, "subsample": 0.88, "loss": "huber",
    },
    {
        "Name": "MetaGBR_Depth2_Tiny", "Family": "gbr",
        "n_estimators": 35, "learning_rate": 0.018, "max_depth": 2,
        "min_samples_leaf": 32, "subsample": 0.82, "loss": "huber",
    },
]
STACK_META2_STRENGTH_GRID = (0.20, 0.35, 0.50, 0.70, 0.90)
STACK_META2_CLIP_MULTIPLIER_GRID = (0.40, 0.65, 0.90, 1.20)
STACK_META2_REQUIRED_GAIN = 0.00010
STACK_META2_MIN_POSITIVE_REPEAT_RATIO = 0.50
STACK_META2_MIN_POSITIVE_FOLD_RATIO = 0.50
STACK_META2_MAX_WORST_REPEAT_LOSS = 0.014
STACK_META2_MAX_WORST_FOLD_LOSS = 0.045
STACK_META2_SOFT_MIN_GAIN = 0.00002
STACK_META2_SOFT_FRACTION = 0.50
STACK_META2_SOFT_MIN_POSITIVE_REPEAT_RATIO = 0.375
STACK_META2_SOFT_MAX_WORST_REPEAT_LOSS = 0.020

STACK_GATE_EXPERTS = ("RF", "ExtraTrees", "LGBM", "XGB", "GBR", "ConsensusMedian", "TreeMedian")
STACK_GATE_C_GRID = (0.25, 1.0, 4.0)
STACK_GATE_THRESHOLD_GRID = (0.55, 0.70)
STACK_GATE_STRENGTH_GRID = (0.20, 0.45, 0.70)
STACK_GATE_CLIP_MULTIPLIER_GRID = (0.70, 1.20)
STACK_FINAL_SCALE_GRID = (0.96, 0.99, 1.00, 1.03, 1.06, 1.10, 1.14)
STACK_FINAL_INTERCEPT_BOUND = 0.10
STACK_GATE_IMPROVEMENT_WEIGHT_CAP = 5.0
STACK_META3_REQUIRED_GAIN = 0.00012
STACK_META3_MIN_POSITIVE_REPEAT_RATIO = 0.625
STACK_META3_MIN_POSITIVE_FOLD_RATIO = 0.55
STACK_META3_MAX_WORST_REPEAT_LOSS = 0.012
STACK_META3_MAX_WORST_FOLD_LOSS = 0.045
STACK_META3_SOFT_MIN_GAIN = 0.00002
STACK_META3_SOFT_FRACTION = 0.50
STACK_META3_SOFT_MIN_POSITIVE_REPEAT_RATIO = 0.50
STACK_META3_SOFT_MAX_WORST_REPEAT_LOSS = 0.018
STACK_META3_FINAL_INTERCEPT_BLEND = 0.50

STACK_META4_MODES = ("Level", "LevelDisagreement", "LevelConsensus")
STACK_META4_LEVEL_BINS = (3, 4)
STACK_META4_SECONDARY_BINS = (1, 2)
STACK_META4_SHRINKAGE_GRID = (0.15, 0.30, 0.50)
STACK_META4_SMOOTHING_GRID = (12.0, 28.0)
STACK_META4_CLIP_MULTIPLIER_GRID = (0.45, 0.80)
STACK_META4_REQUIRED_INCREMENTAL_GAIN = 0.00008
STACK_META4_REQUIRED_TOTAL_GAIN = 0.00012
STACK_META4_MIN_POSITIVE_REPEAT_RATIO = 0.625
STACK_META4_MIN_POSITIVE_FOLD_RATIO = 0.55
STACK_META4_MAX_WORST_REPEAT_LOSS = 0.010
STACK_META4_MAX_WORST_FOLD_LOSS = 0.035
STACK_META4_SOFT_MIN_INCREMENTAL_GAIN = 0.00001
STACK_META4_SOFT_FRACTION = 0.45

STACK_META5_BLEND_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)
STACK_META5_DELTA_CLIP_GRID = (0.35, 0.60, 0.90, 1.20)
STACK_META5_SCALE_GRID = (0.98, 1.00, 1.02, 1.04, 1.06)
STACK_META5_INTERCEPT_BOUND = 0.080
STACK_META5_REQUIRED_TOTAL_GAIN = 0.00012
STACK_META5_REQUIRED_INCREMENTAL_GAIN = 0.00002
STACK_META5_MIN_POSITIVE_REPEAT_RATIO = 0.75
STACK_META5_MIN_POSITIVE_FOLD_RATIO = 0.60
STACK_META5_MAX_WORST_REPEAT_LOSS = 0.007
STACK_META5_MAX_WORST_FOLD_LOSS = 0.030
STACK_META5_SOFT_MIN_TOTAL_GAIN = 0.00002
STACK_META5_SOFT_FRACTION = 0.50
STACK_META5_FINAL_INTERCEPT_BLEND = 0.50

STACK_PARTNER_WEIGHT_BOUNDS = {
    "RF": (-0.05, 0.12),
    "ExtraTrees": (-0.08, 0.18),
    "LGBM": (-0.06, 0.12),
    "XGB": (-0.08, 0.18),
    "GBR": (-0.08, 0.18),
}

FEATURE_TYPES = {
    "Cation type": "categorical",
    "Anion type": "categorical",
    "Temperature": "numeric",
    "Initial Zeta potential": "numeric",
    "Particle type": "categorical",
    "pH": "numeric",
    "Plastic surface modification type": "categorical",
    "Plastic type": "categorical",
    "Plastic concentration": "numeric",
    "Initial particle size": "numeric",
    "Coexisting species type": "categorical",
    "Coexisting species concentration": "numeric",
    "Irradiation time": "numeric",
}

# Retain the original category sort order before one-hot encoding.
# Tree models with feature subsampling must see the same encoded column order.
ORIGINAL_CATEGORY_TOKENS = {
    "Plastic surface modification type": {"Not reported": "\u672a\u62a5\u544a"},
    "Coexisting species type": {"Absent": "\u65e0"},
}


RF_CANDIDATES = [
    {
        "n_estimators": 320, "max_depth": 7, "min_samples_split": 12,
        "min_samples_leaf": 6, "max_features": 0.50, "max_samples": 0.76,
    },
]

ET_CANDIDATES = [
    {
        "n_estimators": 380, "max_depth": 9, "min_samples_split": 12,
        "min_samples_leaf": 5, "max_features": 0.55, "bootstrap": False,
    },
]

GBR_PUBLIC_PARAMS = {
    "n_estimators": 150, "learning_rate": 0.014, "max_depth": 3,
    "min_samples_split": 3, "min_samples_leaf": 2, "subsample": 0.86,
    "max_features": 1.00, "loss": "squared_error",
}
GBR_STACK_PARAMS = {**GBR_PUBLIC_PARAMS, "n_estimators": 750}
GBR_CANDIDATES = [GBR_PUBLIC_PARAMS]
GBR750_CANDIDATES = [GBR_STACK_PARAMS]

LGBM_CANDIDATES = [
    {
        "n_estimators": 520, "learning_rate": 0.025, "num_leaves": 13,
        "max_depth": 4, "min_child_samples": 26, "subsample": 0.84,
        "colsample_bytree": 0.80, "reg_alpha": 0.15, "reg_lambda": 7.0,
    },
]

XGB_CANDIDATES = [
    {
        "n_estimators": 420, "learning_rate": 0.028, "max_depth": 3,
        "min_child_weight": 7.0, "subsample": 0.84, "colsample_bytree": 0.86,
        "reg_alpha": 0.10, "reg_lambda": 10.0, "gamma": 0.010,
    },
]

STACK_ANCHOR_NAME = "GBR750"

BASE_MODEL_NAMES = [
    "RF", "ExtraTrees", "LGBM", "XGB", "GBR", STACK_ANCHOR_NAME,
]
STANDALONE_REPORT_NAMES = (
    ("XGB", "XGB"),
    ("ExtraTrees", "EXT"),
    ("GBR", "GBR"),
    ("RF", "RF"),
    ("LGBM", "LGBM"),
)


def to_numeric_safe(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    cleaned = (
        series.astype(str).str.strip().str.replace(",", "", regex=False)
        .str.replace("，", "", regex=False).str.replace("%", "", regex=False)
        .replace({
            "": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan,
            "NA": np.nan, "N/A": np.nan, "-": np.nan, "--": np.nan,
            "<MISSING>": np.nan, "Not reported": np.nan,
        })
    )
    return pd.to_numeric(cleaned, errors="coerce").replace([np.inf, -np.inf], np.nan)


def normalize_category(value) -> str:
    if pd.isna(value):
        return "<MISSING>"
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text if text else "<MISSING>"


def signature_value(value) -> str:
    if pd.isna(value):
        return "<MISSING>"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return "<MISSING>" if np.isnan(value) else f"{float(value):.12g}"
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) < 2 or float(np.var(y_true)) <= 1e-15:
        return np.nan
    return float(r2_score(y_true, y_pred))


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "R2": safe_r2(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def transform_target(concentration: np.ndarray) -> np.ndarray:
    concentration = np.asarray(concentration, dtype=float)
    if np.any(concentration <= 0):
        raise ValueError("CCC values must be positive for log10 modeling.")
    return np.log10(concentration)


def group_balanced_weights(groups: np.ndarray) -> np.ndarray:
    groups = np.asarray(groups)
    _, codes = np.unique(groups, return_inverse=True)
    counts = np.bincount(codes).astype(float)
    weights = 1.0 / np.maximum(counts[codes], 1.0)
    return weights / np.mean(weights)


def normalized_cv_objective(
    y_full: np.ndarray,
    oof_prediction: np.ndarray,
    fold_records: Sequence[Dict[str, np.ndarray]],
) -> Dict[str, float]:
    global_scale = max(float(np.std(y_full, ddof=0)), 1e-8)
    pooled_nrmse = rmse(y_full, oof_prediction) / global_scale
    fold_nrmse: List[float] = []
    fold_r2: List[float] = []
    for item in fold_records:
        scale = max(float(np.std(item["y_true"], ddof=0)), 1e-8)
        fold_nrmse.append(rmse(item["y_true"], item["y_pred"]) / scale)
        fold_r2.append(safe_r2(item["y_true"], item["y_pred"]))
    mean_fold = float(np.mean(fold_nrmse))
    sd_fold = float(np.std(fold_nrmse, ddof=1)) if len(fold_nrmse) > 1 else 0.0
    worst_fold_nrmse = float(np.max(fold_nrmse))
    valid_fold_r2 = np.asarray([value for value in fold_r2 if np.isfinite(value)], dtype=float)
    mean_fold_r2 = float(np.mean(valid_fold_r2)) if len(valid_fold_r2) else np.nan
    worst_fold_r2 = float(np.min(valid_fold_r2)) if len(valid_fold_r2) else np.nan
    objective = 0.62 * pooled_nrmse + 0.28 * mean_fold + 0.10 * sd_fold
    return {
        "Objective": float(objective),
        "OOF_R2": safe_r2(y_full, oof_prediction),
        "OOF_RMSE": rmse(y_full, oof_prediction),
        "OOF_MAE": float(mean_absolute_error(y_full, oof_prediction)),
        "Pooled_NRMSE": float(pooled_nrmse),
        "Mean_Fold_NRMSE": mean_fold,
        "SD_Fold_NRMSE": sd_fold,
        "Worst_Fold_NRMSE": worst_fold_nrmse,
        "Mean_Fold_R2": mean_fold_r2,
        "Worst_Fold_R2": worst_fold_r2,
    }

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> None:
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first == root_second:
            return
        if self.rank[root_first] < self.rank[root_second]:
            self.parent[root_first] = root_second
        elif self.rank[root_first] > self.rank[root_second]:
            self.parent[root_second] = root_first
        else:
            self.parent[root_second] = root_first
            self.rank[root_first] += 1


def build_strict_groups(literature_ids: np.ndarray, signatures: Sequence[Tuple]) -> np.ndarray:
    uf = UnionFind(len(literature_ids))
    first_literature: Dict[object, int] = {}
    first_signature: Dict[Tuple, int] = {}
    for index, literature in enumerate(literature_ids):
        if literature in first_literature:
            uf.union(index, first_literature[literature])
        else:
            first_literature[literature] = index
    for index, signature in enumerate(signatures):
        if signature in first_signature:
            uf.union(index, first_signature[signature])
        else:
            first_signature[signature] = index
    roots = [uf.find(index) for index in range(len(literature_ids))]
    root_to_group = {root: group_id for group_id, root in enumerate(sorted(set(roots)))}
    return np.asarray([root_to_group[root] for root in roots], dtype=int)


def _regression_bins(y: np.ndarray, n_bins: int) -> np.ndarray:
    unique_count = len(np.unique(y))
    q = max(2, min(int(n_bins), unique_count))
    try:
        return np.asarray(pd.qcut(y, q=q, labels=False, duplicates="drop"), dtype=int)
    except Exception:
        ranks = pd.Series(y).rank(method="average", pct=True).to_numpy()
        return np.floor(np.minimum(ranks, 0.999999) * q).astype(int)


def _assignment_score(
    fold_rows: np.ndarray,
    fold_y_sum: np.ndarray,
    fold_y_sq: np.ndarray,
    fold_bins: np.ndarray,
    fold_group_count: np.ndarray,
    global_mean: float,
    global_sd: float,
    global_bin_prop: np.ndarray,
) -> float:
    ideal_rows = max(float(np.sum(fold_rows)) / len(fold_rows), 1.0)
    size_score = float(np.mean(np.abs(fold_rows - ideal_rows) / ideal_rows))
    valid = fold_rows > 0
    means = np.full(len(fold_rows), global_mean, dtype=float)
    means[valid] = fold_y_sum[valid] / fold_rows[valid]
    mean_score = float(np.mean(np.abs(means - global_mean) / global_sd))
    variances = np.zeros(len(fold_rows), dtype=float)
    variances[valid] = fold_y_sq[valid] / fold_rows[valid] - means[valid] ** 2
    sds = np.sqrt(np.maximum(variances, 0.0))
    sd_score = float(np.mean(np.abs(sds - global_sd) / global_sd))
    bin_scores = []
    for fold in range(len(fold_rows)):
        if fold_rows[fold] <= 0:
            bin_scores.append(1.0)
        else:
            bin_scores.append(float(np.mean(np.abs(fold_bins[fold] / fold_rows[fold] - global_bin_prop))))
    ideal_groups = max(float(np.sum(fold_group_count)) / len(fold_group_count), 1.0)
    group_score = float(np.mean(np.abs(fold_group_count - ideal_groups) / ideal_groups))
    return size_score + 1.25 * mean_score + 0.30 * sd_score + 2.0 * float(np.mean(bin_scores)) + 0.10 * group_score


def target_balanced_group_folds(
    groups: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    seed: int,
    n_restarts: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray(groups, dtype=int)
    y = np.asarray(y, dtype=float)
    unique_groups = np.unique(groups)
    if len(unique_groups) < n_splits:
        raise ValueError(f"Only {len(unique_groups)} independent groups for {n_splits} folds.")
    bins = _regression_bins(y, GROUP_SPLIT_Y_BINS)
    n_actual_bins = int(np.max(bins)) + 1
    records = []
    for group_value in unique_groups:
        index = np.where(groups == group_value)[0]
        group_y = y[index]
        records.append({
            "Group": int(group_value),
            "Rows": int(len(index)),
            "Y_Sum": float(np.sum(group_y)),
            "Y_Sq": float(np.sum(group_y ** 2)),
            "Y_Mean": float(np.mean(group_y)),
            "Bins": np.bincount(bins[index], minlength=n_actual_bins).astype(float),
        })
    global_mean = float(np.mean(y))
    global_sd = max(float(np.std(y, ddof=0)), 1e-8)
    global_bin_prop = np.bincount(bins, minlength=n_actual_bins).astype(float) / len(y)
    master = np.random.default_rng(seed)
    best_score = np.inf
    best_groups: Optional[List[set]] = None
    for _ in range(int(n_restarts)):
        rng = np.random.default_rng(int(master.integers(0, 2**31 - 1)))
        ordered = sorted(
            records,
            key=lambda item: (-item["Rows"], -abs(item["Y_Mean"] - global_mean), rng.random()),
        )
        fold_groups = [set() for _ in range(n_splits)]
        fold_rows = np.zeros(n_splits)
        fold_y_sum = np.zeros(n_splits)
        fold_y_sq = np.zeros(n_splits)
        fold_bins = np.zeros((n_splits, n_actual_bins))
        fold_group_count = np.zeros(n_splits)
        first_order = rng.permutation(n_splits)
        for position in range(n_splits):
            item = ordered[position]
            chosen = int(first_order[position])
            fold_groups[chosen].add(item["Group"])
            fold_rows[chosen] += item["Rows"]
            fold_y_sum[chosen] += item["Y_Sum"]
            fold_y_sq[chosen] += item["Y_Sq"]
            fold_bins[chosen] += item["Bins"]
            fold_group_count[chosen] += 1
        for item in ordered[n_splits:]:
            candidate_scores = []
            for fold in range(n_splits):
                rows_new = fold_rows.copy(); rows_new[fold] += item["Rows"]
                y_sum_new = fold_y_sum.copy(); y_sum_new[fold] += item["Y_Sum"]
                y_sq_new = fold_y_sq.copy(); y_sq_new[fold] += item["Y_Sq"]
                bins_new = fold_bins.copy(); bins_new[fold] += item["Bins"]
                groups_new = fold_group_count.copy(); groups_new[fold] += 1
                candidate_scores.append(_assignment_score(
                    rows_new, y_sum_new, y_sq_new, bins_new, groups_new,
                    global_mean, global_sd, global_bin_prop,
                ))
            minimum = float(np.min(candidate_scores))
            choices = np.where(np.isclose(candidate_scores, minimum, rtol=0, atol=1e-12))[0]
            chosen = int(rng.choice(choices))
            fold_groups[chosen].add(item["Group"])
            fold_rows[chosen] += item["Rows"]
            fold_y_sum[chosen] += item["Y_Sum"]
            fold_y_sq[chosen] += item["Y_Sq"]
            fold_bins[chosen] += item["Bins"]
            fold_group_count[chosen] += 1
        score = _assignment_score(
            fold_rows, fold_y_sum, fold_y_sq, fold_bins, fold_group_count,
            global_mean, global_sd, global_bin_prop,
        )
        if score < best_score:
            best_score = score
            best_groups = [set(values) for values in fold_groups]
    if best_groups is None:
        raise RuntimeError("Could not construct target-balanced group folds.")
    splits = []
    assignment = np.zeros(len(groups), dtype=int)
    for fold_groups in best_groups:
        test_index = np.where(np.isin(groups, list(fold_groups)))[0]
        train_index = np.where(~np.isin(groups, list(fold_groups)))[0]
        if set(groups[train_index]).intersection(set(groups[test_index])):
            raise RuntimeError("Evaluation groups overlap between splits.")
        assignment[test_index] += 1
        splits.append((train_index, test_index))
    if not np.all(assignment == 1):
        raise RuntimeError("Each row must appear in exactly one test fold.")
    return splits


def make_group_folds(
    groups: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    seed: int,
    outer: bool,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    actual_splits = min(int(n_splits), int(pd.Series(groups).nunique()))
    if actual_splits < 2:
        raise ValueError("At least two independent groups are required for cross-validation.")
    restarts = GROUP_SPLIT_RESTARTS_OUTER if outer else GROUP_SPLIT_RESTARTS_INNER
    return target_balanced_group_folds(groups, y, actual_splits, seed, restarts)


def make_onehot() -> OneHotEncoder:
    return OneHotEncoder(
        handle_unknown="ignore", min_frequency=2, sparse_output=False, dtype=np.float32
    )


def make_numeric_imputer(strategy: str) -> SimpleImputer:
    return SimpleImputer(strategy=strategy, keep_empty_features=True)


def make_knn_imputer() -> KNNImputer:
    return KNNImputer(n_neighbors=KNN_NEIGHBORS, weights="distance", keep_empty_features=True)


def build_preprocessor(
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    mode: Mapping,
) -> ColumnTransformer:
    transformers = []
    if numeric_columns:
        if mode["Imputer"] == "knn":
            numeric_pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("imputer", make_knn_imputer()),
            ])
        else:
            numeric_pipeline = Pipeline([
                ("imputer", make_numeric_imputer("median")),
                ("scaler", StandardScaler()),
            ])
        transformers.append(("numeric", numeric_pipeline, list(numeric_columns)))
    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="<MISSING>")),
            ("onehot", make_onehot()),
        ])
        transformers.append(("categorical", categorical_pipeline, list(categorical_columns)))
    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)


def load_and_prepare_data(
    data_path: Path = DEFAULT_DATA_PATH,
    sheet_name: str = DEFAULT_SHEET_NAME,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict]:
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Workbook not found: {data_path.resolve()}. "
            "Place Data.xlsx beside the script, or in the current directory when running "
            "from a notebook cell. Use --data to specify another path."
        )
    with pd.ExcelFile(data_path, engine="openpyxl") as workbook:
        if sheet_name not in workbook.sheet_names:
            raise ValueError(
                f"Worksheet {sheet_name!r} was not found in {data_path.name}. "
                f"Available sheets: {workbook.sheet_names}. "
                "Use the English Data.xlsx from the release package."
            )
        raw = workbook.parse(sheet_name=sheet_name)
    raw.columns = [re.sub(r"\s+", " ", str(column)).strip() for column in raw.columns]
    target_column = TARGET_COLUMN
    group_column = LITERATURE_ID_COLUMN
    required_columns = [target_column, group_column, *FEATURE_TYPES]
    missing_columns = [column for column in required_columns if column not in raw.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {sheet_name!r}: {missing_columns}")
    target_numeric = to_numeric_safe(raw[target_column])
    valid = target_numeric.notna() & (target_numeric > 0)

    source_data = raw.loc[valid].copy().reset_index(drop=True)
    y_model = transform_target(target_numeric.loc[valid].to_numpy(dtype=float))

    literature_ids = []
    for row_index, value in enumerate(source_data[group_column]):
        if pd.isna(value) or not str(value).strip():
            literature_ids.append(f"<MISSING_{row_index}>")
        else:
            literature_ids.append(str(value).strip())
    literature_ids = np.asarray(literature_ids, dtype=object)

    model_columns = [column for column in raw.columns if column in FEATURE_TYPES]
    X = source_data[model_columns].copy()
    numeric_columns = [column for column in model_columns if FEATURE_TYPES[column] == "numeric"]
    categorical_columns = [
        column for column in model_columns if FEATURE_TYPES[column] == "categorical"
    ]
    for column in numeric_columns:
        X[column] = to_numeric_safe(X[column])
    for column in categorical_columns:
        X[column] = X[column].map(normalize_category)
        if column in ORIGINAL_CATEGORY_TOKENS:
            X[column] = X[column].replace(ORIGINAL_CATEGORY_TOKENS[column])
    signatures = [
        tuple(signature_value(value) for value in row)
        for row in X.itertuples(index=False, name=None)
    ]
    strict_groups = build_strict_groups(literature_ids, signatures)

    metadata = {
        "workbook_rows": len(raw),
        "valid_rows": len(X),
        "literature_count": int(pd.Series(literature_ids).nunique()),
        "strict_group_count": int(pd.Series(strict_groups).nunique()),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
    }
    return X, y_model, strict_groups, metadata


def make_rf(seed: int, params: Mapping) -> RandomForestRegressor:
    return RandomForestRegressor(bootstrap=True, random_state=seed, n_jobs=N_JOBS, **params)


def make_extra_trees(seed: int, params: Mapping) -> ExtraTreesRegressor:
    return ExtraTreesRegressor(random_state=seed, n_jobs=N_JOBS, **params)


def make_gbr(seed: int, params: Mapping) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(random_state=seed, **params)


def make_lgbm(seed: int, params: Mapping) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression", random_state=seed, n_jobs=N_JOBS, verbosity=-1,
        force_col_wise=True, subsample_freq=1, **params,
    )


def make_xgb(seed: int, params: Mapping) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=seed,
        n_jobs=N_JOBS,
        tree_method="hist",
        verbosity=0,
        **params,
    )


@dataclass
class InnerFoldData:
    train_index: np.ndarray
    valid_index: np.ndarray
    X_train: np.ndarray
    X_valid: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray


def build_inner_cache(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    seed: int,
    mode: Mapping,
    precomputed_splits: Optional[Sequence[Tuple[np.ndarray, np.ndarray]]] = None,
) -> List[InnerFoldData]:
    inner_splits = (
        list(precomputed_splits)
        if precomputed_splits is not None
        else make_group_folds(groups_train, y_train, INNER_SPLITS, seed, outer=False)
    )
    cache: List[InnerFoldData] = []
    for fit_index, valid_index in inner_splits:
        fit_index = np.asarray(fit_index, dtype=int)
        valid_index = np.asarray(valid_index, dtype=int)
        if set(groups_train[fit_index]).intersection(set(groups_train[valid_index])):
            raise RuntimeError("Inner training and validation groups overlap.")
        preprocessor = build_preprocessor(numeric_columns, categorical_columns, mode)
        X_fit = np.asarray(
            preprocessor.fit_transform(X_train.iloc[fit_index], y_train[fit_index]),
            dtype=np.float32,
        )
        X_valid = np.asarray(preprocessor.transform(X_train.iloc[valid_index]), dtype=np.float32)
        cache.append(InnerFoldData(
            train_index=fit_index,
            valid_index=valid_index,
            X_train=X_fit,
            X_valid=X_valid,
            y_train=y_train[fit_index],
            y_valid=y_train[valid_index],
        ))
    return cache

def evaluate_candidate_grid(
    inner_cache: Sequence[InnerFoldData],
    y_full: np.ndarray,
    candidates: Sequence[Mapping],
    builder: Callable[[int, Mapping], object],
    seed: int,
) -> Dict:
    best: Optional[Dict] = None
    for candidate_index, params in enumerate(candidates, start=1):
        oof = np.full(len(y_full), np.nan, dtype=float)
        fold_records = []
        for fold_index, item in enumerate(inner_cache, start=1):
            model = builder(seed + candidate_index * 10000 + fold_index * 1000, params)
            model.fit(item.X_train, item.y_train)
            prediction = np.asarray(model.predict(item.X_valid), dtype=float)
            oof[item.valid_index] = prediction
            fold_records.append({"y_true": item.y_valid, "y_pred": prediction})
        details = {
            **normalized_cv_objective(y_full, oof, fold_records),
            "Params": dict(params),
            "OOF_Prediction": oof.copy(),
        }
        if best is None or details["Objective"] < best["Objective"]:
            best = details
    if best is None:
        raise RuntimeError("The model parameter grid is empty.")
    return best


def select_preprocessor_and_cache(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    seed: int,
) -> Tuple[Mapping, List[InnerFoldData]]:
    modes = PREPROCESSOR_CANDIDATES if AUTO_SELECT_PREPROCESSOR else PREPROCESSOR_CANDIDATES[:1]
    probe_params = GBR750_CANDIDATES[0]
    best_mode = None
    best_cache = None
    best_objective = np.inf
    for mode_index, mode in enumerate(modes, start=1):
        cache = build_inner_cache(
            X_train, y_train, groups_train, numeric_columns, categorical_columns,
            seed + mode_index * 10000, mode,
        )
        probe = evaluate_candidate_grid(
            cache, y_train, [probe_params], make_gbr,
            seed + mode_index * 100000,
        )
        if probe["Objective"] < best_objective:
            best_objective = probe["Objective"]
            best_mode = mode
            best_cache = cache
    if best_mode is None or best_cache is None:
        raise RuntimeError("Preprocessor selection failed.")
    return best_mode, best_cache


def _safe_corr(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    mask = np.isfinite(first) & np.isfinite(second)
    if int(np.sum(mask)) < 3:
        return 0.0
    first = first[mask]
    second = second[mask]
    if float(np.std(first)) <= 1e-12 or float(np.std(second)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _stack_model_statistics(
    prediction_matrix: np.ndarray,
    y: np.ndarray,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    model_names: Sequence[str],
) -> pd.DataFrame:
    prediction_matrix = np.asarray(prediction_matrix, dtype=float)
    y = np.asarray(y, dtype=float)
    names = list(model_names)
    if STACK_ANCHOR_NAME not in names:
        raise ValueError("The first-level models must include the GBR750 anchor.")
    gbr_index = names.index(STACK_ANCHOR_NAME)
    gbr_prediction = prediction_matrix[:, gbr_index]
    target_residual = y - gbr_prediction
    rows = []
    for model_index, name in enumerate(names):
        fold_r2 = [
            safe_r2(y[valid_index], prediction_matrix[valid_index, model_index])
            for _, valid_index in splits
        ]
        finite = np.asarray([value for value in fold_r2 if np.isfinite(value)], dtype=float)
        pooled = safe_r2(y, prediction_matrix[:, model_index])
        mean_fold = float(np.mean(finite)) if len(finite) else -np.inf
        worst_fold = float(np.min(finite)) if len(finite) else -np.inf
        sd_fold = float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0
        if model_index == gbr_index:
            residual_signal = 0.0
            prediction_corr = 1.0
        else:
            correction_signal = prediction_matrix[:, model_index] - gbr_prediction
            residual_signal = _safe_corr(correction_signal, target_residual)
            prediction_corr = _safe_corr(prediction_matrix[:, model_index], gbr_prediction)
        robust_score = (
            0.50 * pooled
            + 0.25 * mean_fold
            + 0.17 * worst_fold
            - 0.05 * sd_fold
            + 0.10 * abs(residual_signal)
            + 0.03 * (1.0 - abs(prediction_corr))
        )
        rows.append({
            "Model": str(name),
            "Model_Index": int(model_index),
            "Pooled_R2": float(pooled),
            "Mean_Fold_R2": mean_fold,
            "Worst_Fold_R2": worst_fold,
            "SD_Fold_R2": sd_fold,
            "Residual_Signal": float(residual_signal),
            "Corr_vs_GBR": float(prediction_corr),
            "Robust_Score": float(robust_score),
        })
    return pd.DataFrame(rows).sort_values(
        ["Robust_Score", "Pooled_R2"], ascending=[False, False]
    ).reset_index(drop=True)


def _select_residual_partners(
    prediction_matrix: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_names: Sequence[str],
    seed: int,
) -> Dict:
    prediction_matrix = np.asarray(prediction_matrix, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    names = list(model_names)
    gbr_index = names.index(STACK_ANCHOR_NAME)
    split_count = min(META_SPLITS, len(np.unique(groups)))
    splits = make_group_folds(groups, y, split_count, seed, outer=False)
    stats = _stack_model_statistics(prediction_matrix, y, splits, names)
    gbr_r2 = float(stats.loc[stats["Model"] == STACK_ANCHOR_NAME, "Pooled_R2"].iloc[0])

    eligible = stats[
        (stats["Model"] != STACK_ANCHOR_NAME)
        & (stats["Pooled_R2"] >= max(STACK_ABSOLUTE_MIN_R2, gbr_r2 - STACK_PARTNER_R2_TOL))
        & (stats["Worst_Fold_R2"] > -0.10)
    ].copy()
    if not eligible.empty:
        eligible["Partner_Score"] = (
            0.55 * eligible["Robust_Score"]
            + 0.23 * eligible["Residual_Signal"].abs()
            + 0.12 * (1.0 - eligible["Corr_vs_GBR"].abs())
            + 0.10 * eligible["Worst_Fold_R2"]
        )
        eligible = eligible.sort_values(
            ["Partner_Score", "Pooled_R2", "Worst_Fold_R2"],
            ascending=[False, False, False],
        )
    partner_indices = eligible["Model_Index"].astype(int).tolist()[:STACK_MAX_PARTNERS]
    if not partner_indices:
        fallback_partner = stats[stats["Model"] != STACK_ANCHOR_NAME].head(1)
        if not fallback_partner.empty:
            partner_indices = [int(fallback_partner.iloc[0]["Model_Index"])]

    selected_indices = [gbr_index] + [index for index in partner_indices if index != gbr_index]
    return {
        "Stats": stats,
        "GBR_Index": gbr_index,
        "GBR_R2": gbr_r2,
        "Partner_Indices": partner_indices,
        "Selected_Indices": selected_indices,
        "Selected_Models": [names[index] for index in selected_indices],
    }


def _weight_bounds_for_models(model_names: Sequence[str]) -> List[Tuple[float, float]]:
    bounds: List[Tuple[float, float]] = []
    for name in model_names:
        if str(name) == STACK_ANCHOR_NAME:
            bounds.append((STACK_GBR_MIN_WEIGHT, STACK_GBR_MAX_WEIGHT))
        else:
            bounds.append(tuple(STACK_PARTNER_WEIGHT_BOUNDS.get(str(name), (-0.05, 0.10))))
    return bounds


def _minimize_slsqp(
    objective: Callable[[np.ndarray], float],
    initial_parameters: np.ndarray,
    *,
    bounds: Sequence[Tuple[float, float]],
    constraints: Sequence[Mapping],
    options: Mapping,
):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"^Values in x were outside bounds during a minimize step, clipping to bounds$",
            category=RuntimeWarning,
            module=r"^scipy\.optimize\._slsqp_py$",
        )
        return minimize(
            objective,
            initial_parameters,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options=options,
        )


def _project_anchor_weights(candidate_weights: np.ndarray, model_names: Sequence[str]) -> np.ndarray:
    candidate_weights = np.asarray(candidate_weights, dtype=float)
    names = list(model_names)
    gbr_index = names.index(STACK_ANCHOR_NAME)
    bounds = _weight_bounds_for_models(names)
    x0 = np.zeros(len(names), dtype=float)
    for index, (low, high) in enumerate(bounds):
        x0[index] = float(np.clip(candidate_weights[index], low, high))
    x0[gbr_index] += 1.0 - float(np.sum(x0))
    x0[gbr_index] = float(np.clip(x0[gbr_index], *bounds[gbr_index]))
    result = _minimize_slsqp(
        lambda weights: float(np.sum((weights - candidate_weights) ** 2)),
        x0,
        bounds=bounds,
        constraints=[{"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)}],
        options={"maxiter": 700, "ftol": 1e-12, "disp": False},
    )
    if result.success and np.all(np.isfinite(result.x)):
        return np.asarray(result.x, dtype=float)
    fallback = np.zeros(len(names), dtype=float)
    fallback[gbr_index] = 1.0
    return fallback


def fit_gbr_anchored_weights(
    prediction_matrix: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_names: Sequence[str],
    l2: float,
    seed: int,
) -> Dict:
    prediction_matrix = np.asarray(prediction_matrix, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    names = list(model_names)
    context = _select_residual_partners(prediction_matrix, y, groups, names, seed)
    gbr_index = int(context["GBR_Index"])
    selected = list(context["Selected_Indices"])
    selected_names = [names[index] for index in selected]
    selected_bounds = _weight_bounds_for_models(selected_names)
    sample_weight = group_balanced_weights(groups)

    x0_weights = np.zeros(len(selected), dtype=float)
    gbr_position = selected.index(gbr_index)
    x0_weights[gbr_position] = 0.84
    partner_positions = [position for position, index in enumerate(selected) if index != gbr_index]
    if partner_positions:
        per_partner = 0.16 / len(partner_positions)
        for position in partner_positions:
            low, high = selected_bounds[position]
            x0_weights[position] = float(np.clip(per_partner, low, high))
    x0_weights[gbr_position] += 1.0 - float(np.sum(x0_weights))
    x0 = np.concatenate([x0_weights, [0.0]])
    bounds = selected_bounds + [(-STACK_BIAS_BOUND, STACK_BIAS_BOUND)]

    def objective(parameters: np.ndarray) -> float:
        weights = parameters[:-1]
        bias = float(parameters[-1])
        prediction = prediction_matrix[:, selected] @ weights + bias
        residual = y - prediction
        weighted_mse = float(np.average(residual ** 2, weights=sample_weight))
        partner_weights = np.delete(weights, gbr_position)
        return (
            weighted_mse
            + float(l2) * float(np.sum(partner_weights ** 2))
            + STACK_BIAS_L2 * bias * bias
        )

    result = _minimize_slsqp(
        objective,
        x0,
        bounds=bounds,
        constraints=[{"type": "eq", "fun": lambda parameters: float(np.sum(parameters[:-1]) - 1.0)}],
        options={"maxiter": 900, "ftol": 1e-12, "disp": False},
    )

    full_weights = np.zeros(len(names), dtype=float)
    if result.success and np.all(np.isfinite(result.x)):
        selected_weights = np.asarray(result.x[:-1], dtype=float)
        bias = float(np.clip(result.x[-1], -STACK_BIAS_BOUND, STACK_BIAS_BOUND))
        for selected_position, model_index in enumerate(selected):
            full_weights[model_index] = selected_weights[selected_position]
        full_weights = _project_anchor_weights(full_weights, names)
    else:
        full_weights[gbr_index] = 1.0
        bias = 0.0
    return {"Weights": full_weights, "Bias": bias, **context}


def _effective_stage1_parameters(
    learned_weights: np.ndarray,
    learned_bias: float,
    shrinkage: float,
    model_names: Sequence[str],
) -> Tuple[np.ndarray, float]:
    names = list(model_names)
    gbr_index = names.index(STACK_ANCHOR_NAME)
    gbr_only = np.zeros(len(names), dtype=float)
    gbr_only[gbr_index] = 1.0
    effective_weights = gbr_only + float(shrinkage) * (np.asarray(learned_weights) - gbr_only)
    effective_weights = _project_anchor_weights(effective_weights, names)
    effective_bias = float(np.clip(float(shrinkage) * float(learned_bias), -STACK_BIAS_BOUND, STACK_BIAS_BOUND))
    return effective_weights, effective_bias


def _build_meta2_features(prediction_matrix: np.ndarray, model_names: Sequence[str]) -> np.ndarray:
    prediction_matrix = np.asarray(prediction_matrix, dtype=float)
    names = list(model_names)
    gbr_index = names.index(STACK_ANCHOR_NAME)
    gbr = prediction_matrix[:, gbr_index]
    differences = np.column_stack([
        prediction_matrix[:, index] - gbr
        for index in range(prediction_matrix.shape[1]) if index != gbr_index
    ])
    mean_prediction = np.mean(prediction_matrix, axis=1)
    median_prediction = np.median(prediction_matrix, axis=1)
    std_prediction = np.std(prediction_matrix, axis=1)
    min_prediction = np.min(prediction_matrix, axis=1)
    max_prediction = np.max(prediction_matrix, axis=1)
    q25 = np.quantile(prediction_matrix, 0.25, axis=1)
    q75 = np.quantile(prediction_matrix, 0.75, axis=1)
    tree_indices = [index for index, name in enumerate(names) if name in {"RF", "ExtraTrees", "LGBM", "XGB", "GBR", STACK_ANCHOR_NAME}]
    tree_matrix = prediction_matrix[:, tree_indices]
    tree_mean = np.mean(tree_matrix, axis=1)
    tree_std = np.std(tree_matrix, axis=1)
    consensus_gap = gbr - median_prediction
    magnitude_features = np.column_stack([
        gbr,
        gbr ** 2,
        np.abs(gbr - mean_prediction),
        std_prediction,
        std_prediction * np.maximum(np.abs(gbr), 0.25),
    ])
    summary = np.column_stack([
        mean_prediction, median_prediction, std_prediction,
        min_prediction, max_prediction, max_prediction - min_prediction,
        q25, q75, q75 - q25, tree_mean, tree_std, consensus_gap,
    ])
    return np.column_stack([prediction_matrix, differences, summary, magnitude_features])


def _robust_residual_scale(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return 0.20
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 0.03:
        scale = float(np.std(finite))
    if not np.isfinite(scale) or scale < 0.03:
        scale = 0.20
    return float(scale)


def _fit_meta2_corrector(
    features: np.ndarray,
    residual: np.ndarray,
    groups: np.ndarray,
    config: Mapping,
    seed: int,
) -> Dict:
    features = np.asarray(features, dtype=float)
    residual = np.asarray(residual, dtype=float)
    sample_weight = group_balanced_weights(np.asarray(groups))
    family = str(config["Family"])
    if family == "ridge":
        scaler = StandardScaler()
        transformed = scaler.fit_transform(features)
        estimator = Ridge(alpha=float(config["alpha"]), fit_intercept=True)
        estimator.fit(transformed, residual, sample_weight=sample_weight)
        return {"Family": family, "Scaler": scaler, "Estimator": estimator}
    if family == "gbr":
        estimator = GradientBoostingRegressor(
            n_estimators=int(config["n_estimators"]),
            learning_rate=float(config["learning_rate"]),
            max_depth=int(config["max_depth"]),
            min_samples_leaf=int(config["min_samples_leaf"]),
            subsample=float(config["subsample"]),
            loss=str(config["loss"]),
            random_state=int(seed),
            max_features=0.75,
        )
        estimator.fit(features, residual, sample_weight=sample_weight)
        return {"Family": family, "Scaler": None, "Estimator": estimator}
    raise ValueError(f"Unknown stage-2 residual corrector: {family}")


def _predict_meta2_corrector(fitted: Mapping, features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    scaler = fitted.get("Scaler")
    transformed = scaler.transform(features) if scaler is not None else features
    return np.asarray(fitted["Estimator"].predict(transformed), dtype=float)


def _select_stage1_configuration(
    base_oof: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_names: Sequence[str],
    seed: int,
) -> Dict:
    names = list(model_names)
    gbr_index = names.index(STACK_ANCHOR_NAME)
    gbr_oof = base_oof[:, gbr_index]
    gbr_r2 = safe_r2(y, gbr_oof)
    meta_split_count = min(META_SPLITS, len(np.unique(groups)))
    candidate_rows = []
    payloads: Dict[Tuple[float, float], Dict] = {}

    for l2 in STACK_L2_GRID:
        accum = {
            float(shrink): {
                "Prediction_Sum": np.zeros(len(y), dtype=float),
                "Prediction_Count": np.zeros(len(y), dtype=float),
                "Repeat_Gains": [], "Fold_Gains": [],
            }
            for shrink in STACK_STAGE1_SHRINK_GRID
        }
        all_weights: List[np.ndarray] = []
        all_biases: List[float] = []
        for repeat in range(STACK_META_REPEATS):
            repeat_seed = seed + 13007 + repeat * 7919
            splits = _make_distinct_meta_folds(groups, y, meta_split_count, repeat_seed)
            repeat_predictions = {
                float(shrink): np.full(len(y), np.nan, dtype=float)
                for shrink in STACK_STAGE1_SHRINK_GRID
            }
            for fold_index, (train_index, valid_index) in enumerate(splits, start=1):
                fit = fit_gbr_anchored_weights(
                    base_oof[train_index], y[train_index], groups[train_index],
                    names, float(l2), repeat_seed + fold_index * 103,
                )
                all_weights.append(np.asarray(fit["Weights"], dtype=float))
                all_biases.append(float(fit["Bias"]))
                linear_valid = base_oof[valid_index] @ fit["Weights"] + float(fit["Bias"])
                baseline_valid = base_oof[valid_index, gbr_index]
                for shrink in STACK_STAGE1_SHRINK_GRID:
                    shrink = float(shrink)
                    prediction = baseline_valid + shrink * (linear_valid - baseline_valid)
                    repeat_predictions[shrink][valid_index] = prediction
                    accum[shrink]["Fold_Gains"].append(
                        safe_r2(y[valid_index], prediction) - safe_r2(y[valid_index], baseline_valid)
                    )
            for shrink in STACK_STAGE1_SHRINK_GRID:
                shrink = float(shrink)
                repeat_prediction = repeat_predictions[shrink]
                accum[shrink]["Prediction_Sum"] += repeat_prediction
                accum[shrink]["Prediction_Count"] += np.isfinite(repeat_prediction).astype(float)
                accum[shrink]["Repeat_Gains"].append(safe_r2(y, repeat_prediction) - gbr_r2)

        average_weights = _project_anchor_weights(np.mean(np.vstack(all_weights), axis=0), names)
        average_bias = float(np.mean(all_biases))
        for shrink in STACK_STAGE1_SHRINK_GRID:
            shrink = float(shrink)
            item = accum[shrink]
            meta_oof = item["Prediction_Sum"] / np.maximum(item["Prediction_Count"], 1.0)
            metric = metrics(y, meta_oof)
            repeat_array = np.asarray(item["Repeat_Gains"], dtype=float)
            fold_array = np.asarray(item["Fold_Gains"], dtype=float)
            gain = float(metric["R2"] - gbr_r2)
            row = {
                "L2": float(l2), "Stage1_Shrinkage": shrink,
                "R2": float(metric["R2"]), "RMSE": float(metric["RMSE"]), "MAE": float(metric["MAE"]),
                "Meta_Gain": gain,
                "Mean_Repeat_Gain": float(np.mean(repeat_array)),
                "SD_Repeat_Gain": float(np.std(repeat_array, ddof=1)) if len(repeat_array) > 1 else 0.0,
                "Positive_Repeats": int(np.sum(repeat_array > 0.0)),
                "Worst_Repeat_Gain": float(np.min(repeat_array)),
                "Positive_Folds": int(np.sum(fold_array > 0.0)),
                "Total_Folds": int(len(fold_array)),
                "Worst_Fold_Gain": float(np.min(fold_array)),
            }
            candidate_rows.append(row)
            payloads[(float(l2), shrink)] = {
                "Meta_OOF": meta_oof,
                "Average_Weights": average_weights,
                "Average_Bias": average_bias,
            }

    table = pd.DataFrame(candidate_rows)
    table["Robust_Select_Score"] = (
        table["R2"] + 0.16 * table["Mean_Repeat_Gain"]
        + 0.06 * table["Worst_Repeat_Gain"] - 0.04 * table["SD_Repeat_Gain"]
    )
    table = table.sort_values(["Robust_Select_Score", "R2", "RMSE"], ascending=[False, False, True]).reset_index(drop=True)
    best = table.iloc[0]
    key = (float(best["L2"]), float(best["Stage1_Shrinkage"]))
    return {"Table": table, "Best_Row": best, "Payload": payloads[key], "GBR_R2": gbr_r2}


def _select_meta2_configuration(
    base_oof: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_names: Sequence[str],
    selected_l2: float,
    stage1_shrinkage: float,
    seed: int,
) -> Dict:
    names = list(model_names)
    gbr_index = names.index(STACK_ANCHOR_NAME)
    gbr_oof = base_oof[:, gbr_index]
    gbr_r2 = safe_r2(y, gbr_oof)
    meta_split_count = min(META_SPLITS, len(np.unique(groups)))

    candidate_keys: List[Tuple[str, float, float]] = [("None", 0.0, 0.0)]
    for config in STACK_META2_CORRECTOR_CANDIDATES:
        for strength in STACK_META2_STRENGTH_GRID:
            for clip_multiplier in STACK_META2_CLIP_MULTIPLIER_GRID:
                candidate_keys.append((str(config["Name"]), float(strength), float(clip_multiplier)))

    accum = {
        key: {
            "Prediction_Sum": np.zeros(len(y), dtype=float),
            "Prediction_Count": np.zeros(len(y), dtype=float),
            "Repeat_Gains": [], "Fold_Gains": [],
        }
        for key in candidate_keys
    }
    config_map = {str(config["Name"]): config for config in STACK_META2_CORRECTOR_CANDIDATES}

    for repeat in range(STACK_META_REPEATS):
        repeat_seed = seed + 77041 + repeat * 104729
        splits = _make_distinct_meta_folds(groups, y, meta_split_count, repeat_seed)
        repeat_predictions = {key: np.full(len(y), np.nan, dtype=float) for key in candidate_keys}
        for fold_index, (train_index, valid_index) in enumerate(splits, start=1):
            fit = fit_gbr_anchored_weights(
                base_oof[train_index], y[train_index], groups[train_index], names,
                float(selected_l2), repeat_seed + fold_index * 211,
            )
            effective_weights, effective_bias = _effective_stage1_parameters(
                fit["Weights"], fit["Bias"], float(stage1_shrinkage), names
            )
            stage1_train = base_oof[train_index] @ effective_weights + effective_bias
            stage1_valid = base_oof[valid_index] @ effective_weights + effective_bias
            baseline_valid = base_oof[valid_index, gbr_index]
            residual_train = y[train_index] - stage1_train
            residual_scale = _robust_residual_scale(residual_train)
            train_features = _build_meta2_features(base_oof[train_index], names)
            valid_features = _build_meta2_features(base_oof[valid_index], names)

            null_key = ("None", 0.0, 0.0)
            repeat_predictions[null_key][valid_index] = stage1_valid
            accum[null_key]["Fold_Gains"].append(
                safe_r2(y[valid_index], stage1_valid) - safe_r2(y[valid_index], baseline_valid)
            )

            for corrector_index, config in enumerate(STACK_META2_CORRECTOR_CANDIDATES, start=1):
                fitted = _fit_meta2_corrector(
                    train_features, residual_train, groups[train_index], config,
                    repeat_seed + fold_index * 1009 + corrector_index * 97,
                )
                raw_correction = _predict_meta2_corrector(fitted, valid_features)
                for clip_multiplier in STACK_META2_CLIP_MULTIPLIER_GRID:
                    clip_multiplier = float(clip_multiplier)
                    bound = max(0.025, clip_multiplier * residual_scale)
                    clipped = np.clip(raw_correction, -bound, bound)
                    for strength in STACK_META2_STRENGTH_GRID:
                        strength = float(strength)
                        key = (str(config["Name"]), strength, clip_multiplier)
                        prediction = stage1_valid + strength * clipped
                        repeat_predictions[key][valid_index] = prediction
                        accum[key]["Fold_Gains"].append(
                            safe_r2(y[valid_index], prediction) - safe_r2(y[valid_index], baseline_valid)
                        )

        for key in candidate_keys:
            repeat_prediction = repeat_predictions[key]
            accum[key]["Prediction_Sum"] += repeat_prediction
            accum[key]["Prediction_Count"] += np.isfinite(repeat_prediction).astype(float)
            accum[key]["Repeat_Gains"].append(safe_r2(y, repeat_prediction) - gbr_r2)

    rows = []
    payloads = {}
    for key in candidate_keys:
        name, strength, clip_multiplier = key
        item = accum[key]
        meta_oof = item["Prediction_Sum"] / np.maximum(item["Prediction_Count"], 1.0)
        metric = metrics(y, meta_oof)
        repeat_array = np.asarray(item["Repeat_Gains"], dtype=float)
        fold_array = np.asarray(item["Fold_Gains"], dtype=float)
        gain = float(metric["R2"] - gbr_r2)
        rows.append({
            "Corrector": name, "Residual_Strength": float(strength),
            "Clip_Multiplier": float(clip_multiplier),
            "R2": float(metric["R2"]), "RMSE": float(metric["RMSE"]), "MAE": float(metric["MAE"]),
            "Meta_Gain": gain,
            "Mean_Repeat_Gain": float(np.mean(repeat_array)),
            "SD_Repeat_Gain": float(np.std(repeat_array, ddof=1)) if len(repeat_array) > 1 else 0.0,
            "Positive_Repeats": int(np.sum(repeat_array > 0.0)),
            "Worst_Repeat_Gain": float(np.min(repeat_array)),
            "Positive_Folds": int(np.sum(fold_array > 0.0)),
            "Total_Folds": int(len(fold_array)),
            "Worst_Fold_Gain": float(np.min(fold_array)),
        })
        payloads[key] = {"Meta_OOF": meta_oof}

    table = pd.DataFrame(rows)
    table["Robust_Select_Score"] = (
        table["R2"] + 0.20 * table["Mean_Repeat_Gain"]
        + 0.08 * table["Worst_Repeat_Gain"] - 0.06 * table["SD_Repeat_Gain"]
    )
    table = table.sort_values(["Robust_Select_Score", "R2", "RMSE"], ascending=[False, False, True]).reset_index(drop=True)
    best = table.iloc[0]
    best_key = (str(best["Corrector"]), float(best["Residual_Strength"]), float(best["Clip_Multiplier"]))
    return {"Table": table, "Best_Row": best, "Payload": payloads[best_key], "Config_Map": config_map}


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        return float(np.nanmean(values))
    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))


def _get_gate_expert_prediction(
    prediction_matrix: np.ndarray,
    expert_name: str,
    model_names: Sequence[str],
) -> np.ndarray:
    prediction_matrix = np.asarray(prediction_matrix, dtype=float)
    names = list(model_names)
    if expert_name in names:
        return np.asarray(prediction_matrix[:, names.index(expert_name)], dtype=float)
    if expert_name == "ConsensusMedian":
        return np.asarray(np.median(prediction_matrix, axis=1), dtype=float)
    if expert_name == "TreeMedian":
        indices = [index for index, name in enumerate(names) if name in {"RF", "ExtraTrees", "LGBM", "XGB", "GBR", STACK_ANCHOR_NAME}]
        return np.asarray(np.median(prediction_matrix[:, indices], axis=1), dtype=float)
    raise ValueError(f"Unknown gating expert: {expert_name}")


def _make_distinct_meta_folds(
    groups: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Search for group partitions balanced by row count and target mean."""
    groups = np.asarray(groups)
    y = np.asarray(y, dtype=float)
    unique_groups = np.unique(groups)
    n_splits = int(min(max(2, n_splits), len(unique_groups)))
    rng = np.random.default_rng(int(seed))
    global_mean = float(np.mean(y))
    global_sd = max(float(np.std(y)), 1e-8)
    target_size = len(y) / n_splits
    target_group_count = len(unique_groups) / n_splits

    group_indices = [np.flatnonzero(groups == group) for group in unique_groups]
    group_sizes = np.asarray([len(index) for index in group_indices], dtype=float)
    group_sums = np.asarray([float(np.sum(y[index])) for index in group_indices], dtype=float)

    best_assignment = None
    best_score = np.inf
    restarts = max(160, 20 * n_splits)
    for _ in range(restarts):
        permutation = rng.permutation(len(unique_groups))
        fold_order = rng.permutation(n_splits)
        assignment = np.empty(len(unique_groups), dtype=int)
        # Start with balanced group counts, then randomize assignments by swapping groups.
        for position, group_position in enumerate(permutation):
            assignment[group_position] = int(fold_order[position % n_splits])

        for _swap in range(max(30, len(unique_groups) * 2)):
            a, b = rng.choice(len(unique_groups), size=2, replace=False)
            if assignment[a] == assignment[b]:
                continue
            current_a, current_b = assignment[a], assignment[b]
            if rng.random() < 0.55:
                assignment[a], assignment[b] = current_b, current_a

        fold_sizes = np.zeros(n_splits, dtype=float)
        fold_sums = np.zeros(n_splits, dtype=float)
        fold_group_counts = np.zeros(n_splits, dtype=float)
        for group_position, fold in enumerate(assignment):
            fold_sizes[fold] += group_sizes[group_position]
            fold_sums[fold] += group_sums[group_position]
            fold_group_counts[fold] += 1.0
        fold_means = fold_sums / np.maximum(fold_sizes, 1.0)
        size_penalty = float(np.mean(((fold_sizes - target_size) / max(target_size, 1.0)) ** 2))
        mean_penalty = float(np.mean(((fold_means - global_mean) / global_sd) ** 2))
        count_penalty = float(np.mean(((fold_group_counts - target_group_count) / max(target_group_count, 1.0)) ** 2))
        empty_penalty = 1000.0 if np.any(fold_group_counts < 1) else 0.0
        score = 0.48 * size_penalty + 0.47 * mean_penalty + 0.05 * count_penalty + empty_penalty
        if score < best_score:
            best_score = score
            best_assignment = assignment.copy()

    if best_assignment is None:
        raise RuntimeError("Could not construct grouped meta-validation folds.")
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_splits):
        validation_groups = unique_groups[best_assignment == fold]
        valid_mask = np.isin(groups, validation_groups)
        valid_index = np.flatnonzero(valid_mask)
        train_index = np.flatnonzero(~valid_mask)
        if len(valid_index) == 0 or len(train_index) == 0:
            raise RuntimeError("A meta-validation split has an empty training or validation fold.")
        splits.append((train_index, valid_index))
    return splits

def _quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if n_bins <= 1 or len(values) == 0:
        return np.asarray([], dtype=float)
    quantiles = np.linspace(0.0, 1.0, int(n_bins) + 1)[1:-1]
    edges = np.quantile(values, quantiles)
    return np.unique(np.asarray(edges, dtype=float))


def _meta4_secondary_signal(
    base_prediction_matrix: np.ndarray,
    baseline_prediction: np.ndarray,
    model_names: Sequence[str],
    mode: str,
) -> np.ndarray:
    matrix = np.asarray(base_prediction_matrix, dtype=float)
    baseline = np.asarray(baseline_prediction, dtype=float)
    if mode == "Level":
        return np.zeros(len(baseline), dtype=float)
    if mode == "LevelDisagreement":
        return np.std(matrix, axis=1)
    if mode == "LevelConsensus":
        consensus = np.median(matrix, axis=1)
        return consensus - baseline
    raise ValueError(f"Unknown stage-4 calibration mode: {mode}")


def _fit_regime_calibrator(
    base_prediction_matrix: np.ndarray,
    baseline_prediction: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_names: Sequence[str],
    mode: str,
    level_bins: int,
    secondary_bins: int,
    shrinkage: float,
    smoothing: float,
    clip_multiplier: float,
) -> Dict:
    matrix = np.asarray(base_prediction_matrix, dtype=float)
    baseline = np.asarray(baseline_prediction, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    residual = y - baseline
    weights = group_balanced_weights(groups)
    level_edges = _quantile_edges(baseline, int(level_bins))
    secondary_signal = _meta4_secondary_signal(matrix, baseline, model_names, mode)
    effective_secondary_bins = 1 if mode == "Level" else int(secondary_bins)
    secondary_edges = _quantile_edges(secondary_signal, effective_secondary_bins)
    level_id = np.digitize(baseline, level_edges, right=False)
    secondary_id = np.digitize(secondary_signal, secondary_edges, right=False)
    cell_id = level_id * max(1, len(secondary_edges) + 1) + secondary_id
    global_offset = _weighted_mean(residual, weights)
    offsets: Dict[int, float] = {}
    for cell in np.unique(cell_id):
        mask = cell_id == cell
        weight_sum = float(np.sum(weights[mask]))
        weighted_sum = float(np.sum(weights[mask] * residual[mask]))
        offset = (weighted_sum + float(smoothing) * global_offset) / max(weight_sum + float(smoothing), 1e-12)
        offsets[int(cell)] = float(offset)
    residual_scale = _robust_residual_scale(residual)
    bound = max(0.020, float(clip_multiplier) * residual_scale)
    return {
        "Mode": str(mode),
        "Level_Edges": level_edges,
        "Secondary_Edges": secondary_edges,
        "Offsets": offsets,
        "Global_Offset": float(global_offset),
        "Shrinkage": float(shrinkage),
        "Bound": float(bound),
        "Model_Names": list(model_names),
    }


def _predict_regime_correction(
    fitted: Mapping,
    base_prediction_matrix: np.ndarray,
    baseline_prediction: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(base_prediction_matrix, dtype=float)
    baseline = np.asarray(baseline_prediction, dtype=float)
    mode = str(fitted["Mode"])
    secondary_signal = _meta4_secondary_signal(matrix, baseline, fitted["Model_Names"], mode)
    level_id = np.digitize(baseline, np.asarray(fitted["Level_Edges"], dtype=float), right=False)
    secondary_id = np.digitize(
        secondary_signal, np.asarray(fitted["Secondary_Edges"], dtype=float), right=False
    )
    cell_id = level_id * max(1, len(fitted["Secondary_Edges"]) + 1) + secondary_id
    raw = np.asarray([
        float(fitted["Offsets"].get(int(cell), fitted["Global_Offset"]))
        for cell in cell_id
    ], dtype=float)
    raw = np.clip(raw, -float(fitted["Bound"]), float(fitted["Bound"]))
    return float(fitted["Shrinkage"]) * raw


def _select_meta4_configuration(
    base_oof: np.ndarray,
    baseline_oof: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_names: Sequence[str],
    seed: int,
) -> Dict:
    base_oof = np.asarray(base_oof, dtype=float)
    baseline_oof = np.asarray(baseline_oof, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    names = list(model_names)
    gbr_oof = base_oof[:, names.index(STACK_ANCHOR_NAME)]
    gbr_r2 = safe_r2(y, gbr_oof)
    baseline_r2 = safe_r2(y, baseline_oof)
    meta_split_count = min(META_SPLITS, len(np.unique(groups)))

    keys: List[Tuple[str, int, int, float, float, float]] = [
        ("None", 1, 1, 0.0, 0.0, 0.0)
    ]
    for mode in STACK_META4_MODES:
        for level_bins in STACK_META4_LEVEL_BINS:
            secondary_options = (1,) if mode == "Level" else STACK_META4_SECONDARY_BINS
            for secondary_bins in secondary_options:
                for shrinkage in STACK_META4_SHRINKAGE_GRID:
                    for smoothing in STACK_META4_SMOOTHING_GRID:
                        for clip in STACK_META4_CLIP_MULTIPLIER_GRID:
                            keys.append((str(mode), int(level_bins), int(secondary_bins), float(shrinkage), float(smoothing), float(clip)))

    accum = {
        key: {
            "Prediction_Sum": np.zeros(len(y), dtype=float),
            "Prediction_Count": np.zeros(len(y), dtype=float),
            "Repeat_Total_Gains": [], "Repeat_Incremental_Gains": [],
            "Fold_Total_Gains": [], "Fold_Incremental_Gains": [],
        }
        for key in keys
    }
    for repeat in range(STACK_META_REPEATS):
        repeat_seed = int(seed) + 270001 + repeat * 32452843
        splits = _make_distinct_meta_folds(groups, y, meta_split_count, repeat_seed)
        repeat_predictions = {key: np.full(len(y), np.nan, dtype=float) for key in keys}
        for train_index, valid_index in splits:
            null_key = ("None", 1, 1, 0.0, 0.0, 0.0)
            repeat_predictions[null_key][valid_index] = baseline_oof[valid_index]
            for key in keys[1:]:
                mode, level_bins, secondary_bins, shrinkage, smoothing, clip = key
                fitted = _fit_regime_calibrator(
                    base_oof[train_index], baseline_oof[train_index], y[train_index], groups[train_index],
                    names, mode, level_bins, secondary_bins, shrinkage, smoothing, clip,
                )
                correction = _predict_regime_correction(
                    fitted, base_oof[valid_index], baseline_oof[valid_index]
                )
                repeat_predictions[key][valid_index] = baseline_oof[valid_index] + correction

            for key in keys:
                pred = repeat_predictions[key][valid_index]
                total_gain = safe_r2(y[valid_index], pred) - safe_r2(y[valid_index], gbr_oof[valid_index])
                incremental_gain = safe_r2(y[valid_index], pred) - safe_r2(y[valid_index], baseline_oof[valid_index])
                accum[key]["Fold_Total_Gains"].append(float(total_gain))
                accum[key]["Fold_Incremental_Gains"].append(float(incremental_gain))

        for key in keys:
            pred = repeat_predictions[key]
            accum[key]["Prediction_Sum"] += pred
            accum[key]["Prediction_Count"] += np.isfinite(pred).astype(float)
            accum[key]["Repeat_Total_Gains"].append(safe_r2(y, pred) - gbr_r2)
            accum[key]["Repeat_Incremental_Gains"].append(safe_r2(y, pred) - baseline_r2)

    rows, payloads = [], {}
    for key in keys:
        item = accum[key]
        meta_oof = item["Prediction_Sum"] / np.maximum(item["Prediction_Count"], 1.0)
        metric = metrics(y, meta_oof)
        total_repeat = np.asarray(item["Repeat_Total_Gains"], dtype=float)
        incremental_repeat = np.asarray(item["Repeat_Incremental_Gains"], dtype=float)
        total_fold = np.asarray(item["Fold_Total_Gains"], dtype=float)
        incremental_fold = np.asarray(item["Fold_Incremental_Gains"], dtype=float)
        mode, level_bins, secondary_bins, shrinkage, smoothing, clip = key
        rows.append({
            "Regime_Mode": mode, "Level_Bins": level_bins, "Secondary_Bins": secondary_bins,
            "Regime_Shrinkage": shrinkage, "Regime_Smoothing": smoothing,
            "Regime_Clip_Multiplier": clip,
            "R2": metric["R2"], "RMSE": metric["RMSE"], "MAE": metric["MAE"],
            "Total_Gain": float(metric["R2"] - gbr_r2),
            "Incremental_Gain": float(metric["R2"] - baseline_r2),
            "Mean_Repeat_Total_Gain": float(np.mean(total_repeat)),
            "Mean_Repeat_Incremental_Gain": float(np.mean(incremental_repeat)),
            "SD_Repeat_Incremental_Gain": float(np.std(incremental_repeat, ddof=1)) if len(incremental_repeat) > 1 else 0.0,
            "Positive_Repeats": int(np.sum(incremental_repeat > 0.0)),
            "Worst_Repeat_Gain": float(np.min(incremental_repeat)),
            "Positive_Folds": int(np.sum(incremental_fold > 0.0)),
            "Total_Folds": int(len(incremental_fold)),
            "Worst_Fold_Gain": float(np.min(incremental_fold)),
        })
        payloads[key] = {"Meta_OOF": meta_oof}
    table = pd.DataFrame(rows)
    table["Robust_Select_Score"] = (
        table["R2"] + 0.22 * table["Mean_Repeat_Incremental_Gain"]
        + 0.10 * table["Worst_Repeat_Gain"]
        - 0.08 * table["SD_Repeat_Incremental_Gain"]
    )
    table = table.sort_values(["Robust_Select_Score", "R2", "RMSE"], ascending=[False, False, True]).reset_index(drop=True)
    best = table.iloc[0]
    best_key = (
        str(best["Regime_Mode"]), int(best["Level_Bins"]), int(best["Secondary_Bins"]),
        float(best["Regime_Shrinkage"]), float(best["Regime_Smoothing"]),
        float(best["Regime_Clip_Multiplier"]),
    )
    return {
        "Table": table, "Best_Row": best, "Payload": payloads[best_key],
        "GBR_R2": gbr_r2, "Baseline_R2": baseline_r2,
    }


def _select_meta5_configuration(
    gbr_oof: np.ndarray,
    layer4_oof: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
) -> Dict:
    gbr_oof = np.asarray(gbr_oof, dtype=float)
    layer4_oof = np.asarray(layer4_oof, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    gbr_r2 = safe_r2(y, gbr_oof)
    layer4_r2 = safe_r2(y, layer4_oof)
    meta_split_count = min(META_SPLITS, len(np.unique(groups)))
    keys: List[Tuple[float, float, float]] = []
    for blend in STACK_META5_BLEND_GRID:
        clips = (0.0,) if blend == 0.0 else STACK_META5_DELTA_CLIP_GRID
        for delta_clip in clips:
            for scale in STACK_META5_SCALE_GRID:
                keys.append((float(blend), float(delta_clip), float(scale)))
    accum = {
        key: {
            "Prediction_Sum": np.zeros(len(y), dtype=float),
            "Prediction_Count": np.zeros(len(y), dtype=float),
            "Repeat_Total_Gains": [], "Repeat_Incremental_Gains": [],
            "Fold_Total_Gains": [], "Fold_Incremental_Gains": [], "Intercepts": [],
        }
        for key in keys
    }
    for repeat in range(STACK_META_REPEATS):
        repeat_seed = int(seed) + 470001 + repeat * 49979687
        splits = _make_distinct_meta_folds(groups, y, meta_split_count, repeat_seed)
        repeat_predictions = {key: np.full(len(y), np.nan, dtype=float) for key in keys}
        for train_index, valid_index in splits:
            residual_scale = _robust_residual_scale(y[train_index] - gbr_oof[train_index])
            train_weights = group_balanced_weights(groups[train_index])
            for key in keys:
                blend, delta_clip, scale = key
                if blend == 0.0:
                    train_blend = gbr_oof[train_index]
                    valid_blend = gbr_oof[valid_index]
                else:
                    bound = max(0.020, float(delta_clip) * residual_scale)
                    train_delta = np.clip(layer4_oof[train_index] - gbr_oof[train_index], -bound, bound)
                    valid_delta = np.clip(layer4_oof[valid_index] - gbr_oof[valid_index], -bound, bound)
                    train_blend = gbr_oof[train_index] + float(blend) * train_delta
                    valid_blend = gbr_oof[valid_index] + float(blend) * valid_delta
                intercept = np.clip(
                    _weighted_mean(y[train_index] - float(scale) * train_blend, train_weights),
                    -STACK_META5_INTERCEPT_BOUND, STACK_META5_INTERCEPT_BOUND,
                )
                prediction = float(intercept) + float(scale) * valid_blend
                repeat_predictions[key][valid_index] = prediction
                accum[key]["Intercepts"].append(float(intercept))
                accum[key]["Fold_Total_Gains"].append(
                    safe_r2(y[valid_index], prediction) - safe_r2(y[valid_index], gbr_oof[valid_index])
                )
                accum[key]["Fold_Incremental_Gains"].append(
                    safe_r2(y[valid_index], prediction) - safe_r2(y[valid_index], layer4_oof[valid_index])
                )
        for key in keys:
            prediction = repeat_predictions[key]
            accum[key]["Prediction_Sum"] += prediction
            accum[key]["Prediction_Count"] += np.isfinite(prediction).astype(float)
            accum[key]["Repeat_Total_Gains"].append(safe_r2(y, prediction) - gbr_r2)
            accum[key]["Repeat_Incremental_Gains"].append(safe_r2(y, prediction) - layer4_r2)

    rows, payloads = [], {}
    for key in keys:
        blend, delta_clip, scale = key
        item = accum[key]
        meta_oof = item["Prediction_Sum"] / np.maximum(item["Prediction_Count"], 1.0)
        metric = metrics(y, meta_oof)
        total_repeat = np.asarray(item["Repeat_Total_Gains"], dtype=float)
        incremental_repeat = np.asarray(item["Repeat_Incremental_Gains"], dtype=float)
        incremental_fold = np.asarray(item["Fold_Incremental_Gains"], dtype=float)
        rows.append({
            "Final_Blend": blend, "Final_Delta_Clip": delta_clip, "Final_Scale5": scale,
            "Average_Intercept5": float(np.mean(item["Intercepts"])) if item["Intercepts"] else 0.0,
            "R2": metric["R2"], "RMSE": metric["RMSE"], "MAE": metric["MAE"],
            "Total_Gain": float(metric["R2"] - gbr_r2),
            "Incremental_Gain": float(metric["R2"] - layer4_r2),
            "Mean_Repeat_Total_Gain": float(np.mean(total_repeat)),
            "Mean_Repeat_Incremental_Gain": float(np.mean(incremental_repeat)),
            "SD_Repeat_Incremental_Gain": float(np.std(incremental_repeat, ddof=1)) if len(incremental_repeat) > 1 else 0.0,
            "Positive_Repeats": int(np.sum(total_repeat > 0.0)),
            "Worst_Repeat_Gain": float(np.min(total_repeat)),
            "Positive_Folds": int(np.sum(np.asarray(item["Fold_Total_Gains"]) > 0.0)),
            "Total_Folds": int(len(item["Fold_Total_Gains"])),
            "Worst_Fold_Gain": float(np.min(np.asarray(item["Fold_Total_Gains"]))),
        })
        payloads[key] = {
            "Meta_OOF": meta_oof,
            "Average_Intercept": float(np.mean(item["Intercepts"])) if item["Intercepts"] else 0.0,
        }
    table = pd.DataFrame(rows)
    table["Robust_Select_Score"] = (
        table["R2"] + 0.25 * table["Mean_Repeat_Total_Gain"]
        + 0.12 * table["Worst_Repeat_Gain"]
        + 0.10 * table["Mean_Repeat_Incremental_Gain"]
        - 0.08 * table["SD_Repeat_Incremental_Gain"]
    )
    table = table.sort_values(["Robust_Select_Score", "R2", "RMSE"], ascending=[False, False, True]).reset_index(drop=True)
    best = table.iloc[0]
    best_key = (float(best["Final_Blend"]), float(best["Final_Delta_Clip"]), float(best["Final_Scale5"]))
    return {
        "Table": table, "Best_Row": best, "Payload": payloads[best_key],
        "GBR_R2": gbr_r2, "Layer4_R2": layer4_r2,
    }

def _build_gate_features(
    prediction_matrix: np.ndarray,
    base_prediction: np.ndarray,
    expert_prediction: np.ndarray,
    model_names: Sequence[str],
) -> np.ndarray:
    prediction_matrix = np.asarray(prediction_matrix, dtype=float)
    base_prediction = np.asarray(base_prediction, dtype=float)
    expert = np.asarray(expert_prediction, dtype=float)
    gbr_index = list(model_names).index(STACK_ANCHOR_NAME)
    gbr = prediction_matrix[:, gbr_index]
    core = _build_meta2_features(prediction_matrix, model_names)
    delta = expert - base_prediction
    extra = np.column_stack([
        base_prediction,
        expert,
        delta,
        np.abs(delta),
        base_prediction ** 2,
        expert ** 2,
        base_prediction * expert,
        expert - gbr,
        base_prediction - gbr,
        np.abs(expert - gbr),
    ])
    return np.column_stack([core, extra])


def _fit_gate_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    improvement: np.ndarray,
    groups: np.ndarray,
    c_value: float,
    seed: int,
) -> Dict:
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=int)
    improvement = np.asarray(improvement, dtype=float)
    groups = np.asarray(groups)
    positive_rate = float(np.mean(labels)) if len(labels) else 0.0
    unique = np.unique(labels)
    if len(unique) < 2 or np.sum(labels == 1) < 6 or np.sum(labels == 0) < 6:
        return {"Family": "constant", "Probability": positive_rate, "Scaler": None, "Estimator": None}

    scaler = StandardScaler()
    transformed = scaler.fit_transform(features)
    group_weight = group_balanced_weights(groups)
    positive_improvement = np.maximum(improvement, 0.0)
    reference = float(np.median(positive_improvement[positive_improvement > 0])) if np.any(positive_improvement > 0) else 1.0
    reference = max(reference, 1e-6)
    importance = 1.0 + np.clip(
        positive_improvement / reference,
        0.0,
        STACK_GATE_IMPROVEMENT_WEIGHT_CAP,
    )
    sample_weight = group_weight * importance
    estimator = LogisticRegression(
        C=float(c_value),
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        random_state=int(seed),
    )
    estimator.fit(transformed, labels, sample_weight=sample_weight)
    return {"Family": "logistic", "Probability": positive_rate, "Scaler": scaler, "Estimator": estimator}


def _predict_gate_probability(fitted: Mapping, features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    if str(fitted.get("Family")) == "constant":
        return np.full(features.shape[0], float(fitted.get("Probability", 0.0)), dtype=float)
    transformed = fitted["Scaler"].transform(features)
    return np.asarray(fitted["Estimator"].predict_proba(transformed)[:, 1], dtype=float)


def _fit_selected_stage12(
    base_train: np.ndarray,
    base_valid: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    model_names: Sequence[str],
    selected_l2: float,
    stage1_shrinkage: float,
    corrector_name: str,
    corrector_config: Optional[Mapping],
    residual_strength: float,
    clip_multiplier: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    names = list(model_names)
    fit = fit_gbr_anchored_weights(
        base_train, y_train, groups_train, names,
        float(selected_l2), int(seed),
    )
    effective_weights, effective_bias = _effective_stage1_parameters(
        fit["Weights"], fit["Bias"], float(stage1_shrinkage), names
    )
    stage1_train = base_train @ effective_weights + effective_bias
    stage1_valid = base_valid @ effective_weights + effective_bias
    if corrector_name == "None" or corrector_config is None or residual_strength <= 0.0:
        return stage1_train, stage1_valid

    residual_train = np.asarray(y_train, dtype=float) - stage1_train
    residual_scale = _robust_residual_scale(residual_train)
    fitted = _fit_meta2_corrector(
        _build_meta2_features(base_train, names),
        residual_train,
        groups_train,
        corrector_config,
        int(seed) + 4099,
    )
    correction_train = _predict_meta2_corrector(fitted, _build_meta2_features(base_train, names))
    correction_valid = _predict_meta2_corrector(fitted, _build_meta2_features(base_valid, names))
    bound = max(0.025, float(clip_multiplier) * residual_scale)
    correction_train = np.clip(correction_train, -bound, bound)
    correction_valid = np.clip(correction_valid, -bound, bound)
    return (
        stage1_train + float(residual_strength) * correction_train,
        stage1_valid + float(residual_strength) * correction_valid,
    )


def _select_meta3_configuration(
    base_oof: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_names: Sequence[str],
    selected_l2: float,
    stage1_shrinkage: float,
    corrector_name: str,
    corrector_config: Optional[Mapping],
    residual_strength: float,
    residual_clip_multiplier: float,
    seed: int,
) -> Dict:
    names = list(model_names)
    gbr_index = names.index(STACK_ANCHOR_NAME)
    gbr_oof = np.asarray(base_oof[:, gbr_index], dtype=float)
    gbr_r2 = safe_r2(y, gbr_oof)
    meta_split_count = min(META_SPLITS, len(np.unique(groups)))

    candidate_keys: List[Tuple[str, float, float, float, float, float]] = []
    for scale in STACK_FINAL_SCALE_GRID:
        candidate_keys.append(("None", 0.0, 1.0, 0.0, 0.0, float(scale)))
    for expert_name in STACK_GATE_EXPERTS:
        for c_value in STACK_GATE_C_GRID:
            for threshold in STACK_GATE_THRESHOLD_GRID:
                for strength in STACK_GATE_STRENGTH_GRID:
                    for clip_multiplier in STACK_GATE_CLIP_MULTIPLIER_GRID:
                        for scale in STACK_FINAL_SCALE_GRID:
                            candidate_keys.append((
                                str(expert_name), float(c_value), float(threshold),
                                float(strength), float(clip_multiplier), float(scale),
                            ))

    accum = {
        key: {
            "Prediction_Sum": np.zeros(len(y), dtype=float),
            "Prediction_Count": np.zeros(len(y), dtype=float),
            "Repeat_Gains": [], "Fold_Gains": [], "Intercepts": [], "Gate_Rates": [],
        }
        for key in candidate_keys
    }

    for repeat in range(STACK_META_REPEATS):
        repeat_seed = int(seed) + 190001 + repeat * 15485863
        splits = _make_distinct_meta_folds(groups, y, meta_split_count, repeat_seed)
        repeat_predictions = {key: np.full(len(y), np.nan, dtype=float) for key in candidate_keys}

        for fold_index, (train_index, valid_index) in enumerate(splits, start=1):
            stage2_train, stage2_valid = _fit_selected_stage12(
                base_oof[train_index], base_oof[valid_index], y[train_index], groups[train_index],
                names, selected_l2, stage1_shrinkage, corrector_name, corrector_config,
                residual_strength, residual_clip_multiplier,
                repeat_seed + fold_index * 1009,
            )
            baseline_valid = gbr_oof[valid_index]
            train_weights = group_balanced_weights(groups[train_index])
            residual_scale = _robust_residual_scale(y[train_index] - stage2_train)

            for scale in STACK_FINAL_SCALE_GRID:
                key = ("None", 0.0, 1.0, 0.0, 0.0, float(scale))
                intercept = np.clip(
                    _weighted_mean(y[train_index] - float(scale) * stage2_train, train_weights),
                    -STACK_FINAL_INTERCEPT_BOUND,
                    STACK_FINAL_INTERCEPT_BOUND,
                )
                prediction = float(intercept) + float(scale) * stage2_valid
                repeat_predictions[key][valid_index] = prediction
                accum[key]["Intercepts"].append(float(intercept))
                accum[key]["Gate_Rates"].append(0.0)
                accum[key]["Fold_Gains"].append(
                    safe_r2(y[valid_index], prediction) - safe_r2(y[valid_index], baseline_valid)
                )

            for expert_position, expert_name in enumerate(STACK_GATE_EXPERTS, start=1):
                expert_train = _get_gate_expert_prediction(base_oof[train_index], expert_name, names)
                expert_valid = _get_gate_expert_prediction(base_oof[valid_index], expert_name, names)
                base_sq_error = (y[train_index] - stage2_train) ** 2
                expert_sq_error = (y[train_index] - expert_train) ** 2
                improvement = base_sq_error - expert_sq_error
                labels = (improvement > 0.0).astype(int)
                train_features = _build_gate_features(
                    base_oof[train_index], stage2_train, expert_train, names
                )
                valid_features = _build_gate_features(
                    base_oof[valid_index], stage2_valid, expert_valid, names
                )

                for c_position, c_value in enumerate(STACK_GATE_C_GRID, start=1):
                    gate = _fit_gate_classifier(
                        train_features, labels, improvement, groups[train_index],
                        float(c_value), repeat_seed + fold_index * 100003 + expert_position * 1009 + c_position * 97,
                    )
                    probability_train = _predict_gate_probability(gate, train_features)
                    probability_valid = _predict_gate_probability(gate, valid_features)

                    for threshold in STACK_GATE_THRESHOLD_GRID:
                        denominator = max(1e-6, 1.0 - float(threshold))
                        activation_train = np.clip((probability_train - float(threshold)) / denominator, 0.0, 1.0)
                        activation_valid = np.clip((probability_valid - float(threshold)) / denominator, 0.0, 1.0)
                        for clip_multiplier in STACK_GATE_CLIP_MULTIPLIER_GRID:
                            bound = max(0.025, float(clip_multiplier) * residual_scale)
                            delta_train = np.clip(expert_train - stage2_train, -bound, bound)
                            delta_valid = np.clip(expert_valid - stage2_valid, -bound, bound)
                            for strength in STACK_GATE_STRENGTH_GRID:
                                gated_train = stage2_train + float(strength) * activation_train * delta_train
                                gated_valid = stage2_valid + float(strength) * activation_valid * delta_valid
                                gate_rate = float(np.mean(activation_valid))
                                for scale in STACK_FINAL_SCALE_GRID:
                                    key = (
                                        str(expert_name), float(c_value), float(threshold),
                                        float(strength), float(clip_multiplier), float(scale),
                                    )
                                    intercept = np.clip(
                                        _weighted_mean(y[train_index] - float(scale) * gated_train, train_weights),
                                        -STACK_FINAL_INTERCEPT_BOUND,
                                        STACK_FINAL_INTERCEPT_BOUND,
                                    )
                                    prediction = float(intercept) + float(scale) * gated_valid
                                    repeat_predictions[key][valid_index] = prediction
                                    accum[key]["Intercepts"].append(float(intercept))
                                    accum[key]["Gate_Rates"].append(gate_rate)
                                    accum[key]["Fold_Gains"].append(
                                        safe_r2(y[valid_index], prediction) - safe_r2(y[valid_index], baseline_valid)
                                    )

        for key in candidate_keys:
            repeat_prediction = repeat_predictions[key]
            accum[key]["Prediction_Sum"] += repeat_prediction
            accum[key]["Prediction_Count"] += np.isfinite(repeat_prediction).astype(float)
            accum[key]["Repeat_Gains"].append(safe_r2(y, repeat_prediction) - gbr_r2)

    rows: List[Dict] = []
    payloads: Dict[Tuple[str, float, float, float, float, float], Dict] = {}
    for key in candidate_keys:
        expert_name, c_value, threshold, strength, clip_multiplier, scale = key
        item = accum[key]
        meta_oof = item["Prediction_Sum"] / np.maximum(item["Prediction_Count"], 1.0)
        metric = metrics(y, meta_oof)
        repeat_array = np.asarray(item["Repeat_Gains"], dtype=float)
        fold_array = np.asarray(item["Fold_Gains"], dtype=float)
        gain = float(metric["R2"] - gbr_r2)
        rows.append({
            "Gate_Expert": expert_name,
            "Gate_C": float(c_value),
            "Gate_Threshold": float(threshold),
            "Gate_Strength": float(strength),
            "Gate_Clip_Multiplier": float(clip_multiplier),
            "Final_Scale": float(scale),
            "Average_Intercept": float(np.mean(item["Intercepts"])) if item["Intercepts"] else 0.0,
            "Average_Gate_Rate": float(np.mean(item["Gate_Rates"])) if item["Gate_Rates"] else 0.0,
            "R2": float(metric["R2"]), "RMSE": float(metric["RMSE"]), "MAE": float(metric["MAE"]),
            "Meta_Gain": gain,
            "Mean_Repeat_Gain": float(np.mean(repeat_array)),
            "SD_Repeat_Gain": float(np.std(repeat_array, ddof=1)) if len(repeat_array) > 1 else 0.0,
            "Positive_Repeats": int(np.sum(repeat_array > 0.0)),
            "Worst_Repeat_Gain": float(np.min(repeat_array)),
            "Positive_Folds": int(np.sum(fold_array > 0.0)),
            "Total_Folds": int(len(fold_array)),
            "Worst_Fold_Gain": float(np.min(fold_array)),
        })
        payloads[key] = {
            "Meta_OOF": meta_oof,
            "Average_Intercept": float(np.mean(item["Intercepts"])) if item["Intercepts"] else 0.0,
        }

    table = pd.DataFrame(rows)
    table["Robust_Select_Score"] = (
        table["R2"]
        + 0.24 * table["Mean_Repeat_Gain"]
        + 0.10 * table["Worst_Repeat_Gain"]
        - 0.07 * table["SD_Repeat_Gain"]
        + 0.00015 * table["Positive_Repeats"]
    )
    table = table.sort_values(
        ["Robust_Select_Score", "R2", "RMSE"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    best = table.iloc[0]
    best_key = (
        str(best["Gate_Expert"]), float(best["Gate_C"]), float(best["Gate_Threshold"]),
        float(best["Gate_Strength"]), float(best["Gate_Clip_Multiplier"]), float(best["Final_Scale"]),
    )
    return {"Table": table, "Best_Row": best, "Payload": payloads[best_key], "GBR_R2": gbr_r2}


def _fit_meta3_stacking(
    base_oof: np.ndarray,
    base_test: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_names: Sequence[str],
    seed: int,
) -> Dict:
    """Fit the anchored blend, residual calibration, and expert gating stages."""
    base_oof = np.asarray(base_oof, dtype=float)
    base_test = np.asarray(base_test, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    names = list(model_names)
    if base_oof.shape[1] != len(names):
        raise ValueError("base_oof columns must match model_names.")
    if STACK_ANCHOR_NAME not in names:
        raise ValueError("Stacking requires the GBR750 anchor.")
    gbr_index = names.index(STACK_ANCHOR_NAME)
    gbr_oof = base_oof[:, gbr_index]
    gbr_test = base_test[:, gbr_index]

    stage1 = _select_stage1_configuration(base_oof, y, groups, names, seed)
    stage1_best = stage1["Best_Row"]
    selected_l2 = float(stage1_best["L2"])
    stage1_shrinkage = float(stage1_best["Stage1_Shrinkage"])

    meta2 = _select_meta2_configuration(
        base_oof, y, groups, names, selected_l2, stage1_shrinkage, seed + 500003
    )
    meta2_best = meta2["Best_Row"]
    corrector_name = str(meta2_best["Corrector"])
    selected_residual_strength = float(meta2_best["Residual_Strength"])
    selected_clip_multiplier = float(meta2_best["Clip_Multiplier"])
    meta2_gain = float(meta2_best["Meta_Gain"])

    meta2_min_positive_repeats = int(np.ceil(STACK_META_REPEATS * STACK_META2_MIN_POSITIVE_REPEAT_RATIO))
    meta2_min_positive_folds = int(np.ceil(float(meta2_best["Total_Folds"]) * STACK_META2_MIN_POSITIVE_FOLD_RATIO))
    meta2_stable = (
        corrector_name != "None"
        and meta2_gain >= STACK_META2_REQUIRED_GAIN
        and int(meta2_best["Positive_Repeats"]) >= meta2_min_positive_repeats
        and int(meta2_best["Positive_Folds"]) >= meta2_min_positive_folds
        and float(meta2_best["Worst_Repeat_Gain"]) >= -STACK_META2_MAX_WORST_REPEAT_LOSS
        and float(meta2_best["Worst_Fold_Gain"]) >= -STACK_META2_MAX_WORST_FOLD_LOSS
    )
    meta2_soft_min_positive = int(np.ceil(STACK_META_REPEATS * STACK_META2_SOFT_MIN_POSITIVE_REPEAT_RATIO))
    meta2_soft = (
        corrector_name != "None"
        and meta2_gain >= STACK_META2_SOFT_MIN_GAIN
        and int(meta2_best["Positive_Repeats"]) >= meta2_soft_min_positive
        and float(meta2_best["Worst_Repeat_Gain"]) >= -STACK_META2_SOFT_MAX_WORST_REPEAT_LOSS
    )
    if meta2_stable:
        used_corrector_name = corrector_name
        used_residual_strength = selected_residual_strength
    elif meta2_soft:
        used_corrector_name = corrector_name
        used_residual_strength = selected_residual_strength * STACK_META2_SOFT_FRACTION
    else:
        used_corrector_name = "None"
        used_residual_strength = 0.0
    used_corrector_config = meta2["Config_Map"].get(used_corrector_name)

    meta3 = _select_meta3_configuration(
        base_oof, y, groups, names,
        selected_l2, stage1_shrinkage,
        used_corrector_name, used_corrector_config,
        used_residual_strength, selected_clip_multiplier,
        seed + 900001,
    )
    best = meta3["Best_Row"]
    meta3_gain = float(best["Meta_Gain"])
    min_positive_repeats = int(np.ceil(STACK_META_REPEATS * STACK_META3_MIN_POSITIVE_REPEAT_RATIO))
    min_positive_folds = int(np.ceil(float(best["Total_Folds"]) * STACK_META3_MIN_POSITIVE_FOLD_RATIO))
    stable = (
        meta3_gain >= STACK_META3_REQUIRED_GAIN
        and int(best["Positive_Repeats"]) >= min_positive_repeats
        and int(best["Positive_Folds"]) >= min_positive_folds
        and float(best["Worst_Repeat_Gain"]) >= -STACK_META3_MAX_WORST_REPEAT_LOSS
        and float(best["Worst_Fold_Gain"]) >= -STACK_META3_MAX_WORST_FOLD_LOSS
    )
    soft_min_positive = int(np.ceil(STACK_META_REPEATS * STACK_META3_SOFT_MIN_POSITIVE_REPEAT_RATIO))
    soft_eligible = (
        meta3_gain >= STACK_META3_SOFT_MIN_GAIN
        and int(best["Positive_Repeats"]) >= soft_min_positive
        and float(best["Worst_Repeat_Gain"]) >= -STACK_META3_SOFT_MAX_WORST_REPEAT_LOSS
    )

    full_fit = fit_gbr_anchored_weights(base_oof, y, groups, names, selected_l2, seed + 99991)
    average_weights = np.asarray(stage1["Payload"]["Average_Weights"], dtype=float)
    blended_weights = (
        (1.0 - STACK_FINAL_META_WEIGHT) * np.asarray(full_fit["Weights"], dtype=float)
        + STACK_FINAL_META_WEIGHT * average_weights
    )
    blended_weights = _project_anchor_weights(blended_weights, names)
    blended_bias = (
        (1.0 - STACK_FINAL_META_WEIGHT) * float(full_fit["Bias"])
        + STACK_FINAL_META_WEIGHT * float(stage1["Payload"]["Average_Bias"])
    )
    effective_weights, effective_bias = _effective_stage1_parameters(
        blended_weights, blended_bias, stage1_shrinkage, names
    )
    stage1_train = base_oof @ effective_weights + effective_bias
    stage1_test = base_test @ effective_weights + effective_bias

    if used_corrector_name != "None" and used_corrector_config is not None and used_residual_strength > 0.0:
        residual_train = y - stage1_train
        residual_scale = _robust_residual_scale(residual_train)
        fitted_corrector = _fit_meta2_corrector(
            _build_meta2_features(base_oof, names), residual_train, groups,
            used_corrector_config, seed + 700001,
        )
        train_correction = _predict_meta2_corrector(fitted_corrector, _build_meta2_features(base_oof, names))
        test_correction = _predict_meta2_corrector(fitted_corrector, _build_meta2_features(base_test, names))
        correction_bound = max(0.025, selected_clip_multiplier * residual_scale)
        train_correction = np.clip(train_correction, -correction_bound, correction_bound)
        test_correction = np.clip(test_correction, -correction_bound, correction_bound)
        stage2_train = stage1_train + used_residual_strength * train_correction
        stage2_test = stage1_test + used_residual_strength * test_correction
        stage2_crossfit = np.asarray(meta2["Payload"]["Meta_OOF"], dtype=float)
        if meta2_soft and not meta2_stable:
            stage2_crossfit = gbr_oof + STACK_META2_SOFT_FRACTION * (stage2_crossfit - gbr_oof)
    else:
        stage2_train = stage1_train
        stage2_test = stage1_test
        stage2_crossfit = np.asarray(stage1["Payload"]["Meta_OOF"], dtype=float)

    if not stable and not soft_eligible:
        fallback_weights = np.zeros(len(names), dtype=float)
        fallback_weights[gbr_index] = 1.0
        return {
            "Safety_Fallback": True,
            "Soft_Shrinkage": False,
            "Fallback_Model": STACK_ANCHOR_NAME,
            "Selected_L2": selected_l2,
            "Stage1_Shrinkage": stage1_shrinkage,
            "Weights": fallback_weights,
            "Bias": 0.0,
            "Selected_Models": [STACK_ANCHOR_NAME],
            "Residual_Corrector": used_corrector_name,
            "Residual_Strength": float(used_residual_strength),
            "Residual_Clip_Multiplier": float(selected_clip_multiplier),
            "Gate_Expert": "None",
            "Gate_C": 0.0,
            "Gate_Threshold": 1.0,
            "Gate_Strength": 0.0,
            "Gate_Clip_Multiplier": 0.0,
            "Final_Scale": 1.0,
            "Final_Intercept": 0.0,
            "Train_OOF_Prediction": gbr_oof.copy(),
            "Test_Prediction": gbr_test.copy(),
            "Candidate_Table": meta2["Table"],
            "Stage1_Candidate_Table": stage1["Table"],
            "Meta3_Candidate_Table": meta3["Table"],
            "Meta_Gain": meta3_gain,
            "Meta3_Gain": meta3_gain,
            "Meta2_Gain": meta2_gain,
            "Positive_Repeats": int(best["Positive_Repeats"]),
            "Worst_Repeat_Gain": float(best["Worst_Repeat_Gain"]),
            "GBR750_Weight": 1.0,
            "Meta2_Enabled": used_corrector_name != "None",
            "Meta3_Enabled": False,
        }

    expert_name = str(best["Gate_Expert"])
    gate_c = float(best["Gate_C"])
    gate_threshold = float(best["Gate_Threshold"])
    gate_strength = float(best["Gate_Strength"])
    gate_clip_multiplier = float(best["Gate_Clip_Multiplier"])
    final_scale = float(best["Final_Scale"])
    average_intercept = float(meta3["Payload"]["Average_Intercept"])

    if expert_name == "None":
        gated_test = stage2_test.copy()
        gated_oof_for_intercept = stage2_crossfit.copy()
    else:
        expert_oof = _get_gate_expert_prediction(base_oof, expert_name, names)
        expert_test = _get_gate_expert_prediction(base_test, expert_name, names)
        improvement = (y - stage2_crossfit) ** 2 - (y - expert_oof) ** 2
        labels = (improvement > 0.0).astype(int)
        train_features = _build_gate_features(base_oof, stage2_crossfit, expert_oof, names)
        test_features = _build_gate_features(base_test, stage2_test, expert_test, names)
        gate = _fit_gate_classifier(
            train_features, labels, improvement, groups, gate_c, seed + 1200001
        )
        probability_train = _predict_gate_probability(gate, train_features)
        probability_test = _predict_gate_probability(gate, test_features)
        denominator = max(1e-6, 1.0 - gate_threshold)
        activation_train = np.clip((probability_train - gate_threshold) / denominator, 0.0, 1.0)
        activation_test = np.clip((probability_test - gate_threshold) / denominator, 0.0, 1.0)
        residual_scale = _robust_residual_scale(y - stage2_crossfit)
        gate_bound = max(0.025, gate_clip_multiplier * residual_scale)
        train_delta = np.clip(expert_oof - stage2_crossfit, -gate_bound, gate_bound)
        test_delta = np.clip(expert_test - stage2_test, -gate_bound, gate_bound)
        gated_oof_for_intercept = stage2_crossfit + gate_strength * activation_train * train_delta
        gated_test = stage2_test + gate_strength * activation_test * test_delta

    full_intercept = np.clip(
        _weighted_mean(y - final_scale * gated_oof_for_intercept, group_balanced_weights(groups)),
        -STACK_FINAL_INTERCEPT_BOUND,
        STACK_FINAL_INTERCEPT_BOUND,
    )
    final_intercept = (
        (1.0 - STACK_META3_FINAL_INTERCEPT_BLEND) * average_intercept
        + STACK_META3_FINAL_INTERCEPT_BLEND * float(full_intercept)
    )
    test_prediction = final_intercept + final_scale * gated_test
    meta_prediction = np.asarray(meta3["Payload"]["Meta_OOF"], dtype=float)

    if not stable:
        test_prediction = gbr_test + STACK_META3_SOFT_FRACTION * (test_prediction - gbr_test)
        meta_prediction = gbr_oof + STACK_META3_SOFT_FRACTION * (meta_prediction - gbr_oof)

    selected_models = [
        name for name, weight in zip(names, effective_weights) if abs(float(weight)) > 1e-6
    ]
    if used_corrector_name != "None":
        selected_models.append(f"Meta2:{used_corrector_name}")
    if expert_name != "None":
        selected_models.append(f"Gate:{expert_name}")
    if abs(final_scale - 1.0) > 1e-8 or abs(final_intercept) > 1e-8:
        selected_models.append("FinalCalibration")

    return {
        "Safety_Fallback": False,
        "Soft_Shrinkage": bool(not stable),
        "Fallback_Model": None,
        "Selected_L2": selected_l2,
        "Stage1_Shrinkage": stage1_shrinkage,
        "Weights": effective_weights,
        "Bias": effective_bias,
        "Selected_Models": selected_models,
        "Residual_Corrector": used_corrector_name,
        "Residual_Strength": float(used_residual_strength),
        "Residual_Clip_Multiplier": float(selected_clip_multiplier),
        "Gate_Expert": expert_name,
        "Gate_C": gate_c,
        "Gate_Threshold": gate_threshold,
        "Gate_Strength": gate_strength,
        "Gate_Clip_Multiplier": gate_clip_multiplier,
        "Final_Scale": final_scale,
        "Final_Intercept": float(final_intercept),
        "Train_OOF_Prediction": meta_prediction,
        "Test_Prediction": test_prediction,
        "Candidate_Table": meta2["Table"],
        "Stage1_Candidate_Table": stage1["Table"],
        "Meta3_Candidate_Table": meta3["Table"],
        "Meta_Gain": meta3_gain,
        "Meta3_Gain": meta3_gain,
        "Meta2_Gain": meta2_gain,
        "Positive_Repeats": int(best["Positive_Repeats"]),
        "Worst_Repeat_Gain": float(best["Worst_Repeat_Gain"]),
        "GBR750_Weight": float(effective_weights[gbr_index]),
        "Meta2_Enabled": used_corrector_name != "None",
        "Meta3_Enabled": True,
    }


def _evaluate_champion_guard(
    base_oof: np.ndarray,
    y: np.ndarray,
    model_names: Sequence[str],
    stage3: Mapping,
    final_gain: float,
    final_positive_repeats: int,
    final_worst_repeat_gain: float,
) -> Dict:
    """Evaluate the fallback criteria using outer-training predictions."""
    names = list(model_names)
    gbr_index = names.index(STACK_ANCHOR_NAME)
    base_r2 = {
        name: float(safe_r2(y, np.asarray(base_oof[:, index], dtype=float)))
        for index, name in enumerate(names)
    }
    best_base_model = max(base_r2, key=base_r2.get)
    best_base_r2 = float(base_r2[best_base_model])
    gbr_r2 = float(base_r2[STACK_ANCHOR_NAME])
    gbr_gap_to_best = float(best_base_r2 - gbr_r2)

    default_weights = np.eye(1, len(names), gbr_index).ravel()
    weights = np.asarray(stage3.get("Weights", default_weights), dtype=float)
    partner_weights = np.delete(weights, gbr_index)
    positive_partner_weight = float(np.sum(np.clip(partner_weights, 0.0, None)))
    absolute_partner_weight = float(np.sum(np.abs(partner_weights)))
    gbr_weight = float(weights[gbr_index])
    meta2_enabled = bool(stage3.get("Meta2_Enabled", False))
    gate_expert = str(stage3.get("Gate_Expert", "None"))
    gate_enabled = gate_expert != "None"
    intercept = float(stage3.get("Final_Intercept", 0.0))

    strong_evidence = bool(
        final_gain >= STACK_GUARD_STRONG_GAIN
        and int(final_positive_repeats) >= int(STACK_META_REPEATS)
        and float(final_worst_repeat_gain) >= STACK_GUARD_STRONG_WORST_REPEAT_GAIN
    )

    reasons: List[str] = []
    if (
        STACK_GUARD_REQUIRE_GBR_TRAIN_CHAMPION
        and best_base_model != STACK_ANCHOR_NAME
        and gbr_gap_to_best > STACK_GUARD_GBR_CHAMPION_TOL
        and not strong_evidence
    ):
        reasons.append(
            f"GBR750 trails {best_base_model} in training OOF R2 by {gbr_gap_to_best:.6f}"
        )

    if gbr_weight < STACK_GUARD_HARD_MIN_GBR_WEIGHT:
        reasons.append(
            f"GBR750 weight is below the hard minimum: {gbr_weight:.4f} < "
            f"{STACK_GUARD_HARD_MIN_GBR_WEIGHT:.4f}"
        )
    elif (
        gbr_weight < STACK_GUARD_STANDARD_MIN_GBR_WEIGHT
        and not strong_evidence
    ):
        reasons.append(
            f"GBR750 weight is below the standard minimum without a strong repeat-wise gain: {gbr_weight:.4f} < "
            f"{STACK_GUARD_STANDARD_MIN_GBR_WEIGHT:.4f}"
        )

    if positive_partner_weight > STACK_GUARD_MAX_POSITIVE_PARTNER_WEIGHT:
        reasons.append(
            f"Positive partner weights exceed their limit: {positive_partner_weight:.4f} > "
            f"{STACK_GUARD_MAX_POSITIVE_PARTNER_WEIGHT:.4f}"
        )
    if absolute_partner_weight > STACK_GUARD_MAX_ABS_PARTNER_WEIGHT:
        reasons.append(
            f"Absolute partner weights exceed their limit: {absolute_partner_weight:.4f} > "
            f"{STACK_GUARD_MAX_ABS_PARTNER_WEIGHT:.4f}"
        )

    if (
        meta2_enabled and gate_enabled
        and gbr_weight < STACK_GUARD_COMPLEX_MIN_GBR_WEIGHT
        and not strong_evidence
    ):
        reasons.append("Joint residual correction and gating require more GBR750 weight or stronger validation gains")


    # Wide blends require a minimum worst-repeat gain.
    if (
        positive_partner_weight > STACK_GUARD_WIDE_BLEND_PARTNER_WEIGHT
        and final_worst_repeat_gain < STACK_GUARD_WIDE_BLEND_MIN_WORST_REPEAT_GAIN
    ):
        reasons.append(
            f"The worst repeat-wise gain is insufficient for a wide blend: {final_worst_repeat_gain:.6f} < "
            f"{STACK_GUARD_WIDE_BLEND_MIN_WORST_REPEAT_GAIN:.6f}"
        )

    if (
        abs(intercept) >= STACK_GUARD_INTERCEPT_EDGE
        and gbr_weight < STACK_GUARD_COMPLEX_MIN_GBR_WEIGHT
        and final_gain < STACK_GUARD_MIN_GAIN_AT_INTERCEPT_EDGE
        and not strong_evidence
    ):
        reasons.append(
            f"The final intercept is near its bound and meta-validation gains are insufficient: intercept={intercept:.4f}, "
            f"gain={final_gain:.6f}"
        )

    return {
        "Triggered": bool(STACK_CHAMPION_GUARD_ENABLED and len(reasons) > 0),
        "Reasons": reasons,
        "Strong_Evidence_Override": strong_evidence,
        "Base_R2": base_r2,
        "Best_Base_Model": best_base_model,
        "Best_Base_R2": best_base_r2,
        "GBR_Base_R2": gbr_r2,
        "GBR_Gap_To_Best": gbr_gap_to_best,
        "GBR750_Weight_PreGuard": gbr_weight,
        "Positive_Partner_Weight": positive_partner_weight,
        "Absolute_Partner_Weight": absolute_partner_weight,
        "Meta2_Enabled_PreGuard": meta2_enabled,
        "Gate_Enabled_PreGuard": gate_enabled,
        "Final_Positive_Repeats": int(final_positive_repeats),
        "Final_Worst_Repeat_Gain": float(final_worst_repeat_gain),
    }


def fit_safe_stacking(
    base_oof: np.ndarray,
    base_test: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_names: Sequence[str],
    seed: int,
) -> Dict:
    """Fit the five-stage ensemble and apply the fallback criteria."""
    base_oof = np.asarray(base_oof, dtype=float)
    base_test = np.asarray(base_test, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    names = list(model_names)
    gbr_index = names.index(STACK_ANCHOR_NAME)
    gbr_oof = base_oof[:, gbr_index]
    gbr_test = base_test[:, gbr_index]

    stage3 = _fit_meta3_stacking(base_oof, base_test, y, groups, names, int(seed))
    stage3_oof = np.asarray(stage3["Train_OOF_Prediction"], dtype=float)
    stage3_test = np.asarray(stage3["Test_Prediction"], dtype=float)

    meta4 = _select_meta4_configuration(
        base_oof, stage3_oof, y, groups, names, int(seed) + 1500001
    )
    best4 = meta4["Best_Row"]
    mode4 = str(best4["Regime_Mode"])
    meta4_stable = (
        mode4 != "None"
        and float(best4["Incremental_Gain"]) >= STACK_META4_REQUIRED_INCREMENTAL_GAIN
        and float(best4["Total_Gain"]) >= STACK_META4_REQUIRED_TOTAL_GAIN
        and int(best4["Positive_Repeats"]) >= int(np.ceil(STACK_META_REPEATS * STACK_META4_MIN_POSITIVE_REPEAT_RATIO))
        and int(best4["Positive_Folds"]) >= int(np.ceil(float(best4["Total_Folds"]) * STACK_META4_MIN_POSITIVE_FOLD_RATIO))
        and float(best4["Worst_Repeat_Gain"]) >= -STACK_META4_MAX_WORST_REPEAT_LOSS
        and float(best4["Worst_Fold_Gain"]) >= -STACK_META4_MAX_WORST_FOLD_LOSS
    )
    meta4_soft = (
        mode4 != "None"
        and float(best4["Incremental_Gain"]) >= STACK_META4_SOFT_MIN_INCREMENTAL_GAIN
        and float(best4["Total_Gain"]) > 0.0
    )

    if meta4_stable or meta4_soft:
        fitted4 = _fit_regime_calibrator(
            base_oof, stage3_oof, y, groups, names,
            mode4, int(best4["Level_Bins"]), int(best4["Secondary_Bins"]),
            float(best4["Regime_Shrinkage"]), float(best4["Regime_Smoothing"]),
            float(best4["Regime_Clip_Multiplier"]),
        )
        correction_test4 = _predict_regime_correction(fitted4, base_test, stage3_test)
        soft_fraction4 = 1.0 if meta4_stable else STACK_META4_SOFT_FRACTION
        layer4_test = stage3_test + soft_fraction4 * correction_test4
        layer4_oof = np.asarray(meta4["Payload"]["Meta_OOF"], dtype=float)
        if not meta4_stable:
            layer4_oof = stage3_oof + STACK_META4_SOFT_FRACTION * (layer4_oof - stage3_oof)
    else:
        mode4 = "None"
        layer4_oof = stage3_oof.copy()
        layer4_test = stage3_test.copy()

    meta5 = _select_meta5_configuration(
        gbr_oof, layer4_oof, y, groups, int(seed) + 2500001
    )
    best5 = meta5["Best_Row"]
    meta5_stable = (
        float(best5["Total_Gain"]) >= STACK_META5_REQUIRED_TOTAL_GAIN
        and float(best5["Incremental_Gain"]) >= STACK_META5_REQUIRED_INCREMENTAL_GAIN
        and int(best5["Positive_Repeats"]) >= int(np.ceil(STACK_META_REPEATS * STACK_META5_MIN_POSITIVE_REPEAT_RATIO))
        and int(best5["Positive_Folds"]) >= int(np.ceil(float(best5["Total_Folds"]) * STACK_META5_MIN_POSITIVE_FOLD_RATIO))
        and float(best5["Worst_Repeat_Gain"]) >= -STACK_META5_MAX_WORST_REPEAT_LOSS
        and float(best5["Worst_Fold_Gain"]) >= -STACK_META5_MAX_WORST_FOLD_LOSS
    )
    meta5_soft = (
        float(best5["Total_Gain"]) >= STACK_META5_SOFT_MIN_TOTAL_GAIN
        and float(best5["Incremental_Gain"]) > 0.0
    )

    blend5 = float(best5["Final_Blend"])
    clip5 = float(best5["Final_Delta_Clip"])
    scale5 = float(best5["Final_Scale5"])
    residual_scale = _robust_residual_scale(y - gbr_oof)
    if blend5 == 0.0:
        train_blend5 = gbr_oof.copy()
        test_blend5 = gbr_test.copy()
    else:
        bound5 = max(0.020, clip5 * residual_scale)
        train_delta5 = np.clip(layer4_oof - gbr_oof, -bound5, bound5)
        test_delta5 = np.clip(layer4_test - gbr_test, -bound5, bound5)
        train_blend5 = gbr_oof + blend5 * train_delta5
        test_blend5 = gbr_test + blend5 * test_delta5
    full_intercept5 = np.clip(
        _weighted_mean(y - scale5 * train_blend5, group_balanced_weights(groups)),
        -STACK_META5_INTERCEPT_BOUND, STACK_META5_INTERCEPT_BOUND,
    )
    average_intercept5 = float(meta5["Payload"]["Average_Intercept"])
    intercept5 = (
        (1.0 - STACK_META5_FINAL_INTERCEPT_BLEND) * average_intercept5
        + STACK_META5_FINAL_INTERCEPT_BLEND * float(full_intercept5)
    )
    final_test = intercept5 + scale5 * test_blend5
    final_oof = np.asarray(meta5["Payload"]["Meta_OOF"], dtype=float)

    if not meta5_stable and meta5_soft:
        final_test = gbr_test + STACK_META5_SOFT_FRACTION * (final_test - gbr_test)
        final_oof = gbr_oof + STACK_META5_SOFT_FRACTION * (final_oof - gbr_oof)
    elif not meta5_stable and not meta5_soft:
        # Retain the lower-stage prediction when the final blend has insufficient support.
        final_test = layer4_test
        final_oof = layer4_oof
        blend5, clip5, scale5, intercept5 = 1.0, 0.0, 1.0, 0.0

    selected_models = list(stage3.get("Selected_Models", []))
    if mode4 != "None":
        selected_models.append(f"Meta4:{mode4}")
    if meta5_stable or meta5_soft:
        selected_models.append("Meta5:StabilityBlend")
    selected_models = list(dict.fromkeys(selected_models))

    pre_guard_gain = float(safe_r2(y, final_oof) - safe_r2(y, gbr_oof))
    guard = _evaluate_champion_guard(
        base_oof, y, names, stage3, pre_guard_gain,
        int(best5["Positive_Repeats"]),
        float(best5["Worst_Repeat_Gain"]),
    )
    guard_triggered = bool(guard["Triggered"])

    pre_guard_selected_models = list(selected_models)
    pre_guard_weights = np.asarray(stage3.get("Weights"), dtype=float).copy()
    pre_guard_bias = float(stage3.get("Bias", 0.0))
    if guard_triggered:
        final_oof = gbr_oof.copy()
        final_test = gbr_test.copy()
        selected_models = [STACK_ANCHOR_NAME]
        final_weights = np.zeros(len(names), dtype=float)
        final_weights[gbr_index] = 1.0
        final_bias = 0.0
        final_gain = 0.0
    else:
        final_weights = pre_guard_weights
        final_bias = pre_guard_bias
        final_gain = pre_guard_gain

    is_final_gbr = bool(np.allclose(final_oof, gbr_oof, atol=1e-12))
    result = dict(stage3)
    result.update({
        "Safety_Fallback": bool(is_final_gbr),
        "Soft_Shrinkage": bool((not guard_triggered) and (stage3.get("Soft_Shrinkage", False) or (meta4_soft and not meta4_stable) or (meta5_soft and not meta5_stable))),
        "Fallback_Model": STACK_ANCHOR_NAME if is_final_gbr else None,
        "Selected_Models": selected_models,
        "Weights": final_weights,
        "Bias": final_bias,
        "GBR750_Weight": float(final_weights[gbr_index]),
        "Train_OOF_Prediction": final_oof,
        "Test_Prediction": final_test,
        "Champion_Guard_Triggered": guard_triggered,
        "Champion_Guard_Reasons": "; ".join(guard["Reasons"]) if guard["Reasons"] else "None",
        "Champion_Guard_Strong_Evidence_Override": guard["Strong_Evidence_Override"],
        "Best_Base_Model_TrainOOF": guard["Best_Base_Model"],
        "GBR_Base_R2_TrainOOF": guard["GBR_Base_R2"],
        "GBR_Gap_To_Best_TrainOOF": guard["GBR_Gap_To_Best"],
        "Positive_Partner_Weight_PreGuard": guard["Positive_Partner_Weight"],
        "Absolute_Partner_Weight_PreGuard": guard["Absolute_Partner_Weight"],
        "PreGuard_Meta_Gain": pre_guard_gain,
        "PreGuard_Selected_Models": pre_guard_selected_models,
        "PreGuard_Weights": pre_guard_weights,
        "PreGuard_Bias": pre_guard_bias,
        "Meta4_Mode": mode4,
        "Meta4_Level_Bins": int(best4["Level_Bins"]),
        "Meta4_Secondary_Bins": int(best4["Secondary_Bins"]),
        "Meta4_Shrinkage": float(best4["Regime_Shrinkage"]),
        "Meta4_Smoothing": float(best4["Regime_Smoothing"]),
        "Meta4_Clip_Multiplier": float(best4["Regime_Clip_Multiplier"]),
        "Meta4_Gain": float(best4["Total_Gain"]),
        "Meta4_Incremental_Gain": float(best4["Incremental_Gain"]),
        "Meta4_Candidate_Table": meta4["Table"],
        "Meta5_Blend": float(blend5),
        "Meta5_Delta_Clip": float(clip5),
        "Meta5_Scale": float(scale5),
        "Meta5_Intercept": float(intercept5),
        "Meta5_Gain": float(best5["Total_Gain"]),
        "Meta5_Incremental_Gain": float(best5["Incremental_Gain"]),
        "Meta5_Candidate_Table": meta5["Table"],
        "Meta_Gain": final_gain,
        "Positive_Repeats": int(best5["Positive_Repeats"]),
        "Worst_Repeat_Gain": float(best5["Worst_Repeat_Gain"]),
        "Meta4_Enabled": mode4 != "None",
        "Meta5_Enabled": bool(meta5_stable or meta5_soft),
    })
    return result


def fit_predict_model(
    builder: Callable[[int, Mapping], object],
    params: Mapping,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seed: int,
) -> np.ndarray:
    model = builder(seed, params)
    model.fit(X_train, y_train)
    return np.asarray(model.predict(X_test), dtype=float)


def fit_predict_gbr_bagged(
    params: Mapping,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seed: int,
) -> np.ndarray:
    predictions = []
    for offset in GBR_BAG_SEEDS:
        model = make_gbr(seed + int(offset), params)
        model.fit(X_train, y_train)
        predictions.append(np.asarray(model.predict(X_test), dtype=float))
    return np.mean(np.column_stack(predictions), axis=1)


@dataclass
class StandaloneFold:
    number: int
    test_index: np.ndarray
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    groups_train: np.ndarray
    inner_cache: List[InnerFoldData]
    train_oof: Dict[str, np.ndarray]
    test_predictions: Dict[str, np.ndarray]


def fit_standalone_fold(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    train_index: np.ndarray,
    test_index: np.ndarray,
    metadata: Mapping,
    fold_number: int,
) -> StandaloneFold:
    X_train_raw = X.iloc[train_index].reset_index(drop=True)
    X_test_raw = X.iloc[test_index].reset_index(drop=True)
    y_train = y[train_index]
    groups_train = groups[train_index]
    if set(groups_train).intersection(groups[test_index]):
        raise RuntimeError("Outer training and test groups overlap.")

    numeric_columns = metadata["numeric_columns"]
    categorical_columns = metadata["categorical_columns"]
    mode, inner_cache = select_preprocessor_and_cache(
        X_train_raw, y_train, groups_train, numeric_columns, categorical_columns,
        RANDOM_STATE * 10000 + fold_number * 1000,
    )
    model_specs = (
        ("RF", RF_CANDIDATES, make_rf, 101, 1),
        ("ExtraTrees", ET_CANDIDATES, make_extra_trees, 201, 2),
        ("LGBM", LGBM_CANDIDATES, make_lgbm, 401, 4),
        ("XGB", XGB_CANDIDATES, make_xgb, 501, 5),
        ("GBR", GBR_CANDIDATES, make_gbr, 551, 5),
    )
    model_details = {
        name: evaluate_candidate_grid(
            inner_cache, y_train, candidates, builder, fold_number * 10000 + inner_offset,
        )
        for name, candidates, builder, inner_offset, _ in model_specs
    }
    train_oof = {
        name: np.asarray(details["OOF_Prediction"], dtype=float)
        for name, details in model_details.items()
    }

    preprocessor = build_preprocessor(numeric_columns, categorical_columns, mode)
    X_train = np.asarray(preprocessor.fit_transform(X_train_raw, y_train), dtype=np.float32)
    X_test = np.asarray(preprocessor.transform(X_test_raw), dtype=np.float32)

    test_predictions = {}
    for name, _, builder, _, outer_offset in model_specs:
        params = model_details[name]["Params"]
        seed = fold_number * 1000 + outer_offset
        if name == "GBR":
            prediction = fit_predict_gbr_bagged(params, X_train, y_train, X_test, seed)
        else:
            prediction = fit_predict_model(builder, params, X_train, y_train, X_test, seed)
        test_predictions[name] = prediction

    return StandaloneFold(
        number=fold_number,
        test_index=test_index,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        groups_train=groups_train,
        inner_cache=inner_cache,
        train_oof=train_oof,
        test_predictions=test_predictions,
    )


def fit_stacking_fold(fold: StandaloneFold) -> np.ndarray:
    anchor_details = evaluate_candidate_grid(
        fold.inner_cache, fold.y_train, GBR750_CANDIDATES, make_gbr,
        fold.number * 10000 + 601,
    )
    anchor_oof = np.asarray(anchor_details["OOF_Prediction"], dtype=float)
    anchor_test = fit_predict_gbr_bagged(
        anchor_details["Params"], fold.X_train, fold.y_train, fold.X_test,
        fold.number * 1000 + 6,
    )
    train_oof = {**fold.train_oof, STACK_ANCHOR_NAME: anchor_oof}
    test_predictions = {**fold.test_predictions, STACK_ANCHOR_NAME: anchor_test}
    base_oof = np.column_stack([train_oof[name] for name in BASE_MODEL_NAMES])
    base_test = np.column_stack([test_predictions[name] for name in BASE_MODEL_NAMES])
    stack = fit_safe_stacking(
        base_oof, base_test, fold.y_train, fold.groups_train, BASE_MODEL_NAMES,
        RANDOM_STATE * 200000 + fold.number * 1000,
    )
    return stack["Test_Prediction"]


def build_r2_table(
    y: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    folds: Sequence[StandaloneFold],
) -> pd.DataFrame:
    rows = []
    for name, prediction in predictions.items():
        fold_scores = {
            f"Fold {fold.number}": safe_r2(y[fold.test_index], prediction[fold.test_index])
            for fold in folds
        }
        rows.append({
            "Model": name,
            **fold_scores,
            "Fold mean": pd.Series(fold_scores, dtype=float).mean(),
            "Pooled OOF": safe_r2(y, prediction),
        })
    return pd.DataFrame(rows)


def main(
    data_path: Path = DEFAULT_DATA_PATH,
    sheet_name: str = DEFAULT_SHEET_NAME,
) -> None:
    X, y, groups, metadata = load_and_prepare_data(data_path, sheet_name)
    outer_splits = make_group_folds(groups, y, OUTER_SPLITS, RANDOM_STATE, outer=True)
    standalone_oof = {
        display_name: np.full(len(y), np.nan, dtype=float)
        for _, display_name in STANDALONE_REPORT_NAMES
    }
    folds: List[StandaloneFold] = []
    for number, (train_index, test_index) in enumerate(outer_splits, start=1):
        fold = fit_standalone_fold(X, y, groups, train_index, test_index, metadata, number)
        folds.append(fold)
        for model_name, display_name in STANDALONE_REPORT_NAMES:
            standalone_oof[display_name][test_index] = fold.test_predictions[model_name]

    print("Standalone model R² on the log10(CCC) scale", flush=True)
    print(build_r2_table(y, standalone_oof, folds).to_string(
        index=False, float_format=lambda score: f"{score:.4f}",
    ), flush=True)

    stacking_oof = np.full(len(y), np.nan, dtype=float)
    for fold in folds:
        stacking_oof[fold.test_index] = fit_stacking_fold(fold)

    print("\nStacking R² on the log10(CCC) scale", flush=True)
    print(build_r2_table(y, {"Stacking": stacking_oof}, folds).to_string(
        index=False, float_format=lambda score: f"{score:.4f}",
    ), flush=True)


def cli(argv: Optional[Sequence[str]] = None) -> None:
    global N_JOBS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH,
                        help="Input workbook path (default: %(default)s).")
    parser.add_argument("--sheet", default=DEFAULT_SHEET_NAME, help="Input worksheet name.")
    parser.add_argument("--jobs", type=int, default=N_JOBS,
                        help="Workers for parallel base models (default: %(default)s).")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be a positive integer")
    N_JOBS = args.jobs

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"X does not have valid feature names, but LGBMRegressor was fitted with feature names",
            category=UserWarning,
            module=r"sklearn\.utils\.validation",
        )
        main(args.data, args.sheet)


if __name__ == "__main__":
    cli(None if "__file__" in globals() else [])
