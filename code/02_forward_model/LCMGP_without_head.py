import os, json, time, random, argparse, copy, math, py_compile, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch, gpytorch
from scipy import stats
from gpytorch.priors import LogNormalPrior, GammaPrior, NormalPrior
from gpytorch.constraints import Interval
PERM_COL = 'perm (mm/s)'
STRENGTH_COL = 'comp_converted'
DEFAULT_DATA_PATH = str(Path(__file__).resolve().parents[2] / "data" / "data_with_passing_sizes.xlsx")
GRADATION_COLS = [
    'R25-19mm',
    'R19-12.5mm',
    'R12.5-9.5mm',
    'R9.5-6.3mm',
    'R6.3-4.75mm',
    'R4.75-2.36mm',
]
BIN_EDGES_MM = [
    (25.0, 19.0),
    (19.0, 12.5),
    (12.5, 9.5),
    (9.5, 6.3),
    (6.3, 4.75),
    (4.75, 2.36),
]
BIN_DIAMETERS_MM = np.asarray([math.sqrt(upper * lower) for upper, lower in BIN_EDGES_MM], dtype=float)
FEATURES = [
    'porosity',
    'w/c',
    'cem (kg/m3)',
    'SP (kg/m3)',
    'sand (kg/m3)',
    'WAS',
    'I_gap',
    'Cu',
    'Aag(mm^-1)',
]
CONT_IDXS = list(range(len(FEATURES)))
NUM_TASKS = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float32
SPLIT_CACHE = {}
MAX_CG_ITERATIONS = 3000
CG_TOLERANCE = 0.001
SIGMA_FLOOR_S = 1e-06
SIGMA_FLOOR_P = 1e-06
LOCAL_RESIDUAL_K_NEIGHBORS = 10
OOS_RESIDUAL_FEATURE_FILE = 'pooled_oos_features_and_residuals.csv'
TRANSFORM_RECIPE = {
    'log_cols': ['cem (kg/m3)', 'Cu'],
    'log1p_cols': ['SP (kg/m3)', 'sand (kg/m3)', 'Aag(mm^-1)'],
    'raw_cols': ['porosity', 'w/c', 'WAS', 'I_gap'],
    'categorical_cols': [],
}
CALIBRATION_LEVELS = (0.5, 0.8, 0.9, 0.95)
LOWER_BOUND_LEVELS = (0.8, 0.9, 0.95)
REGIME_CALIBRATION_MIN_N = 20
AD_K_NEIGHBORS = 10
AD_THRESHOLD_QUANTILE = 0.95
AD_INFLATION_START = 0.75
AD_INFLATION_ETA = 1.0
AD_INFLATION_CAP = 3.0
UNCERTAINTY_COMBINE_RULE = 'max'
VALID_KERNEL_COMPONENTS = {'rbf', 'rq', 'matern12', 'matern32', 'matern52'}
VALID_ADD_KERNELS = {'rbf', 'matern32', 'matern52'}
STD_UNCERTAINTY_NOTE = (
    'Standard deviation columns are effective base uncertainty estimates. Final empirical calibration is applied '
    'through interval and lower-bound multiplier columns. For reporting, use prediction interval and lower-bound '
    'columns, not raw/effective standard deviations alone.'
)
CALIBRATION_AUDIT_NOTE = (
    'Leave-one-split-out audit using existing repeated-split OOS predictions; no retraining; same original-row '
    'leakage excluded when original_row_index is available.'
)
OOS_STANDARDIZATION_NOTE = (
    'OOS residual reference features are transformed raw features standardized by the final full-data XStandardizer. '
    'They are not split-standardized features.'
)

def set_seed(s):
    random.seed(int(s))
    np.random.seed(int(s))
    torch.manual_seed(int(s))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(s))

def set_runtime_precision(dtype_name):
    global DTYPE
    if str(dtype_name).lower() == 'float64':
        DTYPE = torch.float64
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
            except Exception:
                pass
            try:
                torch.backends.cudnn.allow_tf32 = False
            except Exception:
                pass
        try:
            torch.set_float32_matmul_precision('highest')
        except Exception:
            pass
    else:
        DTYPE = torch.float32
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
            except Exception:
                pass
            try:
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

def _p(level, msg, v):
    if int(v) > 0:
        print(f'[{level}] {msg}', flush=True)

def _gpu_cleanup():
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass

def _finite_mask(*arrs):
    m = None
    for a in arrs:
        mi = np.isfinite(np.asarray(a))
        m = mi if m is None else m & mi
    return m

def mae(t, p):
    t, p = (np.asarray(t).ravel(), np.asarray(p).ravel())
    return float(np.mean(np.abs(t - p)))

def rmse(t, p):
    t, p = (np.asarray(t).ravel(), np.asarray(p).ravel())
    return float(np.sqrt(np.mean((t - p) ** 2)))

def r2_score(t, p):
    t, p = (np.asarray(t).ravel(), np.asarray(p).ravel())
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')

def coverage(t, mu, s, z):
    t, mu, s = (np.asarray(t).ravel(), np.asarray(mu).ravel(), np.asarray(s).ravel())
    return float(np.mean((t >= mu - z * s) & (t <= mu + z * s)))

def nlpd_gaussian(t, mu, s):
    t, mu, s = (np.asarray(t).ravel(), np.asarray(mu).ravel(), np.asarray(s).ravel())
    v = np.maximum(s * s, 1e-12)
    return float(np.mean(0.5 * np.log(2 * math.pi * v) + 0.5 * (t - mu) ** 2 / v))

def _mean_std(x):
    x = np.asarray(x, float)
    return (float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0)

def _nanmean_or_nan(x):
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float('nan')

class XStandardizer:

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        Xc = X[:, CONT_IDXS]
        m = Xc.mean(0)
        s = Xc.std(0, ddof=0)
        s[s == 0] = 1.0
        self.mean, self.std = (m, s)
        return self

    def transform(self, X):
        Xs = X.copy()
        Xs[:, CONT_IDXS] = (X[:, CONT_IDXS] - self.mean) / self.std
        return Xs

class YStandardizer:

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, Y):
        self.mean = Y.mean(0)
        self.std = Y.std(0, ddof=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, Y):
        return (Y - self.mean) / self.std

def _apply_feature_transforms(df):
    out = df.copy()
    for c in TRANSFORM_RECIPE['log_cols']:
        if np.any(out[c] <= 0):
            raise ValueError(f"Column '{c}' must be positive for log transform.")
        out[c] = np.log(out[c])
    for c in TRANSFORM_RECIPE['log1p_cols']:
        if np.any(out[c] < 0):
            raise ValueError(f"Column '{c}' must be non-negative for log1p transform.")
        out[c] = np.log1p(out[c])
    return out

def _resolve_paper_source_column(df, requested=''):
    requested = str(requested or '').strip()
    if requested:
        if requested in df.columns:
            return requested
        matches = [c for c in df.columns if str(c).strip().casefold() == requested.casefold()]
        if len(matches) == 1:
            return matches[0]
        raise ValueError(
            f"Paper-source column '{requested}' was not found. "
            f"Available columns: {[str(c) for c in df.columns]}"
        )
    scored = []
    for position, col in enumerate(df.columns):
        values = df[col].dropna().astype(str).str.strip()
        markers = int(values.str.contains(r'^data\s+source\b', case=False, regex=True).sum())
        name = str(col).strip().casefold()
        name_bonus = int('paper' in name or 'source' in name)
        scored.append((markers, name_bonus, -position, col))
    marker_count, _, _, selected = max(scored, key=lambda item: item[:3])
    if marker_count < 1:
        raise ValueError(
            'Could not detect the paper-source column. '
            'Use --paper_source_column with the exact Excel header.'
        )
    return selected

def _clean_paper_label(value):
    text = ' '.join(str(value).strip().split())
    return re.sub(r'^data\s+source\s*-\s*(?:\d+|new)\s*-\s*', '', text, flags=re.IGNORECASE).strip()

def _safe_filename_token(value):
    token = re.sub(r'[^A-Za-z0-9._-]+', '_', str(value).strip()).strip('._-')
    return token[:100] or 'paper'

def _paper_source_labels(df, requested_column=''):
    source_col = _resolve_paper_source_column(df, requested=requested_column)
    source = df[source_col].copy()
    source = source.where(source.notna(), np.nan)
    source = source.map(lambda value: str(value).strip() if pd.notna(value) else np.nan)
    source = source.replace('', np.nan).ffill()
    label = source.map(lambda value: _clean_paper_label(value) if pd.notna(value) else '')
    return source_col, source, label

def _reconstruct_i_gap(df):
    missing = [c for c in GRADATION_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            "Column 'I_gap' is missing and cannot be reconstructed because these gradation columns "
            f'are absent: {missing}'
        )
    grad = df[GRADATION_COLS].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    result = np.full(len(df), np.nan, dtype=float)
    for row_idx, values in enumerate(grad):
        if not np.isfinite(values).all() or np.any(values < 0):
            continue
        total = float(values.sum())
        if total <= 0:
            continue
        fractions = values / total
        i_gap = 0.0
        for i in range(len(fractions)):
            for j in range(i + 1, len(fractions)):
                if fractions[i] <= 0 or fractions[j] <= 0:
                    continue
                i_gap += fractions[i] * fractions[j] * math.log(BIN_DIAMETERS_MM[i] / BIN_DIAMETERS_MM[j])
        result[row_idx] = float(i_gap)
    return result

def load_and_prepare_data(path, split_mode='random', paper_source_column='', return_metadata=False):
    df = pd.read_excel(path)
    input_columns = list(df.columns)
    i_gap_reconstructed = False
    if 'I_gap' not in df.columns:
        df['I_gap'] = _reconstruct_i_gap(df)
        i_gap_reconstructed = True

    split_mode = str(split_mode).strip().lower()
    paper_source = pd.Series([''] * len(df), index=df.index, dtype=object)
    paper_label = pd.Series([''] * len(df), index=df.index, dtype=object)
    source_col = ''
    if split_mode == 'paperwise':
        source_col, paper_source, paper_label = _paper_source_labels(df, requested_column=paper_source_column)

    needed = FEATURES + [STRENGTH_COL, PERM_COL]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing column: '{c}'")

    numeric = df[needed].copy().replace([np.inf, -np.inf], np.nan)
    numeric = numeric.apply(pd.to_numeric, errors='coerce')
    complete = numeric.notna().all(axis=1)
    n0 = int((complete & numeric[PERM_COL].le(0)).sum())
    keep = complete & numeric[PERM_COL].gt(0)
    if split_mode == 'paperwise':
        source_available = paper_source.notna() & paper_source.astype(str).str.strip().ne('')
        missing_source = keep & ~source_available
        if bool(missing_source.any()):
            rows = (np.flatnonzero(missing_source.to_numpy()) + 2).tolist()
            raise ValueError(f'Usable rows without a paper-source marker were found at Excel rows: {rows[:20]}')
        keep &= source_available

    prepared = numeric.loc[keep].reset_index(drop=True)
    prepared_features = _apply_feature_transforms(prepared[FEATURES])
    X = prepared_features[FEATURES].to_numpy(dtype=float)
    yS = prepared[STRENGTH_COL].to_numpy(dtype=float)
    yP = np.log(prepared[PERM_COL].to_numpy(dtype=float))
    if not return_metadata:
        return (X, yS, yP, n0)

    raw_indices = np.flatnonzero(keep.to_numpy()).astype(int)
    mix_id = (
        df.loc[keep, 'ID'].reset_index(drop=True).astype(str).to_numpy()
        if 'ID' in df.columns
        else np.asarray([''] * len(raw_indices), dtype=object)
    )
    metadata = pd.DataFrame({
        'prepared_row_index': np.arange(len(raw_indices), dtype=int),
        'source_dataframe_row_index': raw_indices,
        'source_excel_row': raw_indices + 2,
        'mix_id': mix_id,
        'paper_source': paper_source.loc[keep].astype(str).to_numpy(),
        'paper_label': paper_label.loc[keep].astype(str).to_numpy(),
    })
    info = {
        'split_mode': split_mode,
        'paper_source_column': str(source_col),
        'i_gap_reconstructed_from_gradation': bool(i_gap_reconstructed),
        'input_columns': [str(c) for c in input_columns],
        'n_rows_raw': int(len(df)),
        'n_rows_used': int(len(prepared)),
        'n_rows_dropped': int(len(df) - len(prepared)),
        'row_metadata': metadata,
    }
    return (X, yS, yP, n0, info)

def build_paperwise_splits(paper_groups, selected_papers=''):
    groups = np.asarray(paper_groups, dtype=object)
    unique_groups = list(pd.unique(pd.Series(groups)))
    if len(unique_groups) < 2:
        raise ValueError('Paperwise evaluation requires at least two distinct papers.')
    all_idx = np.arange(len(groups), dtype=int)
    splits = []
    for fold, source in enumerate(unique_groups):
        test_idx = all_idx[groups == source]
        train_idx = all_idx[groups != source]
        if len(test_idx) < 1 or len(train_idx) < 1:
            raise ValueError(f"Paperwise fold '{source}' has an empty train or test partition.")
        splits.append({
            'split': int(fold),
            'split_type': 'leave_one_paper_out',
            'paper_source': str(source),
            'paper_label': _clean_paper_label(source),
            'train_idx': train_idx,
            'test_idx': test_idx,
            'seed': None,
        })

    requested = [x.strip() for x in str(selected_papers or '').replace(';', ',').split(',') if x.strip()]
    if not requested:
        return splits
    selected = []
    seen = set()
    unmatched = []
    for token in requested:
        token_key = token.casefold()
        matches = [
            split for split in splits
            if token_key in {
                str(split['split']).casefold(),
                split['paper_source'].casefold(),
                split['paper_label'].casefold(),
            }
        ]
        if not matches:
            unmatched.append(token)
            continue
        for split in matches:
            if split['split'] not in seen:
                selected.append(split)
                seen.add(split['split'])
    if unmatched:
        available = ', '.join(split['paper_label'] for split in splits)
        raise ValueError(f'Unknown --paperwise_papers value(s): {unmatched}. Available papers: {available}')
    return selected

def build_evaluation_splits(n_rows, split_mode, n_splits, test_frac, base_seed, paper_groups=None, selected_papers=''):
    split_mode = str(split_mode).strip().lower()
    if split_mode == 'paperwise':
        if paper_groups is None:
            raise ValueError('paper_groups are required for paperwise evaluation.')
        return build_paperwise_splits(paper_groups, selected_papers=selected_papers)
    if split_mode != 'random':
        raise ValueError("--split_mode must be 'paperwise' or 'random'.")
    splits = []
    for i in range(int(n_splits)):
        seed = int(base_seed + i)
        train_idx, test_idx = random_split(n_rows, test_frac, seed)
        splits.append({
            'split': int(i),
            'split_type': 'random_row_split',
            'paper_source': '',
            'paper_label': '',
            'train_idx': train_idx,
            'test_idx': test_idx,
            'seed': seed,
        })
    return splits

def random_split(n, test_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    nt = max(1, int(round(float(test_frac) * n)))
    return (perm[nt:], perm[:nt])

def _kernel_components(kernel_name):
    parts = [p.strip().lower() for p in str(kernel_name).split('+') if p.strip()]
    if not parts:
        raise ValueError('--kernel must be non-empty. Valid components: ' + ', '.join(sorted(VALID_KERNEL_COMPONENTS)))
    bad = [p for p in parts if p not in VALID_KERNEL_COMPONENTS]
    if bad:
        raise ValueError(
            "Unknown --kernel component(s): "
            + ', '.join(bad)
            + '. Valid components: '
            + ', '.join(sorted(VALID_KERNEL_COMPONENTS))
        )
    return parts

def _validate_add_kernel_name(add_kernel_name):
    name = str(add_kernel_name).strip().lower()
    if name not in VALID_ADD_KERNELS:
        raise ValueError(
            f"Unknown --add_kernel '{add_kernel_name}'. Valid options: "
            + ', '.join(sorted(VALID_ADD_KERNELS))
        )
    return name

def _validate_training_args(args, n_rows=None):
    if not (0.0 < float(args.target_coverage) < 1.0):
        raise ValueError('--target_coverage must be > 0 and < 1.')
    if int(args.n_splits) < 1:
        raise ValueError('--n_splits must be >= 1.')
    if not (0.0 < float(args.test_frac) < 1.0):
        raise ValueError('--test_frac must be > 0 and < 1.')
    if int(args.iters) < 1:
        raise ValueError('--iters must be >= 1.')
    if int(args.final_iters) < 1:
        raise ValueError('--final_iters must be >= 1.')
    if int(args.n_restarts) < 1:
        raise ValueError('--n_restarts must be >= 1.')
    if int(args.final_restarts) < 1:
        raise ValueError('--final_restarts must be >= 1.')
    if float(args.lr) <= 0:
        raise ValueError('--lr must be > 0.')
    if int(args.rank) < 1:
        raise ValueError('--rank must be >= 1.')
    _kernel_components(args.kernel)
    _validate_add_kernel_name(args.add_kernel)
    if int(args.max_cg_iterations) < 1:
        raise ValueError('--max_cg_iterations must be >= 1.')
    if float(args.cg_tolerance) <= 0:
        raise ValueError('--cg_tolerance must be > 0.')
    if int(args.plot_dpi) < 50:
        raise ValueError('--plot_dpi must be >= 50.')
    if hasattr(args, 'local_residual_q') and int(args.local_residual_q) < 2:
        raise ValueError('--local_residual_q must be >= 2.')
    selected_rule = _clean_interval_rule(args.uncertainty_combine_rule)
    if selected_rule not in {'gp_only', 'local_only', 'mean', 'max', 'sqrt'}:
        raise ValueError(
            f"Unknown --uncertainty_combine_rule '{args.uncertainty_combine_rule}'. "
            'Use gp_only, local_only, mean, max, or sqrt.'
        )
    _parse_policy_rules(args.policy_compare_rules)
    if n_rows is not None:
        n = int(n_rows)
        if n < 2:
            raise ValueError('Dataset must contain at least two usable rows after filtering.')
        n_test = max(1, int(round(float(args.test_frac) * n)))
        n_train = n - n_test
        if n_train < 1 or n_test < 1:
            raise ValueError(
                f'Dataset/test split leaves train={n_train}, test={n_test}; need at least one train and one test row.'
            )

def _additive_groups():

    def idx(f):
        return FEATURES.index(f)
    return {'Binder': [idx('w/c'), idx('cem (kg/m3)'), idx('SP (kg/m3)')], 'Filler': [idx('porosity'), idx('sand (kg/m3)')], 'Aggregate': [idx('WAS'), idx('I_gap'), idx('Cu'), idx('Aag(mm^-1)')]}

def _reg_ls(k, lo=0.2, hi=4.0, loc=0.0, scale=0.45, init=1.0):
    k.register_constraint('raw_lengthscale', Interval(lo, hi))
    k.register_prior('lengthscale_prior', LogNormalPrior(loc, scale), 'lengthscale')
    with torch.no_grad():
        k.lengthscale = init
    return k

def _make_cont_kernel(kernel_name, use_additive, add_kernel_name):
    nc = len(CONT_IDXS)
    cont = list(CONT_IDXS)
    parts = _kernel_components(kernel_name)
    add_kernel_name = _validate_add_kernel_name(add_kernel_name)
    comps = []
    for part in parts:
        if part == 'rbf':
            k = gpytorch.kernels.RBFKernel(ard_num_dims=nc, active_dims=cont)
            _reg_ls(k)
        elif part == 'rq':
            k = gpytorch.kernels.RQKernel(ard_num_dims=nc, active_dims=cont)
            _reg_ls(k)
            k.register_constraint('raw_alpha', Interval(0.25, 10.0))
            k.register_prior('alpha_prior', GammaPrior(2.0, 1.0), 'alpha')
            with torch.no_grad():
                k.alpha = 2.0
        elif part in ('matern12', 'matern32', 'matern52'):
            nu = {'matern12': 0.5, 'matern32': 1.5, 'matern52': 2.5}[part]
            k = gpytorch.kernels.MaternKernel(nu=nu, ard_num_dims=nc, active_dims=cont)
            _reg_ls(k)
        else:
            raise ValueError(f'Unknown kernel component: {part}')
        comps.append(k)
    k_sum = comps[0]
    for kk in comps[1:]:
        k_sum = k_sum + kk
    if use_additive:
        for idxs in _additive_groups().values():
            n_g = len(idxs)
            if add_kernel_name == 'matern32':
                kg = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=n_g, active_dims=idxs)
            elif add_kernel_name == 'matern52':
                kg = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=n_g, active_dims=idxs)
            elif add_kernel_name == 'rbf':
                kg = gpytorch.kernels.RBFKernel(ard_num_dims=n_g, active_dims=idxs)
            else:
                raise ValueError(f"Unknown additive kernel '{add_kernel_name}'.")
            _reg_ls(kg)
            k_sum = k_sum + kg
    return k_sum

def _scaled_kernel(k_sum, init=0.8, lo=0.05, hi=3.0, prior_log_mean=None, prior_sd=0.45):
    k_data = gpytorch.kernels.ScaleKernel(k_sum)
    if prior_log_mean is None:
        prior_log_mean = math.log(float(init))
    k_data.register_constraint('raw_outputscale', Interval(float(lo), float(hi)))
    k_data.register_prior('outputscale_prior', LogNormalPrior(float(prior_log_mean), float(prior_sd)), 'outputscale')
    with torch.no_grad():
        k_data.outputscale = float(init)
    return k_data

class TwoTaskConstantMean(gpytorch.means.Mean):

    def __init__(self):
        super().__init__()
        self.register_parameter('strength_const', torch.nn.Parameter(torch.tensor(0.0)))
        self.register_parameter('perm_const', torch.nn.Parameter(torch.tensor(0.0)))
        self.register_prior('strength_const_prior', NormalPrior(0.0, 1.0), 'strength_const')
        self.register_prior('perm_const_prior', NormalPrior(0.0, 1.0), 'perm_const')

    def forward(self, x):
        out = x.new_zeros(*x.shape[:-1], NUM_TASKS)
        out[..., 0] = self.strength_const
        out[..., 1] = self.perm_const
        return out

class LCMGPModel(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood, cfg):
        super().__init__(train_x, train_y, likelihood)
        self.cfg = dict(cfg)
        self.mean_module = TwoTaskConstantMean()
        shared_cont = _make_cont_kernel(cfg['kernel'], cfg['use_additive'], cfg['add_kernel'])
        shared_data = _scaled_kernel(shared_cont, init=0.8, lo=0.05, hi=3.0, prior_log_mean=math.log(0.8), prior_sd=0.4)
        self.shared_covar_module = gpytorch.kernels.MultitaskKernel(shared_data, num_tasks=NUM_TASKS, rank=int(cfg['rank']))

    def forward(self, x):
        mean = self.mean_module(x)
        shared_cov = self.shared_covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean, shared_cov)

def _build_likelihood():
    lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=NUM_TASKS, rank=0, has_global_noise=False, has_task_noise=True).to(device=DEVICE, dtype=DTYPE)
    try:
        lik.register_constraint('raw_task_noises', Interval(0.003, 0.1))
    except Exception:
        pass
    try:
        lik.register_prior('strength_task_noise_prior', LogNormalPrior(math.log(0.025), 0.35), lambda m: m.task_noises[..., 0])
        lik.register_prior('perm_task_noise_prior', LogNormalPrior(math.log(0.01), 0.35), lambda m: m.task_noises[..., 1])
    except Exception:
        pass
    with torch.no_grad():
        try:
            lik.task_noises = torch.tensor([0.025, 0.01], dtype=DTYPE, device=DEVICE)
        except Exception:
            pass
    return lik

def _clone_sd(sd, device):
    out = {}
    for k, v in sd.items():
        out[k] = v.detach().to(device=device).contiguous() if torch.is_tensor(v) else copy.deepcopy(v)
    return out

def _train_settings(fast):
    if fast:
        return gpytorch.settings.fast_computations(covar_root_decomposition=True, log_prob=True, solves=True)
    return gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False)

def _max_chol(fast):
    return gpytorch.settings.max_cholesky_size(0 if fast else 10000000)

def train_single(X_train, Y_train, cfg, iters, lr, patience, init_seed, fast_mode, verbosity=0, label=''):
    set_seed(init_seed)
    X_train = X_train.to(device=DEVICE, dtype=DTYPE).contiguous()
    Y_train = Y_train.to(device=DEVICE, dtype=DTYPE).contiguous()
    likelihood = _build_likelihood()
    model = LCMGPModel(X_train, Y_train, likelihood, cfg).to(device=DEVICE, dtype=DTYPE)
    model.train()
    likelihood.train()
    seen, opt_params = (set(), [])
    for p in list(model.parameters()) + list(likelihood.parameters()):
        if id(p) not in seen:
            seen.add(id(p))
            opt_params.append(p)
    optimizer = torch.optim.Adam(opt_params, lr=float(lr))
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    best_loss, best_msd, best_lsd, bad = (float('inf'), None, None, 0)
    mdev = next(model.parameters()).device
    ldev = next(likelihood.parameters()).device
    jitter_seq = (0.001, 0.003, 0.01, 0.03, 0.1) if DTYPE == torch.float32 else (0.0001, 0.0003, 0.001, 0.003, 0.01)
    last_error = None
    for it in range(1, int(iters) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = None
        for jit in jitter_seq:
            try:
                with _train_settings(fast_mode), _max_chol(fast_mode), gpytorch.settings.max_cg_iterations(int(MAX_CG_ITERATIONS)), gpytorch.settings.cholesky_jitter(float(jit)), gpytorch.settings.cg_tolerance(float(CG_TOLERANCE)):
                    loss = -mll(model(X_train), Y_train)
                break
            except Exception as e:
                last_error = e
                loss = None
        if loss is None:
            raise RuntimeError(f'Training failed before a valid loss was computed. Last error: {repr(last_error)}')
        if not torch.isfinite(loss):
            raise RuntimeError(f'Training produced non-finite loss at iteration {it}: {float(loss.detach())}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(opt_params, max_norm=2.0)
        optimizer.step()
        cur = float(loss.detach())
        if cur + 1e-06 < best_loss:
            best_loss, bad = (cur, 0)
            with torch.no_grad():
                best_msd = _clone_sd(model.state_dict(), mdev)
                best_lsd = _clone_sd(likelihood.state_dict(), ldev)
        else:
            bad += 1
        if bad >= int(patience):
            break
    if best_msd is not None:
        model.load_state_dict(best_msd)
    if best_lsd is not None:
        likelihood.load_state_dict(best_lsd)
    model.eval()
    likelihood.eval()
    _p('TRAIN', f'{label} done | best_loss={best_loss:.4f}', verbosity)
    return (model, likelihood, best_loss)

def train_lcmgp(X_train, Y_train, cfg, iters, lr, n_restarts, restart_base_seed, fast_mode=True, verbosity=0, label=''):
    X_train = X_train.to(device=DEVICE, dtype=DTYPE).contiguous()
    Y_train = Y_train.to(device=DEVICE, dtype=DTYPE).contiguous()
    best_loss, best_model, best_lik = (float('inf'), None, None)
    errors = []
    for r in range(int(max(1, n_restarts))):
        _p('RESTART', f'{label} restart {r + 1}/{int(max(1, n_restarts))}', verbosity)
        try:
            model, lik, loss = train_single(X_train, Y_train, cfg, int(iters), float(lr), max(80, int(iters) // 5), int(restart_base_seed + 97 * r), bool(fast_mode), verbosity=0, label=label)
            if float(loss) < best_loss:
                if best_model is not None:
                    try:
                        del best_model, best_lik
                    except Exception:
                        pass
                best_loss, best_model, best_lik = (float(loss), model, lik)
            else:
                try:
                    del model, lik
                except Exception:
                    pass
                _gpu_cleanup()
        except Exception as e:
            errors.append(repr(e))
            _p('ERROR', f'{label} restart {r + 1} failed: {repr(e)}', verbosity)
            _gpu_cleanup()
    if best_model is None:
        raise RuntimeError('All GP training restarts failed. ' + ' | '.join(errors[-3:]))
    return (best_model, best_lik)

@torch.no_grad()
def predict(model, lik, X, fast_mode=True):
    model.eval()
    lik.eval()
    X = X.to(device=next(model.parameters()).device, dtype=DTYPE).contiguous()
    jitters = (0.001, 0.01, 0.05, 0.1) if DTYPE == torch.float32 else (0.0001, 0.001, 0.01, 0.05)
    last_error = None
    for jit in jitters:
        try:
            with gpytorch.settings.detach_test_caches(True), _train_settings(fast_mode), _max_chol(fast_mode), gpytorch.settings.max_cg_iterations(int(MAX_CG_ITERATIONS)), gpytorch.settings.cholesky_jitter(float(jit)), gpytorch.settings.eval_cg_tolerance(float(CG_TOLERANCE)):
                dist = lik(model(X))
                mu = dist.mean
                sig = torch.sqrt(torch.clamp(dist.variance, min=1e-12))
                return (mu.detach().double().cpu().numpy(), sig.detach().double().cpu().numpy())
        except Exception as e:
            last_error = e
    raise RuntimeError(f'Prediction failed. Last error: {repr(last_error)}')

def summarize_metrics(rows, cfg, test_frac):
    out = {'kernel': cfg['kernel'], 'rank': int(cfg['rank']), 'use_additive': bool(cfg['use_additive']), 'add_kernel': cfg['add_kernel'], 'n_splits': len(rows), 'test_frac': float(test_frac)}
    for k in rows[0].keys():
        try:
            vals = np.asarray([r[k] for r in rows], float)
        except (TypeError, ValueError):
            continue
        out[k + '_mean'] = float(np.nanmean(vals))
        out[k + '_std'] = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
    return out

def concat_pooled(preds):
    pooled = {'strength': {}, 'perm': {}}
    for task in ('strength', 'perm'):
        keys = sorted({k for p in preds for k in p[task].keys()})
        for k in keys:
            pooled[task][k] = np.concatenate([np.asarray(p[task][k]) for p in preds])
    if all(('meta' in p for p in preds)):
        pooled['meta'] = {}
        meta_keys = sorted({k for p in preds for k in p['meta'].keys()})
        for k in meta_keys:
            vals = [np.asarray(p['meta'][k]) for p in preds]
            if vals[0].ndim == 2:
                pooled['meta'][k] = np.vstack(vals)
            else:
                pooled['meta'][k] = np.concatenate(vals)
    return pooled

def standardize_pooled_oos_features_with_final_scaler(pooled, final_xsc):
    if 'meta' not in pooled or 'X_oos_model' not in pooled['meta']:
        raise ValueError('Pooled predictions must contain transformed, unstandardized X_oos_model features.')
    X_model = np.asarray(pooled['meta']['X_oos_model'], dtype=float)
    if X_model.ndim != 2 or X_model.shape[1] != len(FEATURES):
        raise ValueError(
            f'X_oos_model must have shape n_oos x {len(FEATURES)}; got {X_model.shape}.'
        )
    # Invariant: every local-residual KNN distance uses final full-data standardized
    # transformed features. Repeated-split scalers are never mixed into deployment KNN.
    pooled['meta']['X_oos_std'] = final_xsc.transform(X_model)
    return pooled

def check_oos_feature_standardization_invariant(pooled, final_xsc, tol=1e-8):
    if 'meta' not in pooled or 'X_oos_model' not in pooled['meta'] or 'X_oos_std' not in pooled['meta']:
        raise RuntimeError('Pooled OOS metadata must contain X_oos_model and X_oos_std before saving oracle.')
    expected = final_xsc.transform(np.asarray(pooled['meta']['X_oos_model'], dtype=float))
    actual = np.asarray(pooled['meta']['X_oos_std'], dtype=float)
    if expected.shape != actual.shape:
        raise RuntimeError(f'OOS feature standardization invariant shape mismatch: expected {expected.shape}, got {actual.shape}.')
    max_abs = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
    if not np.isfinite(max_abs) or max_abs > float(tol):
        raise RuntimeError(
            f'OOS residual feature standardization invariant failed: max_abs_error={max_abs:.6g}, tolerance={float(tol):.1e}.'
        )
    return max_abs

def grouped_oos_diagnostics_by_original_row(pooled):
    if 'meta' not in pooled or 'original_row_index' not in pooled['meta']:
        return {
            'grouped_oos_unique_original_rows': 0,
            'grouped_oos_avg_appearances_per_original_row': float('nan'),
            'grouped_oos_diagnostic_note': 'No original_row_index metadata was available.',
        }
    df = pd.DataFrame({
        'original_row_index': np.asarray(pooled['meta']['original_row_index'], dtype=int),
        'strength_true': np.asarray(pooled['strength']['y_true'], dtype=float),
        'strength_pred': np.asarray(pooled['strength']['y_pred'], dtype=float),
        'logperm_true': np.asarray(pooled['perm']['y_true'], dtype=float),
        'logperm_pred': np.asarray(pooled['perm']['y_pred'], dtype=float),
    })
    grouped = df.groupby('original_row_index', sort=True).agg(
        appearances=('original_row_index', 'size'),
        strength_true=('strength_true', 'mean'),
        strength_pred=('strength_pred', 'mean'),
        logperm_true=('logperm_true', 'mean'),
        logperm_pred=('logperm_pred', 'mean'),
    ).reset_index()
    return {
        'grouped_oos_unique_original_rows': int(len(grouped)),
        'grouped_oos_avg_appearances_per_original_row': float(grouped['appearances'].mean()) if len(grouped) else float('nan'),
        'grouped_oos_rmse_strength': rmse(grouped['strength_true'], grouped['strength_pred']) if len(grouped) else float('nan'),
        'grouped_oos_mae_strength': mae(grouped['strength_true'], grouped['strength_pred']) if len(grouped) else float('nan'),
        'grouped_oos_r2_strength': r2_score(grouped['strength_true'], grouped['strength_pred']) if len(grouped) else float('nan'),
        'grouped_oos_rmse_logperm': rmse(grouped['logperm_true'], grouped['logperm_pred']) if len(grouped) else float('nan'),
        'grouped_oos_mae_logperm': mae(grouped['logperm_true'], grouped['logperm_pred']) if len(grouped) else float('nan'),
        'grouped_oos_r2_logperm': r2_score(grouped['logperm_true'], grouped['logperm_pred']) if len(grouped) else float('nan'),
        'grouped_oos_diagnostic_note': 'Predictions were averaged by original_row_index before computing these metrics.',
    }

def array_to_list_obj(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: array_to_list_obj(v) for k, v in x.items()}
    if isinstance(x, list):
        return [array_to_list_obj(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def _calibration_curve(y_true, mu, s, qs=None):
    if qs is None:
        qs = [0.5, 0.8, 0.9, 0.95]
    y_true, mu, s = (np.asarray(y_true).ravel(), np.asarray(mu).ravel(), np.maximum(np.asarray(s).ravel(), 1e-12))
    m = _finite_mask(y_true, mu, s)
    if m.sum() < 1:
        return (np.asarray([]), np.asarray([]))
    y, u, sv = (y_true[m], mu[m], s[m])
    nom, obs = ([], [])
    for q in qs:
        z = float(stats.norm.ppf((1 + float(q)) / 2))
        nom.append(float(q))
        obs.append(float(np.mean((y >= u - z * sv) & (y <= u + z * sv))))
    return (np.asarray(nom), np.asarray(obs))

def _axis_limits(a, b):
    v = np.asarray(np.concatenate([np.asarray(a).ravel(), np.asarray(b).ravel()]), float)
    v = v[np.isfinite(v)]
    if not v.size:
        return (0.0, 1.0)
    lo, hi = (float(np.min(v)), float(np.max(v)))
    if hi <= lo:
        return (lo - 1.0, hi + 1.0)
    pad = 0.05 * (hi - lo)
    return (lo - pad, hi + pad)

def _read_table(path):
    path = str(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.xlsx', '.xls'):
        return pd.read_excel(path)
    if ext in ('.csv', '.txt'):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input file extension '{ext}'. Use .xlsx, .xls, or .csv.")

def _load_saved_oracle_object(pt_path, device='cpu'):
    try:
        return torch.load(pt_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(pt_path, map_location=device)

def _load_saved_oracle(pt_path, device=None):
    if device is None:
        device = DEVICE
    save_obj = _load_saved_oracle_object(pt_path, device=device)
    set_runtime_precision(save_obj.get('dtype', 'float32'))
    cfg = save_obj['best_config']
    train_x = save_obj['train_x'].to(device=device, dtype=DTYPE).contiguous()
    train_y = save_obj['train_y'].to(device=device, dtype=DTYPE).contiguous()
    likelihood = _build_likelihood()
    model = LCMGPModel(train_x, train_y, likelihood, cfg).to(device=device, dtype=DTYPE)
    model.load_state_dict(save_obj['model_state_dict'])
    likelihood.load_state_dict(save_obj['likelihood_state_dict'])
    model.eval()
    likelihood.eval()
    return (save_obj, model, likelihood)

def predict_saved_oracle_file(pt_path, input_path, output_path=None, target_coverage=None):
    df_new = _read_table(input_path)
    pred = predict_saved_oracle_dataframe(pt_path, df_new, target_coverage=target_coverage, include_input=True)
    if output_path is None:
        base = os.path.splitext(os.path.basename(str(input_path)))[0]
        output_path = os.path.join(os.path.dirname(str(input_path)) or '.', base + '_lcmgp_predictions.csv')
    pred.to_csv(output_path, index=False)
    return (output_path, pred)

def _level_key(level):
    return f'{float(level):.3f}'

def _safe_feature_name(feat):
    return str(feat).replace(' ', '_').replace('/', '_per_').replace('(', '').replace(')', '').replace('^', '')

def _finite_values(*arrs):
    mask = _finite_mask(*arrs)
    return mask

def _quantile_clipped(values, q, default=1.0, lo=0.0, hi=10.0):
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size < 5:
        return float(default)
    val = float(np.quantile(arr, float(q)))
    if not np.isfinite(val):
        return float(default)
    return float(np.clip(val, float(lo), float(hi)))

def apply_sigma_floors(std_orig):
    s = np.asarray(std_orig, dtype=float).copy()
    s[:, 0] = np.maximum(s[:, 0], SIGMA_FLOOR_S)
    s[:, 1] = np.maximum(s[:, 1], SIGMA_FLOOR_P)
    return s

def _calibration_scores(y_true, mu, std):
    y_true = np.asarray(y_true, dtype=float).ravel()
    mu = np.asarray(mu, dtype=float).ravel()
    std = np.maximum(np.asarray(std, dtype=float).ravel(), 1e-12)
    mask = _finite_mask(y_true, mu, std)
    y_true, mu, std = (y_true[mask], mu[mask], std[mask])
    two = np.abs(y_true - mu) / std
    lower = (mu - y_true) / std
    upper = (y_true - mu) / std
    return (two[np.isfinite(two)], lower[np.isfinite(lower)], upper[np.isfinite(upper)])

def _fit_calibration_table_one(y_true, mu, std, levels=CALIBRATION_LEVELS):
    two, lower, upper = _calibration_scores(y_true, mu, std)
    out = {'n': int(len(two)), 'two_sided': {}, 'lower': {}, 'upper': {}}
    for level in levels:
        key = _level_key(level)
        out['two_sided'][key] = _quantile_clipped(two, level, default=1.0, lo=0.0, hi=10.0)
        out['lower'][key] = _quantile_clipped(lower, level, default=1.0, lo=0.0, hi=10.0)
        out['upper'][key] = _quantile_clipped(upper, level, default=1.0, lo=0.0, hi=10.0)
    return out

def _fit_calibration_tables(y_true, mu, std, levels=CALIBRATION_LEVELS):
    return _fit_calibration_table_one(y_true, mu, std, levels=levels)

def _assign_bins(values, edges):
    values = np.asarray(values, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if edges.size < 2:
        return np.array(['all'] * len(values), dtype=object)
    idx = np.searchsorted(edges[1:-1], values, side='right')
    idx = np.where(np.isfinite(values), idx, -1)
    return np.asarray(['missing' if i < 0 else f'bin_{int(i)}' for i in idx], dtype=object)

def _make_quantile_edges(values, n_bins=4):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < max(10, n_bins + 1):
        return np.asarray([], dtype=float)
    qs = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.unique(np.quantile(arr, qs))
    if edges.size < 2:
        return np.asarray([], dtype=float)
    edges[0] -= 1e-12
    edges[-1] += 1e-12
    return edges.astype(float)

def _local_residual_core(
    Xs,
    X_ref,
    res_s,
    res_p,
    q=LOCAL_RESIDUAL_K_NEIGHBORS,
    exclude_self=False,
    candidate_original_row_index=None,
    reference_original_row_index=None,
    exclude_same_original_row=False,
):
    Xs = _to_numpy_float_array(Xs)
    X_ref = _to_numpy_float_array(X_ref)
    res_s = np.asarray(res_s, dtype=float).ravel()
    res_p = np.asarray(res_p, dtype=float).ravel()
    if Xs.ndim != 2:
        raise ValueError(f'Candidate feature matrix must be 2D; got shape {Xs.shape}.')
    if X_ref.ndim != 2:
        raise ValueError(f'OOS residual reference feature matrix must be 2D; got shape {X_ref.shape}.')
    if Xs.shape[1] != X_ref.shape[1]:
        raise ValueError(
            f'Candidate feature dimension {Xs.shape[1]} does not match OOS reference dimension {X_ref.shape[1]}.'
        )
    if X_ref.shape[0] != res_s.shape[0] or X_ref.shape[0] != res_p.shape[0]:
        raise ValueError(
            f'OOS reference rows/features/residuals length mismatch: X_ref={X_ref.shape[0]}, '
            f'strength_residuals={res_s.shape[0]}, logperm_residuals={res_p.shape[0]}.'
        )
    candidate_orig = None
    ref_orig = None
    if candidate_original_row_index is not None:
        candidate_orig = np.asarray(candidate_original_row_index, dtype=int).ravel()
        if candidate_orig.shape[0] != Xs.shape[0]:
            raise ValueError(
                f'candidate_original_row_index length {candidate_orig.shape[0]} does not match candidate rows {Xs.shape[0]}.'
            )
    if reference_original_row_index is not None:
        ref_orig = np.asarray(reference_original_row_index, dtype=int).ravel()
        if ref_orig.shape[0] != X_ref.shape[0]:
            raise ValueError(
                f'reference_original_row_index length {ref_orig.shape[0]} does not match reference rows {X_ref.shape[0]}.'
            )
    if exclude_same_original_row and (candidate_orig is None or ref_orig is None):
        raise ValueError('exclude_same_original_row=True requires candidate and reference original-row indices.')
    if not np.isfinite(Xs).all():
        raise ValueError('Candidate feature matrix contains non-finite values.')
    q_req = int(q)
    if q_req < 2:
        raise ValueError('Local residual uncertainty requires q >= 2 to compute a sample standard deviation.')
    n_ref_before_filter = int(X_ref.shape[0])
    finite = np.isfinite(X_ref).all(axis=1) & np.isfinite(res_s) & np.isfinite(res_p)
    ref_original_idx = np.arange(n_ref_before_filter, dtype=int)[finite]
    if ref_orig is not None:
        ref_orig = ref_orig[finite]
    X_ref = X_ref[finite]
    res_s = res_s[finite]
    res_p = res_p[finite]
    if len(X_ref) < 2:
        raise ValueError(
            f'At least two finite OOS residual references are required; found {len(X_ref)} after filtering.'
        )
    d = np.sqrt(np.sum((Xs[:, None, :] - X_ref[None, :, :]) ** 2, axis=2))
    if exclude_self:
        if Xs.shape[0] != n_ref_before_filter:
            raise ValueError(
                'exclude_self=True requires candidate rows to align one-to-one with the unfiltered OOS reference rows.'
            )
        for row_idx in range(Xs.shape[0]):
            self_cols = np.where(ref_original_idx == row_idx)[0]
            if self_cols.size:
                d[row_idx, self_cols] = np.inf
    if exclude_same_original_row:
        for row_idx, original_idx in enumerate(candidate_orig):
            same_cols = np.where(ref_orig == int(original_idx))[0]
            if same_cols.size:
                d[row_idx, same_cols] = np.inf
    available = np.sum(np.isfinite(d), axis=1)
    min_available = int(np.min(available)) if available.size else 0
    if min_available < 2:
        raise ValueError(
            f'At least two valid non-self OOS neighbors are required for local residual uncertainty; '
            f'minimum available is {min_available}.'
        )
    q_eff = int(min(q_req, min_available))
    nn_unsorted = np.argpartition(d, q_eff - 1, axis=1)[:, :q_eff]
    rows = np.arange(d.shape[0])[:, None]
    order = np.argsort(d[rows, nn_unsorted], axis=1)
    nn_idx = nn_unsorted[rows, order]
    nn_dist = d[rows, nn_idx]
    s_std = np.asarray([np.std(res_s[idx], ddof=1) for idx in nn_idx], dtype=float)
    p_std = np.asarray([np.std(res_p[idx], ddof=1) for idx in nn_idx], dtype=float)
    return {'q_neighbors': q_eff, 'D_res_knn_mean': np.mean(nn_dist, axis=1), 'D_res_knn_max': np.max(nn_dist, axis=1), 'nearest_oos_indices_internal': nn_idx, 'strength_residual_std_local_mpa': s_std, 'logperm_residual_std_local': p_std}

def _clean_interval_rule(rule):
    r = str(rule).strip().lower().replace('-', '_')
    aliases = {
        'gp': 'gp_only',
        'raw': 'gp_only',
        'model': 'gp_only',
        'model_only': 'gp_only',
        'local': 'local_only',
        'residual': 'local_only',
        'residual_only': 'local_only',
        'average': 'mean',
        'blend': 'mean',
        'quadrature': 'sqrt',
        'rss': 'sqrt',
        'conservative': 'max',
    }
    return aliases.get(r, r)

def _combine_std(gp_std, local_std=None, rule=UNCERTAINTY_COMBINE_RULE):
    """Combine the model uncertainty with the local error-history uncertainty.

    Plain English:
    - gp_only: use only the model's own uncertainty. Usually narrowest, but may miss local failures.
    - local_only: use only nearby past errors. Useful as a diagnostic, not always stable.
    - mean: average the two. Often narrower than max.
    - max: use whichever is larger. Safest/conservative default.
    - sqrt: combine both as independent sources. Usually widest.
    """
    gp_std = np.maximum(np.asarray(gp_std, dtype=float), 1e-12)
    if local_std is None:
        return gp_std
    local_std = np.asarray(local_std, dtype=float)
    # During prediction, local_std can be missing if an older saved model is used.
    # Missing local values fall back to the model uncertainty instead of producing NaNs.
    local_std = np.where(np.isfinite(local_std), np.maximum(local_std, 1e-12), gp_std)
    r = _clean_interval_rule(rule)
    if r == 'gp_only':
        return gp_std
    if r == 'local_only':
        return local_std
    if r == 'sqrt':
        return np.sqrt(gp_std ** 2 + local_std ** 2)
    if r == 'mean':
        return 0.5 * (gp_std + local_std)
    if r == 'max':
        return np.maximum(gp_std, local_std)
    raise ValueError(
        f"Unknown uncertainty combine rule '{rule}'. Use one of: gp_only, local_only, mean, max, sqrt."
    )


def _to_numpy_float_array(x):
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)

def _compute_ad_reference(train_x_std, k=AD_K_NEIGHBORS, threshold_quantile=AD_THRESHOLD_QUANTILE):
    X = _to_numpy_float_array(train_x_std)
    if X.ndim != 2 or len(X) < 3:
        return {'ad_k_neighbors': 1, 'ad_threshold_quantile': float(threshold_quantile), 'tau_AD': float('inf'), 'D_AD_train': []}
    k_eff = int(max(1, min(int(k), len(X) - 1)))
    d = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))
    np.fill_diagonal(d, np.inf)
    nn = np.partition(d, k_eff - 1, axis=1)[:, :k_eff]
    D = np.mean(nn, axis=1)
    tau = float(np.quantile(D[np.isfinite(D)], float(threshold_quantile))) if np.isfinite(D).any() else float('inf')
    return {'ad_k_neighbors': int(k_eff), 'ad_threshold_quantile': float(threshold_quantile), 'tau_AD': tau, 'D_AD_train': D.astype(float).tolist()}

def _ad_for_standardized_features(save_obj, Xs):
    Xs = _to_numpy_float_array(Xs)
    if Xs.ndim == 1:
        Xs = Xs.reshape(1, -1)
    X_train = _to_numpy_float_array(save_obj['train_x'])
    if Xs.ndim != 2:
        raise ValueError(f'AD candidate feature matrix must be 2D; got shape {Xs.shape}.')
    if X_train.ndim != 2:
        raise ValueError(f'AD reference train_x must be 2D; got shape {X_train.shape}.')
    if Xs.shape[1] != X_train.shape[1]:
        raise ValueError(
            f'AD candidate feature dimension {Xs.shape[1]} does not match train_x dimension {X_train.shape[1]}.'
        )
    if not np.isfinite(Xs).all():
        raise ValueError('AD candidate feature matrix contains non-finite values.')
    finite_train = np.isfinite(X_train).all(axis=1)
    X_train = X_train[finite_train]
    ad_ref = save_obj.get('ad_reference', {})
    k_eff = int(ad_ref.get('ad_k_neighbors', min(AD_K_NEIGHBORS, max(1, len(X_train)))))
    tau = float(ad_ref.get('tau_AD', np.inf))
    if len(X_train) < 1:
        D = np.full(Xs.shape[0], np.inf, dtype=float)
    else:
        k_eff = int(max(1, min(k_eff, len(X_train))))
        d = np.sqrt(np.sum((Xs[:, None, :] - X_train[None, :, :]) ** 2, axis=2))
        nn = np.partition(d, k_eff - 1, axis=1)[:, :k_eff]
        D = np.mean(nn, axis=1)
    ratio = D / max(tau, 1e-12) if np.isfinite(tau) else np.zeros_like(D)
    return {'D_AD': D, 'tau_AD': np.full_like(D, tau, dtype=float), 'AD_ratio': ratio, 'inside_AD': ratio <= 1.0, 'AD_k_neighbors': int(k_eff)}

def _ad_inflation_from_ratio(ratio, start=AD_INFLATION_START, eta=AD_INFLATION_ETA, cap=AD_INFLATION_CAP):
    ratio = np.asarray(ratio, dtype=float)
    out = np.ones_like(ratio, dtype=float)
    denom = max(1e-12, 1.0 - float(start))
    mask = np.isfinite(ratio) & (ratio > float(start))
    out[mask] = 1.0 + float(eta) * ((ratio[mask] - float(start)) / denom)
    out = np.clip(out, 1.0, float(cap))
    return out

def _prepare_split(X_all, yS_all, yP_all, train_idx, val_idx):
    key = (tuple(train_idx.tolist()), tuple(val_idx.tolist()))
    if key in SPLIT_CACHE:
        return SPLIT_CACHE[key]
    Xtr, Xte = (X_all[train_idx], X_all[val_idx])
    xsc = XStandardizer().fit(Xtr)
    Xtr_s = xsc.transform(Xtr)
    Xte_s = xsc.transform(Xte)
    Ytr = np.stack([yS_all[train_idx], yP_all[train_idx]], 1)
    Yte = np.stack([yS_all[val_idx], yP_all[val_idx]], 1)
    ysc = YStandardizer().fit(Ytr)
    Ytr_s = ysc.transform(Ytr)
    rec = {'Xtr_s': Xtr_s, 'Ytr_s': Ytr_s, 'Xte_s': Xte_s, 'Xte_model': Xte.copy(), 'Yte': Yte, 'y_mean': ysc.mean.copy(), 'y_std': ysc.std.copy()}
    SPLIT_CACHE[key] = rec
    return rec

def eval_split(X_all, yS_all, yP_all, train_idx, val_idx, cfg, iters, lr, n_restarts, restart_seed, fast_mode, verbosity=0, label=''):
    rec = _prepare_split(X_all, yS_all, yP_all, train_idx, val_idx)
    Xt = torch.as_tensor(rec['Xtr_s'], dtype=DTYPE, device=DEVICE).contiguous()
    Yt = torch.as_tensor(rec['Ytr_s'], dtype=DTYPE, device=DEVICE).contiguous()
    Xte = torch.as_tensor(rec['Xte_s'], dtype=DTYPE, device=DEVICE).contiguous()
    model = lik = None
    try:
        model, lik = train_lcmgp(Xt, Yt, cfg, iters, lr, n_restarts, restart_seed, fast_mode=fast_mode, verbosity=verbosity, label=label)
        mu_s, sig_s = predict(model, lik, Xte, fast_mode=fast_mode)
    finally:
        try:
            del Xt, Yt, Xte
        except Exception:
            pass
        try:
            del model, lik
        except Exception:
            pass
        _gpu_cleanup()
    mean = mu_s * rec['y_std'] + rec['y_mean']
    std_raw = apply_sigma_floors(sig_s * rec['y_std'])
    Yte = rec['Yte']
    yt_s, yt_p = (Yte[:, 0], Yte[:, 1])
    yp_s, yp_p = (mean[:, 0], mean[:, 1])
    ss, sp = (np.maximum(std_raw[:, 0], 1e-12), np.maximum(std_raw[:, 1], 1e-12))
    z90, z95 = (1.6449, 1.96)
    metrics = {'rmse_strength': rmse(yt_s, yp_s), 'mae_strength': mae(yt_s, yp_s), 'r2_strength': r2_score(yt_s, yp_s), 'cov90_strength_raw_gaussian': coverage(yt_s, yp_s, ss, z90), 'cov95_strength_raw_gaussian': coverage(yt_s, yp_s, ss, z95), 'nlpd_strength_raw': nlpd_gaussian(yt_s, yp_s, ss), 'rmse_perm': rmse(yt_p, yp_p), 'mae_perm': mae(yt_p, yp_p), 'r2_perm': r2_score(yt_p, yp_p), 'cov90_perm_raw_gaussian': coverage(yt_p, yp_p, sp, z90), 'cov95_perm_raw_gaussian': coverage(yt_p, yp_p, sp, z95), 'nlpd_perm_raw': nlpd_gaussian(yt_p, yp_p, sp)}
    metrics['rmse_logperm'] = metrics['rmse_perm']
    metrics['mae_logperm'] = metrics['mae_perm']
    metrics['r2_logperm'] = metrics['r2_perm']
    metrics['rmse_perm_mm_s'] = rmse(np.exp(yt_p), np.exp(yp_p))
    metrics['mae_perm_mm_s'] = mae(np.exp(yt_p), np.exp(yp_p))
    metrics['r2_perm_mm_s'] = r2_score(np.exp(yt_p), np.exp(yp_p))
    metrics['nlpd_joint_raw'] = 0.5 * (metrics['nlpd_strength_raw'] + metrics['nlpd_perm_raw'])
    pred = {'meta': {'original_row_index': np.asarray(val_idx, dtype=int).copy(), 'X_oos_model': np.asarray(rec['Xte_model'], dtype=float).copy()}, 'strength': {'y_true': yt_s, 'y_pred': yp_s, 'y_std_raw': ss, 'y_std': ss, 'residual': yt_s - yp_s}, 'perm': {'y_true': yt_p, 'y_pred': yp_p, 'y_std_raw': sp, 'y_std': sp, 'residual': yt_p - yp_p}}
    return (metrics, pred)

def build_calibration_bundle(pooled, levels=CALIBRATION_LEVELS, q_neighbors=LOCAL_RESIDUAL_K_NEIGHBORS, regime_min_n=REGIME_CALIBRATION_MIN_N, combine_rule=UNCERTAINTY_COMBINE_RULE):
    # Lower-bound levels are empirical one-sided OOS residual calibration levels.
    # Keep them in the fitted tables even if callers customize interval levels.
    levels = tuple(sorted(set([float(x) for x in levels] + [float(x) for x in LOWER_BOUND_LEVELS])))
    if 'meta' not in pooled or 'X_oos_std' not in pooled['meta']:
        raise ValueError('Calibration requires OOS features standardized by the final full-data scaler.')
    Xstd = np.asarray(pooled['meta'].get('X_oos_std'), dtype=float)
    Xmodel = np.asarray(pooled['meta'].get('X_oos_model', Xstd), dtype=float)
    res_s = np.asarray(pooled['strength'].get('residual', np.asarray(pooled['strength']['y_true']) - np.asarray(pooled['strength']['y_pred'])), dtype=float)
    res_p = np.asarray(pooled['perm'].get('residual', np.asarray(pooled['perm']['y_true']) - np.asarray(pooled['perm']['y_pred'])), dtype=float)
    orig_idx = pooled.get('meta', {}).get('original_row_index', None)
    local_status = 'ok'
    local_warning = ''
    try:
        loc = _local_residual_core(
            Xstd,
            Xstd,
            res_s,
            res_p,
            q=q_neighbors,
            exclude_self=True,
            candidate_original_row_index=orig_idx,
            reference_original_row_index=orig_idx,
            exclude_same_original_row=orig_idx is not None,
        )
    except ValueError as e:
        # Small repeated-split pools may not have two non-self, non-same-row
        # neighbors for every row. Calibration then falls back to raw GP std.
        local_status = 'disabled_fallback_to_gp_std'
        local_warning = str(e)
        n_local = int(Xstd.shape[0])
        loc = {
            'q_neighbors': 0,
            'D_res_knn_mean': np.full(n_local, np.nan, dtype=float),
            'D_res_knn_max': np.full(n_local, np.nan, dtype=float),
            'nearest_oos_indices_internal': np.empty((n_local, 0), dtype=int),
            'strength_residual_std_local_mpa': np.full(n_local, np.nan, dtype=float),
            'logperm_residual_std_local': np.full(n_local, np.nan, dtype=float),
        }
    std_s_raw = np.maximum(np.asarray(pooled['strength'].get('y_std_raw', pooled['strength'].get('y_std')), dtype=float), 1e-12)
    std_p_raw = np.maximum(np.asarray(pooled['perm'].get('y_std_raw', pooled['perm'].get('y_std')), dtype=float), 1e-12)
    std_s_eff = _combine_std(std_s_raw, loc['strength_residual_std_local_mpa'], combine_rule)
    std_p_eff = _combine_std(std_p_raw, loc['logperm_residual_std_local'], combine_rule)
    ys = np.asarray(pooled['strength']['y_true'], dtype=float)
    yps = np.asarray(pooled['strength']['y_pred'], dtype=float)
    yp = np.asarray(pooled['perm']['y_true'], dtype=float)
    ypp = np.asarray(pooled['perm']['y_pred'], dtype=float)
    bundle = {'version': 'empirical_local_oos_ad_v1', 'calibration_fit_scope': 'pooled_repeated_split_oos_predictions', 'independent_validation_estimate': False, 'levels': [float(x) for x in levels], 'level_keys': [_level_key(x) for x in levels], 'lower_bound_levels': [float(x) for x in LOWER_BOUND_LEVELS], 'regime_min_n': int(regime_min_n), 'combine_rule': str(combine_rule), 'local_residual_q_neighbors': int(loc['q_neighbors']), 'global': {'strength': _fit_calibration_tables(ys, yps, std_s_eff, levels), 'perm': _fit_calibration_tables(yp, ypp, std_p_eff, levels)}, 'regime_bins': {}, 'regime': {'strength': {}, 'perm': {}}, 'diagnostics': {}}
    por_edges = _make_quantile_edges(Xmodel[:, FEATURES.index('porosity')], 4)
    wc_edges = _make_quantile_edges(Xmodel[:, FEATURES.index('w/c')], 4)
    lp_edges = _make_quantile_edges(ypp, 4)
    bundle['regime_bins'] = {'porosity_bin': por_edges.tolist(), 'wc_bin': wc_edges.tolist(), 'logperm_pred_bin': lp_edges.tolist()}
    regime_keys = {'porosity_bin': _assign_bins(Xmodel[:, FEATURES.index('porosity')], por_edges), 'wc_bin': _assign_bins(Xmodel[:, FEATURES.index('w/c')], wc_edges), 'logperm_pred_bin': _assign_bins(ypp, lp_edges)}
    task_arrays = {'strength': (ys, yps, std_s_eff), 'perm': (yp, ypp, std_p_eff)}
    for task, arrays in task_arrays.items():
        ytrue, mu, std = arrays
        for group, keys in regime_keys.items():
            bundle['regime'][task].setdefault(group, {})
            for key in sorted(set(map(str, keys))):
                mask = np.asarray(keys == key, dtype=bool)
                if int(mask.sum()) < int(regime_min_n):
                    continue
                bundle['regime'][task][group][key] = _fit_calibration_tables(ytrue[mask], mu[mask], std[mask], levels)
    bundle['diagnostics'] = {'local_residual_status': local_status, 'local_residual_warning': local_warning, 'exclude_same_original_row_for_local_residual': bool(orig_idx is not None), 'strength_std_raw_mean': _nanmean_or_nan(std_s_raw), 'strength_std_local_mean': _nanmean_or_nan(loc['strength_residual_std_local_mpa']), 'strength_std_effective_mean': _nanmean_or_nan(std_s_eff), 'logperm_std_raw_mean': _nanmean_or_nan(std_p_raw), 'logperm_std_local_mean': _nanmean_or_nan(loc['logperm_residual_std_local']), 'logperm_std_effective_mean': _nanmean_or_nan(std_p_eff)}
    return (bundle, {'local': loc, 'strength_std_effective': std_s_eff, 'logperm_std_effective': std_p_eff, 'strength_std_raw': std_s_raw, 'logperm_std_raw': std_p_raw})

def _resolve_level(bundle, level):
    levels = [float(x) for x in bundle.get('levels', CALIBRATION_LEVELS)]
    if not levels:
        return (float(level), _level_key(level))
    arr = np.asarray(levels, dtype=float)
    idx = int(np.argmin(np.abs(arr - float(level))))
    return (float(arr[idx]), _level_key(arr[idx]))

def _regime_key_for_rows(group, Xmodel, mean):
    Xmodel = np.asarray(Xmodel, dtype=float)
    mean = np.asarray(mean, dtype=float)
    return None

def _regime_keys_from_bundle(bundle, Xmodel, mean):
    Xmodel = np.asarray(Xmodel, dtype=float)
    mean = np.asarray(mean, dtype=float)
    edges = bundle.get('regime_bins', {})
    return {'porosity_bin': _assign_bins(Xmodel[:, FEATURES.index('porosity')], edges.get('porosity_bin', [])), 'wc_bin': _assign_bins(Xmodel[:, FEATURES.index('w/c')], edges.get('wc_bin', [])), 'logperm_pred_bin': _assign_bins(mean[:, 1], edges.get('logperm_pred_bin', []))}

def _select_calibration_q(bundle, task, kind, level, Xmodel, mean):
    level_resolved, key = _resolve_level(bundle, level)
    n = len(Xmodel)
    q = np.full(n, float(bundle.get('global', {}).get(task, {}).get(kind, {}).get(key, 1.0)), dtype=float)
    source = np.asarray(['global'] * n, dtype=object)
    keys_by_group = _regime_keys_from_bundle(bundle, Xmodel, mean)
    if task == 'strength':
        priority = ['wc_bin', 'porosity_bin', 'logperm_pred_bin']
    else:
        priority = ['logperm_pred_bin', 'porosity_bin', 'wc_bin']
    for group in priority:
        table = bundle.get('regime', {}).get(task, {}).get(group, {})
        if not table:
            continue
        row_keys = keys_by_group[group]
        for i in range(n):
            if source[i] != 'global':
                continue
            rk = str(row_keys[i])
            if rk in table and kind in table[rk] and (key in table[rk][kind]):
                q[i] = float(table[rk][kind][key])
                source[i] = f'{group}:{rk}'
    return (q, source, level_resolved)

def _coverage_from_q(y_true, mu, std, q):
    y_true = np.asarray(y_true, dtype=float)
    mu = np.asarray(mu, dtype=float)
    std = np.asarray(std, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = _finite_mask(y_true, mu, std, q)
    if not mask.any():
        return float('nan')
    return float(np.mean((y_true[mask] >= mu[mask] - q[mask] * std[mask]) & (y_true[mask] <= mu[mask] + q[mask] * std[mask])))

def _lower_coverage_from_q(y_true, mu, std, q):
    y_true = np.asarray(y_true, dtype=float)
    mu = np.asarray(mu, dtype=float)
    std = np.asarray(std, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = _finite_mask(y_true, mu, std, q)
    if not mask.any():
        return float('nan')
    return float(np.mean(y_true[mask] >= mu[mask] - q[mask] * std[mask]))

def _pooled_subset(pooled, mask):
    mask = np.asarray(mask, dtype=bool)
    out = {'strength': {}, 'perm': {}}
    n = int(mask.shape[0])
    for section in ('strength', 'perm'):
        for key, value in pooled.get(section, {}).items():
            arr = np.asarray(value)
            out[section][key] = arr[mask] if arr.shape[:1] == (n,) else value
    if 'meta' in pooled:
        out['meta'] = {}
        for key, value in pooled['meta'].items():
            arr = np.asarray(value)
            out['meta'][key] = arr[mask] if arr.shape[:1] == (n,) else value
    return out

def calibration_audit_leave_one_split_out(pooled, levels=CALIBRATION_LEVELS, q_neighbors=LOCAL_RESIDUAL_K_NEIGHBORS, combine_rule=UNCERTAINTY_COMBINE_RULE):
    result = {
        'calibration_audit_method': 'leave_one_split_out_using_existing_oos_predictions_only',
        'calibration_audit_cov95_strength_mean': float('nan'),
        'calibration_audit_cov95_strength_std': float('nan'),
        'calibration_audit_cov95_perm_mean': float('nan'),
        'calibration_audit_cov95_perm_std': float('nan'),
        'calibration_audit_per_split': [],
        'calibration_audit_n_ok_splits': 0,
        'calibration_audit_n_ok_low_n_splits': 0,
        'calibration_audit_n_skipped_low_n_splits': 0,
        'calibration_audit_row_leakage_policy': 'original_row_index unavailable; only split-level exclusion was possible',
    }
    if 'meta' not in pooled or 'split' not in pooled['meta']:
        result['calibration_audit_status'] = 'not_run_no_split_metadata'
        return result
    split = np.asarray(pooled['meta']['split'], dtype=int)
    unique_splits = np.unique(split)
    if unique_splits.size < 2:
        result['calibration_audit_status'] = 'not_run_need_at_least_two_splits'
        return result
    original_idx = None
    if 'original_row_index' in pooled.get('meta', {}):
        original_idx = np.asarray(pooled['meta']['original_row_index'], dtype=int)
        result['calibration_audit_row_leakage_policy'] = 'calibration rows with held-out original_row_index are excluded'
    else:
        result['calibration_audit_row_leakage_policy'] = 'original_row_index unavailable; only held-out split rows are excluded'
    for split_id in unique_splits:
        eval_mask = split == split_id
        cal_mask_before = ~eval_mask
        cal_mask = cal_mask_before.copy()
        held_out_original = np.asarray([], dtype=int)
        if original_idx is not None:
            held_out_original = np.unique(original_idx[eval_mask])
            cal_mask = cal_mask & ~np.isin(original_idx, held_out_original)
        n_before = int(cal_mask_before.sum())
        n_after = int(cal_mask.sum())
        detail = {
            'held_out_split': int(split_id),
            'n_calibration_rows': n_after,
            'n_calibration_rows_before_original_row_exclusion': n_before,
            'n_calibration_rows_after_original_row_exclusion': n_after,
            'n_removed_same_original_row': int(n_before - n_after),
            'n_audit_rows': int(eval_mask.sum()),
            'n_held_out_unique_original_rows': int(held_out_original.size) if original_idx is not None else None,
            'row_leakage_exclusion_used': bool(original_idx is not None),
        }
        if n_after < 5:
            detail['status'] = 'skipped_low_n'
            detail['low_n_warning'] = (
                f'Only {n_after} calibration rows remain after held-out original-row exclusion; '
                'at least 5 are required for the audit split.'
            )
            result['calibration_audit_per_split'].append(detail)
            continue
        try:
            pooled_cal = _pooled_subset(pooled, cal_mask)
            pooled_eval = _pooled_subset(pooled, eval_mask)
            bundle, _ = build_calibration_bundle(
                pooled_cal,
                levels=levels,
                q_neighbors=int(q_neighbors),
                regime_min_n=REGIME_CALIBRATION_MIN_N,
                combine_rule=combine_rule,
            )
            X_eval = np.asarray(pooled_eval['meta']['X_oos_std'], dtype=float)
            X_cal = np.asarray(pooled_cal['meta']['X_oos_std'], dtype=float)
            try:
                loc = _local_residual_core(
                    X_eval,
                    X_cal,
                    pooled_cal['strength'].get('residual', np.asarray(pooled_cal['strength']['y_true']) - np.asarray(pooled_cal['strength']['y_pred'])),
                    pooled_cal['perm'].get('residual', np.asarray(pooled_cal['perm']['y_true']) - np.asarray(pooled_cal['perm']['y_pred'])),
                    q=int(q_neighbors),
                    exclude_self=False,
                )
                detail['local_residual_status'] = 'ok'
            except ValueError as e:
                loc = {
                    'q_neighbors': 0,
                    'D_res_knn_mean': np.full(int(eval_mask.sum()), np.nan, dtype=float),
                    'D_res_knn_max': np.full(int(eval_mask.sum()), np.nan, dtype=float),
                    'strength_residual_std_local_mpa': np.full(int(eval_mask.sum()), np.nan, dtype=float),
                    'logperm_residual_std_local': np.full(int(eval_mask.sum()), np.nan, dtype=float),
                }
                detail['local_residual_status'] = 'disabled_fallback_to_gp_std'
                detail['local_residual_warning'] = str(e)
            raw_s = np.maximum(np.asarray(pooled_eval['strength'].get('y_std_raw', pooled_eval['strength'].get('y_std')), dtype=float), 1e-12)
            raw_p = np.maximum(np.asarray(pooled_eval['perm'].get('y_std_raw', pooled_eval['perm'].get('y_std')), dtype=float), 1e-12)
            std_s = _combine_std(raw_s, loc['strength_residual_std_local_mpa'], combine_rule)
            std_p = _combine_std(raw_p, loc['logperm_residual_std_local'], combine_rule)
            Xmodel = np.asarray(pooled_eval['meta'].get('X_oos_model', X_eval), dtype=float)
            mean = np.stack([
                np.asarray(pooled_eval['strength']['y_pred'], dtype=float),
                np.asarray(pooled_eval['perm']['y_pred'], dtype=float),
            ], axis=1)
            q2s, _, _ = _select_calibration_q(bundle, 'strength', 'two_sided', 0.95, Xmodel, mean)
            q2p, _, _ = _select_calibration_q(bundle, 'perm', 'two_sided', 0.95, Xmodel, mean)
            detail['cov95_strength'] = _coverage_from_q(pooled_eval['strength']['y_true'], pooled_eval['strength']['y_pred'], std_s, q2s)
            detail['cov95_perm'] = _coverage_from_q(pooled_eval['perm']['y_true'], pooled_eval['perm']['y_pred'], std_p, q2p)
            detail['q_neighbors_used'] = int(loc['q_neighbors'])
            if n_after < int(REGIME_CALIBRATION_MIN_N):
                detail['status'] = 'ok_low_n'
                detail['low_n_warning'] = (
                    f'Only {n_after} calibration rows remain after held-out original-row exclusion; '
                    'global calibration is audited, but regime-specific calibration may be unavailable or weak.'
                )
            else:
                detail['status'] = 'ok'
        except Exception as e:
            detail['status'] = 'skipped'
            detail['error'] = str(e)
        result['calibration_audit_per_split'].append(detail)
    result['calibration_audit_n_ok_splits'] = int(sum(1 for d in result['calibration_audit_per_split'] if d.get('status') == 'ok'))
    result['calibration_audit_n_ok_low_n_splits'] = int(sum(1 for d in result['calibration_audit_per_split'] if d.get('status') == 'ok_low_n'))
    result['calibration_audit_n_skipped_low_n_splits'] = int(sum(1 for d in result['calibration_audit_per_split'] if d.get('status') == 'skipped_low_n'))
    ok = [d for d in result['calibration_audit_per_split'] if d.get('status') in {'ok', 'ok_low_n'}]
    if not ok:
        result['calibration_audit_status'] = 'not_run_no_valid_held_out_split'
        return result
    cov_s = np.asarray([d['cov95_strength'] for d in ok], dtype=float)
    cov_p = np.asarray([d['cov95_perm'] for d in ok], dtype=float)
    result['calibration_audit_status'] = 'ok'
    result['calibration_audit_cov95_strength_mean'] = float(np.nanmean(cov_s))
    result['calibration_audit_cov95_strength_std'] = float(np.nanstd(cov_s, ddof=1)) if cov_s.size > 1 else 0.0
    result['calibration_audit_cov95_perm_mean'] = float(np.nanmean(cov_p))
    result['calibration_audit_cov95_perm_std'] = float(np.nanstd(cov_p, ddof=1)) if cov_p.size > 1 else 0.0
    return result

def _make_conditional_coverage_table(pooled, calib_diag, bundle):
    Xmodel = np.asarray(pooled['meta'].get('X_oos_model', pooled['meta'].get('X_oos_std')), dtype=float)
    mean = np.stack([np.asarray(pooled['strength']['y_pred'], dtype=float), np.asarray(pooled['perm']['y_pred'], dtype=float)], axis=1)
    keys = _regime_keys_from_bundle(bundle, Xmodel, mean)
    rows = []
    for task, ykey, mkey, skey in [('strength', 'strength', 'strength', 'strength_std_effective'), ('perm', 'perm', 'perm', 'logperm_std_effective')]:
        y = np.asarray(pooled[ykey]['y_true'], dtype=float)
        mu = np.asarray(pooled[mkey]['y_pred'], dtype=float)
        std = np.asarray(calib_diag[skey], dtype=float)
        for group, vals in keys.items():
            for val in sorted(set(map(str, vals))):
                mask = np.asarray(vals == val, dtype=bool)
                if int(mask.sum()) < 5:
                    continue
                row = {'task': task, 'group': group, 'key': val, 'n': int(mask.sum()), 'diagnostic_scope': 'calibration_fit_same_pooled_oos_not_independent_validation'}
                for level in bundle.get('levels', CALIBRATION_LEVELS):
                    q, _, _ = _select_calibration_q(bundle, task, 'two_sided', float(level), Xmodel[mask], mean[mask])
                    lab = int(round(100 * float(level)))
                    row[f'calibration_fit_diagnostic_coverage_two_sided_{lab}'] = _coverage_from_q(y[mask], mu[mask], std[mask], q)
                    row[f'calibration_fit_diagnostic_mean_width_two_sided_{lab}'] = float(np.nanmean(2.0 * q * std[mask]))
                rows.append(row)
    return pd.DataFrame(rows)


def _parse_policy_rules(rule_text):
    rules = []
    for raw in str(rule_text).split(','):
        r = _clean_interval_rule(raw)
        if not r:
            continue
        if r not in {'gp_only', 'local_only', 'mean', 'max', 'sqrt'}:
            raise ValueError(f"Unknown rule in --policy_compare_rules: '{raw}'. Use gp_only, local_only, mean, max, sqrt.")
        if r not in rules:
            rules.append(r)
    if not rules:
        rules = ['gp_only', 'mean', 'max', 'sqrt']
    if 'max' not in rules:
        rules.append('max')
    return rules

def _policy_explanation(rule):
    r = _clean_interval_rule(rule)
    if r == 'gp_only':
        return 'uses only the model uncertainty; usually narrowest, but can miss local trouble spots'
    if r == 'local_only':
        return 'uses only nearby past errors; useful diagnostic, but can be noisy'
    if r == 'mean':
        return 'averages model uncertainty and nearby-error uncertainty; candidate for narrower intervals'
    if r == 'max':
        return 'uses the larger of model uncertainty and nearby-error uncertainty; conservative default'
    if r == 'sqrt':
        return 'adds both uncertainty sources strongly; usually widest'
    return 'unknown rule'

def make_interval_policy_comparison(pooled, rules='gp_only,mean,max,sqrt,local_only', levels=CALIBRATION_LEVELS, q_neighbors=LOCAL_RESIDUAL_K_NEIGHBORS, regime_min_n=REGIME_CALIBRATION_MIN_N):
    """Compare interval-width options on the same repeated-split OOS predictions.

    This does not retrain the GP. It only reuses the already-created OOS predictions and asks:
    after calibration, which interval rule gives narrower intervals while still matching the requested coverage?
    This table is a fitted-on-same-OOS diagnostic, not an independent validation estimate.
    """
    rules = _parse_policy_rules(rules)
    Xmodel = np.asarray(pooled['meta'].get('X_oos_model', pooled['meta']['X_oos_std']), dtype=float)
    mean = np.stack([np.asarray(pooled['strength']['y_pred'], dtype=float), np.asarray(pooled['perm']['y_pred'], dtype=float)], axis=1)
    rows = []
    for rule in rules:
        bundle, diag = build_calibration_bundle(
            pooled,
            levels=levels,
            q_neighbors=int(q_neighbors),
            regime_min_n=int(regime_min_n),
            combine_rule=rule,
        )
        for task, ykey, skey in [
            ('strength', 'strength', 'strength_std_effective'),
            ('perm', 'perm', 'logperm_std_effective'),
        ]:
            y = np.asarray(pooled[ykey]['y_true'], dtype=float)
            mu = np.asarray(pooled[ykey]['y_pred'], dtype=float)
            std = np.asarray(diag[skey], dtype=float)
            for level in bundle.get('levels', levels):
                lab = int(round(100 * float(level)))
                q2, _, _ = _select_calibration_q(bundle, task, 'two_sided', float(level), Xmodel, mean)
                ql, _, _ = _select_calibration_q(bundle, task, 'lower', float(level), Xmodel, mean)
                width = float(np.nanmean(2.0 * q2 * std))
                rows.append({
                    'rule': _clean_interval_rule(rule),
                    'plain_english': _policy_explanation(rule),
                    'diagnostic_scope': 'fitted_on_same_pooled_oos_not_independent_validation',
                    'independent_validation_estimate': False,
                    'task': task,
                    'level': float(level),
                    'level_percent': lab,
                    'calibration_fit_diagnostic_two_sided_coverage': _coverage_from_q(y, mu, std, q2),
                    'mean_two_sided_width': width,
                    'calibration_fit_diagnostic_lower_bound_success': _lower_coverage_from_q(y, mu, std, ql),
                    'mean_effective_std': float(np.nanmean(std)),
                    'mean_q_two_sided': float(np.nanmean(q2)),
                    'mean_q_lower': float(np.nanmean(ql)),
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        base = df[df['rule'] == 'max'][['task', 'level', 'mean_two_sided_width']].rename(columns={'mean_two_sided_width': 'max_rule_width'})
        df = df.merge(base, on=['task', 'level'], how='left')
        df['width_change_vs_max_percent'] = 100.0 * (df['mean_two_sided_width'] - df['max_rule_width']) / df['max_rule_width']
        df['narrower_than_max'] = df['width_change_vs_max_percent'] < 0
        df['calibration_fit_diagnostic_coverage_minus_requested'] = df['calibration_fit_diagnostic_two_sided_coverage'] - df['level']
    return df

def _write_plain_english_interval_report(outdir, selected_rule, policy_df):
    path = os.path.join(outdir, 'READ_ME_INTERVAL_WIDTH_RESULTS.txt')
    selected_rule = _clean_interval_rule(selected_rule)
    lines = []
    lines.append('WHAT THIS FILE MEANS')
    lines.append('====================')
    lines.append('The code now tests several ways to make the prediction intervals. It does not guess blindly.')
    lines.append('It writes a table called interval_policy_comparison.csv.')
    lines.append('Important: this is a fitted-on-same-OOS diagnostic width comparison, not independent validation.')
    lines.append('For validation-style coverage, use the calibration_audit_* fields in summary_results.json.')
    lines.append(CALIBRATION_AUDIT_NOTE)
    lines.append('Raw GP standard deviations are diagnostic only.')
    lines.append(STD_UNCERTAINTY_NOTE)
    lines.append('Legacy plot filenames containing validation are retained for compatibility but should be read as diagnostics.')
    lines.append('')
    lines.append('Columns to look at:')
    lines.append('1. rule: the interval option being tested.')
    lines.append('2. task: strength or permeability.')
    lines.append('3. level_percent: 90 means 90% interval, 95 means 95% interval.')
    lines.append('4. calibration_fit_diagnostic_two_sided_coverage: fitted-on-same-OOS coverage diagnostic; it is not independent validation.')
    lines.append('5. mean_two_sided_width: smaller is narrower.')
    lines.append('6. width_change_vs_max_percent: negative means narrower than the old conservative max rule.')
    lines.append('')
    lines.append(f'The final saved model will use this rule: {selected_rule}')
    lines.append(f'Plain meaning: {_policy_explanation(selected_rule)}')
    lines.append('')
    lines.append('My blunt recommendation:')
    lines.append('- Keep max if you want the safest intervals.')
    lines.append('- Try mean only if it gives clearly smaller widths here and the calibration_audit_* fields remain acceptable.')
    lines.append('- Do not use gp_only just because it is narrow. If fitted diagnostic coverage drops, it is fake confidence.')
    lines.append('- sqrt is usually too wide unless your safety requirement is extreme.')
    lines.append('')
    if policy_df is not None and len(policy_df):
        lines.append('QUICK 90/95 SUMMARY')
        lines.append('-------------------')
        focus = policy_df[policy_df['level_percent'].isin([90, 95])].copy()
        cols = ['rule', 'task', 'level_percent', 'calibration_fit_diagnostic_two_sided_coverage', 'mean_two_sided_width', 'width_change_vs_max_percent']
        for _, row in focus[cols].sort_values(['task', 'level_percent', 'mean_two_sided_width']).iterrows():
            lines.append(
                f"{row['task']}, {int(row['level_percent'])}% | {row['rule']} | fitted diagnostic coverage={row['calibration_fit_diagnostic_two_sided_coverage']:.3f} | "
                f"width={row['mean_two_sided_width']:.4g} | vs max={row['width_change_vs_max_percent']:.1f}%"
            )
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return path

def make_pooled_oos_residual_dataframe(pooled, calib_diag=None):
    if 'meta' not in pooled or 'X_oos_std' not in pooled['meta']:
        raise ValueError('Pooled predictions do not contain OOS standardized feature vectors.')
    X_oos_std = np.asarray(pooled['meta']['X_oos_std'], dtype=float)
    X_oos_model = np.asarray(pooled['meta'].get('X_oos_model', X_oos_std), dtype=float)
    if X_oos_std.ndim != 2 or X_oos_std.shape[1] != len(FEATURES):
        raise ValueError('Invalid OOS feature matrix shape.')
    n = X_oos_std.shape[0]
    df = pd.DataFrame()
    if 'split' in pooled['meta']:
        df['split'] = np.asarray(pooled['meta']['split'], dtype=int)
    if 'split_seed' in pooled['meta']:
        df['split_seed'] = np.asarray(pooled['meta']['split_seed'], dtype=int)
    if 'original_row_index' in pooled['meta']:
        df['original_row_index'] = np.asarray(pooled['meta']['original_row_index'], dtype=int)
    for key in ['paper_source', 'paper_label', 'mix_id']:
        if key in pooled['meta']:
            df[key] = np.asarray(pooled['meta'][key], dtype=object)
    if 'source_excel_row' in pooled['meta']:
        df['source_excel_row'] = np.asarray(pooled['meta']['source_excel_row'], dtype=int)
    for j, feat in enumerate(FEATURES):
        safe_feat = _safe_feature_name(feat)
        df[f'xstd_{safe_feat}'] = X_oos_std[:, j]
        df[f'xmodel_{safe_feat}'] = X_oos_model[:, j]
    strength_true = np.asarray(pooled['strength']['y_true'], dtype=float)
    strength_pred = np.asarray(pooled['strength']['y_pred'], dtype=float)
    logperm_true = np.asarray(pooled['perm']['y_true'], dtype=float)
    logperm_pred = np.asarray(pooled['perm']['y_pred'], dtype=float)
    strength_residual = np.asarray(pooled['strength'].get('residual', strength_true - strength_pred), dtype=float)
    logperm_residual = np.asarray(pooled['perm'].get('residual', logperm_true - logperm_pred), dtype=float)
    df['strength_true_mpa'] = strength_true
    df['strength_pred_mpa'] = strength_pred
    df['strength_residual_mpa'] = strength_residual
    df['strength_abs_residual_mpa'] = np.abs(strength_residual)
    df['strength_std_oos_raw_mpa'] = np.asarray(pooled['strength'].get('y_std_raw', pooled['strength'].get('y_std')), dtype=float)
    df['logperm_true'] = logperm_true
    df['logperm_pred'] = logperm_pred
    df['logperm_residual'] = logperm_residual
    df['logperm_abs_residual'] = np.abs(logperm_residual)
    df['logperm_std_oos_raw'] = np.asarray(pooled['perm'].get('y_std_raw', pooled['perm'].get('y_std')), dtype=float)
    df['perm_true_mm_s'] = np.exp(logperm_true)
    df['perm_pred_mm_s'] = np.exp(logperm_pred)
    if calib_diag is not None:
        loc = calib_diag.get('local', {})
        for k in ['D_res_knn_mean', 'D_res_knn_max', 'strength_residual_std_local_mpa', 'logperm_residual_std_local']:
            if k in loc:
                df[k] = np.asarray(loc[k], dtype=float)
        if 'strength_std_effective' in calib_diag:
            df['strength_std_oos_effective_mpa'] = np.asarray(calib_diag['strength_std_effective'], dtype=float)
        if 'logperm_std_effective' in calib_diag:
            df['logperm_std_oos_effective'] = np.asarray(calib_diag['logperm_std_effective'], dtype=float)
    if len(df) != n:
        raise RuntimeError('Internal error while building OOS residual dataframe.')
    return df

def _calibration_curve_empirical(y_true, mu, std, bundle, task, Xmodel, mean, levels=None):
    if levels is None:
        levels = bundle.get('levels', CALIBRATION_LEVELS)
    nom, obs, width = ([], [], [])
    for level in levels:
        q, _, level_used = _select_calibration_q(bundle, task, 'two_sided', float(level), Xmodel, mean)
        nom.append(float(level_used))
        obs.append(_coverage_from_q(y_true, mu, std, q))
        width.append(float(np.nanmean(2.0 * q * std)))
    return (np.asarray(nom), np.asarray(obs), np.asarray(width))

def make_validation_plots(outdir, pooled, calibration_bundle, calib_diag, dpi=300):
    plot_dir = ensure_dir(os.path.join(outdir, 'plots'))
    ys = np.asarray(pooled['strength']['y_true'], dtype=float)
    yps = np.asarray(pooled['strength']['y_pred'], dtype=float)
    yp = np.asarray(pooled['perm']['y_true'], dtype=float)
    ypp = np.asarray(pooled['perm']['y_pred'], dtype=float)
    ss = np.asarray(calib_diag['strength_std_effective'], dtype=float)
    sp = np.asarray(calib_diag['logperm_std_effective'], dtype=float)
    Xmodel = np.asarray(pooled['meta'].get('X_oos_model', pooled['meta']['X_oos_std']), dtype=float)
    mean = np.stack([yps, ypp], axis=1)
    fig, ax = plt.subplots(2, 2, figsize=(11.0, 9.0), dpi=150)
    lo, hi = _axis_limits(ys, yps)
    ax[0, 0].scatter(ys, yps, s=18, alpha=0.55, edgecolors='none')
    ax[0, 0].plot([lo, hi], [lo, hi], 'k--', lw=1.4)
    ax[0, 0].set_xlim(lo, hi)
    ax[0, 0].set_ylim(lo, hi)
    ax[0, 0].set_xlabel('Measured compressive strength (MPa)')
    ax[0, 0].set_ylabel('Predicted compressive strength (MPa)')
    ax[0, 0].set_title('(a) Strength: pooled OOS fit diagnostic')
    lo, hi = _axis_limits(yp, ypp)
    ax[0, 1].scatter(yp, ypp, s=18, alpha=0.55, edgecolors='none')
    ax[0, 1].plot([lo, hi], [lo, hi], 'k--', lw=1.4)
    ax[0, 1].set_xlim(lo, hi)
    ax[0, 1].set_ylim(lo, hi)
    ax[0, 1].set_xlabel('Measured log(permeability)')
    ax[0, 1].set_ylabel('Predicted log(permeability)')
    ax[0, 1].set_title('(b) Permeability: pooled OOS fit diagnostic')
    ax[1, 0].scatter(yps, ys - yps, s=18, alpha=0.55, edgecolors='none')
    ax[1, 0].axhline(0.0, color='k', ls='--', lw=1.4)
    ax[1, 0].set_xlabel('Predicted compressive strength (MPa)')
    ax[1, 0].set_ylabel('Residual: measured - predicted')
    ax[1, 0].set_title('(c) Strength residuals')
    ax[1, 1].scatter(ypp, yp - ypp, s=18, alpha=0.55, edgecolors='none')
    ax[1, 1].axhline(0.0, color='k', ls='--', lw=1.4)
    ax[1, 1].set_xlabel('Predicted log(permeability)')
    ax[1, 1].set_ylabel('Residual: measured - predicted')
    ax[1, 1].set_title('(d) Permeability residuals')
    for a in ax.ravel():
        a.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'lcmgp_without_head_pooled_oos_validation_2x2.png'), dpi=int(dpi), bbox_inches='tight')
    fig.savefig(os.path.join(plot_dir, 'lcmgp_without_head_pooled_oos_diagnostic_2x2.png'), dpi=int(dpi), bbox_inches='tight')
    plt.close(fig)
    ns, os_, ws = _calibration_curve_empirical(ys, yps, ss, calibration_bundle, 'strength', Xmodel, mean)
    np_, op, wp = _calibration_curve_empirical(yp, ypp, sp, calibration_bundle, 'perm', Xmodel, mean)
    raw_s = np.asarray(pooled['strength'].get('y_std_raw', pooled['strength'].get('y_std')), dtype=float)
    raw_p = np.asarray(pooled['perm'].get('y_std_raw', pooled['perm'].get('y_std')), dtype=float)
    ns_raw, os_raw = _calibration_curve(ys, yps, raw_s, qs=calibration_bundle.get('levels', CALIBRATION_LEVELS))
    np_raw, op_raw = _calibration_curve(yp, ypp, raw_p, qs=calibration_bundle.get('levels', CALIBRATION_LEVELS))
    rows = []
    for label, nom, obs in [('strength_raw_gp', ns_raw, os_raw), ('permeability_raw_gp', np_raw, op_raw), ('strength_empirical', ns, os_), ('permeability_empirical', np_, op)]:
        for a, b in zip(np.asarray(nom, dtype=float), np.asarray(obs, dtype=float)):
            rows.append({'curve': label, 'diagnostic_scope': 'calibration_fit_same_pooled_oos_not_independent_validation', 'nominal_coverage': float(a), 'calibration_fit_diagnostic_observed_coverage': float(b)})
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(plot_dir, 'lcmgp_without_head_calibration_curve_points.csv'), index=False)
    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=150)
    ax.plot([0, 1], [0, 1], 'k--', lw=1.4, label='Ideal')
    if ns_raw.size:
        ax.plot(ns_raw, os_raw, marker='o', lw=1.2, ls=':', alpha=0.8, label='Strength raw GP')
    if np_raw.size:
        ax.plot(np_raw, op_raw, marker='s', lw=1.2, ls=':', alpha=0.8, label='Permeability raw GP')
    offset = 0.006
    if ns.size:
        ax.plot(ns - offset, os_, marker='o', mfc='none', lw=1.8, label='Strength empirical')
    if np_.size:
        ax.plot(np_ + offset, op, marker='s', mfc='none', lw=1.8, label='Permeability empirical')
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel('Nominal coverage')
    ax.set_ylabel('Observed coverage')
    ax.set_title('Calibration-fit diagnostic: raw vs empirical uncertainty')
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'lcmgp_without_head_uncertainty_calibration.png'), dpi=int(dpi), bbox_inches='tight')
    fig.savefig(os.path.join(plot_dir, 'lcmgp_without_head_uncertainty_calibration_raw_vs_empirical.png'), dpi=int(dpi), bbox_inches='tight')
    fig.savefig(os.path.join(plot_dir, 'lcmgp_without_head_uncertainty_calibration_fit_diagnostic.png'), dpi=int(dpi), bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 5.4), dpi=150)
    if ns.size:
        ax.plot(ns, ws, marker='o', lw=1.6, label='Strength')
    if np_.size:
        ax.plot(np_, wp, marker='s', lw=1.6, label='Log-permeability')
    ax.set_xlabel('Nominal coverage')
    ax.set_ylabel('Mean interval width')
    ax.set_title('Calibration-fit diagnostic: interval sharpness')
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'lcmgp_without_head_interval_width.png'), dpi=int(dpi), bbox_inches='tight')
    plt.close(fig)
    k = np.exp(yp)
    kp = np.exp(ypp)
    lo, hi = _axis_limits(k, kp)
    fig, ax = plt.subplots(figsize=(6.5, 5.8), dpi=150)
    ax.scatter(k, kp, s=18, alpha=0.55, edgecolors='none')
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.4)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel('Measured permeability (mm/s)')
    ax.set_ylabel('Predicted permeability (mm/s)')
    ax.set_title('Permeability: original scale')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, 'lcmgp_without_head_original_scale_permeability.png'), dpi=int(dpi), bbox_inches='tight')
    plt.close(fig)
    return plot_dir

def oos_residual_reference_from_save_obj(save_obj):
    required = ['oos_residual_features_std', 'oos_residual_strength', 'oos_residual_logperm']
    missing = [k for k in required if k not in save_obj]
    if missing:
        raise ValueError('This oracle does not contain OOS residual-feature references. Missing: ' + ', '.join(missing))
    X_oos_std = np.asarray(save_obj['oos_residual_features_std'], dtype=float)
    X_oos_model = np.asarray(save_obj.get('oos_residual_features_model', X_oos_std), dtype=float)
    df = pd.DataFrame()
    if 'oos_residual_split' in save_obj:
        df['split'] = np.asarray(save_obj['oos_residual_split'], dtype=int)
    if 'oos_residual_split_seed' in save_obj:
        df['split_seed'] = np.asarray(save_obj['oos_residual_split_seed'], dtype=int)
    if 'oos_residual_original_row_index' in save_obj:
        df['original_row_index'] = np.asarray(save_obj['oos_residual_original_row_index'], dtype=int)
    if 'oos_residual_paper_source' in save_obj:
        df['paper_source'] = np.asarray(save_obj['oos_residual_paper_source'], dtype=object)
    if 'oos_residual_paper_label' in save_obj:
        df['paper_label'] = np.asarray(save_obj['oos_residual_paper_label'], dtype=object)
    feature_names = list(save_obj.get('features', FEATURES))
    for j, feat in enumerate(feature_names):
        safe_feat = _safe_feature_name(feat)
        df[f'xstd_{safe_feat}'] = X_oos_std[:, j]
        if X_oos_model.shape == X_oos_std.shape:
            df[f'xmodel_{safe_feat}'] = X_oos_model[:, j]
    df['strength_residual_mpa'] = np.asarray(save_obj['oos_residual_strength'], dtype=float)
    df['logperm_residual'] = np.asarray(save_obj['oos_residual_logperm'], dtype=float)
    for key, col in [('oos_residual_strength_true', 'strength_true_mpa'), ('oos_residual_strength_pred', 'strength_pred_mpa'), ('oos_residual_logperm_true', 'logperm_true'), ('oos_residual_logperm_pred', 'logperm_pred')]:
        if key in save_obj:
            df[col] = np.asarray(save_obj[key], dtype=float)
    return df

def local_residual_uncertainty_from_standardized_features(save_obj, Xs, q=LOCAL_RESIDUAL_K_NEIGHBORS):
    if 'oos_residual_features_std' not in save_obj:
        raise ValueError('No OOS residual-feature table found in oracle.')
    # Coordinate-system invariant: Xs and oos_residual_features_std must both be
    # transformed features standardized by the final full-data XStandardizer.
    return _local_residual_core(Xs, save_obj['oos_residual_features_std'], save_obj['oos_residual_strength'], save_obj['oos_residual_logperm'], q=q, exclude_self=False)

def _prediction_local_residual_lookup(save_obj, Xs, q):
    n = int(np.asarray(Xs).shape[0])
    base = {
        'D_res_knn_mean': np.full(n, np.nan, dtype=float),
        'D_res_knn_max': np.full(n, np.nan, dtype=float),
        'q_neighbors': int(q),
        'strength_residual_std_local_mpa': np.full(n, np.nan, dtype=float),
        'logperm_residual_std_local': np.full(n, np.nan, dtype=float),
    }
    required = ['oos_residual_features_std', 'oos_residual_strength', 'oos_residual_logperm']
    missing = [k for k in required if k not in save_obj]
    if missing:
        return {
            'loc': base,
            'strength_local': base['strength_residual_std_local_mpa'],
            'logperm_local': base['logperm_residual_std_local'],
            'status': 'missing_reference_fallback_to_gp_std',
            'warning': 'Missing OOS residual reference keys: ' + ', '.join(missing),
            'q_neighbors_used': 0,
        }
    try:
        loc = local_residual_uncertainty_from_standardized_features(save_obj, Xs, q=int(q))
        return {
            'loc': loc,
            'strength_local': loc['strength_residual_std_local_mpa'],
            'logperm_local': loc['logperm_residual_std_local'],
            'status': 'ok',
            'warning': '',
            'q_neighbors_used': int(loc['q_neighbors']),
        }
    except ValueError as e:
        return {
            'loc': base,
            'strength_local': base['strength_residual_std_local_mpa'],
            'logperm_local': base['logperm_residual_std_local'],
            'status': 'invalid_reference_fallback_to_gp_std',
            'warning': str(e),
            'q_neighbors_used': 0,
        }

def _prediction_features_model(df_in, save_obj):
    missing = [c for c in FEATURES if c not in df_in.columns]
    if missing:
        raise ValueError('Prediction input is missing required feature columns: ' + ', '.join(missing))
    df_feat = df_in[FEATURES].copy().replace([np.inf, -np.inf], np.nan)
    if df_feat.isna().any().any():
        bad_cols = df_feat.columns[df_feat.isna().any()].tolist()
        raise ValueError('Prediction input contains missing/invalid values in: ' + ', '.join(bad_cols))
    df_feat = _apply_feature_transforms(df_feat)
    return df_feat[FEATURES].to_numpy(dtype=float)

def _prepare_prediction_features(df_in, save_obj):
    X = _prediction_features_model(df_in, save_obj)
    x_mean = np.asarray(save_obj['x_cont_mean'], dtype=float)
    x_std = np.asarray(save_obj['x_cont_std'], dtype=float)
    x_std = np.where(x_std == 0, 1.0, x_std)
    Xs = X.copy()
    Xs[:, CONT_IDXS] = (Xs[:, CONT_IDXS] - x_mean) / x_std
    return Xs

def _add_level_outputs(out, prefix, pred, std_eff, bundle, Xmodel, mean, task, levels):
    for level in levels:
        q2, src2, level_used = _select_calibration_q(bundle, task, 'two_sided', float(level), Xmodel, mean)
        ql, srcl, _ = _select_calibration_q(bundle, task, 'lower', float(level), Xmodel, mean)
        lab = int(round(100 * float(level_used)))
        out[f'{prefix}_pi{lab}_lower'] = pred - q2 * std_eff
        out[f'{prefix}_pi{lab}_upper'] = pred + q2 * std_eff
        out[f'{prefix}_LB{lab}'] = pred - ql * std_eff
        out[f'{prefix}_q_two_sided_{lab}'] = q2
        out[f'{prefix}_q_lower_{lab}'] = ql
        out[f'{prefix}_cal_source_two_sided_{lab}'] = src2
        out[f'{prefix}_cal_source_lower_{lab}'] = srcl
    return out

def predict_saved_oracle_dataframe(pt_path, df_new, target_coverage=None, include_input=True, local_residual_q=None, apply_ad_inflation=True):
    if target_coverage is not None and not (0.0 < float(target_coverage) < 1.0):
        raise ValueError('target_coverage must be > 0 and < 1.')
    save_obj, model, likelihood = _load_saved_oracle(pt_path, DEVICE)
    Xmodel = _prediction_features_model(df_new, save_obj)
    Xs = _prepare_prediction_features(df_new, save_obj)
    Xt_new = torch.as_tensor(Xs, dtype=DTYPE, device=DEVICE).contiguous()
    mu_s, sig_s = predict(model, likelihood, Xt_new, fast_mode=bool(save_obj.get('fast_mode', True)))
    y_mean = np.asarray(save_obj['y_mean'], dtype=float)
    y_std = np.asarray(save_obj['y_std'], dtype=float)
    mean = mu_s * y_std + y_mean
    std_raw = apply_sigma_floors(sig_s * y_std)
    if local_residual_q is None:
        local_residual_q = max(2, int(save_obj.get('local_residual_default_q_neighbors', LOCAL_RESIDUAL_K_NEIGHBORS)))
    local_info = _prediction_local_residual_lookup(save_obj, Xs, int(local_residual_q))
    loc = local_info['loc']
    local_s = local_info['strength_local']
    local_p = local_info['logperm_local']
    std_eff_s_pre = _combine_std(std_raw[:, 0], local_s, save_obj.get('uncertainty_combine_rule', UNCERTAINTY_COMBINE_RULE))
    std_eff_p_pre = _combine_std(std_raw[:, 1], local_p, save_obj.get('uncertainty_combine_rule', UNCERTAINTY_COMBINE_RULE))
    ad = _ad_for_standardized_features(save_obj, Xs) if 'ad_reference' in save_obj else {'D_AD': np.full(len(Xs), np.nan), 'tau_AD': np.full(len(Xs), np.nan), 'AD_ratio': np.full(len(Xs), np.nan), 'inside_AD': np.full(len(Xs), False), 'AD_k_neighbors': AD_K_NEIGHBORS}
    infl = _ad_inflation_from_ratio(ad['AD_ratio'], eta=float(save_obj.get('ad_inflation_eta', AD_INFLATION_ETA))) if apply_ad_inflation else np.ones(len(Xs), dtype=float)
    std_eff_s = std_eff_s_pre * infl
    std_eff_p = std_eff_p_pre * infl
    bundle = save_obj.get('calibration_bundle', None)
    if bundle is None:
        old = save_obj.get('sigma_scale_used', {})
        scale_s = float(old.get('strength', 1.0))
        scale_p = float(old.get('perm', 1.0))
        bundle = {'levels': [0.95], 'global': {'strength': {'two_sided': {'0.950': 1.96 * scale_s}, 'lower': {'0.950': 1.645 * scale_s}}, 'perm': {'two_sided': {'0.950': 1.96 * scale_p}, 'lower': {'0.950': 1.645 * scale_p}}}, 'regime': {'strength': {}, 'perm': {}}, 'regime_bins': {}}
    if target_coverage is None:
        target_coverage = 0.95
    target_coverage = float(target_coverage)
    if not (0.0 < target_coverage < 1.0):
        raise ValueError('target_coverage must be > 0 and < 1.')
    level_used, _ = _resolve_level(bundle, target_coverage)
    strength_pred = mean[:, 0]
    logperm_pred = mean[:, 1]
    out = pd.DataFrame({'strength_pred_mpa': strength_pred, 'logperm_pred': logperm_pred, 'perm_pred_mm_s': np.exp(logperm_pred), 'strength_std_raw_diagnostic_mpa': std_raw[:, 0], 'logperm_std_raw_diagnostic': std_raw[:, 1], 'strength_residual_std_local_mpa': local_s, 'logperm_residual_std_local': local_p, 'strength_std_effective_before_ad_mpa': std_eff_s_pre, 'logperm_std_effective_before_ad': std_eff_p_pre, 'AD_uncertainty_inflation': infl, 'strength_std_effective_mpa': std_eff_s, 'logperm_std_effective': std_eff_p, 'strength_std_calibrated_mpa': std_eff_s, 'logperm_std_calibrated': std_eff_p, 'D_res_knn_mean': loc['D_res_knn_mean'], 'D_res_knn_max': loc['D_res_knn_max'], 'local_residual_q_neighbors': int(loc['q_neighbors']), 'local_residual_status': str(local_info['status']), 'local_residual_warning': str(local_info['warning']), 'local_residual_q_neighbors_used': int(local_info['q_neighbors_used']), 'D_AD': ad['D_AD'], 'tau_AD': ad['tau_AD'], 'AD_ratio': ad['AD_ratio'], 'inside_AD': ad['inside_AD'], 'AD_k_neighbors': int(ad['AD_k_neighbors']), 'target_coverage_requested': target_coverage, 'target_coverage_used': level_used})
    lower_bound_levels = [float(x) for x in bundle.get('lower_bound_levels', LOWER_BOUND_LEVELS)]
    all_levels = sorted(set([float(x) for x in bundle.get('levels', CALIBRATION_LEVELS)] + lower_bound_levels + [float(target_coverage)]))
    all_levels = [float(_resolve_level(bundle, x)[0]) for x in all_levels]
    all_levels = sorted(set(all_levels))
    out = _add_level_outputs(out, 'strength', strength_pred, std_eff_s, bundle, Xmodel, mean, 'strength', all_levels)
    out = _add_level_outputs(out, 'logperm', logperm_pred, std_eff_p, bundle, Xmodel, mean, 'perm', all_levels)
    for level in all_levels:
        lab = int(round(100 * float(level)))
        out[f'perm_pi{lab}_lower_mm_s'] = np.exp(out[f'logperm_pi{lab}_lower'])
        out[f'perm_pi{lab}_upper_mm_s'] = np.exp(out[f'logperm_pi{lab}_upper'])
        out[f'perm_LB{lab}_mm_s'] = np.exp(out[f'logperm_LB{lab}'])
        out[f'strength_lower_bound_{lab}_mpa'] = out[f'strength_LB{lab}']
        out[f'logperm_lower_bound_{lab}'] = out[f'logperm_LB{lab}']
        out[f'perm_lower_bound_{lab}_mm_s'] = out[f'perm_LB{lab}_mm_s']
    labt = int(round(100 * float(level_used)))
    out['strength_pi_lower_mpa'] = out[f'strength_pi{labt}_lower']
    out['strength_pi_upper_mpa'] = out[f'strength_pi{labt}_upper']
    out['strength_LB_mpa'] = out[f'strength_LB{labt}']
    out['logperm_pi_lower'] = out[f'logperm_pi{labt}_lower']
    out['logperm_pi_upper'] = out[f'logperm_pi{labt}_upper']
    out['logperm_LB'] = out[f'logperm_LB{labt}']
    out['perm_pi_lower_mm_s'] = out[f'perm_pi{labt}_lower_mm_s']
    out['perm_pi_upper_mm_s'] = out[f'perm_pi{labt}_upper_mm_s']
    out['perm_LB_mm_s'] = out[f'perm_LB{labt}_mm_s']
    if include_input:
        out = pd.concat([df_new.reset_index(drop=True), out], axis=1)
    return out

def local_residual_uncertainty_for_dataframe(pt_path, df_new, q=LOCAL_RESIDUAL_K_NEIGHBORS, include_input=True):
    save_obj = _load_saved_oracle_object(pt_path, device='cpu')
    Xs = _prepare_prediction_features(df_new, save_obj)
    loc = local_residual_uncertainty_from_standardized_features(save_obj, Xs, q=q)
    out = pd.DataFrame({'D_res_knn_mean': loc['D_res_knn_mean'], 'D_res_knn_max': loc['D_res_knn_max'], 'strength_residual_std_local_mpa': loc['strength_residual_std_local_mpa'], 'logperm_residual_std_local': loc['logperm_residual_std_local'], 'local_residual_q_neighbors': int(loc['q_neighbors'])})
    if include_input:
        out = pd.concat([df_new.reset_index(drop=True), out], axis=1)
    return out

def write_prediction_contract(outdir, pt_path, summary, calibration_bundle):
    paperwise = str(summary.get('split_mode', '')).lower() == 'paperwise'
    audit_policy = (
        'Leave-one-paper-fold-out calibration audit using existing paperwise OOS predictions; no retraining.'
        if paperwise
        else CALIBRATION_AUDIT_NOTE
    )
    oos_pool_label = 'leave-one-paper-out OOS pool' if paperwise else 'repeated-random-split OOS pool'
    main_cols = [
        'strength_pred_mpa', 'strength_std_calibrated_mpa',
        'strength_pi_lower_mpa', 'strength_pi_upper_mpa',
        'logperm_pred', 'logperm_std_calibrated',
        'logperm_pi_lower', 'logperm_pi_upper',
        'perm_pred_mm_s', 'perm_pi_lower_mm_s', 'perm_pi_upper_mm_s',
        'strength_LB80', 'strength_LB90', 'strength_LB95',
        'logperm_LB80', 'logperm_LB90', 'logperm_LB95',
        'perm_LB80_mm_s', 'perm_LB90_mm_s', 'perm_LB95_mm_s',
        'strength_std_effective_mpa', 'logperm_std_effective',
        'strength_std_raw_diagnostic_mpa', 'logperm_std_raw_diagnostic',
        'strength_residual_std_local_mpa', 'logperm_residual_std_local',
        'local_residual_status', 'local_residual_warning', 'local_residual_q_neighbors_used',
        'strength_lower_bound_80_mpa', 'strength_lower_bound_90_mpa', 'strength_lower_bound_95_mpa',
        'logperm_lower_bound_80', 'logperm_lower_bound_90', 'logperm_lower_bound_95',
        'perm_lower_bound_80_mm_s', 'perm_lower_bound_90_mm_s', 'perm_lower_bound_95_mm_s',
        'D_AD', 'AD_ratio', 'inside_AD',
    ]
    contract = {
        'model_file': os.path.basename(str(pt_path)),
        'official_prediction_function': 'predict_saved_oracle_dataframe',
        'official_prediction_cli': 'python <this_script.py> --mode predict --model_path <model.pt> --predict_data <new_data.xlsx> --pred_out <predictions.csv>',
        'uncertainty_policy': 'Use empirical OOS calibrated intervals and one-sided lower bounds. Raw GP standard deviations are diagnostic only.',
        'std_column_semantics': STD_UNCERTAINTY_NOTE,
        'raw_gp_std_policy': 'Raw GP standard deviations are diagnostic only.',
        'pooled_fitted_coverage_policy': 'Pooled fitted coverage is a calibration-fit diagnostic, not independent validation.',
        'calibration_audit_policy': audit_policy,
        'calibration_levels': calibration_bundle.get('levels', []),
        'lower_bound_levels': calibration_bundle.get('lower_bound_levels', []),
        'oos_residual_feature_file': OOS_RESIDUAL_FEATURE_FILE,
        'oos_residual_feature_coordinate_system': 'final_full_data_standardized_transformed_features',
        'official_local_residual_function': 'local_residual_uncertainty_from_standardized_features',
        'official_local_residual_dataframe_function': 'local_residual_uncertainty_for_dataframe',
        'ad_policy': {
            'ad_k_neighbors': AD_K_NEIGHBORS,
            'ad_threshold_quantile': AD_THRESHOLD_QUANTILE,
            'ad_inflation_start': AD_INFLATION_START,
            'ad_inflation_eta': AD_INFLATION_ETA,
            'ad_inflation_cap': AD_INFLATION_CAP,
        },
        'main_report_columns': main_cols,
        'validation_summary': summary,
    }
    json_path = os.path.join(outdir, 'deployment_prediction_contract.json')
    with open(json_path, 'w') as f:
        json.dump(array_to_list_obj(contract), f, indent=2)
    readme_path = os.path.join(outdir, 'HOW_TO_PREDICT_WITH_CALIBRATED_UNCERTAINTY.txt')
    with open(readme_path, 'w') as f:
        f.write(
            'Use prediction mode in this same Python file.\n\n'
            'python <this_script.py> --mode predict --model_path ' + os.path.basename(str(pt_path)) + ' --predict_data new_data.xlsx --pred_out predictions.csv\n\n'
            'Reportable uncertainty lives in the prediction interval and lower-bound columns.\n'
            'Use strength_pi_lower_mpa / strength_pi_upper_mpa and perm_pi_lower_mm_s / perm_pi_upper_mm_s for interval bands.\n'
            'Use strength_LB80/90/95 and perm_LB80/90/95_mm_s, or their lower_bound aliases, for one-sided robust feasibility.\n\n'
            'Raw GP standard deviations are diagnostic only.\n'
            + STD_UNCERTAINTY_NOTE + '\n\n'
            'Pooled fitted coverage and interval-policy tables are calibration-fit diagnostics fitted on the same '
            + oos_pool_label + '; they are not independent validation.\n'
            + audit_policy + ' This audit is internal, not external validation.\n'
            'When original_row_index is available, same original-row leakage is excluded from the calibration audit.\n'
            'Files with legacy names containing validation are retained for compatibility but should be read as diagnostic plots.\n'
        )
    return (json_path, readme_path)

def _assert_raises_value_error(label, fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(f'Self-check failed: {label} did not raise ValueError.')

def _assert_raises_runtime_error(label, fn):
    try:
        fn()
    except RuntimeError:
        return
    raise AssertionError(f'Self-check failed: {label} did not raise RuntimeError.')

def _run_internal_self_checks():
    py_compile.compile(__file__, doraise=True)
    _assert_raises_value_error('bad kernel validation', lambda: _kernel_components('rbf+badkernel'))
    _assert_raises_value_error('bad add_kernel validation', lambda: _validate_add_kernel_name('badkernel'))
    dummy_args = argparse.Namespace(
        target_coverage=0.95,
        n_splits=1,
        test_frac=0.2,
        iters=1,
        final_iters=1,
        n_restarts=1,
        final_restarts=1,
        lr=0.01,
        rank=1,
        kernel='rbf',
        add_kernel='rbf',
        max_cg_iterations=1,
        cg_tolerance=0.001,
        plot_dpi=100,
        local_residual_q=2,
        uncertainty_combine_rule='max',
        policy_compare_rules='gp_only,max',
    )
    bad_rule_args = copy.copy(dummy_args)
    bad_rule_args.uncertainty_combine_rule = 'bad_rule'
    _assert_raises_value_error('bad uncertainty combine rule validation', lambda: _validate_training_args(bad_rule_args))
    bad_target_args = copy.copy(dummy_args)
    bad_target_args.target_coverage = 1.5
    _assert_raises_value_error('bad target coverage validation', lambda: _validate_training_args(bad_target_args))
    save_obj = {
        'oos_residual_features_std': np.zeros((3, 2), dtype=float),
        'oos_residual_strength': np.asarray([0.0, 1.0, 2.0]),
        'oos_residual_logperm': np.asarray([0.0, -1.0, -2.0]),
    }
    _assert_raises_value_error(
        'local residual feature-dimension mismatch',
        lambda: local_residual_uncertainty_from_standardized_features(save_obj, np.zeros((1, 3), dtype=float), q=2),
    )
    X_ref = np.asarray([[0.0], [1.0], [2.0]], dtype=float)
    loc = _local_residual_core(
        X_ref,
        X_ref,
        np.asarray([0.0, 1.0, 2.0]),
        np.asarray([0.0, -1.0, -2.0]),
        q=2,
        exclude_self=True,
    )
    for i, idx in enumerate(loc['nearest_oos_indices_internal']):
        if i in set(map(int, idx)):
            raise AssertionError('Self-check failed: local residual KNN included the self row.')
    loc_same = _local_residual_core(
        np.asarray([[0.0], [10.0]], dtype=float),
        np.asarray([[0.0], [1.0], [10.0], [11.0]], dtype=float),
        np.asarray([0.0, 1.0, 2.0, 3.0]),
        np.asarray([0.0, -1.0, -2.0, -3.0]),
        q=2,
        candidate_original_row_index=np.asarray([7, 8]),
        reference_original_row_index=np.asarray([7, 9, 8, 10]),
        exclude_same_original_row=True,
    )
    if 0 in set(map(int, loc_same['nearest_oos_indices_internal'][0])) or 2 in set(map(int, loc_same['nearest_oos_indices_internal'][1])):
        raise AssertionError('Self-check failed: same original row entered local residual neighbors.')
    pooled_audit = {
        'meta': {
            'split': np.asarray([0, 0, 1, 1], dtype=int),
            'original_row_index': np.asarray([0, 1, 0, 2], dtype=int),
            'X_oos_std': np.arange(4 * len(FEATURES), dtype=float).reshape(4, len(FEATURES)),
            'X_oos_model': np.arange(4 * len(FEATURES), dtype=float).reshape(4, len(FEATURES)),
        },
        'strength': {'y_true': np.asarray([1.0, 2.0, 1.2, 3.0]), 'y_pred': np.asarray([1.1, 1.8, 1.3, 2.8]), 'y_std': np.ones(4), 'y_std_raw': np.ones(4), 'residual': np.asarray([-0.1, 0.2, -0.1, 0.2])},
        'perm': {'y_true': np.asarray([0.0, 0.2, 0.1, 0.4]), 'y_pred': np.asarray([0.1, 0.1, 0.2, 0.3]), 'y_std': np.ones(4), 'y_std_raw': np.ones(4), 'residual': np.asarray([-0.1, 0.1, -0.1, 0.1])},
    }
    audit = calibration_audit_leave_one_split_out(pooled_audit, q_neighbors=2)
    n_cal_by_split = {d['held_out_split']: d['n_calibration_rows'] for d in audit['calibration_audit_per_split']}
    if n_cal_by_split.get(0) != 1 or n_cal_by_split.get(1) != 1:
        raise AssertionError('Self-check failed: calibration audit did not remove held-out original rows.')
    missing_local = _prediction_local_residual_lookup({}, np.zeros((2, 2), dtype=float), 2)
    if missing_local['status'] != 'missing_reference_fallback_to_gp_std' or missing_local['strength_local'].shape[0] != 2:
        raise AssertionError('Self-check failed: missing local residual reference did not fall back cleanly.')
    invalid_local = _prediction_local_residual_lookup(save_obj, np.zeros((2, 3), dtype=float), 2)
    if invalid_local['status'] != 'invalid_reference_fallback_to_gp_std' or invalid_local['strength_local'].shape[0] != 2:
        raise AssertionError('Self-check failed: invalid local residual reference did not fall back cleanly.')
    X_all = (np.arange(4 * len(FEATURES), dtype=float).reshape(4, len(FEATURES)) + 1.0)
    final_xsc = XStandardizer().fit(X_all)
    pooled = {'meta': {'X_oos_model': X_all[[0, 2]].copy()}}
    standardize_pooled_oos_features_with_final_scaler(pooled, final_xsc)
    expected = final_xsc.transform(X_all[[0, 2]].copy())
    if not np.allclose(pooled['meta']['X_oos_std'], expected):
        raise AssertionError('Self-check failed: OOS residual features were not standardized by the final full-data scaler.')
    bad_pooled = {'meta': {'X_oos_model': X_all[[0, 2]].copy(), 'X_oos_std': expected + 1.0}}
    _assert_raises_runtime_error('OOS final-scaler invariant mismatch', lambda: check_oos_feature_standardization_invariant(bad_pooled, final_xsc))
    paper_groups = np.asarray(['paper_a', 'paper_a', 'paper_b', 'paper_b', 'paper_b'], dtype=object)
    paper_splits = build_paperwise_splits(paper_groups)
    if len(paper_splits) != 2:
        raise AssertionError('Self-check failed: paperwise split count is wrong.')
    held_out_rows = np.concatenate([split['test_idx'] for split in paper_splits])
    if sorted(held_out_rows.tolist()) != list(range(len(paper_groups))):
        raise AssertionError('Self-check failed: each row must appear in exactly one paperwise test fold.')
    for split in paper_splits:
        if np.intersect1d(split['train_idx'], split['test_idx']).size:
            raise AssertionError('Self-check failed: paperwise train/test overlap.')
        held_out_sources = set(paper_groups[split['test_idx']])
        training_sources = set(paper_groups[split['train_idx']])
        if len(held_out_sources) != 1 or held_out_sources & training_sources:
            raise AssertionError('Self-check failed: held-out paper leaked into training.')
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', type=str, default='train', choices=['train', 'predict', 'selfcheck'])
    ap.add_argument('--model_path', type=str, default='')
    ap.add_argument('--predict_data', type=str, default='')
    ap.add_argument('--pred_out', type=str, default='')
    ap.add_argument('--target_coverage', type=float, default=0.95)
    ap.add_argument('--local_residual_q', type=int, default=LOCAL_RESIDUAL_K_NEIGHBORS)
    ap.add_argument('--uncertainty_combine_rule', type=str, default=UNCERTAINTY_COMBINE_RULE,
                    help='Final interval rule saved in the oracle: gp_only, local_only, mean, max, or sqrt. Default=max.')
    ap.add_argument('--policy_compare_rules', type=str, default='gp_only,mean,max,sqrt,local_only',
                    help='Comma-separated rules to compare in interval_policy_comparison.csv.')
    ap.add_argument('--data', type=str, default=DEFAULT_DATA_PATH)
    ap.add_argument('--outdir', type=str, default='lcmgp_without_head_out')
    ap.add_argument(
        '--split_mode',
        type=str,
        default='paperwise',
        choices=['paperwise', 'random'],
        help='paperwise performs leave-one-paper-out evaluation; random preserves the previous repeated row-split study.',
    )
    ap.add_argument(
        '--paper_source_column',
        type=str,
        default='',
        help="Exact paper-source column header. Empty means auto-detect values beginning with 'data source'.",
    )
    ap.add_argument(
        '--paperwise_papers',
        type=str,
        default='',
        help='Optional comma-separated paper labels, full source labels, or fold numbers. Empty runs every paper.',
    )
    ap.add_argument(
        '--paperwise_dry_run',
        type=int,
        default=0,
        help='Set to 1 to validate groups and write fold definitions without fitting models.',
    )
    ap.add_argument(
        '--skip_final_oracle',
        type=int,
        default=0,
        help='Set to 1 to stop after the validation study instead of fitting the final all-data deployment oracle.',
    )
    ap.add_argument('--deployment_seed', type=int, default=42)
    ap.add_argument('--n_splits', type=int, default=5)
    ap.add_argument('--test_frac', type=float, default=0.2)
    ap.add_argument('--base_seed', type=int, default=1000)
    ap.add_argument('--iters', type=int, default=700)
    ap.add_argument('--final_iters', type=int, default=900)
    ap.add_argument('--lr', type=float, default=0.02)
    ap.add_argument('--n_restarts', type=int, default=2)
    ap.add_argument('--final_restarts', type=int, default=2)
    ap.add_argument('--kernel', type=str, default='rbf+rq')
    ap.add_argument('--rank', type=int, default=1)
    ap.add_argument('--use_additive', type=int, default=1)
    ap.add_argument('--add_kernel', type=str, default='rbf')
    ap.add_argument('--restart_base_seed', type=int, default=12345)
    ap.add_argument('--max_cg_iterations', type=int, default=3000)
    ap.add_argument('--cg_tolerance', type=float, default=0.001)
    ap.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float64'])
    ap.add_argument('--fast_mode', type=int, default=1)
    ap.add_argument('--verbosity', type=int, default=1)
    ap.add_argument('--make_plots', type=int, default=1)
    ap.add_argument('--plot_dpi', type=int, default=300)
    ap.add_argument('--self_check', type=int, default=0)
    args, _ = ap.parse_known_args()
    if int(args.self_check) or args.mode == 'selfcheck':
        _run_internal_self_checks()
        print('Internal self-checks passed.', flush=True)
        return
    if args.mode == 'predict':
        if not args.model_path:
            raise ValueError('--model_path is required when --mode predict')
        if not args.predict_data:
            raise ValueError('--predict_data is required when --mode predict')
        if int(args.local_residual_q) < 2:
            raise ValueError('--local_residual_q must be >= 2.')
        if not (0.0 < float(args.target_coverage) < 1.0):
            raise ValueError('--target_coverage must be > 0 and < 1.')
        df_new = _read_table(args.predict_data)
        pred = predict_saved_oracle_dataframe(args.model_path, df_new, target_coverage=float(args.target_coverage), include_input=True, local_residual_q=int(args.local_residual_q))
        out_path = args.pred_out if args.pred_out else os.path.join(os.path.dirname(str(args.predict_data)) or '.', os.path.splitext(os.path.basename(str(args.predict_data)))[0] + '_lcmgp_predictions.csv')
        pred.to_csv(out_path, index=False)
        print(f'Saved calibrated predictions -> {out_path}', flush=True)
        print('Use strength_LB80/90/95 and perm_LB80/90/95_mm_s for lower-bound feasibility.', flush=True)
        return
    _validate_training_args(args)
    os.makedirs(args.outdir, exist_ok=True)
    global MAX_CG_ITERATIONS, CG_TOLERANCE
    MAX_CG_ITERATIONS = int(args.max_cg_iterations)
    CG_TOLERANCE = float(args.cg_tolerance)
    set_runtime_precision(args.dtype)
    set_seed(args.deployment_seed)
    t0 = time.time()
    fast_mode = bool(int(args.fast_mode))
    _p('START', 'LCMGP without-head engine', args.verbosity)
    _p('INFO', f"Device: {DEVICE} | DTYPE: {str(DTYPE).replace('torch.', '')} | fast_mode={int(fast_mode)}", args.verbosity)
    X_all, yS_all, yP_all, zero_removed, data_info = load_and_prepare_data(
        args.data,
        split_mode=args.split_mode,
        paper_source_column=args.paper_source_column,
        return_metadata=True,
    )
    _validate_training_args(args, n_rows=len(X_all))
    _p('DATA', f'n={len(X_all)} | zero_perm_removed={zero_removed}', args.verbosity)
    _p('DATA', f'strength mean={yS_all.mean():.3f} std={yS_all.std():.3f} | log_perm mean={yP_all.mean():.3f} std={yP_all.std():.3f}', args.verbosity)
    cfg = {'kernel': '+'.join(_kernel_components(args.kernel)), 'rank': int(args.rank), 'use_additive': bool(int(args.use_additive)), 'add_kernel': _validate_add_kernel_name(args.add_kernel)}
    row_metadata = data_info['row_metadata'].reset_index(drop=True)
    evaluation_splits = build_evaluation_splits(
        len(X_all),
        split_mode=args.split_mode,
        n_splits=args.n_splits,
        test_frac=args.test_frac,
        base_seed=args.base_seed,
        paper_groups=row_metadata['paper_source'].to_numpy(dtype=object),
        selected_papers=args.paperwise_papers,
    )
    if not evaluation_splits:
        raise ValueError('No evaluation folds were selected.')
    split_records = {}
    for split in evaluation_splits:
        key = f"split_{int(split['split']):03d}"
        split_records[key] = {
            'split': int(split['split']),
            'split_type': split['split_type'],
            'paper_source': split['paper_source'],
            'paper_label': split['paper_label'],
            'seed': split['seed'],
            'n_train': int(len(split['train_idx'])),
            'n_test': int(len(split['test_idx'])),
            'train_idx': split['train_idx'].astype(int).tolist(),
            'test_idx': split['test_idx'].astype(int).tolist(),
        }

    with open(os.path.join(args.outdir, 'evaluation_split_indices.json'), 'w') as f:
        json.dump(split_records, f, indent=2)
    if args.split_mode == 'paperwise':
        paper_counts = (
            row_metadata.groupby(['paper_source', 'paper_label'], sort=False)
            .size()
            .rename('n_rows')
            .reset_index()
        )
        selected_sources = {split['paper_source'] for split in evaluation_splits}
        paper_counts['selected_for_run'] = paper_counts['paper_source'].isin(selected_sources)
        paper_counts.to_csv(os.path.join(args.outdir, 'paper_counts.csv'), index=False)
        with open(os.path.join(args.outdir, 'paperwise_fold_indices.json'), 'w') as f:
            json.dump(split_records, f, indent=2)
        _p(
            'GROUPS',
            f"papers_total={paper_counts['paper_source'].nunique()} | folds_selected={len(evaluation_splits)} | "
            f"source_column={data_info['paper_source_column']} | I_gap_reconstructed={int(data_info['i_gap_reconstructed_from_gradation'])}",
            args.verbosity,
        )
    if int(args.paperwise_dry_run):
        dry_summary = {
            'status': 'dry_run_complete',
            'split_mode': args.split_mode,
            'data': os.path.abspath(args.data),
            'n_rows_raw': int(data_info['n_rows_raw']),
            'n_rows_used': int(data_info['n_rows_used']),
            'n_rows_dropped': int(data_info['n_rows_dropped']),
            'n_evaluation_folds': int(len(evaluation_splits)),
            'paper_source_column': data_info['paper_source_column'],
            'i_gap_reconstructed_from_gradation': bool(data_info['i_gap_reconstructed_from_gradation']),
        }
        with open(os.path.join(args.outdir, 'paperwise_dry_run_summary.json'), 'w') as f:
            json.dump(dry_summary, f, indent=2)
        print(json.dumps(dry_summary, indent=2), flush=True)
        return

    metric_rows, pred_rows = ([], [])
    paperwise_fold_dir = os.path.join(args.outdir, 'paperwise_folds')
    if args.split_mode == 'paperwise':
        os.makedirs(paperwise_fold_dir, exist_ok=True)
    _p(
        'EVAL',
        f'fixed config={cfg} | split_mode={args.split_mode} | folds={len(evaluation_splits)} | '
        f'iters={args.iters} | restarts={args.n_restarts}',
        args.verbosity,
    )
    for run_position, split in enumerate(evaluation_splits):
        i = int(split['split'])
        tr_idx = split['train_idx']
        te_idx = split['test_idx']
        fold_name = split['paper_label'] if args.split_mode == 'paperwise' else f'split {i + 1}'
        seed_text = '' if split['seed'] is None else f" | seed={split['seed']}"
        _p(
            'SPLIT',
            f"{run_position + 1}/{len(evaluation_splits)} started | held_out={fold_name} | "
            f"train={len(tr_idx)} | test={len(te_idx)}{seed_text}",
            args.verbosity,
        )
        m, p = eval_split(
            X_all,
            yS_all,
            yP_all,
            tr_idx,
            te_idx,
            cfg,
            int(args.iters),
            float(args.lr),
            int(args.n_restarts),
            int(args.restart_base_seed + 100000 * i),
            fast_mode,
            verbosity=max(0, args.verbosity - 1),
            label=f'held out {fold_name}',
        )
        p['meta']['split'] = np.full(len(te_idx), i, dtype=int)
        p['meta']['split_seed'] = np.full(len(te_idx), -1 if split['seed'] is None else int(split['seed']), dtype=int)
        test_metadata = row_metadata.iloc[te_idx].reset_index(drop=True)
        for key in ['paper_source', 'paper_label', 'mix_id']:
            p['meta'][key] = test_metadata[key].astype(str).to_numpy(dtype=object)
        p['meta']['source_excel_row'] = test_metadata['source_excel_row'].to_numpy(dtype=int)
        metric_row = {
            'split': i,
            'split_type': split['split_type'],
            'paper_source': split['paper_source'],
            'paper_label': split['paper_label'],
            'n_train': int(len(tr_idx)),
            'n_test': int(len(te_idx)),
            'test_fraction': float(len(te_idx) / len(X_all)),
            **m,
        }
        metric_rows.append(metric_row)
        pred_rows.append(p)
        if args.split_mode == 'paperwise':
            fold_token = f"{i:03d}_{_safe_filename_token(split['paper_label'])}"
            fold_predictions = test_metadata.copy()
            fold_predictions['strength_true_mpa'] = np.asarray(p['strength']['y_true'], dtype=float)
            fold_predictions['strength_pred_mpa'] = np.asarray(p['strength']['y_pred'], dtype=float)
            fold_predictions['strength_std_raw_mpa'] = np.asarray(p['strength']['y_std_raw'], dtype=float)
            fold_predictions['strength_residual_mpa'] = np.asarray(p['strength']['residual'], dtype=float)
            fold_predictions['logperm_true'] = np.asarray(p['perm']['y_true'], dtype=float)
            fold_predictions['logperm_pred'] = np.asarray(p['perm']['y_pred'], dtype=float)
            fold_predictions['logperm_std_raw'] = np.asarray(p['perm']['y_std_raw'], dtype=float)
            fold_predictions['logperm_residual'] = np.asarray(p['perm']['residual'], dtype=float)
            fold_predictions['perm_true_mm_s'] = np.exp(fold_predictions['logperm_true'])
            fold_predictions['perm_pred_mm_s'] = np.exp(fold_predictions['logperm_pred'])
            fold_predictions.to_csv(os.path.join(paperwise_fold_dir, fold_token + '_predictions.csv'), index=False)
            with open(os.path.join(paperwise_fold_dir, fold_token + '_metrics.json'), 'w') as f:
                json.dump(metric_row, f, indent=2)
        pd.DataFrame(metric_rows).to_csv(os.path.join(args.outdir, 'evaluation_metrics_in_progress.csv'), index=False)
        _p(
            'SPLIT',
            f"{run_position + 1}/{len(evaluation_splits)} done | held_out={fold_name} | "
            f"rmse_strength={m['rmse_strength']:.4f} r2_strength={m['r2_strength']:.4f} | "
            f"rmse_logperm={m['rmse_perm']:.4f} r2_logperm={m['r2_perm']:.4f} | "
            f"elapsed={(time.time() - t0) / 60:.1f}min",
            args.verbosity,
        )
    summary = summarize_metrics(
        metric_rows,
        cfg,
        args.test_frac if args.split_mode == 'random' else float(np.mean([row['test_fraction'] for row in metric_rows])),
    )
    summary['split_mode'] = str(args.split_mode)
    summary['evaluation_protocol'] = (
        'leave_one_paper_out_train_all_other_papers'
        if args.split_mode == 'paperwise'
        else 'repeated_random_row_splits'
    )
    summary['n_evaluation_folds'] = int(len(evaluation_splits))
    summary['n_rows_raw'] = int(data_info['n_rows_raw'])
    summary['n_rows_used'] = int(data_info['n_rows_used'])
    summary['n_rows_dropped'] = int(data_info['n_rows_dropped'])
    summary['paper_source_column'] = data_info['paper_source_column']
    summary['i_gap_reconstructed_from_gradation'] = bool(data_info['i_gap_reconstructed_from_gradation'])
    if args.split_mode == 'paperwise':
        summary['n_papers_total'] = int(row_metadata['paper_source'].nunique())
        summary['n_papers_evaluated'] = int(len(evaluation_splits))
    pooled = concat_pooled(pred_rows)
    Yall = np.stack([yS_all, yP_all], 1)
    xsc = XStandardizer().fit(X_all)
    ysc = YStandardizer().fit(Yall)
    pooled = standardize_pooled_oos_features_with_final_scaler(pooled, xsc)
    oos_invariant_error = check_oos_feature_standardization_invariant(pooled, xsc, tol=1e-8)
    summary.update(grouped_oos_diagnostics_by_original_row(pooled))
    summary['pooled_oos_rmse_strength'] = rmse(pooled['strength']['y_true'], pooled['strength']['y_pred'])
    summary['pooled_oos_mae_strength'] = mae(pooled['strength']['y_true'], pooled['strength']['y_pred'])
    summary['pooled_oos_r2_strength'] = r2_score(pooled['strength']['y_true'], pooled['strength']['y_pred'])
    summary['pooled_oos_rmse_logperm'] = rmse(pooled['perm']['y_true'], pooled['perm']['y_pred'])
    summary['pooled_oos_mae_logperm'] = mae(pooled['perm']['y_true'], pooled['perm']['y_pred'])
    summary['pooled_oos_r2_logperm'] = r2_score(pooled['perm']['y_true'], pooled['perm']['y_pred'])
    summary['pooled_oos_rmse_perm_mm_s'] = rmse(np.exp(pooled['perm']['y_true']), np.exp(pooled['perm']['y_pred']))
    summary['pooled_oos_mae_perm_mm_s'] = mae(np.exp(pooled['perm']['y_true']), np.exp(pooled['perm']['y_pred']))
    summary['pooled_oos_r2_perm_mm_s'] = r2_score(np.exp(pooled['perm']['y_true']), np.exp(pooled['perm']['y_pred']))
    summary['metric_scale_note'] = (
        'Strength metrics use MPa. rmse_logperm/mae_logperm/r2_logperm use log(mm/s). '
        'Metrics ending in _perm_mm_s use permeability in the original mm/s scale.'
    )
    selected_rule = _clean_interval_rule(args.uncertainty_combine_rule)
    compare_rules = args.policy_compare_rules
    calibration_bundle, calib_diag = build_calibration_bundle(
        pooled,
        q_neighbors=int(args.local_residual_q),
        combine_rule=selected_rule,
    )
    calibration_fit_scope = (
        'pooled_leave_one_paper_out_oos_predictions'
        if args.split_mode == 'paperwise'
        else 'pooled_repeated_random_split_oos_predictions'
    )
    calibration_bundle['calibration_fit_scope'] = calibration_fit_scope
    calibration_bundle['evaluation_split_mode'] = str(args.split_mode)
    summary['deployment_calibration_fit_scope'] = calibration_fit_scope
    summary['deployment_calibration_independent_validation_estimate'] = False
    summary['point_prediction_external_paper_validation'] = bool(args.split_mode == 'paperwise')
    summary['local_residual_calibration_status'] = calibration_bundle.get('diagnostics', {}).get('local_residual_status', 'unknown')
    summary['local_residual_calibration_warning'] = calibration_bundle.get('diagnostics', {}).get('local_residual_warning', '')
    summary['local_residual_exclude_same_original_row'] = calibration_bundle.get('diagnostics', {}).get('exclude_same_original_row_for_local_residual', False)
    Xmodel_pool = np.asarray(pooled['meta'].get('X_oos_model', pooled['meta']['X_oos_std']), dtype=float)
    mean_pool = np.stack([np.asarray(pooled['strength']['y_pred'], dtype=float), np.asarray(pooled['perm']['y_pred'], dtype=float)], axis=1)
    legacy_pooled_aliases = {}
    for task, store_key, std_key in [('strength', 'strength', 'strength_std_effective'), ('perm', 'perm', 'logperm_std_effective')]:
        y = np.asarray(pooled[store_key]['y_true'], dtype=float)
        mu = np.asarray(pooled[store_key]['y_pred'], dtype=float)
        std = np.asarray(calib_diag[std_key], dtype=float)
        for level in calibration_bundle.get('levels', CALIBRATION_LEVELS):
            lab = int(round(100 * float(level)))
            q2, _, _ = _select_calibration_q(calibration_bundle, task, 'two_sided', float(level), Xmodel_pool, mean_pool)
            ql, _, _ = _select_calibration_q(calibration_bundle, task, 'lower', float(level), Xmodel_pool, mean_pool)
            coverage_key = f'pooled_calibration_fit_diagnostic_coverage_{lab}_{task}_two_sided'
            width_key = f'pooled_calibration_fit_diagnostic_width_{lab}_{task}_two_sided_mean'
            lower_key = f'pooled_calibration_fit_diagnostic_lower_coverage_{lab}_{task}'
            summary[coverage_key] = _coverage_from_q(y, mu, std, q2)
            summary[width_key] = float(np.nanmean(2.0 * q2 * std))
            summary[lower_key] = _lower_coverage_from_q(y, mu, std, ql)
            legacy_pooled_aliases[f'pooled_coverage_{lab}_{task}_two_sided'] = coverage_key
            legacy_pooled_aliases[f'pooled_width_{lab}_{task}_two_sided_mean'] = width_key
            legacy_pooled_aliases[f'pooled_lower_coverage_{lab}_{task}'] = lower_key
    summary['legacy_pooled_calibrated_coverage_key_aliases'] = legacy_pooled_aliases
    summary.update(calibration_audit_leave_one_split_out(
        pooled,
        levels=calibration_bundle.get('levels', CALIBRATION_LEVELS),
        q_neighbors=int(args.local_residual_q),
        combine_rule=selected_rule,
    ))
    summary['pooled_calibrated_coverage_note'] = (
        'Pooled calibrated coverage is fitted and evaluated on the same leave-one-paper-out OOS pool; '
        'use calibration_audit_* fields for the held-out-fold calibration audit estimate.'
        if args.split_mode == 'paperwise'
        else
        'Pooled calibrated coverage is fitted and evaluated on the same repeated-random-split OOS pool; '
        'use calibration_audit_* fields for the split-held-out audit estimate.'
    )
    summary['pooled_calibrated_coverage_is_independent_validation'] = False
    summary['calibration_audit_is_external_validation'] = False
    summary['calibration_audit_note'] = (
        'Leave-one-paper-fold-out calibration audit using existing paperwise OOS predictions; no retraining.'
        if args.split_mode == 'paperwise'
        else CALIBRATION_AUDIT_NOTE
    )
    summary['oos_dependence_note'] = (
        'Each usable row appears once in OOS predictions, from the fold where its entire paper was held out.'
        if args.split_mode == 'paperwise'
        else
        'OOS rows may repeat the same original_row_index across random splits; grouped_oos_* metrics '
        'average predictions by original row before scoring.'
    )
    summary['std_column_semantics_note'] = (
        'Prediction std columns named strength_std_calibrated_mpa/logperm_std_calibrated are effective base '
        'uncertainties after raw/local/AD combination. Final empirical calibration is applied through q multipliers '
        'in PI/LB columns; use PI/LB columns for calibrated intervals.'
    )
    summary['sigma_floor_strength'] = float(SIGMA_FLOOR_S)
    summary['sigma_floor_logperm'] = float(SIGMA_FLOOR_P)
    summary['uncertainty_combine_rule'] = str(selected_rule)
    summary['uncertainty_combine_rule_plain_english'] = _policy_explanation(selected_rule)
    summary['policy_compare_rules'] = _parse_policy_rules(compare_rules)
    summary['local_residual_q_neighbors'] = int(calibration_bundle.get('local_residual_q_neighbors', args.local_residual_q))
    policy_df = make_interval_policy_comparison(
        pooled,
        rules=compare_rules,
        levels=calibration_bundle.get('levels', CALIBRATION_LEVELS),
        q_neighbors=int(args.local_residual_q),
        regime_min_n=REGIME_CALIBRATION_MIN_N,
    )
    policy_csv_path = os.path.join(args.outdir, 'interval_policy_comparison.csv')
    policy_df.to_csv(policy_csv_path, index=False)
    interval_readme_path = _write_plain_english_interval_report(args.outdir, selected_rule, policy_df)
    summary['interval_policy_comparison_file'] = os.path.basename(policy_csv_path)
    summary['plain_english_interval_report_file'] = os.path.basename(interval_readme_path)
    summary['interval_policy_comparison_scope'] = 'fitted_on_same_pooled_oos_diagnostic_not_independent_validation'
    with open(os.path.join(args.outdir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(metric_rows, f, indent=2)
    with open(os.path.join(args.outdir, 'calibration_bundle.json'), 'w') as f:
        json.dump(array_to_list_obj(calibration_bundle), f, indent=2)
    pd.DataFrame(metric_rows).to_csv(os.path.join(args.outdir, 'evaluation_metrics.csv'), index=False)
    if args.split_mode == 'paperwise':
        with open(os.path.join(args.outdir, 'paperwise_fold_metrics.json'), 'w') as f:
            json.dump(metric_rows, f, indent=2)
        pd.DataFrame(metric_rows).to_csv(os.path.join(args.outdir, 'paperwise_fold_metrics.csv'), index=False)
    else:
        with open(os.path.join(args.outdir, 'repeated_splits_indices.json'), 'w') as f:
            json.dump(split_records, f, indent=2)
        with open(os.path.join(args.outdir, 'repeated_split_metrics.json'), 'w') as f:
            json.dump(metric_rows, f, indent=2)
        pd.DataFrame(metric_rows).to_csv(os.path.join(args.outdir, 'split_metrics.csv'), index=False)
    oos_residual_df = make_pooled_oos_residual_dataframe(pooled, calib_diag=calib_diag)
    oos_residual_csv_path = os.path.join(args.outdir, OOS_RESIDUAL_FEATURE_FILE)
    oos_residual_df.to_csv(oos_residual_csv_path, index=False)
    conditional_df = _make_conditional_coverage_table(pooled, calib_diag, calibration_bundle)
    conditional_df.to_csv(os.path.join(args.outdir, 'conditional_coverage_by_regime.csv'), index=False)
    rows_out = {
        'strength_true': pooled['strength']['y_true'],
        'strength_pred': pooled['strength']['y_pred'],
        'strength_std_raw': calib_diag['strength_std_raw'],
        'strength_std_effective': calib_diag['strength_std_effective'],
        'logperm_true': pooled['perm']['y_true'],
        'logperm_pred': pooled['perm']['y_pred'],
        'logperm_std_raw': calib_diag['logperm_std_raw'],
        'logperm_std_effective': calib_diag['logperm_std_effective'],
        'perm_true': np.exp(pooled['perm']['y_true']),
        'perm_pred': np.exp(pooled['perm']['y_pred']),
    }
    for key in ['split', 'split_seed', 'original_row_index', 'paper_source', 'paper_label', 'mix_id', 'source_excel_row']:
        if key in pooled.get('meta', {}):
            rows_out[key] = pooled['meta'][key]
    for level in calibration_bundle.get('levels', CALIBRATION_LEVELS):
        lab = int(round(100 * float(level)))
        q2s, _, _ = _select_calibration_q(calibration_bundle, 'strength', 'two_sided', float(level), Xmodel_pool, mean_pool)
        qls, _, _ = _select_calibration_q(calibration_bundle, 'strength', 'lower', float(level), Xmodel_pool, mean_pool)
        q2p, _, _ = _select_calibration_q(calibration_bundle, 'perm', 'two_sided', float(level), Xmodel_pool, mean_pool)
        qlp, _, _ = _select_calibration_q(calibration_bundle, 'perm', 'lower', float(level), Xmodel_pool, mean_pool)
        rows_out[f'strength_{lab}_lower'] = pooled['strength']['y_pred'] - q2s * calib_diag['strength_std_effective']
        rows_out[f'strength_{lab}_upper'] = pooled['strength']['y_pred'] + q2s * calib_diag['strength_std_effective']
        rows_out[f'strength_LB{lab}'] = pooled['strength']['y_pred'] - qls * calib_diag['strength_std_effective']
        rows_out[f'logperm_{lab}_lower'] = pooled['perm']['y_pred'] - q2p * calib_diag['logperm_std_effective']
        rows_out[f'logperm_{lab}_upper'] = pooled['perm']['y_pred'] + q2p * calib_diag['logperm_std_effective']
        rows_out[f'logperm_LB{lab}'] = pooled['perm']['y_pred'] - qlp * calib_diag['logperm_std_effective']
        rows_out[f'perm_{lab}_lower'] = np.exp(rows_out[f'logperm_{lab}_lower'])
        rows_out[f'perm_{lab}_upper'] = np.exp(rows_out[f'logperm_{lab}_upper'])
        rows_out[f'perm_LB{lab}'] = np.exp(rows_out[f'logperm_LB{lab}'])
        rows_out[f'perm_LB{lab}_mm_s'] = rows_out[f'perm_LB{lab}']
    pd.DataFrame(rows_out).to_csv(os.path.join(args.outdir, 'pooled_oos_predictions.csv'), index=False)
    summary['oos_residual_feature_file'] = OOS_RESIDUAL_FEATURE_FILE
    summary['oos_residual_feature_rows'] = int(len(oos_residual_df))
    summary['oos_residual_feature_columns'] = list(oos_residual_df.columns)
    summary['oos_residual_feature_coordinate_system'] = 'final_full_data_standardized_transformed_features'
    summary['oos_residual_feature_standardization_invariant_max_abs_error'] = float(oos_invariant_error)
    summary['oos_residual_feature_standardization_note'] = OOS_STANDARDIZATION_NOTE
    with open(os.path.join(args.outdir, 'summary_results.json'), 'w') as f:
        json.dump(array_to_list_obj(summary), f, indent=2)
    _p('SAVE', f'OOS residual-feature table saved -> {oos_residual_csv_path}', args.verbosity)
    if int(args.make_plots):
        plot_dir = make_validation_plots(args.outdir, pooled, calibration_bundle, calib_diag, dpi=int(args.plot_dpi))
        _p('PLOTS', f'saved -> {plot_dir}', args.verbosity)
    _p('SUMMARY', json.dumps(array_to_list_obj(summary), indent=2), args.verbosity)
    if int(args.skip_final_oracle):
        _p('DONE', f'validation study complete; final oracle skipped | total runtime={(time.time() - t0) / 60:.1f}min', args.verbosity)
        return
    _p('FINAL', f'training full-data oracle | iters={args.final_iters} | restarts={args.final_restarts}', args.verbosity)
    Xall_s = xsc.transform(X_all)
    Yall_s = ysc.transform(Yall)
    Xt = torch.as_tensor(Xall_s, dtype=DTYPE, device=DEVICE).contiguous()
    Yt = torch.as_tensor(Yall_s, dtype=DTYPE, device=DEVICE).contiguous()
    model, lik = train_lcmgp(Xt, Yt, cfg, int(args.final_iters), float(args.lr), int(args.final_restarts), int(args.restart_base_seed + 999999), fast_mode=fast_mode, verbosity=args.verbosity, label='full data')
    oos_feature_cols = [c for c in oos_residual_df.columns if c.startswith('xstd_')]
    oos_model_cols = [c for c in oos_residual_df.columns if c.startswith('xmodel_')]
    ad_reference = _compute_ad_reference(Xall_s, k=AD_K_NEIGHBORS, threshold_quantile=AD_THRESHOLD_QUANTILE)
    save_obj = {'model_state_dict': model.state_dict(), 'likelihood_state_dict': lik.state_dict(), 'train_x': Xt.detach().cpu(), 'train_y': Yt.detach().cpu(), 'features': FEATURES, 'cont_idxs': CONT_IDXS, 'x_cont_mean': xsc.mean, 'x_cont_std': xsc.std, 'y_mean': ysc.mean, 'y_std': ysc.std, 'transform_recipe': TRANSFORM_RECIPE, 'best_config': cfg, 'sigma_floor_s': SIGMA_FLOOR_S, 'sigma_floor_p': SIGMA_FLOOR_P, 'calibration_bundle': calibration_bundle, 'deployment_seed': int(args.deployment_seed), 'n_total': int(len(X_all)), 'dtype': str(DTYPE).replace('torch.', ''), 'fast_mode': fast_mode, 'summary_results': summary, 'prediction_contract_version': 'empirical_local_regime_ad_v2_interval_policy_diagnostics', 'official_prediction_function': 'predict_saved_oracle_dataframe', 'official_local_residual_function': 'local_residual_uncertainty_from_standardized_features', 'uncertainty_combine_rule': selected_rule, 'ad_reference': ad_reference, 'ad_inflation_start': AD_INFLATION_START, 'ad_inflation_eta': AD_INFLATION_ETA, 'ad_inflation_cap': AD_INFLATION_CAP, 'official_prediction_outputs': ['strength_pred_mpa', 'strength_LB80', 'strength_LB90', 'strength_LB95', 'logperm_pred', 'logperm_LB80', 'logperm_LB90', 'logperm_LB95', 'perm_pred_mm_s', 'perm_LB80_mm_s', 'perm_LB90_mm_s', 'perm_LB95_mm_s', 'strength_std_effective_mpa', 'logperm_std_effective', 'D_AD', 'AD_ratio', 'inside_AD'], 'oos_residual_reference_available': True, 'oos_residual_feature_file': OOS_RESIDUAL_FEATURE_FILE, 'oos_residual_feature_columns': oos_feature_cols, 'oos_residual_model_feature_columns': oos_model_cols, 'oos_residual_features_std': oos_residual_df[oos_feature_cols].to_numpy(dtype=float), 'oos_residual_features_model': oos_residual_df[oos_model_cols].to_numpy(dtype=float), 'oos_residual_original_row_index': oos_residual_df['original_row_index'].to_numpy(dtype=int), 'oos_residual_split': oos_residual_df['split'].to_numpy(dtype=int), 'oos_residual_split_seed': oos_residual_df['split_seed'].to_numpy(dtype=int), 'oos_residual_strength': oos_residual_df['strength_residual_mpa'].to_numpy(dtype=float), 'oos_residual_logperm': oos_residual_df['logperm_residual'].to_numpy(dtype=float), 'oos_residual_strength_true': oos_residual_df['strength_true_mpa'].to_numpy(dtype=float), 'oos_residual_strength_pred': oos_residual_df['strength_pred_mpa'].to_numpy(dtype=float), 'oos_residual_logperm_true': oos_residual_df['logperm_true'].to_numpy(dtype=float), 'oos_residual_logperm_pred': oos_residual_df['logperm_pred'].to_numpy(dtype=float), 'local_residual_default_q_neighbors': max(2, int(args.local_residual_q)), 'raw_uncertainty_warning': 'Raw GP standard deviations are diagnostic only. Use empirical OOS calibrated intervals, lower bounds, local residual uncertainty, and AD diagnostics from predict_saved_oracle_dataframe.'}
    save_obj['official_prediction_outputs'] = [
        'strength_pred_mpa', 'strength_std_calibrated_mpa',
        'strength_pi_lower_mpa', 'strength_pi_upper_mpa',
        'logperm_pred', 'logperm_std_calibrated',
        'logperm_pi_lower', 'logperm_pi_upper',
        'perm_pred_mm_s', 'perm_pi_lower_mm_s', 'perm_pi_upper_mm_s',
        'strength_LB80', 'strength_LB90', 'strength_LB95',
        'logperm_LB80', 'logperm_LB90', 'logperm_LB95',
        'perm_LB80_mm_s', 'perm_LB90_mm_s', 'perm_LB95_mm_s',
        'strength_std_effective_mpa', 'logperm_std_effective',
        'strength_std_raw_diagnostic_mpa', 'logperm_std_raw_diagnostic',
        'strength_residual_std_local_mpa', 'logperm_residual_std_local',
        'local_residual_status', 'local_residual_warning', 'local_residual_q_neighbors_used',
        'strength_lower_bound_80_mpa', 'strength_lower_bound_90_mpa', 'strength_lower_bound_95_mpa',
        'logperm_lower_bound_80', 'logperm_lower_bound_90', 'logperm_lower_bound_95',
        'perm_lower_bound_80_mm_s', 'perm_lower_bound_90_mm_s', 'perm_lower_bound_95_mm_s',
        'D_AD', 'AD_ratio', 'inside_AD',
    ]
    save_obj['oos_residual_feature_coordinate_system'] = 'final_full_data_standardized_transformed_features'
    save_obj['std_column_semantics'] = (
        STD_UNCERTAINTY_NOTE
    )
    save_obj['oos_residual_feature_standardization_invariant_max_abs_error'] = float(oos_invariant_error)
    save_obj['oos_residual_feature_standardization_note'] = OOS_STANDARDIZATION_NOTE
    save_obj['evaluation_split_mode'] = str(args.split_mode)
    save_obj['evaluation_protocol'] = summary.get('evaluation_protocol', '')
    if 'paper_source' in oos_residual_df.columns:
        save_obj['oos_residual_paper_source'] = oos_residual_df['paper_source'].astype(str).to_numpy(dtype=object)
    if 'paper_label' in oos_residual_df.columns:
        save_obj['oos_residual_paper_label'] = oos_residual_df['paper_label'].astype(str).to_numpy(dtype=object)
    save_obj['pooled_calibrated_coverage_is_independent_validation'] = False
    save_obj['calibration_audit_is_external_validation'] = False
    save_obj['calibration_audit_note'] = summary.get('calibration_audit_note', CALIBRATION_AUDIT_NOTE)
    pt_path = os.path.join(args.outdir, 'lcmgp_without_head_oracle_full_data.pt')
    torch.save(save_obj, pt_path)
    contract_path, readme_path = write_prediction_contract(args.outdir, pt_path, summary, calibration_bundle)
    _p('SAVE', f'oracle saved -> {pt_path}', args.verbosity)
    _p('SAVE', f'prediction contract saved -> {contract_path}', args.verbosity)
    _p('SAVE', f'prediction instructions saved -> {readme_path}', args.verbosity)
    _p('DONE', f'total runtime={(time.time() - t0) / 60:.1f}min', args.verbosity)
if __name__ == '__main__':
    main()
