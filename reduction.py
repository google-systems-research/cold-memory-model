# Copyright 2025 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import logging
import math
import time
from scipy.optimize import minimize, Bounds, curve_fit
from ortools.sat.python import cp_model
from skimage.feature import peak_local_max
from tqdm import tqdm
import plotting
import file_io

# Access count reduction
def reconstruct_counts_from_intervals(interval_edges, interval_sums, interval_weighted_sums, time_limit=300, tol=0.01):
    """
    Reconstructs counts from interval sums and weighted sums using a CP-SAT model.

    Args:
        interval_edges (list): The edges of the intervals.
        interval_sums (list): The sum of counts in each interval.
        interval_weighted_sums (list): The weighted sum of counts in each interval.
        time_limit (int): The time limit for the solver.
        tol (float): The tolerance for the constraints.

    Returns:
        dict: A dictionary mapping each value to its reconstructed count.
    """
    model = cp_model.CpModel()
    all_x = [x for a, b in zip(interval_edges[:-1], interval_edges[1:]) for x in range(a, b)]
    UB = max(interval_sums) * 2
    n = {x: model.NewIntVar(0, UB, f"n{x}") for x in all_x}
    q = {}

    idx_of = {x: i for i, (a, b) in enumerate(zip(interval_edges[:-1], interval_edges[1:]))
              for x in range(a, b)}

    for i in range(len(interval_sums)):
        xs = [x for x in all_x if idx_of[x] == i]
        model.Add(sum(n[x] for x in xs) >= int((1 - tol) * interval_sums[i]))
        model.Add(sum(n[x] for x in xs) <= int((1 + tol) * interval_sums[i]))
        model.Add(sum(x * n[x] for x in xs) >= int((1 - tol) * interval_weighted_sums[i]))
        model.Add(sum(x * n[x] for x in xs) <= int((1 + tol) * interval_weighted_sums[i]))

    for x in all_x[:-1]:
        diff = model.NewIntVar(-UB, UB, f"diff_{x}")
        model.Add(diff == n[x] - n[x + 1])
        q[x] = model.NewIntVar(0, UB * UB, f"q_{x}")
        model.AddMultiplicationEquality(q[x], [diff, diff])

    model.Minimize(sum(q.values()))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible solution")

    return {x: int(solver.Value(n[x])) for x in all_x if solver.Value(n[x]) > 0}

def _fit_log_distribution(x_arr, y_arr, end, num_intervals=5, bypass_solver=False):
    """
    Fits a log distribution to the data.

    Args:
        x_arr (np.array): The x-values of the distribution.
        y_arr (np.array): The y-values of the distribution.
        end (int): The end value for the bins.
        num_intervals (int): The number of intervals to create.
        bypass_solver (bool): If True, skip the CP-SAT solver and return the
            original data as-is. Interval sums/weighted sums are still computed
            for access_params (used in JSON export and synthesis).

    Returns:
        tuple: A tuple containing the edges, interval sums, interval weighted sums,
               and the reconstructed distribution.
    """
    edges = [1, 2, 10, 50, 100, 1000, end]
    num_bins = len(edges) - 1
    interval_sums = np.zeros(num_bins, dtype=int)
    interval_weighted_sums = np.zeros(num_bins, dtype=int)

    idx = np.digitize(x_arr, edges[:-1], right=False) - 1

    for i in range(num_bins):
        mask = idx == i
        interval_sums[i] = y_arr[mask].sum()
        interval_weighted_sums[i] = (x_arr[mask] * y_arr[mask]).sum()

    if bypass_solver:
        return edges, interval_sums, interval_weighted_sums, x_arr, y_arr

    dist = reconstruct_counts_from_intervals(edges, interval_sums.tolist(), interval_weighted_sums.tolist())

    x_dist = np.array(sorted(dist))
    y_dist = np.array([dist[x] for x in x_dist])

    return edges, interval_sums, interval_weighted_sums, x_dist, y_dist

# Markov matrix reduction
(s1_init, s2_init, s3_init, s4_init, s1_diag_init, s2_diag_init, s3_diag_init, s4_diag_init) = 3, 0.2, 5, 3, 3, 0.2, 2, 2
def _find_peaks(line):
    """
    Finds peaks in a 1D array.

    Args:
        line (np.array): The 1D array to find peaks in.

    Returns:
        list: A list of peak indices.
    """
    peaks = []
    for i in range(1, len(line) - 1):
        if line[i] > line[i - 1] and line[i] > line[i + 1]:
            peaks.append(i)
    # detect edge peaks as well
    if line[0] > line[1]:
        peaks.append(0)
    if line[-1] > line[-2]:
        peaks.append(len(line) - 1)
    return peaks

def _elliptical_gaussian(x, y, cx, cy, amp, sigx, sigy, theta):
    """
    Calculates a 2D elliptical Gaussian.

    Args:
        x (np.array): The x-coordinates.
        y (np.array): The y-coordinates.
        cx (float): The x-coordinate of the center.
        cy (float): The y-coordinate of the center.
        amp (float): The amplitude.
        sigx (float): The standard deviation in the x-direction.
        sigy (float): The standard deviation in the y-direction.
        theta (float): The rotation angle.

    Returns:
        np.array: The calculated Gaussian.
    """
    xr = x - cx
    yr = y - cy
    ct = np.cos(theta)
    st = np.sin(theta)

    x_rot = ct * xr + st * yr
    y_rot = -st * xr + ct * yr

    expo = np.exp(-(x_rot ** 2) / (2 * sigx ** 2) - (y_rot ** 2) / (2 * sigy ** 2))
    return amp * expo

def _model_markov_matrix(params, peaks, N):
    """
    Models the Markov matrix using a sum of elliptical Gaussians.

    Args:
        params (list): The parameters of the Gaussians.
        peaks (list): The list of peaks.
        N (int): The size of the matrix.

    Returns:
        np.array: The modeled Markov matrix.
    """
    K = len(peaks)
    s1, s2, s3, s4, s1_diag, s2_diag, s3_diag, s4_diag = params[0:8]
    peak_params = params[8:]
    assert len(peak_params) == 2 * K

    xx, yy = np.meshgrid(np.arange(N), np.arange(N))
    M = np.zeros((N, N), dtype=float)

    for i in range(K):
        ptype, rr, cc, theta = peaks[i]

        x0 = float(rr)
        y0 = float(cc)
        A1 = peak_params[2 * i]
        A2 = peak_params[2 * i + 1]

        if ptype == 'col':
            G1 = _elliptical_gaussian(xx, yy, x0, y0, A1, s1, s2, 0)
            G2 = _elliptical_gaussian(xx, yy, x0, y0, A2, s3, s4, 0)
        elif (rr == 0) or (rr == N - 1):
            G1 = _elliptical_gaussian(xx, yy, x0, y0, A1, s1_diag, s2_diag, theta)
            G2 = _elliptical_gaussian(xx, yy, x0, y0, A2, s1, s2, 0)
        else:
            G1 = _elliptical_gaussian(xx, yy, x0, y0, A1, s1_diag, s2_diag, theta)
            G2 = _elliptical_gaussian(xx, yy, x0, y0, A2, s3_diag, s4_diag, 0)

        M += G1 + G2
    for i in range(N):
        if (M[:, i].sum()):
            M[:, i] = M[:, i] / M[:, i].sum()
    return M

def _calculate_markov_matrix_objective(params, original_matrix, original_col_sums, peak_list, alpha, N):
    """
    Calculates the objective function for the Markov matrix optimization.

    Args:
        params (list): The parameters of the Gaussians.
        original_matrix (np.array): The original Markov matrix.
        original_col_sums (np.array): The original column sums of the Markov matrix.
        peak_list (list): The list of peaks.
        alpha (float): The weight of the peak penalty.
        N (int): The size of the matrix.

    Returns:
        float: The objective value.
    """
    M = _model_markov_matrix(params, peak_list, N)
    diff = M - original_matrix
    sq_error = np.sum(diff ** 2)

    current_col_sums = M.sum(axis=1)
    col_diff = current_col_sums - original_col_sums
    col_penalty = np.sum(col_diff ** 2)

    peak_penalty = 0
    for ptype, rr, cc, theta in peak_list:
        peak_penalty += (M[rr, cc] - original_matrix[rr, cc]) ** 2

    return sq_error + alpha * (peak_penalty)

def _build_markov_matrix_initial_guess(peak_list, original_matrix):
    """
    Builds the initial guess for the Markov matrix optimization.

    Args:
        peak_list (list): The list of peaks.
        original_matrix (np.array): The original Markov matrix.

    Returns:
        np.array: The initial guess for the parameters.
    """
    guess_list = [s1_init, s2_init, s3_init, s4_init, s1_diag_init, s2_diag_init, s3_diag_init, s4_diag_init]
    for (ptype, x, y, theta) in peak_list:
        val = original_matrix[x, y]
        if x == y and (x == 0 or x == 11):
            A1_init = val * 0.5
            A2_init = val * 0.5
        else:
            A1_init = val * 0.9
            A2_init = val * 0.1
        guess_list += [A1_init, A2_init]
    return np.array(guess_list, dtype=float)

# Bit Flip Matrix reduction
(s1_init_bfr, s2_init_bfr, s3_init_bfr, s4_init_bfr) = 1, 1, 5, 5

def _model_bit_flip_matrix(params, peak_coords, N, M):
    """
    Models the bit flip matrix using a sum of elliptical Gaussians.

    Args:
        params (list): The parameters of the Gaussians.
        peak_coords (list): The coordinates of the peaks.
        N (int): The number of rows.
        M (int): The number of columns.

    Returns:
        np.array: The modeled bit flip matrix.
    """
    K = len(peak_coords)
    s1, s2, s3, s4 = params[0:4]
    peak_params = params[4:]
    assert len(peak_params) == 2 * K

    yy, xx = np.indices((N, M))
    MM = np.zeros((N, M), dtype=float)

    for i in range(K):
        rr, cc = peak_coords[i]

        x0 = float(cc)
        y0 = float(rr)
        A1 = peak_params[2 * i]
        A2 = peak_params[2 * i + 1]

        G1 = _elliptical_gaussian(xx, yy, x0, y0, A1, s1, s2, 0)
        G2 = _elliptical_gaussian(xx, yy, x0, y0, A2, s3, s4, math.pi / 4)
        MM += G1 + G2

    return MM

def _combined_bit_flip_model(coords, *params, peak_coords, N, M):
    """
    A wrapper for `_model_bit_flip_matrix` to be used with `curve_fit`.
    """
    M2D = _model_bit_flip_matrix(np.array(params), peak_coords, N, M)
    return M2D.ravel()

def _build_bit_flip_matrix_initial_guess(peak_coords, original_matrix):
    """
    Builds the initial guess for the bit flip matrix optimization.

    Args:
        peak_coords (list): The coordinates of the peaks.
        original_matrix (np.array): The original bit flip matrix.

    Returns:
        np.array: The initial guess for the parameters.
    """
    guess_list = [s1_init_bfr, s2_init_bfr, s3_init_bfr, s4_init_bfr]
    for (r, c) in peak_coords:
        val = original_matrix[r, c]
        A1_init = val * 0.9
        A2_init = val * 0.1
        guess_list += [A1_init, A2_init]
    return np.array(guess_list, dtype=float)

def _reduce_access_count_distribution(full_params, trace_name, output_dir, bypass_solver=True):
    """Reduces the access count distribution."""
    logging.info("Reducing access count distribution...")
    t0 = time.time()
    (count, count_freq) = full_params['access_count_dist']
    max_x = count[-1] + 1
    x_data = np.asarray(count, dtype=int)
    y_data = np.asarray(count_freq)

    edges, s, w, x_dist, y_dist = _fit_log_distribution(
        x_data, y_data, end=max_x, num_intervals=5, bypass_solver=bypass_solver
    )

    y_full = np.zeros(max_x, dtype=y_data.dtype)
    y_full[x_data] = y_data

    if bypass_solver:
        y_reduced = None
        logging.info("Skipping CP-SAT solver (bypass_solver=True).")
    else:
        y_reduced = np.zeros(max_x, dtype=y_dist.dtype)
        y_reduced[x_dist] = y_dist

    plotting.plot_AD(max_x, y_reduced, y_full, trace_name, output_dir)
    logging.info(f"Completed reducing access count distribution. ({time.time() - t0:.1f}s)")
    return (x_dist, y_dist), (s, w, edges)

def _reduce_markov_matrix(full_params, trace_name, output_dir):
    """Reduces the Markov matrix."""
    logging.info("Reducing Markov matrix...")
    t0 = time.time()
    N = 12
    markov_model = full_params['markov_matrix']
    original_matrix_mm = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if j in markov_model and i in markov_model[j]:
                original_matrix_mm[i, j] = markov_model[j][i]

    if np.all(original_matrix_mm[10, :] == 0.0) and np.all(original_matrix_mm[:, 10] == 0.0):
        N_mm = 11
        original_matrix_mm = np.delete(original_matrix_mm, 10, axis=0)
        original_matrix_mm = np.delete(original_matrix_mm, 10, axis=1)
    else:
        N_mm = N
    original_col_sums_mm = original_matrix_mm.sum(axis=1)

    # 1. Peak Detection
    col0_vals = original_matrix_mm[0, :]
    col11_vals = original_matrix_mm[N_mm - 1, :]
    diag_vals = np.diag(original_matrix_mm)

    peaks_col0 = set(_find_peaks(col0_vals))
    peaks_col11 = set(_find_peaks(col11_vals))
    peaks_diag = set(_find_peaks(diag_vals))

    peak_list_mm = []

    if 0 in peaks_diag:
        peaks_col0.discard(0)
    if N_mm - 1 in peaks_diag:
        peaks_col11.discard(N_mm - 1)

    for r in peaks_col0:
        peak_list_mm.append(('col', r, 0, 0))
    for r in peaks_col11:
        peak_list_mm.append(('col', r, N_mm - 1, 0))
    for r in peaks_diag:
        peak_list_mm.append(('diag', r, r, math.pi / 4))

    # 2. Fitting into sum of Gaussian Matrix
    K_mm = len(peak_list_mm)
    init_params_mm = _build_markov_matrix_initial_guess(peak_list_mm, original_matrix_mm)
    lb_mm = [1e-3, 1e-3, 1e-3, 1e-3] * 2
    ub_mm = [20, 20, 20, 20] * 2
    lb_mm += [0.0] * 2 * K_mm
    ub_mm += [2] * 2 * K_mm
    bounds_mm = Bounds(lb_mm, ub_mm, keep_feasible=True)

    alpha_mm = 0
    res = minimize(
        fun=_calculate_markov_matrix_objective,
        args=(original_matrix_mm, original_col_sums_mm, peak_list_mm, alpha_mm, N_mm),
        x0=init_params_mm,
        method='SLSQP',
        bounds=bounds_mm,
        options={'maxiter': 5000, 'disp': True, 'ftol': 1e-9}
    )

    best_params_mm = res.x

    # 3. Display the fitted result
    fitted_M_mm = _model_markov_matrix(best_params_mm, peak_list_mm, N_mm)
    for i in range(N_mm):
        if fitted_M_mm[:, i].sum() > 0:
            fitted_M_mm[:, i] = fitted_M_mm[:, i] / fitted_M_mm[:, i].sum()

    if N_mm == 11:
        fitted_M_mm = np.insert(fitted_M_mm, 10, 0, axis=0)
        fitted_M_mm = np.insert(fitted_M_mm, 10, 0, axis=1)
        original_matrix_mm = np.insert(original_matrix_mm, 10, 0, axis=0)
        original_matrix_mm = np.insert(original_matrix_mm, 10, 0, axis=1)

    plotting.plot_MM(fitted_M_mm, original_matrix_mm, trace_name, output_dir)
    logging.info(f"Completed reducing Markov matrix. ({time.time() - t0:.1f}s)")
    diff = fitted_M_mm - original_matrix_mm
    sq_error = np.sum(diff ** 2)
    rms_error = np.sqrt(sq_error / (N_mm * N_mm))
    current_col_sums = fitted_M_mm.sum(axis=1)
    original_col_sums = original_matrix_mm.sum(axis=1)
    col_diff = current_col_sums - original_col_sums
    col_error = np.sum(col_diff**2)
    return fitted_M_mm, (best_params_mm, peak_list_mm), rms_error, col_error

def _reduce_bit_flip_matrix(full_params, trace_name, output_dir):
    """Reduces the bit flip matrix."""
    logging.info("Reducing bit flip matrix...")
    t0 = time.time()
    spatial_param = full_params['bit_flip_matrix']
    original_matrix_bfr = np.vstack(spatial_param[:-1])
    original_matrix_bfr = 1 - original_matrix_bfr

    # Replace any NaN/Inf value with the closest finite entry.
    invalid_mask = ~np.isfinite(original_matrix_bfr)
    if np.any(invalid_mask):
        finite_coords = np.argwhere(~invalid_mask)
        if finite_coords.size == 0:
            raise ValueError("Bit flip matrix has no finite values to substitute NaN/Inf entries.")
        invalid_coords = np.argwhere(invalid_mask)
        for invalid_row, invalid_col in invalid_coords:
            distances = (finite_coords[:, 0] - invalid_row) ** 2 + (finite_coords[:, 1] - invalid_col) ** 2
            nearest_idx = np.argmin(distances)
            nearest_row, nearest_col = finite_coords[nearest_idx]
            original_matrix_bfr[invalid_row, invalid_col] = original_matrix_bfr[nearest_row, nearest_col]
    N_bfr = 6
    M_bfr = 11
    peak_coords_bfr = peak_local_max(
        original_matrix_bfr,
        min_distance=1,
        threshold_abs=0.2,
        num_peaks=3,
        exclude_border=False
    )
    K_bfr = len(peak_coords_bfr)
    logging.info(f"Detected peak coords (row, col): {peak_coords_bfr}")
    init_params_bfr = _build_bit_flip_matrix_initial_guess(peak_coords_bfr, original_matrix_bfr)
    y_flat, x_flat = np.indices((N_bfr, M_bfr))
    z_data = original_matrix_bfr.ravel()
    lb_bfr = [1e-3] * 4
    ub_bfr = [10] * 4
    lb_bfr += [0, 0] * K_bfr
    ub_bfr += [1, 1] * K_bfr
    try:
        popt, pcov = curve_fit(
            f=lambda coords, *params: _combined_bit_flip_model(coords, *params, peak_coords=peak_coords_bfr, N=N_bfr, M=M_bfr),
            xdata=(x_flat.ravel(), y_flat.ravel()),
            ydata=z_data,
            p0=init_params_bfr,
            maxfev=10000,
            bounds=(lb_bfr, ub_bfr)
        )
    except RuntimeError:
        logging.error("Optimization failed!")
        popt = init_params_bfr

    fitted_M_bfr = _model_bit_flip_matrix(popt, peak_coords_bfr, N_bfr, M_bfr)
    diff = fitted_M_bfr - original_matrix_bfr
    sq_error = math.sqrt(np.sum(diff ** 2) / 66)
    spatial_list = [row for row in 1 - fitted_M_bfr]
    plotting.plot_BFR(spatial_list, spatial_param, trace_name, output_dir)
    logging.info(f"Completed reducing bit flip matrix. ({time.time() - t0:.1f}s)")
    return spatial_list, (popt, peak_coords_bfr), sq_error

def _reduce_short_interval_ratio(full_params, trace_name, output_dir):
    """Reduces the short interval ratio."""
    logging.info("Reducing short interval ratio...")
    t0 = time.time()
    short_interval_ratio = full_params['short_ratio']
    x_data = [0, 5, 11]
    y_data = [short_interval_ratio[0], short_interval_ratio[5], short_interval_ratio[11]]
    p = np.polyfit(x_data, y_data, 2)
    poly = np.poly1d(p)
    reduced_short_interval_ratio = [float(poly(x)) for x in range(0, 12)]
    plotting.plot_SIR(reduced_short_interval_ratio, trace_name, "reduced", output_dir)
    logging.info(f"Completed reducing short interval ratio. ({time.time() - t0:.1f}s)")
    return reduced_short_interval_ratio, p

def reduce_parameters(full_params, trace_name, output_dir):
  """
  Reduces the parameters of the memory trace.

  Args:
      full_params (dict): The full parameters of the memory trace.
      trace_name (str): The name of the trace.
      output_dir (str): The output directory.

  Returns:
      dict: The reduced parameters.
  """
  logging.info("Reducing parameters...")
  t_total = time.time()

  num_blocks = full_params['num_pages']

  reduction_steps = [
      "Access count distribution",
      "Markov matrix",
      "Bit flip matrix",
      "Short interval ratio",
  ]
  pbar = tqdm(total=len(reduction_steps), desc=f"[{trace_name}] Reducing parameters", unit="step")

  # Reduce access count distribution
  pbar.set_postfix_str(reduction_steps[0])
  (x_dist, y_dist), (s, w, edges) = _reduce_access_count_distribution(
      full_params, trace_name, output_dir
  )
  pbar.update(1)

  # Reduce Markov Matrix
  pbar.set_postfix_str(reduction_steps[1])
  fitted_M_mm, markov_params, mm_rms_error, mm_col_error = _reduce_markov_matrix(full_params, trace_name, output_dir)
  pbar.update(1)

  # Reduce bit flip matrix
  pbar.set_postfix_str(reduction_steps[2])
  spatial_list, bit_flip_params, bfr_rms_error = _reduce_bit_flip_matrix(
      full_params, trace_name, output_dir
  )
  pbar.update(1)

  # Reduce short interval ratio
  pbar.set_postfix_str(reduction_steps[3])
  reduced_short_interval_ratio, short_ratio_params = _reduce_short_interval_ratio(
      full_params, trace_name, output_dir
  )
  pbar.update(1)
  pbar.close()

  logging.info(f"Parameter reduction complete. Total time: {time.time() - t_total:.1f}s")

  errors = [
        ('Markov_matrix_RMS', mm_rms_error),
        ('Markov_matrix_col_error', mm_col_error),
        ('BFR_RMS', bfr_rms_error)
  ]
  file_io.write_errors_to_csv(trace_name, errors, output_dir)

  reduced_params = {
    'num_pages': num_blocks,
    'reduced_access_count_dist': (x_dist, y_dist),
    'access_params': (s,w,edges),
    'reduced_markov_matrix': fitted_M_mm,
    'markov_params': markov_params,
    'reduced_bit_flip_matrix': spatial_list,
    'bit_flip_params': bit_flip_params,
    'short_ratio_params': short_ratio_params,
    'reduced_short_interval_ratio': reduced_short_interval_ratio
  }
  return reduced_params
