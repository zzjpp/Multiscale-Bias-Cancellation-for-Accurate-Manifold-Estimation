"""Explicit fused-series implementation; original radial quadrature is unchanged.

Only the exp(-x) I_nu(x) small-x series is fused into a CUDA elementwise kernel.
For x>=16 the series is not used by the original scaled_iv and is skipped.
Function globals are copied; no original module or frozen file is monkeypatched.
"""
from __future__ import annotations

from functools import lru_cache
import types
from scipy.special import gammaln
from e1_theory_path import gpu_population as original


@lru_cache(maxsize=1)
def _kernel():
    import cupy as cp
    return cp.ElementwiseKernel(
        "float64 x, float64 nu, float64 log_gamma, int32 terms", "float64 answer",
        r'''
        if (x >= 16.0) { answer = 0.0; }
        else {
            double safe_x = x > 2.2250738585072014e-308 ? x : 2.2250738585072014e-308;
            double leading = exp(-safe_x + nu*log(safe_x/2.0) - log_gamma);
            double term = 1.0, total = 1.0, z = safe_x*safe_x/4.0;
            for (int j=1; j<terms; ++j) {
                term = term*z / ((double)j*(nu+(double)j)); total += term;
            }
            answer = leading*total;
            if (x == 0.0) answer = (nu == 0.0 ? 1.0 : 0.0);
        }
        ''', "ms_stsp_exact_small_bessel_series_v1", options=("--fmad=false",))


def fused_series(cp, nu, x, terms=80):
    return _kernel()(x, float(nu), float(gammaln(nu+1.0)), int(terms))


scaled_globals = dict(original.scaled_iv.__globals__, _series_scaled_iv=fused_series)
scaled_iv = types.FunctionType(original.scaled_iv.__code__, scaled_globals, "fused_scaled_iv", original.scaled_iv.__defaults__)
radial_globals = dict(original.radial_mass_gpu.__globals__, scaled_iv=scaled_iv)
radial_mass_gpu = types.FunctionType(original.radial_mass_gpu.__code__, radial_globals, "same_gl_radial_mass_gpu", original.radial_mass_gpu.__defaults__)
evaluate_globals = dict(original.GpuPopulationSTSP.evaluate.__globals__, radial_mass_gpu=radial_mass_gpu)
evaluate = types.FunctionType(original.GpuPopulationSTSP.evaluate.__code__, evaluate_globals, "same_population_evaluate", original.GpuPopulationSTSP.evaluate.__defaults__)


class FastGpuPopulationSTSP(original.GpuPopulationSTSP):
    evaluate = evaluate
