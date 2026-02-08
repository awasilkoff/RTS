"""
Test script for time-varying ARUC integration.

This script verifies that the ARUC model correctly handles both static and
time-varying uncertainty sets from the UncertaintySetProvider.

Usage:
    python test_time_varying_aruc.py

Expected output:
    - Static mode test passes
    - Time-varying mode test passes
    - Regression test (constant time-varying equals static) passes
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aruc_model import align_uncertainty_to_aruc, build_aruc_ldr_model
from io_rts import build_damdata_from_rts
from run_rts_aruc import build_uncertainty_set
from uncertainty_set_provider import HorizonUncertaintySet, UncertaintySetProvider


def test_static_mode():
    """Test that static mode still works (backward compatibility)."""
    print("Testing static mode (backward compatibility)...")

    data = build_damdata_from_rts(
        source_dir=Path("RTS_Data/SourceData"),
        ts_dir=Path("RTS_Data/timeseries_data_files"),
        start_time=pd.Timestamp(year=2020, month=1, day=1, hour=0),
        horizon_hours=4,
    )

    Sigma, rho = build_uncertainty_set(data, rho=2.0, wind_std_fraction=0.15)
    model, vars_dict = build_aruc_ldr_model(data, Sigma, rho, M_p=1e4)
    model.update()

    print(f"  Model: {model.NumVars} vars, {model.NumConstrs} constrs")
    print("  Static mode test PASSED")
    return model


def test_time_varying_mode():
    """Test time-varying mode with provider integration."""
    print("\nTesting time-varying mode...")

    data = build_damdata_from_rts(
        source_dir=Path("RTS_Data/SourceData"),
        ts_dir=Path("RTS_Data/timeseries_data_files"),
        start_time=pd.Timestamp(year=2020, month=1, day=1, hour=0),
        horizon_hours=4,
    )

    npz_path = Path(
        "uncertainty_sets_refactored/data/uncertainty_sets_rts4/sigma_rho.npz"
    )
    if not npz_path.exists():
        print(f"  Skipping: {npz_path} not found")
        print("  Run uncertainty calibration pipeline first")
        return None

    provider = UncertaintySetProvider.from_npz(npz_path)
    horizon = provider.get_by_indices(0, 4, compute_sqrt=True)

    Sigma, rho, sqrt_Sigma = align_uncertainty_to_aruc(
        horizon, data, provider.get_wind_gen_ids()
    )

    print(f"  Sigma shape: {Sigma.shape}")
    print(f"  rho range: [{rho.min():.3f}, {rho.max():.3f}]")

    model, vars_dict = build_aruc_ldr_model(
        data, Sigma, rho, sqrt_Sigma=sqrt_Sigma, M_p=1e4
    )
    model.update()

    print(f"  Model: {model.NumVars} vars, {model.NumConstrs} constrs")
    print("  Time-varying mode test PASSED")
    return model


def test_alignment_permutation():
    """Test that align_uncertainty_to_aruc correctly permutes wind generators."""
    print("\nTesting alignment permutation...")

    from models import DAMData

    I, T = 4, 4
    dummy_data = DAMData(
        gen_ids=["G_THERMAL", "WIND_A", "WIND_B", "G_HYDRO"],
        bus_ids=["B0", "B1", "B2"],
        line_ids=["L0"],
        time=list(range(T)),
        gen_type=["THERMAL", "WIND", "WIND", "HYDRO"],
        gen_to_bus=np.array([0, 1, 2, 0], dtype=int),
        Pmin=np.zeros(I),
        Pmax=np.ones(I),
        RU=np.ones(I) * 10.0,
        RD=np.ones(I) * 10.0,
        MUT=np.ones(I) * 2,
        MDT=np.ones(I) * 2,
        startup_cost=np.ones(I) * 5.0,
        shutdown_cost=np.ones(I) * 2.0,
        no_load_cost=np.ones(I) * 1.0,
        u_init=np.zeros(I),
        init_up_time=np.zeros(I),
        init_down_time=np.zeros(I),
        block_cap=np.ones((I, 2)) * 50.0,
        block_cost=np.ones((I, 2)) * 10.0,
        PTDF=np.zeros((1, 3)),
        Fmax=np.ones(1) * 100.0,
        d=np.ones((3, T)) * 10.0,
        gens_df=None,
        buses_df=None,
        lines_df=None,
    )

    # Provider has wind in reversed order
    provider_wind_ids = ["WIND_B", "WIND_A"]
    K = 2
    sigma = np.zeros((T, K, K))
    for t in range(T):
        sigma[t] = np.array([[1.0 + t, 0.1], [0.1, 10.0 + t]])

    horizon = HorizonUncertaintySet(
        indices=list(range(T)),
        sigma=sigma,
        rho=np.array([1.0, 2.0, 3.0, 4.0]),
        mu=np.zeros((T, K)),
        sqrt_sigma=None,
    )

    Sigma_aligned, rho_aligned, _ = align_uncertainty_to_aruc(
        horizon, dummy_data, provider_wind_ids
    )

    # Check permutation: WIND_A should get the variance that was at index 1
    for t in range(T):
        expected_00 = 10.0 + t  # WIND_A was at provider index 1
        expected_11 = 1.0 + t  # WIND_B was at provider index 0
        assert abs(Sigma_aligned[t, 0, 0] - expected_00) < 1e-10
        assert abs(Sigma_aligned[t, 1, 1] - expected_11) < 1e-10

    print("  Permutation test PASSED")


def test_regression_constant_equals_static():
    """Test that time-varying with constant values equals static mode."""
    print("\nTesting regression: constant time-varying equals static...")

    from models import DAMData

    I, N, L, T, B, K = 2, 3, 1, 4, 2, 1

    dummy_data = DAMData(
        gen_ids=["G0", "G1"],
        bus_ids=["B0", "B1", "B2"],
        line_ids=["L0"],
        time=list(range(T)),
        gen_type=["THERMAL", "WIND"],
        gen_to_bus=np.array([0, 2], dtype=int),
        Pmin=np.zeros(I),
        Pmax=np.ones(I),
        RU=np.ones(I) * 10.0,
        RD=np.ones(I) * 10.0,
        MUT=np.ones(I) * 2,
        MDT=np.ones(I) * 2,
        startup_cost=np.ones(I) * 5.0,
        shutdown_cost=np.ones(I) * 2.0,
        no_load_cost=np.ones(I) * 1.0,
        u_init=np.array([1.0, 0.0]),
        init_up_time=np.array([1.0, 0.0]),
        init_down_time=np.array([0.0, 3.0]),
        block_cap=np.ones((I, B)) * 50.0,
        block_cost=np.array([[10.0, 20.0], [12.0, 25.0]]),
        PTDF=np.zeros((L, N)),
        Fmax=np.ones(L) * 100.0,
        d=np.ones((N, T)) * 10.0,
        gens_df=None,
        buses_df=None,
        lines_df=None,
    )

    # Static mode
    Sigma_static = np.array([[1.0]])
    rho_static = 2.0
    model_static, _ = build_aruc_ldr_model(dummy_data, Sigma_static, rho_static, M_p=1e3)
    model_static.Params.OutputFlag = 0
    model_static.optimize()
    obj_static = model_static.ObjVal

    # Time-varying with constant values
    Sigma_tv = np.tile(Sigma_static[np.newaxis, :, :], (T, 1, 1))
    rho_tv = np.full(T, rho_static)
    model_tv, _ = build_aruc_ldr_model(dummy_data, Sigma_tv, rho_tv, M_p=1e3)
    model_tv.Params.OutputFlag = 0
    model_tv.optimize()
    obj_tv = model_tv.ObjVal

    print(f"  Static objective: {obj_static:.2f}")
    print(f"  Time-varying objective: {obj_tv:.2f}")

    assert abs(obj_static - obj_tv) < 1e-4, f"Objectives differ: {obj_static} vs {obj_tv}"
    print("  Regression test PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("ARUC Time-Varying Uncertainty Integration Tests")
    print("=" * 60)

    test_static_mode()
    test_time_varying_mode()
    test_alignment_permutation()
    test_regression_constant_equals_static()

    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
