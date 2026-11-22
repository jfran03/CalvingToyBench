import json, os
import numpy as np, pandas as pd, pytest
import calvingdetection_baseline as cb

T = pd.Timestamp("2023-03-01 12:00", tz="UTC")


def test_fold_for_cow():
    cows = [c for k in cb.HELD_OUT for c in cb.HELD_OUT[k]]
    assert len(cows) == 15 and len(set(cows)) == 15
    assert cb.fold_for_cow("09B") == 1 and cb.fold_for_cow("12B") == 5
    with pytest.raises(ValueError):
        cb.fold_for_cow("ZZZ")


def test_window_rf_prob_max_and_empty():
    h = lambda n: T - pd.Timedelta(hours=n)
    fp = pd.Series([0.9, 0.3, 0.6], index=pd.DatetimeIndex([h(12), h(11), h(0)]))
    assert cb.window_rf_prob(fp, T) == 0.6          # t-12h excluded, t included
    assert cb.window_rf_prob(fp.iloc[:1], T) == 0.0  # only an out-of-range hour
    assert cb.window_rf_prob(fp.iloc[:0], T) == 0.0


def test_window_ae_frac():
    idx = pd.DatetimeIndex([T - pd.Timedelta(hours=n) for n in (0, 5, 11, 12, 30)])
    assert cb.window_ae_frac(idx, T) == pytest.approx(3 / 12)


def test_sliding_features_columns():
    n = 30
    df = pd.DataFrame({c: np.arange(n, dtype=float) for c in
                       ["KNN2", "KNN3", "KNN4", "KNN5", "Distance_moved", "ActivityCount_Difference"]})
    out = cb.add_sliding_features(df)
    assert len(cb.feature_names()) == 42 and set(cb.feature_names()) <= set(out.columns)
    assert out["KNN2_MA6"].iloc[10] == pytest.approx(df["KNN2"].iloc[5:11].mean())
    assert out["KNN2_Diff3"].iloc[10] == 3.0


CD = os.environ.get("CD_REPO", "CalvingDetection")


@pytest.mark.skipif(not os.path.isdir(CD), reason="CalvingDetection clone not found (set CD_REPO)")
def test_integration_real_repo():
    W = [json.loads(l) for l in open("windows.jsonl")]
    rf, ae = cb.score_windows(CD, W)
    assert len(rf) == len(ae) == 735
    assert all(0 <= r["prob"] <= 1 for r in rf + ae)
    assert any(r["prob"] > 0 for r in rf) and any(r["prob"] > 0 for r in ae)
    for c in cb.COWS:
        assert any(r["prob"] > 0 for r in ae if r["cow"] == c), c   # flags aligned (tz) for every cow


@pytest.mark.skipif(not os.path.isdir(CD), reason="CalvingDetection clone not found (set CD_REPO)")
def test_feature_names_match_pickle():
    assert set(cb.load_rf(CD, 1).feature_names_in_) == set(cb.feature_names())


@pytest.mark.skipif(not os.path.isdir(CD), reason="CalvingDetection clone not found (set CD_REPO)")
def test_fold_map_is_out_of_sample():
    held, trained = cb.fold_sanity(CD)
    assert trained > 0.97 and held < trained - 0.05
