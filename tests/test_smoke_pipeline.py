from hcc_survival.config import load_config
from hcc_survival.evaluation import run_nested_experiment
from hcc_survival.reporting import generate_report


def test_fast_nested_pipeline_runs_on_fixture(synthetic_data, tmp_path):
    features, target = synthetic_data
    config = load_config("configs/fast.yaml")
    config["experiment"]["outer_folds"] = 3
    config["experiment"]["outer_repeats"] = 1
    config["experiment"]["bootstrap_resamples"] = 10
    config["models"]["include"] = ["dummy", "logistic_regression"]
    run_dir = run_nested_experiment(features, target, config, artifact_root=tmp_path)
    assert (run_dir / "oof_predictions_all.csv").exists()
    assert generate_report(run_dir).exists()
