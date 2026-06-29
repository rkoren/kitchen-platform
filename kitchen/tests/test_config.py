"""Tests for KitchenConfig and sub-models."""

import pytest
import yaml
from pydantic import ValidationError

from kitchen.config import DataConfig, KitchenConfig, MLflowConfig, MonitorConfig, ThresholdSpec

# --- KitchenConfig top-level ---


def test_minimal_valid_config():
    cfg = KitchenConfig(experiment="my-exp")
    assert cfg.experiment == "my-exp"
    assert cfg.mlflow.tracking_uri == "sqlite:///mlruns.db"
    assert cfg.data is None
    assert cfg.monitor is None


def test_experiment_required():
    with pytest.raises(ValidationError, match="experiment"):
        KitchenConfig()


def test_project_sections_pass_through():
    cfg = KitchenConfig(experiment="x", model={"depth": 5}, train={"lr": 0.01})
    assert cfg.model_extra["model"] == {"depth": 5}
    assert cfg.model_extra["train"] == {"lr": 0.01}


def test_from_yaml(tmp_path):
    params = {"experiment": "test-exp", "mlflow": {"tracking_uri": "sqlite:///test.db"}}
    p = tmp_path / "params.yaml"
    p.write_text(yaml.dump(params))
    cfg = KitchenConfig.from_yaml(str(p))
    assert cfg.experiment == "test-exp"
    assert cfg.mlflow.tracking_uri == "sqlite:///test.db"


# --- INT-007: from_yaml transparently loads a menu.yaml ---


def test_from_yaml_loads_menu_by_content(tmp_path):
    """A file whose content is a menu is bridged even when passed via the params path."""
    menu = {
        "project": "cbb-model",
        "pipeline": ["train"],
        "recipes": {"train": {"kind": "stage", "source": "src/train/run.py"}},
        "thresholds": {"val_accuracy": 0.8},
    }
    p = tmp_path / "menu.yaml"
    p.write_text(yaml.dump(menu))
    cfg = KitchenConfig.from_yaml(str(p))
    assert cfg.experiment == "cbb-model"  # menu's project defaults the experiment
    assert cfg.thresholds["val_accuracy"] == 0.8


def test_from_yaml_falls_back_to_sibling_menu(tmp_path):
    """A missing params.yaml resolves to a sibling menu.yaml with no --params flag."""
    menu = {"project": "p", "recipes": {}}
    (tmp_path / "menu.yaml").write_text(yaml.dump(menu))
    cfg = KitchenConfig.from_yaml(str(tmp_path / "params.yaml"))
    assert cfg.experiment == "p"


def test_from_yaml_prefers_params_when_both_present(tmp_path):
    (tmp_path / "params.yaml").write_text(yaml.dump({"experiment": "from-params"}))
    (tmp_path / "menu.yaml").write_text(yaml.dump({"project": "from-menu", "recipes": {}}))
    cfg = KitchenConfig.from_yaml(str(tmp_path / "params.yaml"))
    assert cfg.experiment == "from-params"  # no auto-prefer surprise


# --- INT-007: shared menu-aware path resolver (used by every top-level command) ---


def test_resolve_params_path_falls_back_to_menu(tmp_path, monkeypatch):
    from kitchen.config import resolve_params_path

    monkeypatch.chdir(tmp_path)
    assert resolve_params_path("params.yaml") == "params.yaml"  # neither exists → unchanged
    (tmp_path / "menu.yaml").write_text("project: p\nrecipes: {}\n")
    assert resolve_params_path("params.yaml") == "menu.yaml"  # default absent → sibling menu
    (tmp_path / "params.yaml").write_text("experiment: p\n")
    assert resolve_params_path("params.yaml") == "params.yaml"  # present → respected


def test_resolve_params_path_respects_explicit_nondefault(tmp_path, monkeypatch):
    from kitchen.config import resolve_params_path

    monkeypatch.chdir(tmp_path)
    (tmp_path / "menu.yaml").write_text("project: p\nrecipes: {}\n")
    # an explicit non-"params.yaml" path is never rewritten to menu.yaml
    assert resolve_params_path("custom.yaml") == "custom.yaml"


# --- DataConfig ---


def test_kaggle_source_valid():
    cfg = DataConfig(source="kaggle", competition="titanic")
    assert cfg.competition == "titanic"


def test_kaggle_source_missing_competition():
    with pytest.raises(ValidationError, match="competition"):
        DataConfig(source="kaggle")


def test_s3_source_valid():
    cfg = DataConfig(source="s3", bucket="my-bucket", prefix="raw/")
    assert cfg.bucket == "my-bucket"


def test_s3_source_missing_bucket():
    with pytest.raises(ValidationError, match="bucket"):
        DataConfig(source="s3")


def test_local_source_valid():
    cfg = DataConfig(source="local", path="/data")
    assert cfg.path == "/data"


def test_local_source_missing_path():
    with pytest.raises(ValidationError, match="path"):
        DataConfig(source="local")


def test_unknown_source_rejected():
    with pytest.raises(ValidationError):
        DataConfig(source="gcs")


def test_data_extra_fields_allowed():
    cfg = DataConfig(source="kaggle", competition="titanic", raw_file="train.csv")
    assert cfg.model_extra["raw_file"] == "train.csv"


# --- MLflowConfig ---


def test_mlflow_defaults():
    cfg = MLflowConfig()
    assert cfg.tracking_uri == "sqlite:///mlruns.db"
    assert cfg.artifact_bucket is None
    assert cfg.model_artifact_path == "model"  # CBB-002 default


def test_mlflow_custom_model_artifact_path():
    cfg = MLflowConfig(model_artifact_path="cbb_model")
    assert cfg.model_artifact_path == "cbb_model"


def test_mlflow_custom_uri():
    cfg = MLflowConfig(tracking_uri="http://localhost:5000")
    assert "5000" in cfg.tracking_uri


# --- MonitorConfig ---


def test_monitor_with_bucket():
    cfg = MonitorConfig(report_bucket="my-bucket")
    assert cfg.report_bucket == "my-bucket"


def test_monitor_with_local_path():
    cfg = MonitorConfig(local_path="/tmp/report.html")
    assert cfg.local_path == "/tmp/report.html"


def test_monitor_with_both():
    cfg = MonitorConfig(report_bucket="b", local_path="/tmp/r.html")
    assert cfg.report_bucket == "b"
    assert cfg.local_path == "/tmp/r.html"


def test_monitor_missing_output_raises():
    with pytest.raises(
        ValidationError, match="report_bucket.*local_path|local_path.*report_bucket"
    ):
        MonitorConfig()


def test_monitor_defaults():
    cfg = MonitorConfig(report_bucket="b")
    assert cfg.reference_file == "reference.parquet"
    assert cfg.report_key == "monitoring/drift_report.html"


# --- Nested in KitchenConfig ---


def test_full_config_with_all_sections():
    cfg = KitchenConfig(
        experiment="titanic",
        data={"source": "kaggle", "competition": "spaceship-titanic"},
        mlflow={"tracking_uri": "sqlite:///mlruns.db"},
        monitor={"report_bucket": "my-bucket"},
        run_name="baseline",
        model={"n_estimators": 100},
    )
    assert cfg.data.source == "kaggle"
    assert cfg.monitor.report_bucket == "my-bucket"
    assert cfg.model_extra["model"]["n_estimators"] == 100


def test_invalid_data_section_propagates():
    with pytest.raises(ValidationError, match="competition"):
        KitchenConfig(experiment="x", data={"source": "kaggle"})


# --- ThresholdSpec ---


def test_threshold_spec_min_only():
    spec = ThresholdSpec(min=0.80)
    assert spec.min == 0.80
    assert spec.max is None


def test_threshold_spec_max_only():
    spec = ThresholdSpec(max=0.40)
    assert spec.max == 0.40
    assert spec.min is None


def test_threshold_spec_both():
    spec = ThresholdSpec(min=0.60, max=0.95)
    assert spec.min == 0.60
    assert spec.max == 0.95


def test_threshold_spec_neither_raises():
    with pytest.raises(ValidationError, match="at least one"):
        ThresholdSpec()


def test_threshold_spec_extra_field_raises():
    with pytest.raises(ValidationError):
        ThresholdSpec(min=0.80, typo=0.5)


# --- thresholds in KitchenConfig ---


def test_config_plain_float_thresholds():
    cfg = KitchenConfig(experiment="x", thresholds={"val_accuracy": 0.85})
    assert cfg.thresholds["val_accuracy"] == 0.85


def test_config_spec_threshold():
    cfg = KitchenConfig(experiment="x", thresholds={"val_logloss": {"max": 0.40}})
    assert isinstance(cfg.thresholds["val_logloss"], ThresholdSpec)
    assert cfg.thresholds["val_logloss"].max == 0.40


def test_config_mixed_thresholds():
    cfg = KitchenConfig(
        experiment="x",
        thresholds={"val_accuracy": 0.80, "val_logloss": {"max": 0.40}},
    )
    assert cfg.thresholds["val_accuracy"] == 0.80
    assert isinstance(cfg.thresholds["val_logloss"], ThresholdSpec)


def test_config_thresholds_from_yaml(tmp_path):
    params = tmp_path / "params.yaml"
    params.write_text(
        "experiment: titanic\nthresholds:\n  val_accuracy: 0.80\n  val_logloss:\n    max: 0.45\n"
    )
    cfg = KitchenConfig.from_yaml(str(params))
    assert cfg.thresholds["val_accuracy"] == 0.80
    assert isinstance(cfg.thresholds["val_logloss"], ThresholdSpec)
    assert cfg.thresholds["val_logloss"].max == 0.45


def test_config_empty_thresholds_default():
    cfg = KitchenConfig(experiment="x")
    assert cfg.thresholds == {}


# --- ci section (CFG-002) ---


def test_config_ci_defaults_absent():
    cfg = KitchenConfig(experiment="x")
    assert cfg.ci is None


def test_config_ci_field_defaults():
    cfg = KitchenConfig(experiment="x", ci={})
    assert cfg.ci.auto_submit is False
    assert cfg.ci.fail_on_threshold is True
    assert cfg.ci.notifications is None


def test_config_ci_from_yaml(tmp_path):
    params = tmp_path / "params.yaml"
    params.write_text(
        "experiment: titanic\n"
        "ci:\n"
        "  auto_submit: true\n"
        "  fail_on_threshold: false\n"
        "  notifications:\n"
        "    slack_webhook_secret: SLACK_WEBHOOK_URL\n"
        "    when: always\n"
    )
    cfg = KitchenConfig.from_yaml(str(params))
    assert cfg.ci.auto_submit is True
    assert cfg.ci.fail_on_threshold is False
    assert cfg.ci.notifications.slack_webhook_secret == "SLACK_WEBHOOK_URL"
    assert cfg.ci.notifications.when == "always"


def test_config_ci_rejects_bad_notify_when():
    with pytest.raises(ValidationError):
        KitchenConfig(experiment="x", ci={"notifications": {"when": "sometimes"}})


# --- CheckConfig (CBB-012) ---


def test_check_config_defaults_empty():
    from kitchen.config import CheckConfig

    assert CheckConfig().required_env == []


def test_check_section_parsed_on_kitchen_config():
    cfg = KitchenConfig(experiment="e", check={"required_env": ["KENPOM_API_KEY"]})
    assert cfg.check is not None
    assert cfg.check.required_env == ["KENPOM_API_KEY"]


def test_check_section_absent_is_none():
    cfg = KitchenConfig(experiment="e")
    assert cfg.check is None


# --- SecretSpec + secrets manifest (SECR-001) ---


def test_secret_spec_env_only_default_required():
    from kitchen.config import SecretSpec

    s = SecretSpec()
    assert s.required is True
    assert s.source == "env"


def test_secret_spec_sm_bundle_source():
    from kitchen.config import SecretSpec

    s = SecretSpec(aws_secret="cbb-model/prod", key="KENPOM_API_KEY", required=False)
    assert s.required is False
    assert s.source == "SM cbb-model/prod#KENPOM_API_KEY"


def test_secret_spec_ssm_source():
    from kitchen.config import SecretSpec

    assert SecretSpec(ssm="/cbb/kenpom").source == "SSM /cbb/kenpom"


def test_secret_spec_key_requires_aws_secret():
    from kitchen.config import SecretSpec

    with pytest.raises(ValidationError, match="requires `aws_secret`"):
        SecretSpec(key="X")


def test_secret_spec_aws_and_ssm_mutually_exclusive():
    from kitchen.config import SecretSpec

    with pytest.raises(ValidationError, match="not both"):
        SecretSpec(aws_secret="a", ssm="b")


def test_secrets_parsed_on_kitchen_config():
    cfg = KitchenConfig(
        experiment="e",
        secrets={"KENPOM_API_KEY": {"aws_secret": "cbb-model/prod", "key": "KENPOM_API_KEY"}},
    )
    assert cfg.secrets["KENPOM_API_KEY"].source == "SM cbb-model/prod#KENPOM_API_KEY"


def test_effective_secrets_folds_legacy_required_env():
    cfg = KitchenConfig(experiment="e", check={"required_env": ["LEGACY_VAR"]})
    eff = cfg.effective_secrets()
    assert eff["LEGACY_VAR"].source == "env"
    assert eff["LEGACY_VAR"].required is True
    assert cfg.uses_legacy_required_env is True


def test_effective_secrets_manifest_wins_over_legacy():
    cfg = KitchenConfig(
        experiment="e",
        secrets={"DUP": {"ssm": "/x"}},
        check={"required_env": ["DUP"]},
    )
    # secrets: entry wins on a name conflict (migrating one secret at a time).
    assert cfg.effective_secrets()["DUP"].source == "SSM /x"


def test_uses_legacy_required_env_false_without_check():
    assert KitchenConfig(experiment="e").uses_legacy_required_env is False


# KG-014: feature schema spec
def test_feature_schema_spec_parses():
    from kitchen.config import FeatureSchemaSpec, KitchenConfig

    cfg = KitchenConfig(
        experiment="e",
        feature_schema={"file": "matchups.parquet", "columns": {"a": "int64", "b": "float64"}},
    )
    assert isinstance(cfg.feature_schema, FeatureSchemaSpec)
    assert cfg.feature_schema.file == "matchups.parquet"
    assert cfg.feature_schema.columns == {"a": "int64", "b": "float64"}


def test_feature_schema_spec_rejects_unknown_key():
    import pytest

    from kitchen.config import KitchenConfig

    with pytest.raises(Exception, match="typo"):
        KitchenConfig(
            experiment="e",
            feature_schema={"file": "f.parquet", "columns": {}, "typo": 1},
        )
