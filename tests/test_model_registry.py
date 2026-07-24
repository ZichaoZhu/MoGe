import json
from pathlib import Path

from moge.model import import_model_class_by_version


def test_all_model_versions_remain_registered():
    for version in ("v1", "v2", "v3"):
        model_class = import_model_class_by_version(version)
        assert model_class.__name__ == "MoGeModel"


def test_v3_configuration_disables_normal_prediction():
    config_path = Path(__file__).parents[1] / "configs" / "train" / "v3.json"
    config = json.loads(config_path.read_text())
    assert config["model"]["predict_normal"] is False
    assert "normal_head" not in config["model"]
    assert config["training"]["normal_prediction"] is False
