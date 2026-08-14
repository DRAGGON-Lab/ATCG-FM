import json

from pytest import CaptureFixture

from atcg.eval import main


def test_cli_lists_versioned_modern_protocol(capsys: CaptureFixture[str]) -> None:
    assert main(["list-tasks"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["protocol"] == "modern-v1"
    assert "songlab_clinvar" in {task["name"] for task in output["tasks"]}


def test_cli_lists_isolated_model_providers(capsys: CaptureFixture[str]) -> None:
    assert main(["list-providers"]) == 0

    output = json.loads(capsys.readouterr().out)
    assert set(output["providers"]) == {"atcg", "carbon", "evo2", "jepa-dna", "ntv3"}
    assert output["providers"]["carbon"]["default_revision"]
    assert output["providers"]["evo2"]["environment"].endswith("/evo2")
