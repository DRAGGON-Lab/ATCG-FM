import json
from pathlib import Path

from pytest import CaptureFixture

from atcg.cli import main


def test_cli_trains_and_scores_a_tiny_model(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    fasta = tmp_path / "tiny.fa"
    fasta.write_text(">synthetic\nACGTACGTACGTACGT\n", encoding="utf-8")
    run_dir = tmp_path / "run"

    assert (
        main(
            [
                "train",
                "--fasta",
                str(fasta),
                "--run-dir",
                str(run_dir),
                "--architecture",
                "attention",
                "--steps",
                "1",
                "--batch-size",
                "2",
                "--context-length",
                "8",
                "--d-model",
                "16",
                "--layers",
                "1",
                "--heads",
                "4",
            ]
        )
        == 0
    )
    train_output = json.loads(capsys.readouterr().out)
    checkpoint = Path(train_output["checkpoint"])
    assert checkpoint.is_file()

    assert main(["score", "--checkpoint", str(checkpoint), "--sequence", "ACGT"]) == 0
    score_output = json.loads(capsys.readouterr().out)
    assert score_output["token_count"] == 5


def test_cli_trains_a_stateful_titans_memory_candidate(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    fasta = tmp_path / "streams.fa"
    fasta.write_text(">a\nACGTACGTACGT\n>b\nTGCATGCATGCA\n", encoding="utf-8")
    run_dir = tmp_path / "titans-run"

    assert (
        main(
            [
                "train",
                "--fasta",
                str(fasta),
                "--run-dir",
                str(run_dir),
                "--architecture",
                "titans-memory",
                "--steps",
                "1",
                "--batch-size",
                "2",
                "--context-length",
                "4",
                "--gradient-horizon",
                "2",
                "--d-model",
                "8",
                "--layers",
                "1",
                "--heads",
                "2",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["steps"] == 1
    checkpoint = Path(output["checkpoint"])
    assert checkpoint.is_file()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stateful_execution"]["gradient_horizon"] == 2
    assert manifest["stateful_execution"]["memory_mode"] == "adaptive"

    assert (
        main(
            [
                "score",
                "--checkpoint",
                str(checkpoint),
                "--sequence",
                "ACGTAC",
                "--memory-mode",
                "adaptive",
            ]
        )
        == 0
    )
    score_output = json.loads(capsys.readouterr().out)
    assert score_output["memory_mode"] == "adaptive"
    assert score_output["token_count"] == 7

    assert (
        main(
            [
                "generate",
                "--checkpoint",
                str(checkpoint),
                "--prompt",
                "AC",
                "--tokens",
                "5",
                "--memory-mode",
                "adaptive",
            ]
        )
        == 0
    )
    generation_output = json.loads(capsys.readouterr().out)
    assert generation_output["generated_tokens"] == 5
    assert generation_output["memory_mode"] == "adaptive"
