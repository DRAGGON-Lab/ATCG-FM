import json
from pathlib import Path

from pytest import CaptureFixture

from atcg.cli import main


def test_cli_trains_scores_and_evaluates_a_tiny_model(
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

    assert (
        main(
            [
                "evaluate",
                "--checkpoint",
                str(checkpoint),
                "--fasta",
                str(fasta),
            ]
        )
        == 0
    )
    evaluation_output = json.loads(capsys.readouterr().out)
    assert evaluation_output["tokens"] == 16
