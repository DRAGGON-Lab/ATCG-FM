"""Command-line entry points for small, inspectable ATCG-FM workflows."""

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

from atcg.eval import evaluate_language_model
from atcg.models import GenomicLanguageModel, attention_tiny, hybrid_tiny
from atcg.runtime.checkpoint import load_model_checkpoint
from atcg.runtime.inference import generate, score_sequence
from atcg.runtime.training import TrainingConfig, fit
from atcg.sequence import CausalWindowDataset, FixedAlphabetTokenizer, read_fasta


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atcg",
        description="Train and inspect small genomic language-model experiments.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train a reference model from FASTA records")
    train.add_argument("--fasta", type=Path, required=True)
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument("--architecture", choices=("attention", "hybrid"), default="hybrid")
    train.add_argument("--steps", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--context-length", type=int, default=128)
    train.add_argument("--d-model", type=int, default=64)
    train.add_argument("--layers", type=int, default=4)
    train.add_argument("--heads", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=3e-4)
    train.add_argument("--seed", type=int, default=17)
    train.add_argument("--device", default="cpu")
    train.add_argument("--reverse-complement", action="store_true")
    train.set_defaults(handler=_train)

    evaluate = commands.add_parser("evaluate", help="evaluate a checkpoint on FASTA records")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--fasta", type=Path, required=True)
    evaluate.add_argument("--batch-size", type=int, default=8)
    evaluate.add_argument("--device", default="cpu")
    evaluate.set_defaults(handler=_evaluate)

    score = commands.add_parser("score", help="score one nucleotide sequence")
    score.add_argument("--checkpoint", type=Path, required=True)
    score.add_argument("--sequence", required=True)
    score.add_argument("--device", default="cpu")
    score.set_defaults(handler=_score)

    generation = commands.add_parser("generate", help="continue a nucleotide prompt")
    generation.add_argument("--checkpoint", type=Path, required=True)
    generation.add_argument("--prompt", required=True)
    generation.add_argument("--tokens", type=int, default=32)
    generation.add_argument("--temperature", type=float, default=1.0)
    generation.add_argument("--top-k", type=int, default=4)
    generation.add_argument("--seed", type=int, default=0)
    generation.add_argument("--device", default="cpu")
    generation.set_defaults(handler=_generate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch an ATCG-FM command."""

    arguments = _parser().parse_args(list(argv) if argv is not None else None)
    handler = cast(Callable[[argparse.Namespace], int], arguments.handler)
    return handler(arguments)


def _train(arguments: argparse.Namespace) -> int:
    fasta = cast(Path, arguments.fasta)
    run_dir = cast(Path, arguments.run_dir)
    architecture = cast(str, arguments.architecture)
    steps = cast(int, arguments.steps)
    batch_size = cast(int, arguments.batch_size)
    context_length = cast(int, arguments.context_length)
    d_model = cast(int, arguments.d_model)
    layers = cast(int, arguments.layers)
    heads = cast(int, arguments.heads)
    learning_rate = cast(float, arguments.learning_rate)
    seed = cast(int, arguments.seed)
    device = cast(str, arguments.device)
    reverse_complement = cast(bool, arguments.reverse_complement)

    tokenizer = FixedAlphabetTokenizer()
    records = read_fasta(fasta)
    dataset = CausalWindowDataset(
        records,
        tokenizer,
        context_length=context_length,
        stride=context_length,
        include_eos=True,
        include_reverse_complements=reverse_complement,
    )
    preset = hybrid_tiny if architecture == "hybrid" else attention_tiny
    model = GenomicLanguageModel(
        preset(
            tokenizer.vocab_size,
            max_seq_len=context_length,
            d_model=d_model,
            n_layers=layers,
            n_heads=heads,
        )
    )
    run = fit(
        model,
        dataset,
        pad_id=tokenizer.pad_id,
        config=TrainingConfig(
            max_steps=steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            device=device,
        ),
        run_dir=run_dir,
    )
    final_metrics = run.metrics[-1]
    _print_json(
        {
            "checkpoint": str(run.checkpoint_path),
            "examples": len(dataset),
            "final_loss": final_metrics.loss,
            "parameters": model.parameter_count(),
            "steps": run.state.step,
            "tokens_seen": run.state.tokens_seen,
        }
    )
    return 0


def _evaluate(arguments: argparse.Namespace) -> int:
    checkpoint = cast(Path, arguments.checkpoint)
    fasta = cast(Path, arguments.fasta)
    batch_size = cast(int, arguments.batch_size)
    device = cast(str, arguments.device)
    model, _ = load_model_checkpoint(checkpoint, device=device)
    tokenizer = _tokenizer_for(model)
    records = read_fasta(fasta)
    dataset = CausalWindowDataset(
        records,
        tokenizer,
        context_length=model.config.max_seq_len,
        stride=model.config.max_seq_len,
        include_eos=True,
    )
    metrics = evaluate_language_model(
        model,
        dataset,
        pad_id=tokenizer.pad_id,
        batch_size=batch_size,
        device=device,
    )
    _print_json(
        {
            "bits_per_token": metrics.bits_per_token,
            "examples": metrics.example_count,
            "mean_nll": metrics.mean_nll,
            "perplexity": metrics.perplexity,
            "token_accuracy": metrics.token_accuracy,
            "tokens": metrics.token_count,
        }
    )
    return 0


def _score(arguments: argparse.Namespace) -> int:
    checkpoint = cast(Path, arguments.checkpoint)
    sequence = cast(str, arguments.sequence)
    device = cast(str, arguments.device)
    model, _ = load_model_checkpoint(checkpoint, device=device)
    score = score_sequence(model, _tokenizer_for(model), sequence)
    _print_json(
        {
            "bits_per_token": score.bits_per_token,
            "mean_nll": score.mean_nll,
            "token_count": score.token_count,
            "total_nll": score.total_nll,
        }
    )
    return 0


def _generate(arguments: argparse.Namespace) -> int:
    checkpoint = cast(Path, arguments.checkpoint)
    prompt = cast(str, arguments.prompt)
    tokens = cast(int, arguments.tokens)
    temperature = cast(float, arguments.temperature)
    top_k = cast(int, arguments.top_k)
    seed = cast(int, arguments.seed)
    device = cast(str, arguments.device)
    model, _ = load_model_checkpoint(checkpoint, device=device)
    tokenizer = _tokenizer_for(model)
    result = generate(
        model,
        tokenizer,
        prompt,
        max_new_tokens=tokens,
        temperature=temperature,
        top_k=top_k,
        allowed_token_ids=tuple(range(len(tokenizer.alphabet))),
        seed=seed,
    )
    _print_json(
        {
            "generated_tokens": len(result.generated_token_ids),
            "prompt": result.prompt,
            "sequence": result.sequence,
        }
    )
    return 0


def _tokenizer_for(model: GenomicLanguageModel) -> FixedAlphabetTokenizer:
    tokenizer = FixedAlphabetTokenizer()
    if tokenizer.vocab_size != model.config.vocab_size:
        raise ValueError("checkpoint vocabulary is not compatible with the CLI's IUPAC tokenizer")
    return tokenizer


def _print_json(values: dict[str, object]) -> None:
    print(json.dumps(values, indent=2, sort_keys=True))
