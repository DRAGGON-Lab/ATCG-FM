from atcg.eval.registry import MODERN_V1, task_spec


def test_modern_protocol_resolves_real_gfmbench_tasks() -> None:
    assert {task.family for task in MODERN_V1} == {
        "supervised",
        "zero_shot_embedding",
        "zero_shot_likelihood",
    }
    for task in MODERN_V1:
        assert task.load().__name__ == task.class_name
        assert task_spec(task.name) == task
