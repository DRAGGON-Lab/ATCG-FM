"""Curated GFMBench task protocols for ATCG studies."""

from dataclasses import dataclass
from importlib import import_module
from typing import Literal, cast

from gfmbench_api.tasks.base import BaseGFMTask

TaskFamily = Literal["supervised", "zero_shot_embedding", "zero_shot_likelihood"]


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Stable ATCG name for one upstream GFMBench task class."""

    name: str
    module: str
    class_name: str
    family: TaskFamily

    def load(self) -> type[BaseGFMTask]:
        value = getattr(import_module(self.module), self.class_name)
        if not isinstance(value, type) or not issubclass(value, BaseGFMTask):
            raise TypeError(f"{self.module}.{self.class_name} is not a GFMBench task")
        return cast(type[BaseGFMTask], value)


MODERN_V1 = (
    TaskSpec(
        "gue_promoter_all",
        "gfmbench_api.tasks.concrete.gue_promoter_all_task",
        "GuePromoterAllTask",
        "supervised",
    ),
    TaskSpec(
        "gue_splice_site",
        "gfmbench_api.tasks.concrete.gue_splice_site_task",
        "GueSpliceSiteTask",
        "supervised",
    ),
    TaskSpec(
        "variant_benchmarks_non_coding",
        "gfmbench_api.tasks.concrete.variant_benchmarks_non_coding_task",
        "VariantBenchmarksNonCodingTask",
        "supervised",
    ),
    TaskSpec(
        "vepeval_clinvar",
        "gfmbench_api.tasks.concrete.clinvar_vepeval_task",
        "VepevalClinvarTask",
        "zero_shot_embedding",
    ),
    TaskSpec(
        "songlab_clinvar",
        "gfmbench_api.tasks.concrete.songlab_clinvar_task",
        "SonglabClinvarTask",
        "zero_shot_likelihood",
    ),
    TaskSpec(
        "brca1",
        "gfmbench_api.tasks.concrete.brca1_task",
        "BRCA1Task",
        "zero_shot_likelihood",
    ),
    TaskSpec(
        "traitgym_mendelian",
        "gfmbench_api.tasks.concrete.traitgym_mendelian_task",
        "TraitGymMendelianTask",
        "zero_shot_likelihood",
    ),
)

TASKS = {task.name: task for task in MODERN_V1}


def task_spec(name: str) -> TaskSpec:
    """Resolve one task in the versioned modern protocol."""

    try:
        return TASKS[name]
    except KeyError as error:
        available = ", ".join(sorted(TASKS))
        raise ValueError(f"unknown task {name!r}; choose one of: {available}") from error
