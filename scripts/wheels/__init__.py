"""
Reusable “wheel” modules consolidating常用攻击、训练与工具逻辑。

后续实验请优先从此处导入复用组件，避免重复造轮子。
"""

from .attacks import (  # noqa: F401
    BIMAttack,
    CarliniWagnerAttack,
    FGSMAttack,
    PGDAttack,
    SAM2ForwardHelper,
)
from .trainer import (  # noqa: F401
    AggregateMetrics,
    EvaluationSummary,
    match_sample,
    SampleEvaluation,
    StepRecord,
    UAPSample,
    UniversalPatchTrainer,
    load_uap_samples,
)
from .utils import (  # noqa: F401
    AttackConfig,
    AttackLogger,
    AttackSummary,
    BestWorstTracker,
    IMAGENET_MEAN,
    IMAGENET_STD,
    bce_loss,
    compute_perturbation_norms,
    denormalize_image,
    dice_loss,
    ensure_dir,
    eval_masks_numpy,
    load_mask_tensor,
    load_rgb_tensor,
    mask_probs_to_numpy,
    mask_to_binary,
    resize_image_tensor,
    resize_mask_tensor,
    restore_image_tensor,
    save_perturbation_image,
    save_rgb_tensor,
)
from .dataset import (  # noqa: F401
    find_frame_path,
    normalize_masks,
    normalize_object_ids,
)
from .pipeline import run_bear_uap_experiment  # noqa: F401

__all__ = [
    # attacks
    "FGSMAttack",
    "PGDAttack",
    "BIMAttack",
    "CarliniWagnerAttack",
    "SAM2ForwardHelper",
    # trainer
    "UniversalPatchTrainer",
    "UAPSample",
    "load_uap_samples",
    "match_sample",
    "AggregateMetrics",
    "EvaluationSummary",
    "SampleEvaluation",
    "StepRecord",
    # utils
    "AttackConfig",
    "AttackLogger",
    "AttackSummary",
    "BestWorstTracker",
    "compute_perturbation_norms",
    "ensure_dir",
    "mask_probs_to_numpy",
    "resize_image_tensor",
    "resize_mask_tensor",
    "restore_image_tensor",
    "save_perturbation_image",
    "save_rgb_tensor",
    "load_rgb_tensor",
    "load_mask_tensor",
    "mask_to_binary",
    "dice_loss",
    "bce_loss",
    "eval_masks_numpy",
    "denormalize_image",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    # dataset helpers
    "find_frame_path",
    "normalize_object_ids",
    "normalize_masks",
    # pipeline
    "run_bear_uap_experiment",
]
