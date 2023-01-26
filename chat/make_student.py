import warnings
from pathlib import Path
from typing import List, Tuple, Union

import fire
from torch import nn

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.utils import logging


logger = logging.get_logger(__name__)


def copy_layers(src_layers: nn.ModuleList, dest_layers: nn.ModuleList, layers_to_copy: List[int]):# -> None:
    layers_to_copy = nn.ModuleList([src_layers[i] for i in layers_to_copy])
    assert len(dest_layers) == len(layers_to_copy), f"{len(dest_layers)} != {len(layers_to_copy)}"
    dest_layers.load_state_dict(layers_to_copy.state_dict())


LAYERS_TO_COPY = {
    # maps  num layers in teacher -> num_layers in student -> which teacher layers to copy.
    # 12: bart, 16: pegasus, 6: marian/Helsinki-NLP
    12: {
        1: [0],  # This says that if the teacher has 12 layers and the student has 1, copy layer 0 of the teacher
        2: [0, 6],
        3: [0, 6, 11],
        4: [0, 4, 8, 11],
        6: [0, 2, 4, 7, 9, 11],
        9: [0, 1, 2, 4, 5, 7, 9, 10, 11],
        12: list(range(12)),
    },
    16: {  # maps  num layers in student -> which teacher layers to copy
        1: [0],
        2: [0, 15],
        3: [0, 8, 15],
        4: [0, 5, 10, 15],
        6: [0, 3, 6, 9, 12, 15],
        8: [0, 2, 4, 6, 8, 10, 12, 15],
        9: [0, 1, 3, 5, 7, 9, 11, 13, 15],
        12: [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15],
        16: list(range(16)),
    },
    6: {1: [0], 2: [0, 5], 3: [0, 2, 5], 4: [0, 1, 3, 5], 6: list(range(6))},
    24:{
        1: [0],  # This says that if the teacher has 12 layers and the student has 1, copy layer 0 of the teacher
        2: [0, 23],
        3: [0, 11, 23],
        4: [0, 8, 16, 23],
        5: [0, 5, 11, 17, 23],
        6: [0, 5, 10, 15, 20, 23],
        7: [0, 4, 8, 12, 16, 20, 23],
        #7: [0, 1, 2, 3, 4, 5, 6],
        9: [0, 3, 6, 9, 12, 15, 18, 21, 23],
        12: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23],
        24: list(range(24)),
    }
}
LAYERS_TO_SUPERVISE = {
    # maps  num layers in student -> which teacher layers to copy.
    #6: {1: [5], 2: [3, 5], 3: [1, 4, 5], 4: [1, 2, 4, 5]},
    #12: {1: [11], 2: [5, 11], 3: [3, 7, 11], 6: [1, 3, 5, 8, 10, 11]},
    #16: {1: [15], 4: [4, 9, 12, 15], 8: [1, 3, 5, 7, 9, 11, 13, 15]},
    24: {
        2: [1, 23],
        3: [1, 11, 23],
        4: [0, 9, 17, 23],
        6: [1, 6, 11, 16, 21, 23],
        7: [1, 5, 9, 13, 17, 21, 23]},
        9: [1, 4, 7, 10, 13, 16, 19, 22, 23],
        12: [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
}


def pick_layers_to_copy(n_student, n_teacher):
    try:
        val = LAYERS_TO_COPY[n_teacher][n_student]
        return val
    except KeyError:
        if n_student != n_teacher:
            warnings.warn(
                f"no hardcoded layers to copy for teacher {n_teacher} -> student {n_student}, defaulting to first {n_student}"
            )
        return list(range(n_student))


def get_layers_to_supervise(n_student, n_teacher):# -> List[int]:
    """Used or the --supervise_forward kwarg"""
    if n_student > n_teacher:
        raise ValueError(f"Cannot perform intermediate supervision for student {n_student} > teacher {n_teacher}")
    elif n_teacher == n_student:
        return list(range(n_teacher))
    elif n_student == 1:
        return [n_teacher - 1]
    else:
        return LAYERS_TO_SUPERVISE[n_teacher][n_student]


def create_student_by_copying_alternating_layers(
    teacher: Union[str, PreTrainedModel],
    save_path: Union[str, Path] = "student",
    nlayer=None,
    copy_first_teacher_layers=False,
    layers_to_copy=None,
    **extra_config_kwargs
):# -> Tuple[PreTrainedModel, List[int], List[int]]:
    """Make a student by copying alternating layers from a teacher, save it to save_path.
    Args:
        teacher: str or PreTrainedModel if str, this will call AutoModelForCausalLM.from_pretrained(teacher) before
        copying layers
        save_path: where to save the student, defaults to student directory.
        e: how many Encoder layers should the student have, default is fully copy of teacher
        d: how many Decoder layers should the student have, default is fully copy of teacher
        copy_first_teacher_layers: [bool] dont copy alternating layers, just the first e/d.
        **extra_config_kwargs: extra kwargs to pass to the student, by default the teacher config is used.

    Returns:
        student: new, smaller model.  (Also saves it to save_path)
        layers_to_copy: list of which teacher layers were used
    """
    if isinstance(teacher, str):
        AutoTokenizer.from_pretrained(teacher).save_pretrained(save_path)  # purely for convenience
        teacher = AutoModelForCausalLM.from_pretrained(teacher).eval()
    else:

        assert isinstance(teacher, PreTrainedModel), f"teacher must be a model or string got type {type(teacher)}"
    init_kwargs = teacher.config.to_diff_dict()

    try:
        #teacher_e, teacher_d = teacher.config.encoder_layers, teacher.config.decoder_layers
        n_teacher_layer = teacher.config.n_layer
        if nlayer is None:
            nlayer = n_teacher_layer
        init_kwargs.update({"n_layer": nlayer})
    except AttributeError:  # T5
        assert False
        teacher_e, teacher_d = teacher.config.num_layers, teacher.config.num_decoder_layers
        if e is None:
            e = teacher_e
        if d is None:
            d = teacher_d
        init_kwargs.update({"num_layers": e, "num_decoder_layers": d})

    # Kwargs to instantiate student: teacher kwargs with updated layer numbers + **extra_config_kwargs
    init_kwargs.update(extra_config_kwargs)

    # Copy weights
    student_cfg = teacher.config_class(**init_kwargs)
    student = AutoModelForCausalLM.from_config(student_cfg)
    # Start by copying the full teacher state dict this will copy the first N teacher layers to the student.
    info = student.load_state_dict(teacher.state_dict(), strict=False)
    assert info.missing_keys == [], info.missing_keys  # every student key should have a teacher keys.

    if copy_first_teacher_layers:  # Our copying is done. We just log and save
        layers_to_copy = list(range(nlayer))
        logger.info(
            f"Copied layers {layers_to_copy}. Saving them to {save_path}"
        )
        student.save_pretrained(save_path)
        return student, layers_to_copy

    # Decide which layers of the teacher to copy. Not exactly alternating -- we try to keep first and last layer.
    if layers_to_copy is None:
        layers_to_copy: List[int] = pick_layers_to_copy(nlayer, n_teacher_layer)

    #student.transformer.wte.load_state_dict(teacher.transformer.wte.state_dict())
    #student.transformer.wpe.load_state_dict(teacher.transformer.wpe.state_dict())
    try:
        copy_layers(teacher.transformer.h, student.transformer.h, layers_to_copy)
    except AttributeError:  # For t5, student.model.encoder.layers is called student.encoder.block
        assert False, 'AttributeError'
        copy_layers(teacher.encoder.block, student.encoder.block, e_layers_to_copy)
        copy_layers(teacher.decoder.block, student.decoder.block, d_layers_to_copy)
    logger.info(
        f"Copied encoder layers {layers_to_copy}. Saving them to {save_path}"
    )
    student.config.init_metadata = dict(
        teacher_type=teacher.config.model_type,
        copied_layers=layers_to_copy
    )
    student.save_pretrained(save_path)
    # Save information about copying for easier reproducibility

    return student, layers_to_copy


if __name__ == "__main__":
    fire.Fire(create_student_by_copying_alternating_layers)
