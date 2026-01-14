from __future__ import annotations

import torch


def distortion_loss(
    anchor_teacher: torch.Tensor,
    anchor_student: torch.Tensor,
    neighbor_teacher: torch.Tensor,
    neighbor_student: torch.Tensor,
) -> torch.Tensor:
    anchor_teacher = torch.nn.functional.normalize(anchor_teacher, dim=-1)
    anchor_student = torch.nn.functional.normalize(anchor_student, dim=-1)
    neighbor_teacher = torch.nn.functional.normalize(neighbor_teacher, dim=-1)
    neighbor_student = torch.nn.functional.normalize(neighbor_student, dim=-1)
    teacher_sim = torch.einsum("bd,bkd->bk", anchor_teacher, neighbor_teacher)
    student_sim = torch.einsum("bd,bkd->bk", anchor_student, neighbor_student)
    return torch.mean((teacher_sim - student_sim) ** 2)
