from __future__ import annotations

import torch


def _cosine_loss(teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
    teacher = torch.nn.functional.normalize(teacher, dim=-1)
    student = torch.nn.functional.normalize(student, dim=-1)
    return 1.0 - torch.sum(teacher * student, dim=-1).mean()


def _relation_loss(teacher: torch.Tensor, student: torch.Tensor) -> torch.Tensor:
    teacher = torch.nn.functional.normalize(teacher, dim=-1)
    student = torch.nn.functional.normalize(student, dim=-1)
    teacher_sim = teacher @ teacher.T
    student_sim = student @ student.T
    mse = torch.mean((teacher_sim - student_sim) ** 2)
    std = torch.std(student, dim=0).mean()
    collapse_penalty = torch.relu(0.05 - std)
    return mse + collapse_penalty


def distill_loss(
    teacher: torch.Tensor, student: torch.Tensor, mode: str
) -> torch.Tensor:
    if teacher.shape[-1] == student.shape[-1]:
        if mode == "cosine":
            return _cosine_loss(teacher, student)
        if mode == "mse":
            return torch.mean((teacher - student) ** 2)
    return _relation_loss(teacher, student)
