from ecd.train.trainer import build_train_config


def test_build_train_config_allows_null_eval_and_save_every_steps() -> None:
    cfg = {
        "seed": 0,
        "device": "cpu",
        "train": {
            "steps": 1,
            "train_steps": 1,
            "batch_size": 2,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "lr_schedule": "constant",
            "warmup_steps": 0,
            "amp": "none",
            "resume_path": None,
            "log_every_steps": 1,
            "save_every_steps": None,
            "eval_every_steps": None,
            "hard_negative": {
                "enabled": False,
                "mode": "teacher_tail",
                "tail_from": 1,
                "tail_to": 1,
                "mix_random_ratio": 0.0,
            },
            "rank": {
                "kind": "info_nce",
                "multi_positive": {"enabled": False, "num_positives": 1},
            },
        },
        "loss": {
            "distill": False,
            "rank": False,
            "struct": False,
            "distill_mode": "mse",
            "rank_temperature": 0.07,
            "alpha": 1.0,
            "beta": 0.0,
            "gamma": 0.0,
            "warmup_frac": 0.0,
        },
        "eval": {
            "query_n": 1,
            "k_values": [10],
            "backend": "brute",
            "metric": "cosine",
            "force_normalize": True,
            "use_two_stage": False,
            "candidate_n": 100,
            "m": 16,
            "ef_construction": 200,
            "ef_search": 64,
            "eval_every": 1,
        },
    }

    train_config, eval_config = build_train_config(cfg)
    assert train_config.save_every == 0
    assert eval_config.eval_every == 0
