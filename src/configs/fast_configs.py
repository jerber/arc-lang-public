from src.configs.models import Model, RunConfig, Step, StepRevision, StepRevisionPool

model = Model.grok_3_mini_fast

mini_config_big = RunConfig(
    final_follow_model=model,
    final_follow_times=10,
    max_concurrent_tasks=2,
    steps=[
        Step(
            instruction_model=model,
            follow_model=model,
            times=30,
            timeout_secs=300,
            include_base64=False,
            use_diffs=True,
        ),
        Step(
            instruction_model=model,
            follow_model=model,
            times=30,
            timeout_secs=300,
            include_base64=False,
            use_diffs=True,
        ),
        Step(
            instruction_model=model,
            follow_model=model,
            times=40,
            timeout_secs=300,
            include_base64=False,
            use_diffs=True,
        ),
        StepRevision(
            top_scores_used=10,
            instruction_model=model,
            follow_model=model,
            times_per_top_score=5,
            timeout_secs=300,
            include_base64=False,
            use_diffs=True,
        ),
        StepRevisionPool(
            top_scores_used=5,
            instruction_model=model,
            follow_model=model,
            times=10,
            timeout_secs=300,
            include_base64=False,
            use_diffs=True,
        ),
        StepRevision(
            top_scores_used=10,
            instruction_model=model,
            follow_model=model,
            times_per_top_score=5,
            timeout_secs=300,
            include_base64=False,
            use_diffs=True,
        ),
        StepRevisionPool(
            top_scores_used=5,
            instruction_model=model,
            follow_model=model,
            times=10,
            timeout_secs=300,
            include_base64=False,
            use_diffs=True,
        ),
    ],
)

mini_config = RunConfig(
    final_follow_model=model,
    final_follow_times=10,
    max_concurrent_tasks=2,
    steps=[
        Step(
            instruction_model=model,
            follow_model=model,
            times=20,
            timeout_secs=300,
            include_base64=False,
            use_diffs=True,
        ),
        StepRevision(
            top_scores_used=5,
            instruction_model=model,
            follow_model=model,
            times_per_top_score=2,
            timeout_secs=300,
            include_base64=False,
            use_diffs=True,
        ),
        StepRevisionPool(
            top_scores_used=5,
            instruction_model=model,
            follow_model=model,
            times=10,
            timeout_secs=300,
            include_base64=False,
            use_diffs=True,
        ),
    ],
)
