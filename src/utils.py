import os
import random
import string

import asyncpg

from src.llms.models import Model


def random_str(k: int) -> str:
    return "".join(random.choices(string.ascii_uppercase, k=k))


async def get_random_unattempted_task_ids(
    *, model: Model, limit: int, all_task_ids: set[str]
) -> list[str]:
    if limit <= 0:
        return []

    model_value = model.value
    conn = await asyncpg.connect(os.environ["NEON_DSN"])
    try:
        rows = await conn.fetch(
            """
            SELECT DISTINCT i.task_id
            FROM guess AS g
            JOIN instructions AS i ON g.instructions_score_id = i.id
            WHERE g.model = $1
            """,
            model_value,
        )
    finally:
        await conn.close()

    attempted = {row["task_id"] for row in rows}
    remaining = list(all_task_ids - attempted)
    if not remaining:
        return []
    return random.sample(remaining, k=min(limit, len(remaining)))
