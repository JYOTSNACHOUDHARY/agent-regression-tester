import random
import zlib
from langchain_core.tools import tool
from typing import List, Tuple


@tool(response_format="content_and_artifact")
def risk_score_calculator(idea_details: str, mock: bool = True) -> int:
    """
    Calculate total risk score for an idea.
    If mock=True, simulate risk score randomly.
    """
    if mock:
        # Seeded by the idea so recorded runs are reproducible
        response = random.Random(zlib.crc32(idea_details.encode())).randint(3, 8)
        content = f"Total risk score for an idea - {idea_details} are: {response}"
        return content, response

    else:
        # Future: Real risk score calculation based on model
        raise NotImplementedError("Real model-based risk scoring not implemented yet.")
