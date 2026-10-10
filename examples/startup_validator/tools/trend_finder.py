import random
from langchain_core.tools import tool
from typing import List, Tuple


@tool(response_format="content_and_artifact")
def trend_finder(domain: str, mock: bool = True) -> list:
    """
    Find trends for a domain.
    If mock=True, returns fake trends.
    """
    if mock:
        trends = {
            "healthcare": ["AI Diagnostics", "Remote Patient Monitoring"],
            "fitness": ["Wearable Health Devices", "Virtual Personal Training"],
            "education": ["AI Tutors", "Microlearning Platforms"]
        }
        response = trends.get(domain.lower(), ["Emerging Trend 1", "Emerging Trend 2"])
        content = f"Found trends for domain -  {domain} are: {response}"
        return content, response
    else:
        raise NotImplementedError("Real API integration not implemented yet.")
