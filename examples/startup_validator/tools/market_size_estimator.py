from langchain_core.tools import tool
from typing import List, Tuple


@tool(response_format="content_and_artifact")
def market_size_estimator(domain: str, mock: bool = True) -> dict:
    """
    Estimate market size and CAGR for a domain.
    If mock=True, returns fake data. Otherwise, real API can be plugged in later.
    """
    if mock:
        market_data = {
            "healthcare": {"market_size": "$10B", "cagr": "12%"},
            "fitness": {"market_size": "$5B", "cagr": "8%"},
            "education": {"market_size": "$7B", "cagr": "10%"}
        }
        response = market_data.get(domain.lower(), {"market_size": "$1B", "cagr": "5%"})
        content = f"Estimate market size and CAGR for domain -  {domain} are: {response}"
        return content, response

    else:
        # Future: Call real market analysis API here
        raise NotImplementedError("Real API integration not implemented yet.")
