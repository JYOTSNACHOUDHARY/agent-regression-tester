from langchain_core.tools import tool
from typing import List, Tuple


@tool(response_format="content_and_artifact")
def legal_risk_checker(domain: str, mock: bool = True) -> list:
    """
    Check legal risks for a domain.
    If mock=True, returns fake risks.
    """
    if mock:
        legal_risks = {
            "healthcare": ["HIPAA compliance", "Patient Data Privacy"],
            "fitness": ["Injury Liability", "User Data Protection"],
            "education": ["Student Data Privacy", "Content Licensing Issues"]
        }
        response = legal_risks.get(domain.lower(), ["General Compliance Risk"])
        content = f"Legal risk for domain {domain} are: {response}"
        return content, response
    else:
        # Future: Call real legal risk database API
        raise NotImplementedError("Real API integration not implemented yet.")
