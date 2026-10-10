from langchain_core.tools import tool
from typing import List, Tuple


@tool(response_format="content_and_artifact")
def competitor_finder(domain: str, mock: bool = True) -> Tuple[str, List[str]]:
    """
    Find competitors for a domain.
    If mock=True, returns fake competitors.
    """
    if mock:
        competitors = {
            "healthcare": ["MediTech", "HealthFirst", "CareWell"],
            "fitness": ["FitTrack", "GymSmart", "MoveWell"],
            "education": ["EduBright", "Learnify", "SkillUp"]
        }
        response =  competitors.get(domain.lower(), ["GenericCompetitor1", "GenericCompetitor2"])
        content = f"Competitors for domain {domain} are: {response}"
        return content, response
    else:
        # Future: Call real competitor lookup API
        raise NotImplementedError("Real API integration not implemented yet.")
