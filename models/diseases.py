from pydantic import BaseModel

# disease collection model
class Disease(BaseModel):
    name: str
    overview: str
    treatment_recommendations: str