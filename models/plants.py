from pydantic import BaseModel

# plant collection model
class Plant(BaseModel):
	name: str
	scientific_name: str
	description: str
	watering_needs: str
	light_needs: str

