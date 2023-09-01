# plant collection schema
def plants_serial(plant) -> dict:
    return {
     "_id": str(plant["_id"]),
     "name": plant["name"],
     "scientific_name": plant["scientific_name"],
     "description": plant["description"],
     "watering_needs": plant["watering_needs"],
     "light_needs": plant["light_needs"],
     "difficulty": plant["difficulty"],
     }

def list_plants_serial(plants) -> list:
    return [plants_serial(plant) for plant in plants]

# disease collection schema
def disease_serial(disease) -> dict:
    return {
    "_id": str(disease["_id"]),
    "name": disease["name"],
    "overview": disease["overview"],
    "treatment_recommendations": disease["treatment_recommendations"]
    }

def list_diseases_serial(diseases) -> list:
    return [disease_serial(disease) for disease in diseases]


# user collection schema
def user_serial(user) -> dict:
    return {
    "_id": str(user["_id"]),
    "username": user["username"],
    "email": user["email"],
    "password": user["password"]
    }

def list_users_serial(users) -> list:
    return [user_serial(user) for user in users]


