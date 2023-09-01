from fastapi import APIRouter, HTTPException, UploadFile, File
from models.plants import Plant
from models.diseases import Disease
from models.users import User
from config.database import plant_collection, disease_collection, user_collection
from schema.schemas import plants_serial, list_plants_serial, disease_serial, list_diseases_serial, user_serial, list_users_serial
from bson import ObjectId
from typing import List
from converted_tflite_quantized.ai import classify_image_with_info
import os

router = APIRouter()

# GET request method for plant 
@router.get("/plant", response_model=List[Plant], tags=["plant"])
async def get_plants():
    plants = plant_collection.find()
    return list_plants_serial(plants)

# POST request method for plant
@router.post("/plant", response_model=Plant, tags=["plant"])
async def create_plant(plant: Plant):
    plant_dict = plant.dict()
    result = plant_collection.insert_one(plant_dict)
    plant_id = result.inserted_id
    return {**plant_dict, "_id": str(plant_id)}

# PUT request method for plant
@router.put("/plant/{id}", response_model=Plant, tags=["plant"])
async def update_plant(id: str, plant: Plant):
    plant_dict = plant.dict()
    plant_collection.find_one_and_update({"_id": ObjectId(id)}, {"$set": plant_dict})
    return {**plant_dict, "_id": id}

# DELETE request method for plant
@router.delete("/plant/{id}", tags=["plant"])
async def delete_plant(id: str):
    result = plant_collection.delete_one({"_id": ObjectId(id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Plant not found")
    return {"message": "Plant deleted successfully"}

# GET request method for disease 
@router.get("/diseases/", response_model=List[Disease], tags=["disease"])
async def get_diseases():
    diseases = disease_collection.find()
    return list_diseases_serial(diseases)

@router.post("/diseases/", tags=["disease"])
async def create_disease(upload_file: UploadFile = File(...)):
    # Construct absolute image path
    temp_dir = "./temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    image_path = os.path.join(temp_dir, upload_file.filename)
    
    # Save the uploaded image temporarily
    with open(image_path, "wb") as f:
        f.write(upload_file.file.read())

    # Perform inference using your TensorFlow Lite model and get disease details
    predicted_class, disease_details = classify_image_with_info(image_path)

    # Create a new record in the database
    new_disease_record = {
        "image_path": image_path,
        "predicted_class": predicted_class,
        "disease_details": disease_details,
    }
    result = disease_collection.insert_one(new_disease_record)
    disease_id = str(result.inserted_id)

    response_data = {
        "message": "Disease prediction and record created",
        "disease_id": disease_id,
        "predicted_class": predicted_class,
        "disease_details": disease_details,
    }
    
    return response_data



# PUT request method for disease
@router.put("/diseases/{id}", response_model=Disease, tags=["disease"])
async def update_disease(id: str, disease: Disease):
    disease_dict = disease.dict()
    disease_collection.find_one_and_update({"_id": ObjectId(id)}, {"$set": disease_dict})
    return {**disease_dict, "_id": id}

# DELETE request method for disease
@router.delete("/diseases/{id}", tags=["disease"])
async def delete_disease(id: str):
    result = disease_collection.delete_one({"_id": ObjectId(id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Disease not found")
    return {"message": "Disease deleted successfully"}


# GET request method for user
@router.get("/", response_model=List[User], tags=["users"])
async def get_users():
    users = user_collection.find()
    return list_users_serial(users)

# POST request method for user
@router.post("/users/", response_model=User, tags=["users"])
async def create_user(user: User):
    user_dict = user.dict()
    result = user_collection.insert_one(user_dict)
    user_id = result.inserted_id
    return {**user_dict, "_id": str(user_id)}

# PUT request method for user
@router.put("/users/{id}", response_model=User, tags=["users"])
async def update_user(id: str, user: User):
    user_dict = user.dict()
    user_collection.find_one_and_update({"_id": ObjectId(id)}, {"$set": user_dict})
    return {**user_dict, "_id": id}

# DELETE request method for user
@router.delete("/users/{id}", tags=["users"])
async def delete_user(id: str):
    result = user_collection.delete_one({"_id": ObjectId(id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User deleted successfully"}



