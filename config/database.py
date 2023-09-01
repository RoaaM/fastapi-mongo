from pymongo import MongoClient

client = MongoClient("mongodb+srv://admin:cpzk9kox5x2dSHds@cluster0.15znqwt.mongodb.net/?retryWrites=true&w=majority")

db = client.plant_db

plant_collection = db["plants"]

disease_collection = db["diseases"]

user_collection = db["users"]