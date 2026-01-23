from pymongo import MongoClient
import streamlit as st
import os

def get_database():
    try:
        mongo_uri = st.secrets["mongo"]["uri"]
        db_name = st.secrets["mongo"]["database"]
    except:
        mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        db_name = os.getenv("DATABASE_NAME", "herbal_disease_db")
    
    client = MongoClient(mongo_uri)
    return client[db_name]

def get_users_collection():
    db = get_database()
    return db["users"]

def get_detections_collection():
    db = get_database()
    return db["detections"]
 
