import bcrypt
import streamlit as st
from database.db_config import get_users_collection
from datetime import datetime

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def register_user(username, email, password, name):
    users = get_users_collection()
    
    if users.find_one({"username": username}):
        return False, "Username already exists"
    
    if users.find_one({"email": email}):
        return False, "Email already registered"
    
    user_doc = {
        "username": username,
        "email": email,
        "password": hash_password(password),
        "name": name,
        "created_at": datetime.now()
    }
    
    try:
        users.insert_one(user_doc)
        return True, "Registration successful"
    except Exception as e:
        return False, f"Registration failed: {str(e)}"

def login_user(username, password):
    users = get_users_collection()
    user = users.find_one({"username": username})
    
    if user and verify_password(password, user['password']):
        user.pop('password', None)
        return True, user
    
    return False, None

def is_authenticated():
    return 'user' in st.session_state and st.session_state['user'] is not None
 
