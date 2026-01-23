from database.db_config import get_users_collection, get_detections_collection
from datetime import datetime
from bson.objectid import ObjectId

def save_detection_result(user_id, image_path, prediction_result):
    detections = get_detections_collection()
    
    if isinstance(user_id, str):
        user_id = ObjectId(user_id)
    
    detection_doc = {
        "user_id": user_id,
        "image_path": image_path,
        "plant_name": prediction_result['plant_name'],
        "botanical_name": prediction_result.get('botanical_name', ''),
        "confidence": prediction_result['confidence'],
        "timestamp": datetime.now()
    }
    
    result = detections.insert_one(detection_doc)
    return result.inserted_id

def get_user_history(user_id, limit=10):
    detections = get_detections_collection()
    
    if isinstance(user_id, str):
        user_id = ObjectId(user_id)
    
    history = list(detections.find({"user_id": user_id})
                   .sort("timestamp", -1)
                   .limit(limit))
    return history

def get_detection_count(user_id):
    detections = get_detections_collection()
    
    if isinstance(user_id, str):
        user_id = ObjectId(user_id)
    
    return detections.count_documents({"user_id": user_id})

def get_all_detections(user_id):
    detections = get_detections_collection()
    
    if isinstance(user_id, str):
        user_id = ObjectId(user_id)
    
    return list(detections.find({"user_id": user_id})
                .sort("timestamp", -1))
 
