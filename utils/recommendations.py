import pandas as pd
import os

def get_plant_info(plant_name):
    csv_path = 'data/metadata/plant_info.csv'
    
    if not os.path.exists(csv_path):
        return None
    
    try:
        df = pd.read_csv(csv_path)
        plant_info = df[df['botanical_name'] == plant_name]
        
        if not plant_info.empty:
            return plant_info.iloc[0].to_dict()
    except Exception as e:
        print(f"Error reading plant info: {e}")
    
    return None

def get_treatments(plant_name):
    csv_path = 'data/metadata/treatments.csv'
    
    if not os.path.exists(csv_path):
        return []
    
    try:
        df = pd.read_csv(csv_path)
        treatments = df[df['botanical_name'] == plant_name]
        
        if not treatments.empty:
            return treatments.to_dict('records')
    except Exception as e:
        print(f"Error reading treatments: {e}")
    
    return []

def get_all_plants():
    csv_path = 'data/metadata/plant_info.csv'
    
    if not os.path.exists(csv_path):
        return []
    
    try:
        df = pd.read_csv(csv_path)
        return df.to_dict('records')
    except Exception as e:
        print(f"Error reading plants: {e}")
        return []
 
