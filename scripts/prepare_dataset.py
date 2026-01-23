import os
import shutil
from sklearn.model_selection import train_test_split

SOURCE_DIR = "data/raw/segmented_dataset/MED 117 Leaf Species/Segmented leaf set using UNET"
TRAIN_DIR = "data/raw/train"
TEST_DIR = "data/raw/test"
TRAIN_SPLIT = 0.80
TEST_SPLIT = 0.20

print("🌿 MED 117 Dataset Preparation")
print("=" * 70)

if not os.path.exists(SOURCE_DIR):
    print(f"❌ ERROR: Source directory not found!")
    print(f"Expected location: {SOURCE_DIR}")
    print("\n💡 Solution:")
    print("1. Download MED 117 dataset from:")
    print("   https://data.mendeley.com/datasets/dtvbwrhznz/4")
    print(f"2. Extract and place 'Segmented leaf set' folder at: {SOURCE_DIR}")
    exit(1)

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

print(f"✅ Created train directory: {TRAIN_DIR}")
print(f"✅ Created test directory: {TEST_DIR}")
print("=" * 70)

plant_folders = [f for f in os.listdir(SOURCE_DIR) 
                 if os.path.isdir(os.path.join(SOURCE_DIR, f))]

print(f"\n📊 Found {len(plant_folders)} medicinal plant species")
print("=" * 70)
print(f"{'Species Name':<50} | {'Train':<6} | {'Test':<6}")
print("=" * 70)

total_train = 0
total_test = 0

for idx, plant_folder in enumerate(plant_folders, 1):
    plant_path = os.path.join(SOURCE_DIR, plant_folder)
    
    clean_name = (plant_folder
                  .replace(' ', '_')
                  .replace('(', '')
                  .replace(')', '')
                  .replace("'", '')
                  .replace('-', '_'))
    
    images = [f for f in os.listdir(plant_path) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(images) == 0:
        print(f"⚠️  [{idx:>3}/{len(plant_folders)}] {clean_name[:45]:<45} | No images!")
        continue
    
    train_imgs, test_imgs = train_test_split(
        images, 
        test_size=TEST_SPLIT, 
        random_state=42
    )
    
    train_species_dir = os.path.join(TRAIN_DIR, clean_name)
    test_species_dir = os.path.join(TEST_DIR, clean_name)
    
    os.makedirs(train_species_dir, exist_ok=True)
    os.makedirs(test_species_dir, exist_ok=True)
    
    for img in train_imgs:
        src = os.path.join(plant_path, img)
        dst = os.path.join(train_species_dir, img)
        shutil.copy2(src, dst)
    
    for img in test_imgs:
        src = os.path.join(plant_path, img)
        dst = os.path.join(test_species_dir, img)
        shutil.copy2(src, dst)
    
    total_train += len(train_imgs)
    total_test += len(test_imgs)
    
    print(f"✅ [{idx:>3}/{len(plant_folders)}] {clean_name[:45]:<45} | {len(train_imgs):<6} | {len(test_imgs):<6}")

print("=" * 70)
print("\n🎉 Dataset Preparation Complete!")
print("=" * 70)
print(f"📊 Total Species Processed: {len(plant_folders)}")
print(f"📁 Training Images: {total_train:,}")
print(f"📁 Testing Images: {total_test:,}")
print(f"📁 Total Images: {total_train + total_test:,}")
print("=" * 70)
print(f"\n✅ Training data saved to: {TRAIN_DIR}")
print(f"✅ Testing data saved to: {TEST_DIR}")
print("\n📌 Next Step: Train the model")
print("   Command: python scripts/train_model.py")
print("=" * 70)
 
