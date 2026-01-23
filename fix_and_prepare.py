import os
import shutil
from sklearn.model_selection import train_test_split

print("🔧 Finding and Preparing Dataset...\n")

# All possible paths
possible_paths = [
    "data/raw/segmented_dataset/MED117_Medicinal Plant Leaf Dataset & Name Table/MED 117 Leaf Species/Segmented leaf set using UNET segmentation",
    "data/raw/segmented_dataset/MED117_Medicinal Plant Leaf Dataset & Name Table/MED 117 Leaf Species/Segmented leaf set using UNET",
]


SOURCE_DIR = None

# Find which path exists
for path in possible_paths:
    if os.path.exists(path):
        # Check if it has folders with images
        try:
            items = os.listdir(path)
            folders = [f for f in items if os.path.isdir(os.path.join(path, f))]
            if len(folders) > 50:  # Should have ~117 folders
                SOURCE_DIR = path
                print(f"✅ Found dataset at: {path}")
                print(f"   Folders: {len(folders)}")
                break
        except:
            pass

if not SOURCE_DIR:
    print("❌ Dataset not found in any expected location!")
    print("\n📁 Let's check what exists:")
    
    base = "data/raw/segmented_dataset"
    if os.path.exists(base):
        print(f"\n✅ {base} exists")
        items = os.listdir(base)
        print(f"   Contains: {items}")
        
        # Check each item
        for item in items:
            item_path = os.path.join(base, item)
            if os.path.isdir(item_path):
                print(f"\n   📁 {item}")
                sub_items = os.listdir(item_path)
                print(f"      Contains: {sub_items[:5]}")
    else:
        print(f"❌ {base} doesn't exist!")
    
    exit(1)

# Now prepare the dataset
TRAIN_DIR = "data/raw/train"
TEST_DIR = "data/raw/test"

# Clean old folders
for folder in [TRAIN_DIR, TEST_DIR]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

print("\n🌿 Preparing MED 117 Dataset")
print("="*70)

# Get plant folders
plant_folders = [f for f in os.listdir(SOURCE_DIR) 
                 if os.path.isdir(os.path.join(SOURCE_DIR, f))]

print(f"📊 Found {len(plant_folders)} plant species\n")

total_train = 0
total_test = 0

# Process each plant
for idx, plant in enumerate(plant_folders, 1):
    plant_path = os.path.join(SOURCE_DIR, plant)
    
    # Clean name
    clean_name = (plant.replace(' ', '_')
                      .replace('(', '')
                      .replace(')', '')
                      .replace("'", '')
                      .replace('-', '_'))
    
    # Get images
    images = [f for f in os.listdir(plant_path) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(images) == 0:
        print(f"⚠️  [{idx:>3}/{len(plant_folders)}] {plant[:50]:<50} | No images!")
        continue
    
    # Split
    train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)
    
    # Create folders
    train_folder = os.path.join(TRAIN_DIR, clean_name)
    test_folder = os.path.join(TEST_DIR, clean_name)
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    
    # Copy images
    for img in train_imgs:
        shutil.copy2(
            os.path.join(plant_path, img),
            os.path.join(train_folder, img)
        )
    
    for img in test_imgs:
        shutil.copy2(
            os.path.join(plant_path, img),
            os.path.join(test_folder, img)
        )
    
    total_train += len(train_imgs)
    total_test += len(test_imgs)
    
    print(f"✅ [{idx:>3}/{len(plant_folders)}] {clean_name[:45]:<45} | {len(train_imgs):<6} | {len(test_imgs):<6}")

print("="*70)
print(f"\n🎉 Complete!")
print(f"📁 Training images: {total_train:,}")
print(f"📁 Testing images: {total_test:,}")
print(f"📁 Total images: {total_train + total_test:,}")
print("\n📌 Next: python scripts/train_model.py")
print("="*70)
