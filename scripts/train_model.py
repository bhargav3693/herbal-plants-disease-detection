import sys
import os
sys.path.append('.')

from models.model import MED117Model

print("🔍 Performing pre-flight checks...")

if not os.path.exists('data/raw/train'):
    print("❌ Training data not found!")
    print("   Expected location: data/raw/train/")
    print("\n💡 Solution: Run the dataset preparation script first:")
    print("   Command: python scripts/prepare_dataset.py")
    exit(1)

if not os.path.exists('data/raw/test'):
    print("❌ Testing data not found!")
    print("   Expected location: data/raw/test/")
    print("\n💡 Solution: Run the dataset preparation script first:")
    print("   Command: python scripts/prepare_dataset.py")
    exit(1)

train_folders = [f for f in os.listdir('data/raw/train') 
                 if os.path.isdir(os.path.join('data/raw/train', f))]
test_folders = [f for f in os.listdir('data/raw/test') 
                if os.path.isdir(os.path.join('data/raw/test', f))]

print(f"✅ Found {len(train_folders)} species in training set")
print(f"✅ Found {len(test_folders)} species in testing set")

if len(train_folders) < 100:
    print("⚠️  WARNING: Expected ~117 species folders")
    response = input("Continue anyway? (y/n): ")
    if response.lower() != 'y':
        exit(0)

print("\n" + "=" * 70)
print("Starting Model Training...")
print("=" * 70)

try:
    print("\n📦 Creating MED 117 CNN model...")
    model = MED117Model(num_classes=114)
    
    print("\n📋 Model Architecture:")
    model.model.summary()
    
    print("\n🚀 Starting training process...")
    print("   This will take 2-3 hours depending on your hardware")
    print("   You can monitor the progress below:")
    print("=" * 70)
    
    history = model.train(
        train_dir='data/raw/train',
        test_dir='data/raw/test',
        epochs=30,
        batch_size=32
    )
    
    print("\n" + "=" * 70)
    print("📊 TRAINING RESULTS")
    print("=" * 70)
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
    print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
    print("=" * 70)
    
    print("\n✅ Model training completed successfully!")
    print(f"📁 Model saved at: models/med117_model.h5")
    print(f"📁 Class names saved at: models/class_names.json")
    
    print("\n📌 Next Step: Run the Streamlit application")
    print("   Command: streamlit run app.py")
    print("=" * 70)

except KeyboardInterrupt:
    print("\n\n⚠️  Training interrupted by user")
    print("   Progress has been saved in checkpoints")
    exit(0)

except Exception as e:
    print(f"\n❌ Error during training: {str(e)}")
    import traceback
    traceback.print_exc()
    exit(1)
 
