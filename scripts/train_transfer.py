import sys
import os

# Add project root to path
sys.path.append('.')

from models.model_transfer import MED117TransferModel

print("🔍 Pre-flight checks...")
print("="*70)

# Check if train and test directories exist
if not os.path.exists('data/raw/train'):
    print("❌ Training data not found!")
    print("   Run: python fix_and_prepare.py")
    exit(1)

if not os.path.exists('data/raw/test'):
    print("❌ Testing data not found!")
    print("   Run: python fix_and_prepare.py")
    exit(1)

# Count folders
train_folders = len([f for f in os.listdir('data/raw/train') 
                     if os.path.isdir(os.path.join('data/raw/train', f))])
test_folders = len([f for f in os.listdir('data/raw/test') 
                    if os.path.isdir(os.path.join('data/raw/test', f))])

print(f"✅ Found {train_folders} species in training set")
print(f"✅ Found {test_folders} species in testing set")
print("="*70)

if train_folders < 50:
    print("\n⚠️  WARNING: Expected more plant species")
    response = input("Continue anyway? (y/n): ")
    if response.lower() != 'y':
        exit(0)

print("\n🚀 Starting Transfer Learning Training...\n")

try:
    # Create and train model
    model = MED117TransferModel(num_classes=train_folders)
    
    history = model.train(
        train_dir='data/raw/train',
        test_dir='data/raw/test',
        epochs=15,
        batch_size=32
    )
    
    # Print final results
    print("\n" + "="*70)
    print("📊 TRAINING RESULTS")
    print("="*70)
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%")
    print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
    print("="*70)
    
    print("\n✅ Model training completed successfully!")
    print(f"📁 Model saved at: models/med117_model.h5")
    print(f"📁 Class names saved at: models/class_names.json")
    
    print("\n📌 Next Step: Run the Streamlit application")
    print("   Command: streamlit run app.py")
    print("="*70)

except KeyboardInterrupt:
    print("\n\n⚠️  Training interrupted by user")
    print("   Best model checkpoint has been saved")
    exit(0)

except Exception as e:
    print(f"\n❌ Error during training: {str(e)}")
    import traceback
    traceback.print_exc()
    exit(1)
