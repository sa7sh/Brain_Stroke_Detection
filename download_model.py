"""
Download best_unet_model.h5 from Hugging Face
Run this to get the missing model file
"""

from huggingface_hub import hf_hub_download
import os
import shutil

print("=" * 70)
print("Downloading UNet Model from Hugging Face...")
print("=" * 70)

try:
    # Download from Hugging Face
    print("\n📥 Downloading from: Sharvarihk/CNNBasedBrainStrokeDetection")
    print("⏳ This may take 1-2 minutes depending on your connection...")
    
    downloaded_path = hf_hub_download(
        repo_id="Sharvarihk/CNNBasedBrainStrokeDetection",
        filename="best_unet_model.h5",
        local_dir=".",  # Download to current directory
        local_dir_use_symlinks=False  # Don't use symlinks on Windows
    )
    
    print(f"\n✅ Downloaded to: {downloaded_path}")
    
    # Check if file exists
    if os.path.exists("best_unet_model.h5"):
        file_size = os.path.getsize("best_unet_model.h5") / (1024 * 1024)  # MB
        print(f"✅ File verified: best_unet_model.h5 ({file_size:.2f} MB)")
        print("\n" + "=" * 70)
        print("🎉 SUCCESS! Model downloaded successfully!")
        print("=" * 70)
        print("\nNext step: Run the app:")
        print("   python app.py")
        print("=" * 70)
    else:
        print("\n⚠️ File downloaded but not found in current directory")
        print(f"File is at: {downloaded_path}")
        print("Copying to current directory...")
        
        if os.path.exists(downloaded_path):
            shutil.copy2(downloaded_path, "best_unet_model.h5")
            print("✅ Copied successfully!")
        
except Exception as e:
    print(f"\n❌ Error downloading model: {e}")
    print("\n" + "=" * 70)
    print("Alternative: Download manually")
    print("=" * 70)
    print("1. Go to: https://huggingface.co/Sharvarihk/CNNBasedBrainStrokeDetection")
    print("2. Find 'Files and versions' tab")
    print("3. Download 'best_unet_model.h5'")
    print("4. Save it to: C:\\Users\\4DiN\\Desktop\\Brain_Stroke_Detection\\")
    print("=" * 70)
