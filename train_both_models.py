#!/usr/bin/env python3
"""
Comprehensive PPO Training Script
Trains both Standard PPO and Maskable PPO models for 30 minutes each,
then automatically copies the trained models to Sentio's models directory.
"""

import os
import sys
import subprocess
import time
import shutil
from pathlib import Path
from datetime import datetime

def run_command(command, description):
    """Run a command and handle output"""
    print(f"\n🚀 {description}")
    print(f"📋 Command: {command}")
    print("-" * 60)
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.stdout:
            print("📤 Output:")
            print(result.stdout)
        
        if result.stderr:
            print("⚠️  Errors:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
        else:
            print(f"❌ {description} failed with return code {result.returncode}")
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False

def find_latest_model_files(pattern_base):
    """Find the latest model files matching the pattern"""
    model_files = []
    metadata_files = []
    
    # Look for .pth files
    for pth_file in Path(".").glob(f"{pattern_base}*.pth"):
        model_files.append(pth_file)
    
    # Look for corresponding metadata files
    for pth_file in model_files:
        metadata_file = pth_file.with_suffix("") / "-metadata.json"
        if not metadata_file.exists():
            metadata_file = Path(str(pth_file).replace(".pth", "-metadata.json"))
        if metadata_file.exists():
            metadata_files.append(metadata_file)
    
    return model_files, metadata_files

def copy_models_to_sentio(model_files, metadata_files):
    """Copy trained models to Sentio's models directory"""
    sentio_models_dir = Path("../Sentio/models")
    
    if not sentio_models_dir.exists():
        print(f"❌ Sentio models directory not found: {sentio_models_dir}")
        return False
    
    print(f"\n📦 Copying models to Sentio directory: {sentio_models_dir}")
    
    copied_files = []
    
    # Copy model files
    for model_file in model_files:
        try:
            dest_file = sentio_models_dir / model_file.name
            shutil.copy2(model_file, dest_file)
            copied_files.append(dest_file)
            print(f"✅ Copied: {model_file} → {dest_file}")
        except Exception as e:
            print(f"❌ Failed to copy {model_file}: {e}")
            return False
    
    # Copy metadata files
    for metadata_file in metadata_files:
        try:
            dest_file = sentio_models_dir / metadata_file.name
            shutil.copy2(metadata_file, dest_file)
            copied_files.append(dest_file)
            print(f"✅ Copied: {metadata_file} → {dest_file}")
        except Exception as e:
            print(f"❌ Failed to copy {metadata_file}: {e}")
            return False
    
    print(f"\n🎉 Successfully copied {len(copied_files)} files to Sentio!")
    return True

def main():
    """Main training and deployment workflow"""
    print("🚀 PPO Model Training & Deployment Pipeline")
    print("=" * 60)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"⏱️  Training duration: 30 minutes per model")
    print("=" * 60)
    
    # Change to training directory
    training_dir = Path("/Users/yeogirlyun/Python/Sentio_PPO_Trainer")
    if not training_dir.exists():
        print(f"❌ Training directory not found: {training_dir}")
        return False
    
    os.chdir(training_dir)
    print(f"📂 Changed to training directory: {training_dir}")
    
    # Generate unique timestamp for this training session
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Training configurations
    training_jobs = [
        {
            "name": "Maskable PPO (Primary Model)",
            "command": f"python -m models.ppo_trainer --model-type maskable_ppo --minutes 30 --output maskable_ppo_30min_{timestamp}",
            "output_pattern": f"maskable_ppo_30min_{timestamp}",
            "priority": "HIGH"
        },
        {
            "name": "Standard PPO (Baseline Model)", 
            "command": f"python -m models.ppo_trainer --model-type ppo --minutes 30 --output standard_ppo_30min_{timestamp}",
            "output_pattern": f"standard_ppo_30min_{timestamp}",
            "priority": "MEDIUM"
        }
    ]
    
    successful_models = []
    
    # Train each model
    for i, job in enumerate(training_jobs, 1):
        print(f"\n{'='*60}")
        print(f"🎯 Training Job {i}/2: {job['name']}")
        print(f"⭐ Priority: {job['priority']}")
        print(f"📋 Output Pattern: {job['output_pattern']}")
        print(f"{'='*60}")
        
        start_time = time.time()
        success = run_command(job['command'], f"Training {job['name']}")
        end_time = time.time()
        
        duration_minutes = (end_time - start_time) / 60
        print(f"⏱️  Training duration: {duration_minutes:.1f} minutes")
        
        if success:
            successful_models.append(job)
            print(f"✅ {job['name']} training completed successfully!")
        else:
            print(f"❌ {job['name']} training failed!")
    
    # Find and copy trained models
    if successful_models:
        print(f"\n{'='*60}")
        print(f"📦 MODEL DEPLOYMENT PHASE")
        print(f"✅ Successfully trained {len(successful_models)} models")
        print(f"{'='*60}")
        
        all_model_files = []
        all_metadata_files = []
        
        # Collect all trained model files
        for job in successful_models:
            print(f"\n🔍 Looking for {job['name']} files...")
            model_files, metadata_files = find_latest_model_files(job['output_pattern'])
            
            if model_files:
                print(f"📄 Found model files: {[f.name for f in model_files]}")
                all_model_files.extend(model_files)
            else:
                print(f"⚠️  No model files found for pattern: {job['output_pattern']}")
            
            if metadata_files:
                print(f"📋 Found metadata files: {[f.name for f in metadata_files]}")
                all_metadata_files.extend(metadata_files)
            else:
                print(f"⚠️  No metadata files found for pattern: {job['output_pattern']}")
        
        # Copy to Sentio
        if all_model_files or all_metadata_files:
            success = copy_models_to_sentio(all_model_files, all_metadata_files)
            
            if success:
                print(f"\n🎉 DEPLOYMENT SUCCESSFUL!")
                print(f"📊 Models ready for testing in Sentio Strategy Hub")
                print(f"🔬 You can now run walk-forward and other tests")
                
                # Display summary
                print(f"\n📋 TRAINING SUMMARY:")
                print(f"   ✅ Trained Models: {len(successful_models)}")
                print(f"   📄 Model Files: {len(all_model_files)}")
                print(f"   📋 Metadata Files: {len(all_metadata_files)}")
                print(f"   📁 Deployed to: ../Sentio/models/")
                
                if len(successful_models) == 2:
                    print(f"\n🏆 COMPLETE SUCCESS: Both PPO and Maskable PPO models trained!")
                    print(f"🎯 Primary Focus: Maskable PPO (with action masking & risk management)")
                    print(f"📊 Baseline: Standard PPO (for performance comparison)")
                else:
                    print(f"\n⚠️  PARTIAL SUCCESS: {len(successful_models)}/2 models trained")
                
                return True
            else:
                print(f"\n❌ DEPLOYMENT FAILED!")
                return False
        else:
            print(f"\n❌ No model files found to deploy!")
            return False
    else:
        print(f"\n❌ NO MODELS TRAINED SUCCESSFULLY!")
        return False

if __name__ == "__main__":
    success = main()
    
    print(f"\n{'='*60}")
    if success:
        print(f"🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print(f"🚀 Ready to test models in Sentio Strategy Hub")
    else:
        print(f"❌ PIPELINE FAILED!")
        print(f"🔧 Check the error messages above for troubleshooting")
    print(f"📅 Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    sys.exit(0 if success else 1)
