#!/usr/bin/env python3
"""
Setup script for Diabetes Prediction Web App
"""

import os
import sys
import subprocess

def create_directory_structure():
    """Create necessary directories"""
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("✓ Created templates directory")
    else:
        print("✓ Templates directory already exists")

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✓ All packages installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages. Please install manually using:")
        print("pip install -r requirements.txt")
        return False
    return True

def check_dataset():
    """Check if dataset exists"""
    if os.path.exists('diabetes_prediction_dataset.xlsx'):
        print("✓ Dataset file found")
        return True
    else:
        print("❌ Dataset file 'diabetes_prediction_dataset.xlsx' not found")
        print("Please place your Excel dataset file in the same directory")
        return False

def main():
    print("🩺 Setting up Diabetes Prediction Web App")
    print("=" * 50)
    
    # Create directory structure
    create_directory_structure()
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Check dataset
    if not check_dataset():
        print("\n📋 Please ensure you have the following files:")
        print("   - diabetes_prediction_dataset.xlsx (your dataset)")
        print("   - app.py (Flask backend)")
        print("   - templates/index.html (frontend)")
        print("   - requirements.txt (dependencies)")
        return False
    
    print("\n🎉 Setup complete!")
    print("\n🚀 To run the application:")
    print("   python app.py")
    print("\n🌐 Then open your browser and go to:")
    print("   http://localhost:5000")
    
    return True

if __name__ == "__main__":
    if main():
        print("\n" + "=" * 50)
        print("Ready to start the web application! 🎊")
    else:
        print("\n" + "=" * 50)
        print("Setup incomplete. Please fix the issues above.")
        sys.exit(1)