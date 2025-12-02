# test_complete.py
import os
import sys

print("=" * 60)
print("FINAL PROJECT STRUCTURE CHECK")
print("=" * 60)

required_files = [
    ('app.py', 'Main Streamlit app'),
    ('requirements.txt', 'Dependencies list'),
    ('pages/02_batch_processing.py', 'Batch processing page'),
    ('utils/batch_processor.py', 'Batch processor module'),
    ('model/', 'Model folder (can be empty)'),
    ('assets/', 'Assets folder (can be empty)'),
]

print("\n📁 Checking project structure:")
all_good = True
for file_path, description in required_files:
    exists = os.path.exists(file_path)
    status = "✅" if exists else "❌"
    print(f"  {status} {file_path:30} - {description}")
    if not exists:
        all_good = False

print("\n📦 Checking requirements.txt:")
if os.path.exists('requirements.txt'):
    with open('requirements.txt', 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f"  Found {len(lines)} packages")
    for line in lines[:5]:  # Show first 5
        print(f"    {line}")
    if len(lines) > 5:
        print(f"    ... and {len(lines)-5} more")
else:
    print("  ❌ requirements.txt not found")

print("\n🔧 Testing imports:")
try:
    sys.path.append('.')
    from utils.batch_processor import BatchProcessor
    print("  ✅ utils.batch_processor imports successfully")
    
    # Test creating processor
    processor = BatchProcessor()
    print(f"  ✅ BatchProcessor created - {len(processor.required_columns)} required columns")
    
except Exception as e:
    print(f"  ❌ Import test failed: {e}")
    all_good = False

print("\n🌐 Testing Streamlit page:")
try:
    with open('pages/02_batch_processing.py', 'r') as f:
        content = f.read()
        checks = [
            ('import streamlit', 'Streamlit import'),
            ('BatchProcessor', 'BatchProcessor import'),
            ('st.file_uploader', 'File uploader'),
            ('st.button', 'Buttons'),
            ('plotly', 'Plotly charts'),
        ]
        
        for check, desc in checks:
            if check in content:
                print(f"  ✅ {desc} found")
            else:
                print(f"  ⚠️  {desc} not found")
except Exception as e:
    print(f"  ❌ Page test failed: {e}")
    all_good = False

print("\n" + "=" * 60)
if all_good:
    print("🎉 ALL CHECKS PASSED! Your project is ready!")
    print("\nNext steps:")
    print("1. Run: streamlit run app.py")
    print("2. Test the batch processing page")
    print("3. Deploy to Streamlit Cloud")
else:
    print("⚠️  Some checks failed. Please fix the issues above.")
print("=" * 60)