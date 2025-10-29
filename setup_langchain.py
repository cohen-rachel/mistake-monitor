#!/usr/bin/env python3
"""
Setup script for LangChain Language Tutor
Installs dependencies and tests the system
"""
import subprocess
import sys
import os
import asyncio
import httpx

def install_requirements():
    """Install LangChain requirements"""
    print("📦 Installing LangChain requirements...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements_langchain.txt"], check=True)
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_ollama():
    """Check if Ollama is running and has the required model"""
    print("🔍 Checking Ollama...")
    
    try:
        # Check if Ollama is running
        response = httpx.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model.get("name", "") for model in models]
            
            if "llama2:latest" in model_names:
                print("✅ Ollama is running with llama2:latest model")
                return True
            else:
                print("⚠️  Ollama is running but llama2:latest not found")
                print(f"Available models: {model_names}")
                print("Run: ollama pull llama2:latest")
                return False
        else:
            print("❌ Ollama is not responding")
            return False
            
    except httpx.ConnectError:
        print("❌ Ollama is not running")
        print("Start Ollama with: ollama serve")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    required_modules = [
        "langchain",
        "langchain.llms",
        "langchain.prompts",
        "langchain.chains",
        "langchain.memory",
        "langchain.vectorstores",
        "langchain.embeddings",
        "pydantic",
        "chromadb"
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {failed_imports}")
        return False
    else:
        print("✅ All imports successful")
        return True

async def test_langchain_tutor():
    """Test the LangChain tutor with a simple example"""
    print("🎯 Testing LangChain tutor...")
    
    try:
        from langchain_tutor import AdvancedLanguageTutor
        
        # Initialize tutor
        tutor = AdvancedLanguageTutor()
        
        # Test analysis
        test_text = "I goed to the store yesterday."
        analysis = await tutor.analyze_transcript(test_text, "test_session")
        
        print(f"✅ Analysis successful: {len(analysis.mistakes)} mistakes found")
        
        # Test practice generation
        if analysis.mistakes:
            exercise = await tutor.generate_practice_exercise(
                analysis.mistakes[0].type,
                analysis.difficulty_level,
                "test_user"
            )
            print(f"✅ Practice exercise generated: {exercise.prompt[:50]}...")
        
        # Test conversation
        response = await tutor.conversational_practice("Hello, I want to practice English")
        print(f"✅ Conversation test successful: {response[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ LangChain tutor test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 LangChain Language Tutor Setup")
    print("=" * 50)
    
    # Step 1: Install requirements
    if not install_requirements():
        print("❌ Setup failed at requirements installation")
        return False
    
    # Step 2: Test imports
    if not test_imports():
        print("❌ Setup failed at import testing")
        return False
    
    # Step 3: Check Ollama
    if not check_ollama():
        print("❌ Setup failed at Ollama check")
        print("\nTo fix Ollama issues:")
        print("1. Install Ollama: https://ollama.com/download")
        print("2. Start Ollama: ollama serve")
        print("3. Pull model: ollama pull llama2:latest")
        return False
    
    # Step 4: Test LangChain tutor
    print("\n🧪 Running LangChain tutor test...")
    try:
        result = asyncio.run(test_langchain_tutor())
        if not result:
            print("❌ Setup failed at LangChain tutor test")
            return False
    except Exception as e:
        print(f"❌ Setup failed at LangChain tutor test: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Test with audio: python langchain_tutor.py audio_file.wav")
    print("2. Try conversation mode: python langchain_tutor.py --conversation")
    print("3. Run feature tests: python test_langchain_features.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
