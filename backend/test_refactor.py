#!/usr/bin/env python3
"""
Simple test script for the refactored French sustainability transcript analyzer.
Run from the backend directory with: python3 test_refactor.py
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all our refactored modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        # Test basic imports
        from agent.state import OverallState, SegmentationState, AnalysisResult
        print("✅ State classes imported")
        
        from agent.tools_and_schemas import (
            SegmentsList, TensionExtraction, Categorization, 
            FullAnalysisResult, CONCEPT_CODE_MAPPING, THEME_KEYWORDS
        )
        print("✅ Schemas and tools imported")
        
        from agent.prompts import (
            segmentation_instructions, tension_extraction_instructions,
            categorization_instructions, synthesis_instructions,
            imaginaire_classification_instructions
        )
        print("✅ Prompts imported")
        
        from agent.utils import (
            clean_transcript, detect_period, determine_theme, 
            assign_code, format_csv
        )
        print("✅ Utilities imported")
        
        from agent.configuration import Configuration
        print("✅ Configuration imported")
        
        # Test graph import (this might fail if dependencies are missing)
        try:
            from agent.graph import graph
            print("✅ Graph imported successfully")
            return True
        except Exception as e:
            print(f"⚠️  Graph import failed: {e}")
            print("This might be due to missing API key or dependencies")
            return False
            
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_utilities():
    """Test utility functions."""
    print("\n🔧 Testing utility functions...")
    
    try:
        from agent.utils import clean_transcript, detect_period, determine_theme, assign_code
        
        # Test clean_transcript
        dirty = "[Music] Speaker: Hello 12:34 world [Applause]"
        clean = clean_transcript(dirty)
        print(f"   Clean transcript: '{clean}'")
        
        # Test period detection
        period_2050 = detect_period("En 2050, nous devrons...")
        period_2023 = detect_period("Actuellement en 2023...")
        print(f"   Period detection: 2050='{period_2050}', 2023='{period_2023}'")
        
        # Test theme determination
        theme_perf = determine_theme("profit rentabilité efficacité")
        theme_legit = determine_theme("transparence équité environnement")
        print(f"   Theme detection: performance='{theme_perf}', legitimacy='{theme_legit}'")
        
        # Test code assignment
        code = assign_code("Accumulation / Partage", "MODELES SOCIO-ECONOMIQUES")
        print(f"   Code assignment: '{code}'")
        
        print("✅ All utility functions working")
        return True
        
    except Exception as e:
        print(f"❌ Utility test failed: {e}")
        return False

def test_configuration():
    """Test configuration."""
    print("\n⚙️  Testing configuration...")
    
    try:
        from agent.configuration import Configuration
        
        config = Configuration()
        print(f"   Default model: {config.analysis_model}")
        print(f"   Max segments: {config.max_segments}")
        print(f"   Analysis temperature: {config.analysis_temperature}")
        
        print("✅ Configuration working")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_schemas():
    """Test Pydantic schemas."""
    print("\n📋 Testing schemas...")
    
    try:
        from agent.tools_and_schemas import SegmentsList, TensionExtraction, Categorization
        
        # Test SegmentsList
        segments = SegmentsList(segments=["segment 1", "segment 2"])
        print(f"   SegmentsList: {len(segments.segments)} segments")
        
        # Test TensionExtraction
        tension = TensionExtraction(
            original_item="original text",
            reformulated_item="A vs B"
        )
        print(f"   TensionExtraction: {tension.reformulated_item}")
        
        # Test Categorization
        cat = Categorization(concept="MODELES SOCIO-ECONOMIQUES", code="test.code")
        print(f"   Categorization: {cat.concept}")
        
        print("✅ All schemas working")
        return True
        
    except Exception as e:
        print(f"❌ Schema test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🎯 French Sustainability Transcript Analyzer - Refactor Test")
    print("=" * 65)
    
    # Check if we're in the right directory
    if not Path("src/agent").exists():
        print("❌ Please run this script from the backend directory")
        print("   cd backend && python3 test_refactor.py")
        return
    
    tests = [
        ("Imports", test_imports),
        ("Utilities", test_utilities), 
        ("Configuration", test_configuration),
        ("Schemas", test_schemas),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Test Summary:")
    print("=" * 20)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Refactor is working correctly.")
        print("\nNext steps:")
        print("1. Set GEMINI_API_KEY environment variable")
        print("2. Test with actual transcript data")
        print("3. Refactor the frontend")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
