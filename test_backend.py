#!/usr/bin/env python3
"""
Test script for the refactored French sustainability transcript analyzer.
"""

import os
import sys
import json
from pathlib import Path

# Add the backend source to the path
backend_path = Path(__file__).parent / "backend" / "src"
sys.path.insert(0, str(backend_path))

from agent.graph import graph
from agent.state import OverallState
from langchain_core.messages import HumanMessage

# Sample French transcript for testing
SAMPLE_TRANSCRIPT = """
3CJ : Mais alors d'un point de vue macro-économique, on n'est plus dans l'entreprise, parce que ça va... ça requestionne après le revenu universel, etc., si les gens ont plus de travail, si on veut pas que ce soit la guerre avec la destruction, il faut leur donner à manger pour qu'ils puissent... il faut leur donner de l'argent pour qu'ils puissent vivre et pas se tuer les uns les autres. Donc là, le revenu universel, quoi, mais ça, c'est sociétal, c'est politique, ça. Si tu veux pas que les gens se mettent sur la gueule, il faut leur... il faut leur donner un revenu. C'est normal. 

1CJ : Bah oui, oui. 

2CJ : Oui, mais bon, c'était ça aussi, quoi. 

3CJ : Donc c'est bien la société au sens large qui prend soin des... des plus démunis. 

1CJ : Oui, complètement, oui. 

3CJ : À condition de créer de la richesse pour la redistribuer. 

1CJ : Oui, on va pas... on va pas redistribuer ce qu'on n'a pas. 

3CJ : Parce que si on part du principe qu'il faut redistribuer des revenus, c'est le capital.
"""

def test_backend():
    """Test the refactored backend with a sample transcript."""
    
    print("🧪 Testing French Sustainability Transcript Analyzer Backend")
    print("=" * 60)
    
    # Check if GEMINI_API_KEY is set
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not found in environment variables")
        print("Please set your Gemini API key:")
        print("export GEMINI_API_KEY='your-api-key-here'")
        return False
    
    print("✅ GEMINI_API_KEY found")
    
    # Prepare the initial state
    initial_state = {
        "messages": [HumanMessage(content="Analyze this French sustainability transcript")],
        "transcript": SAMPLE_TRANSCRIPT,
        "preprocessed_data": {},
        "segments": [],
        "analysis_results": [],
        "final_results": [],
        "max_segments": 10
    }
    
    print(f"📄 Sample transcript length: {len(SAMPLE_TRANSCRIPT)} characters")
    print("🚀 Starting analysis...")
    
    try:
        # Run the graph
        result = graph.invoke(initial_state)
        
        print("✅ Analysis completed successfully!")
        print(f"📊 Results summary:")
        print(f"   - Segments identified: {len(result.get('segments', []))}")
        print(f"   - Tensions analyzed: {len(result.get('analysis_results', []))}")
        print(f"   - Final results: {len(result.get('final_results', []))}")
        
        # Display some results
        if result.get('final_results'):
            print("\n📋 Sample results:")
            for i, tension in enumerate(result['final_results'][:2]):  # Show first 2
                print(f"\n   Tension {i+1}:")
                print(f"   - Concept: {tension.get('Concepts de 2nd ordre', 'N/A')}")
                print(f"   - Reformulated: {tension.get('Items de 1er ordre reformulé', 'N/A')}")
                print(f"   - Theme: {tension.get('Thème', 'N/A')}")
                print(f"   - Period: {tension.get('Période', 'N/A')}")
        
        # Display final message
        if result.get('messages'):
            final_message = result['messages'][-1]
            print(f"\n💬 Final message: {final_message.content}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def test_individual_components():
    """Test individual components of the system."""
    
    print("\n🔧 Testing Individual Components")
    print("=" * 40)
    
    try:
        # Test utilities
        from agent.utils import clean_transcript, detect_period, determine_theme, assign_code
        
        print("✅ Utilities imported successfully")
        
        # Test clean_transcript
        dirty_text = "[Music] Speaker 1: Hello 12:34 world [Applause]"
        clean_text = clean_transcript(dirty_text)
        print(f"   Clean transcript: '{clean_text}'")
        
        # Test period detection
        period_2050 = detect_period("En 2050, nous devrons...")
        period_2023 = detect_period("Actuellement en 2023...")
        print(f"   Period detection: 2050='{period_2050}', 2023='{period_2023}'")
        
        # Test theme determination
        theme_perf = determine_theme("profit rentabilité efficacité")
        theme_legit = determine_theme("transparence équité environnement")
        print(f"   Theme detection: performance='{theme_perf}', legitimacy='{theme_legit}'")
        
        # Test code assignment
        code = assign_code("Accumulation vs Partage", "MODELES SOCIO-ECONOMIQUES")
        print(f"   Code assignment: '{code}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("🎯 French Sustainability Transcript Analyzer - Backend Test")
    print("=" * 70)
    
    # Test individual components first
    components_ok = test_individual_components()
    
    if components_ok:
        # Test full backend
        backend_ok = test_backend()
        
        if backend_ok:
            print("\n🎉 All tests passed! Backend is ready.")
            print("\nNext steps:")
            print("1. Test with your data engineering pipeline output")
            print("2. Refactor the frontend")
            print("3. Test end-to-end workflow")
        else:
            print("\n⚠️  Backend test failed. Check the errors above.")
    else:
        print("\n⚠️  Component tests failed. Check the errors above.")
