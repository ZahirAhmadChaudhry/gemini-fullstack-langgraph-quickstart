"""Test runner for the French transcript preprocessing pipeline."""

import unittest
import sys
import json
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from test.config import TEST_RESULTS_DIR, TEST_LOGS_DIR, logger

def run_test_suite() -> dict:
    """Run all test cases and return results."""
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=str(Path(__file__).parent / "cases"),
        pattern="test_*.py",
        top_level_dir=str(PROJECT_ROOT)
    )
    
    # Configure test result collection
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Compile results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    test_results = {
        "timestamp": timestamp,
        "total_tests": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "success": result.wasSuccessful(),
        "failure_details": [
            {
                "test": test[0].id(),
                "error": str(test[1])
            }
            for test in result.failures + result.errors
        ]
    }
    
    # Save results
    results_file = TEST_RESULTS_DIR / f"test_results_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    return test_results

def print_results(results: dict) -> None:
    """Print test results in a readable format."""
    print("\n=== Test Execution Summary ===")
    print(f"Timestamp: {results['timestamp']}")
    print(f"Total Tests Run: {results['total_tests']}")
    print(f"Successes: {results['total_tests'] - results['failures'] - results['errors']}")
    print(f"Failures: {results['failures']}")
    print(f"Errors: {results['errors']}")
    print(f"Skipped: {results['skipped']}")
    print(f"Overall Success: {'Yes' if results['success'] else 'No'}")
    
    if results['failure_details']:
        print("\n=== Failure Details ===")
        for detail in results['failure_details']:
            print(f"\nTest: {detail['test']}")
            print(f"Error: {detail['error']}")

if __name__ == '__main__':
    logger.info("Starting test execution")
    results = run_test_suite()
    print_results(results)
    sys.exit(0 if results["success"] else 1)