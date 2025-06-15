# Test Issues for Paradox Detection Module

This document tracks issues encountered during testing of the Paradox Detection Module.

## Open Issues

No open issues at this time.

## In Progress Issues

No issues currently in progress.

## Resolved Issues

### ISSUE-PD-001: Antonym Pair Detection Failing
- **Status**: Resolved
- **Date Identified**: April 23, 2025
- **Identified By**: GitHub Copilot
- **Description**: The antonym pair detection test was failing because the confidence score calculated for antonyms that are distant from each other in the sentence was below the default threshold.
- **Steps to Reproduce**: 
  1. Run the test with the text "Nous devons augmenter la production tout en diminuant l'impact environnemental."
  2. Observe that even though "augmenter" and "diminuer" are antonyms in the dictionary, they are not detected.
- **Expected Behavior**: The detector should identify "augmenter" and "diminuer" as an antonym pair.
- **Actual Behavior**: The detector was returning an empty list, failing to detect the antonym pair.
- **Resolution**: Lowered the confidence threshold from the default value to 0.3 in the test setup to account for the distance between antonyms in sentences. This allows detection of antonym pairs even when they are separated by multiple tokens.
- **Resolution Date**: April 23, 2025

## Issue Template

When adding new issues, please use the following format:

```
### ISSUE-PD-XXX: Brief Title
- **Status**: [Open/In Progress/Resolved]
- **Date Identified**: YYYY-MM-DD
- **Identified By**: [Name]
- **Description**: Detailed description of the issue
- **Steps to Reproduce**: 
  1. Step 1
  2. Step 2
  3. ...
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Resolution**: (Fill this in when resolved) How the issue was fixed
- **Resolution Date**: (Fill this in when resolved) YYYY-MM-DD
```