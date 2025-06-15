#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Progress Updater for French Transcript Preprocessing

This module provides functionality to update the progress.md file
as the preprocessing steps are completed.
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional

logger = logging.getLogger(__name__)

class ProgressUpdater:
    """Class to update progress.md file during preprocessing."""
    def __init__(self, progress_file_path: Path = Path("memory/progress.md")):
        """Initialize with path to progress.md file."""
        self.progress_file_path = progress_file_path
        self.processed_files: Set[str] = set()
        
        # Create the directory if it doesn't exist
        if not self.progress_file_path.parent.exists():
            self.progress_file_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {self.progress_file_path.parent}")
            
        # Create the file if it doesn't exist
        if not self.progress_file_path.exists():
            with open(self.progress_file_path, 'w', encoding='utf-8') as f:
                f.write("# Preprocessing Progress\n\n## Files\n\n## Phases\n\n## Notes\n\n")
            logger.info(f"Created progress file: {self.progress_file_path}")
            
        logger.info(f"Initialized ProgressUpdater with file: {self.progress_file_path}")
    
    def read_progress_file(self) -> str:
        """Read the current content of the progress file."""
        try:
            with open(self.progress_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except Exception as e:
            logger.error(f"Error reading progress file: {e}")
            return ""
    
    def write_progress_file(self, content: str):
        """Write updated content to the progress file."""
        try:
            with open(self.progress_file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info("Progress file updated successfully")
        except Exception as e:
            logger.error(f"Error writing progress file: {e}")
    
    def update_file_progress(self, filename: str, phases_completed: List[str]):
        """
        Update progress for a specific file.
        
        Args:
            filename: Name of the processed transcript file
            phases_completed: List of phases completed for this file 
                              (e.g., ["encoding", "tokenization", "segmentation"])
        """
        logger.info(f"Updating progress for {filename} - Phases: {phases_completed}")
        
        # Add to processed files
        self.processed_files.add(filename)
        
        # Read current content
        content = self.read_progress_file()
        if not content:
            logger.warning("Progress file is empty or cannot be read")
            return
        
        # Update processed files section or create if it doesn't exist
        processed_files_section = "**Processed Files:**\n\n"
        
        if "**Processed Files:**" not in content:
            # Add section before Phase 6
            phase6_index = content.find("**Phase 6: Quality Control and Refinement**")
            if phase6_index != -1:
                content = content[:phase6_index] + processed_files_section + content[phase6_index:]
            else:
                # If Phase 6 not found, add at the end
                content += "\n\n" + processed_files_section
        
        # Update file status in processed files section
        file_entry_pattern = rf"\* \[.\] {re.escape(filename)}.*"
        file_entry_replacement = f"* [x] {filename} - Completed phases: {', '.join(phases_completed)}"
        
        processed_section_pattern = r"(\*\*Processed Files:\*\*\s*\n\n)((?:.|\n)*?)(\n\n\*\*|$)"
        processed_section_match = re.search(processed_section_pattern, content)
        
        if processed_section_match:
            # CRUDE FIX: Instead of replacing the whole section, just append to it or append the new line
            # This avoids loading a potentially huge section into memory for replacement
            # This will make progress.md less tidy but aims to prevent MemoryError
            
            # Find the end of the "Processed Files:" line to append after it
            processed_files_header_end = content.find("**Processed Files:**")
            if processed_files_header_end != -1:
                # Find the next blank line or next major section to insert before
                insert_point = content.find("\\n\\n**", processed_files_header_end)
                if insert_point == -1: # If no other major section, append at the end of "Processed Files" block
                    # Try to find where the list of files ends more loosely
                    end_of_files_list_match = re.search(r"(\\*\\s*\\[[x ]\\]\\s*\\S+[\\s\\S]*?)(\\n\\n\\*\\*|$)", content[processed_files_header_end:])
                    if end_of_files_list_match:
                        insert_point = processed_files_header_end + end_of_files_list_match.end(1)
                    else: # Absolute fallback, just append after header (might be messy)
                         insert_point = content.find("\\n", processed_files_header_end + len("**Processed Files:**"))
                         if insert_point == -1: insert_point = len(content) # append to end of file

                # Simpler append:
                # Find the line "**Processed Files:**" and insert the new entry after that line.
                # This is not ideal as it doesn't update existing entries, but aims to prevent MemoryError.
                header_line_end = content.find("\\n", content.find("**Processed Files:**"))
                if header_line_end != -1:
                    content = content[:header_line_end+1] + file_entry_replacement + "\\n" + content[header_line_end+1:]
                else: # Fallback if structure is unexpected
                    content += f"\\n{file_entry_replacement}\\n"

            else: # Fallback: add to end if "**Processed Files:**" header not found (should not happen based on init)
                content += f"\\n\\n**Processed Files:**\\n\\n{file_entry_replacement}\\n"
            
            # Original problematic code:
            # processed_section = processed_section_match.group(2)
            # # Check if file already in list
            # if re.search(file_entry_pattern, processed_section):
            #     # Update existing entry
            #     updated_section = re.sub(file_entry_pattern, file_entry_replacement, processed_section)
            # else:
            #     # Add new entry
            #     updated_section = processed_section + f"* [x] {filename} - Completed phases: {', '.join(phases_completed)}\\n"
            # # Replace the old section with updated one
            # content = content.replace(processed_section_match.group(2), updated_section)

        else:
            # If we can't find the processed files section in the expected format
            processed_files_index = content.find("**Processed Files:**")
            if processed_files_index != -1:
                # Find the end of this section
                next_section_index = content.find("**", processed_files_index + 17)
                if next_section_index != -1:
                    content = content[:processed_files_index + 19] + "\n" + file_entry_replacement + "\n\n" + content[next_section_index:]
                else:
                    content += f"\n{file_entry_replacement}\n"
            else:
                # Fallback: add to end
                content += f"\n\n**Processed Files:**\n\n{file_entry_replacement}\n"
        
        # Update checkbox for each phase based on completed phases
        if "encoding" in phases_completed:
            content = self._update_checkbox(content, "Determined the current character encoding")
            content = self._update_checkbox(content, "Converted the encoding to UTF-8")
        
        if "tokenization" in phases_completed:
            content = self._update_checkbox(content, "Performed tokenization")
        
        if "sentence" in phases_completed:
            content = self._update_checkbox(content, "Performed sentence segmentation")
        
        if "lemmatization" in phases_completed:
            content = self._update_checkbox(content, "Performed lemmatization")
        
        if "segmentation" in phases_completed:
            content = self._update_checkbox(content, "Segmented the text into chunks")
            content = self._update_checkbox(content, "Ensured semantic coherence")
        
        if "temporal" in phases_completed:
            content = self._update_checkbox(content, "Identified temporal markers")
        
        # Write updated content
        self.write_progress_file(content)
    
    def _update_checkbox(self, content: str, task_text: str) -> str:
        """Update a checkbox in the progress file."""
        # Replace [ ] with [x] for the given task
        pattern = rf"(\* \[) (\] {re.escape(task_text)})"
        return re.sub(pattern, r"\1x\2", content)
    
    def mark_phase_complete(self, phase: str):
        """Mark a complete phase as done in the progress file."""
        logger.info(f"Marking phase {phase} as complete")
        
        content = self.read_progress_file()
        if not content:
            logger.warning("Progress file is empty or cannot be read")
            return
        
        # Phase mapping to section titles
        phase_titles = {
            "phase1": "**Phase 1: Data Ingestion and Encoding Standardization**",
            "phase2": "**Phase 2: Preprocessing Tasks**",
            "phase3": "**Phase 3: Text Segmentation for Analysis**",
            "phase4": "**Phase 4: Temporal Marker Identification**",
            "phase5": "**Phase 5: Handling PDF Transcripts (If Applicable)**",
            "phase6": "**Phase 6: Quality Control and Refinement**"
        }
        
        if phase not in phase_titles:
            logger.warning(f"Unknown phase: {phase}")
            return
        
        phase_title = phase_titles[phase]
        phase_start = content.find(phase_title)
        
        if phase_start == -1:
            logger.warning(f"Could not find section for {phase} in progress file")
            return
        
        # Find the next phase or end of file
        next_phase_start = content.find("**Phase", phase_start + len(phase_title))
        if next_phase_start == -1:
            next_phase_start = len(content)
        
        # Extract phase section
        phase_section = content[phase_start:next_phase_start]
        
        # Update all checkboxes in this section
        updated_phase_section = re.sub(r"\* \[ \]", "* [x]", phase_section)
        
        # Replace the section in the content
        content = content[:phase_start] + updated_phase_section + content[next_phase_start:]
        
        self.write_progress_file(content)
    
    def update_located_files(self, files: List[str]):
        """Update the 'Located all French language transcript files' task."""
        content = self.read_progress_file()
        if not content:
            logger.warning("Progress file is empty or cannot be read")
            return
        
        # Update the checkbox
        content = self._update_checkbox(content, "Located all French language transcript files")
        
        # Add the list of files if not already present
        files_list_pattern = r"\* \[x\] Located all French language transcript files\.\s*\n(\s*\* Files: .*)?"
        files_list_text = f"* Files: {', '.join(files)}"
        
        if re.search(files_list_pattern, content):
            content = re.sub(files_list_pattern, f"* [x] Located all French language transcript files.\n  {files_list_text}", content)
        
        self.write_progress_file(content)
    
    def add_notes(self, note: str):
        """Add a note to the Notes section."""
        content = self.read_progress_file()
        if not content:
            logger.warning("Progress file is empty or cannot be read")
            return
        
        notes_pattern = r"(\*\*Notes:\*\*\s*\n\n)((?:.|\n)*?)($)"
        notes_match = re.search(notes_pattern, content)
        
        if notes_match:
            # Add note to existing notes section
            notes_section = notes_match.group(2)
            updated_notes = notes_section + f"* {note}\n"
            content = content.replace(notes_section, updated_notes)
        else:
            # If Notes section not found, add it at the end
            notes_section_index = content.find("**Notes:**")
            if notes_section_index != -1:
                # Notes section exists but not in expected format
                next_line_index = content.find("\n", notes_section_index)
                if next_line_index != -1:
                    content = content[:next_line_index + 1] + "\n* " + note + "\n" + content[next_line_index + 1:]
                else:
                    content += f"\n* {note}\n"
            else:
                # No Notes section, add it
                content += f"\n\n**Notes:**\n\n* {note}\n"
        
        self.write_progress_file(content)