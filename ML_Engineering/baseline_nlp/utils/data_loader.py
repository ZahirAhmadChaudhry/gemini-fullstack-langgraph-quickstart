"""
Data loading utilities for the Baseline NLP System.
"""

import os
import json
import logging
from glob import glob
from typing import Dict, List, Any, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Handles loading of preprocessed data from Data Engineer.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize DataLoader with directory containing preprocessed data.
        
        Args:
            data_dir (str): Path to directory with preprocessed JSON files
        """
        self.data_dir = data_dir
        self._validate_data_dir()
        
    def _validate_data_dir(self) -> None:
        """Validate that data directory exists and contains JSON files."""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory {self.data_dir} not found")
        
        json_files = glob(os.path.join(self.data_dir, "*.json"))
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {self.data_dir}")
        
        logger.info(f"Found {len(json_files)} JSON files in {self.data_dir}")
    
    def load_file(self, filename: str) -> Dict[str, Any]:
        """
        Load a single preprocessed JSON file.
        
        Args:
            filename (str): Name of the JSON file to load
            
        Returns:
            dict: The loaded JSON data
        """
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filepath} not found")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Successfully loaded {filepath}")
                return data
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading {filepath}: {str(e)}")
            raise
    
    def load_all_files(self) -> List[Dict[str, Any]]:
        """
        Load all preprocessed JSON files in the data directory.
        
        Returns:
            list: A list of dictionaries containing the loaded JSON data
        """
        json_files = glob(os.path.join(self.data_dir, "*.json"))
        data = []
        
        for file_path in json_files:
            try:
                filename = os.path.basename(file_path)
                file_data = self.load_file(filename)
                data.append(file_data)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
        
        logger.info(f"Loaded {len(data)} files successfully")
        return data
    
    def extract_segments(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Extract segments from loaded data for processing.

        Args:
            data: Either a single loaded JSON file or list of loaded JSON files

        Returns:
            list: A list of segment dictionaries ready for processing
        """
        if isinstance(data, dict):
            data = [data]

        all_segments = []

        for doc_data in data: # doc_data is the loaded preprocessed JSON dictionary
            # Extract document metadata
            doc_metadata = self._extract_document_metadata(doc_data)
            doc_identifier = doc_metadata["doc_id"]

            if not doc_identifier:
                logger.error(f"Cannot identify document - skipping")
                continue

            segments_list = doc_data.get("segments", [])

            if not segments_list:
                logger.warning(f"No segments found in document '{doc_identifier}'")
                continue

            for i, segment_content_dict in enumerate(segments_list):
                # Process and validate segment
                processed_segment = self._process_segment(
                    segment_content_dict, doc_metadata, i
                )

                if processed_segment:
                    all_segments.append(processed_segment)

        logger.info(f"Extracted {len(all_segments)} segments for processing")
        return all_segments

    def _extract_document_metadata(self, doc_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and validate document-level metadata.

        Args:
            doc_data: Document data from JSON

        Returns:
            Dictionary with document metadata
        """
        # Prioritize "doc_id" if present (should be the raw_file_id UUID)
        # Fallback to "original_filename_display" or "filename" if "doc_id" is missing.
        doc_identifier = doc_data.get("doc_id")
        if not doc_identifier:
            doc_identifier = doc_data.get("original_filename_display",
                                        doc_data.get("filename",
                                                   doc_data.get("source_file", "unknown_document")))
            if doc_identifier != "unknown_document":
                logger.warning(f"Preprocessed JSON missing 'doc_id'. Using '{doc_identifier}' as document identifier.")
            else:
                logger.error(f"Preprocessed JSON missing 'doc_id' and any fallback filename.")
                return {"doc_id": None}

        return {
            "doc_id": doc_identifier,
            "source_file": doc_data.get("source_file", ""),
            "processed_timestamp": doc_data.get("processed_timestamp", ""),
            "total_segments": len(doc_data.get("segments", []))
        }

    def _process_segment(self, segment_dict: Dict[str, Any],
                        doc_metadata: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
        """
        Process and validate a single segment.

        Args:
            segment_dict: Raw segment data
            doc_metadata: Document metadata
            index: Segment index

        Returns:
            Processed segment or None if invalid
        """
        segment = segment_dict.copy()
        doc_id = doc_metadata["doc_id"]

        # Handle field name compatibility
        if "segment_text" in segment and "text" not in segment:
            segment["text"] = segment["segment_text"]
            logger.info(f"Converted 'segment_text' to 'text' field for segment {index} in document {doc_id}")

        # Validate required text field
        if not segment.get("text"):
            logger.warning(f"Empty text in segment {index} of document {doc_id}")
            return None

        # Add enhanced metadata
        segment["source_doc_id"] = doc_id
        segment["segment_id"] = segment.get("id", f"{doc_id}_seg_{index:03d}")
        segment["segment_index"] = index

        # Validate and enhance features
        segment = self._validate_and_enhance_features(segment, doc_id, index)

        return segment

    def _validate_and_enhance_features(self, segment: Dict[str, Any],
                                     doc_id: str, index: int) -> Dict[str, Any]:
        """
        Validate and enhance segment features from new JSON format.

        Args:
            segment: Segment data
            doc_id: Document identifier
            index: Segment index

        Returns:
            Enhanced segment with validated features
        """
        # Extract and validate features object
        features = segment.get("features", {})

        # Validate temporal context
        temporal_context = features.get("temporal_context", "unknown")
        if temporal_context not in ["unknown", "2023", "2050", "present", "future"]:
            logger.warning(f"Invalid temporal_context '{temporal_context}' in segment {index} of {doc_id}")
            features["temporal_context"] = "unknown"

        # Validate discourse markers
        discourse_markers = features.get("discourse_markers", [])
        if not isinstance(discourse_markers, list):
            logger.warning(f"Invalid discourse_markers format in segment {index} of {doc_id}")
            features["discourse_markers"] = []

        # Validate counts
        for count_field in ["sentence_count", "word_count"]:
            count_value = features.get(count_field, 0)
            if not isinstance(count_value, int) or count_value < 0:
                logger.warning(f"Invalid {count_field} in segment {index} of {doc_id}")
                features[count_field] = 0

        # Validate noun phrases
        noun_phrases = features.get("noun_phrases", [])
        if not isinstance(noun_phrases, list):
            logger.warning(f"Invalid noun_phrases format in segment {index} of {doc_id}")
            features["noun_phrases"] = []

        # Update segment with validated features
        segment["features"] = features

        return segment
    
    def load_segments(self) -> List[Dict[str, Any]]:
        """
        Load and extract all segments from all files in one step.
        
        Returns:
            list: A list of all segments ready for processing
        """
        data = self.load_all_files()
        return self.extract_segments(data)