#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration Management for French Transcript Preprocessing Pipeline

This module provides centralized configuration management for all pipeline components.
Supports environment-specific configurations and runtime parameter adjustment.

Author: Enhanced French Transcript Preprocessing Pipeline
Version: 2.0.0
Date: 2025-06-12
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum

class ProcessingMode(Enum):
    """Processing modes for different use cases."""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"
    BENCHMARK = "benchmark"

class SegmentationStrategy(Enum):
    """Segmentation strategies for different content types."""
    AUTO = "auto"  # Automatically detect and choose strategy
    SENTENCE_BASED = "sentence_based"  # Traditional sentence-based segmentation
    WORD_BASED = "word_based"  # Word-based segmentation for transcripts
    HYBRID = "hybrid"  # Combination approach

@dataclass
class SegmentationConfig:
    """Configuration for text segmentation."""
    strategy: SegmentationStrategy = SegmentationStrategy.AUTO
    min_segment_lines: int = 2
    max_segment_lines: int = 10
    target_words_per_segment: int = 150
    min_words_per_segment: int = 80
    max_words_per_segment: int = 300
    transcript_detection_threshold: float = 500.0  # avg words per sentence
    coherence_threshold: float = 0.8

@dataclass
class MLConfig:
    """Configuration for ML-ready output generation."""
    enable_enhanced_features: bool = True
    enable_target_format: bool = True
    enable_tension_detection: bool = True
    enable_thematic_analysis: bool = True
    enable_temporal_classification: bool = True
    enable_conceptual_markers: bool = True
    quality_threshold: float = 0.5
    confidence_threshold: float = 0.7

@dataclass
class ProcessingConfig:
    """Configuration for file processing."""
    supported_formats: List[str] = None
    max_file_size_mb: int = 100
    encoding_detection_confidence: float = 0.8
    enable_memory_optimization: bool = True
    enable_parallel_processing: bool = False
    max_workers: int = 4

    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = [".txt", ".docx", ".pdf", ".json"]

@dataclass
class OutputConfig:
    """Configuration for output generation."""
    base_output_dir: str = "preprocessed_data"
    create_standard_output: bool = True
    create_ml_ready_output: bool = True
    create_target_format_output: bool = True
    preserve_backward_compatibility: bool = True
    enable_progress_tracking: bool = True

@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "preprocessing.log"
    enable_console_output: bool = True
    enable_file_output: bool = True
    max_file_size_mb: int = 10
    backup_count: int = 5

@dataclass
class NLPConfig:
    """Configuration for NLP processing."""
    spacy_model: str = "fr_core_news_lg"
    enable_stanza: bool = False
    stanza_model: str = "fr"
    enable_noun_phrase_extraction: bool = True
    enable_discourse_analysis: bool = True
    enable_temporal_analysis: bool = True

class PipelineConfig:
    """Main configuration class for the preprocessing pipeline."""
    
    def __init__(self, mode: ProcessingMode = ProcessingMode.PRODUCTION, config_file: Optional[str] = None):
        """
        Initialize pipeline configuration.
        
        Args:
            mode: Processing mode (development, production, testing, benchmark)
            config_file: Optional path to JSON configuration file
        """
        self.mode = mode
        self.config_file = config_file
        
        # Initialize default configurations
        self.segmentation = SegmentationConfig()
        self.ml = MLConfig()
        self.processing = ProcessingConfig()
        self.output = OutputConfig()
        self.logging = LoggingConfig()
        self.nlp = NLPConfig()
        
        # Apply mode-specific configurations
        self._apply_mode_config()
        
        # Load from file if provided
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
        
        # Apply environment variable overrides
        self._apply_env_overrides()
    
    def _apply_mode_config(self):
        """Apply mode-specific configuration adjustments."""
        if self.mode == ProcessingMode.DEVELOPMENT:
            self.logging.level = "DEBUG"
            self.processing.enable_parallel_processing = False
            self.output.enable_progress_tracking = True
            
        elif self.mode == ProcessingMode.PRODUCTION:
            self.logging.level = "INFO"
            self.processing.enable_parallel_processing = True
            self.processing.enable_memory_optimization = True
            self.output.enable_progress_tracking = False
            
        elif self.mode == ProcessingMode.TESTING:
            self.logging.level = "WARNING"
            self.output.base_output_dir = "test/preprocessed_data"
            self.processing.enable_parallel_processing = False
            
        elif self.mode == ProcessingMode.BENCHMARK:
            self.logging.level = "ERROR"
            self.output.enable_progress_tracking = False
            self.processing.enable_parallel_processing = True
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        # Logging level override
        if "PIPELINE_LOG_LEVEL" in os.environ:
            self.logging.level = os.environ["PIPELINE_LOG_LEVEL"]
        
        # Output directory override
        if "PIPELINE_OUTPUT_DIR" in os.environ:
            self.output.base_output_dir = os.environ["PIPELINE_OUTPUT_DIR"]
        
        # Processing mode override
        if "PIPELINE_MODE" in os.environ:
            try:
                self.mode = ProcessingMode(os.environ["PIPELINE_MODE"])
                self._apply_mode_config()
            except ValueError:
                pass
        
        # Parallel processing override
        if "PIPELINE_PARALLEL" in os.environ:
            self.processing.enable_parallel_processing = os.environ["PIPELINE_PARALLEL"].lower() == "true"
        
        # Max workers override
        if "PIPELINE_MAX_WORKERS" in os.environ:
            try:
                self.processing.max_workers = int(os.environ["PIPELINE_MAX_WORKERS"])
            except ValueError:
                pass
    
    def load_from_file(self, config_file: str):
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Update configurations from file
            if "segmentation" in config_data:
                self._update_dataclass(self.segmentation, config_data["segmentation"])
            if "ml" in config_data:
                self._update_dataclass(self.ml, config_data["ml"])
            if "processing" in config_data:
                self._update_dataclass(self.processing, config_data["processing"])
            if "output" in config_data:
                self._update_dataclass(self.output, config_data["output"])
            if "logging" in config_data:
                self._update_dataclass(self.logging, config_data["logging"])
            if "nlp" in config_data:
                self._update_dataclass(self.nlp, config_data["nlp"])
                
        except Exception as e:
            print(f"Warning: Failed to load configuration from {config_file}: {e}")
    
    def _update_dataclass(self, obj, data: Dict[str, Any]):
        """Update dataclass object with dictionary data."""
        for key, value in data.items():
            if hasattr(obj, key):
                # Handle enum conversions
                if key == "strategy" and hasattr(obj, "strategy"):
                    try:
                        setattr(obj, key, SegmentationStrategy(value))
                    except ValueError:
                        pass
                else:
                    setattr(obj, key, value)
    
    def save_to_file(self, config_file: str):
        """Save current configuration to JSON file."""
        config_data = {
            "mode": self.mode.value,
            "segmentation": asdict(self.segmentation),
            "ml": asdict(self.ml),
            "processing": asdict(self.processing),
            "output": asdict(self.output),
            "logging": asdict(self.logging),
            "nlp": asdict(self.nlp)
        }
        
        # Convert enums to strings
        if "strategy" in config_data["segmentation"]:
            config_data["segmentation"]["strategy"] = config_data["segmentation"]["strategy"].value
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
    
    def get_paths(self) -> Dict[str, Path]:
        """Get all configured paths as Path objects."""
        base_dir = Path(self.output.base_output_dir)
        return {
            "base": base_dir,
            "standard": base_dir / "standard",
            "ml_ready": base_dir / "ml_ready_data",
            "target_format": base_dir / "target_format_data",
            "logs": Path("logs"),
            "memory": Path("memory")
        }
    
    def setup_logging(self) -> logging.Logger:
        """Setup logging based on configuration."""
        logger = logging.getLogger("preprocessing_pipeline")
        logger.setLevel(getattr(logging, self.logging.level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        formatter = logging.Formatter(self.logging.format)
        
        # Console handler
        if self.logging.enable_console_output:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if self.logging.enable_file_output:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                self.logging.file_path,
                maxBytes=self.logging.max_file_size_mb * 1024 * 1024,
                backupCount=self.logging.backup_count
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate segmentation config
        if self.segmentation.min_segment_lines >= self.segmentation.max_segment_lines:
            issues.append("min_segment_lines must be less than max_segment_lines")
        
        if self.segmentation.min_words_per_segment >= self.segmentation.max_words_per_segment:
            issues.append("min_words_per_segment must be less than max_words_per_segment")
        
        # Validate processing config
        if self.processing.max_workers < 1:
            issues.append("max_workers must be at least 1")
        
        # Validate paths
        try:
            paths = self.get_paths()
            for name, path in paths.items():
                if name != "base":  # base path will be created automatically
                    path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Path validation failed: {e}")
        
        return issues

# Global configuration instance
_config_instance: Optional[PipelineConfig] = None

def get_config() -> PipelineConfig:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = PipelineConfig()
    return _config_instance

def set_config(config: PipelineConfig):
    """Set the global configuration instance."""
    global _config_instance
    _config_instance = config

def load_config(config_file: str, mode: ProcessingMode = ProcessingMode.PRODUCTION) -> PipelineConfig:
    """Load configuration from file and set as global instance."""
    config = PipelineConfig(mode=mode, config_file=config_file)
    set_config(config)
    return config
