"""
ML Integration module that orchestrates all ML components.

This module provides a unified interface for running the complete ML pipeline
including topic modeling, semantic search, feature engineering, and evaluation.
"""

import logging
import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

# Lazy imports to avoid slow numba compilation
def _import_topic_modeling():
    try:
        from .unsupervised_learning.topic_modeling import TopicModeling
        return TopicModeling
    except ImportError:
        return None

def _import_semantic_search():
    try:
        from .unsupervised_learning.semantic_search import SemanticSearch
        return SemanticSearch
    except ImportError:
        return None

def _import_feature_engineering():
    try:
        from .unsupervised_learning.feature_engineering import FeatureEngineering
        return FeatureEngineering
    except ImportError:
        return None

def _import_metrics_calculator():
    try:
        from .evaluation.metrics import MetricsCalculator
        return MetricsCalculator
    except ImportError:
        return None

def _import_data_splitter():
    try:
        from .dataset_management.splitter import DataSplitter
        return DataSplitter
    except ImportError:
        return None

def _import_excel_exporter():
    try:
        from .utils.excel_exporter import ExcelExporter
        return ExcelExporter
    except ImportError:
        return None

def _import_target_format_generator():
    try:
        from .target_format.target_format_generator import TargetFormatGenerator
        return TargetFormatGenerator
    except ImportError:
        return None

# Configure logging
logger = logging.getLogger(__name__)

class MLPipeline:
    """
    Unified ML pipeline orchestrator.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ML pipeline.
        
        Args:
            config: Configuration dictionary for ML components
        """
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.topic_modeling = None
        self.semantic_search = None
        self.feature_engineering = None
        self.metrics_calculator = None
        self.data_splitter = None
        self.target_format_generator = None

        self._initialize_components()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for ML pipeline."""
        return {
            "topic_modeling": {
                "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
                "min_topic_size": 2,
                "language": "multilingual"
            },
            "semantic_search": {
                "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
                "index_type": "flat"
            },
            "feature_engineering": {
                "spacy_model": "fr_core_news_lg"
            },
            "dataset_splitting": {
                "train_ratio": 0.7,
                "validation_ratio": 0.2,
                "test_ratio": 0.1,
                "random_seed": 42
            }
        }
    
    def _initialize_components(self):
        """Initialize all ML components."""
        try:
            # Topic modeling
            TopicModeling = _import_topic_modeling()
            if TopicModeling:
                topic_config = self.config.get("topic_modeling", {})
                self.topic_modeling = TopicModeling(
                    embedding_model=topic_config.get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2"),
                    min_topic_size=topic_config.get("min_topic_size", 2),
                    language=topic_config.get("language", "multilingual")
                )
            else:
                logger.warning("TopicModeling not available")
                self.topic_modeling = None

            # Semantic search
            SemanticSearch = _import_semantic_search()
            if SemanticSearch:
                search_config = self.config.get("semantic_search", {})
                self.semantic_search = SemanticSearch(
                    embedding_model=search_config.get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2"),
                    index_type=search_config.get("index_type", "flat")
                )
            else:
                logger.warning("SemanticSearch not available")
                self.semantic_search = None

            # Feature engineering
            FeatureEngineering = _import_feature_engineering()
            if FeatureEngineering:
                feature_config = self.config.get("feature_engineering", {})
                self.feature_engineering = FeatureEngineering(
                    spacy_model=feature_config.get("spacy_model", "fr_core_news_lg")
                )
            else:
                logger.warning("FeatureEngineering not available")
                self.feature_engineering = None

            # Metrics calculator
            MetricsCalculator = _import_metrics_calculator()
            if MetricsCalculator:
                self.metrics_calculator = MetricsCalculator()
            else:
                logger.warning("MetricsCalculator not available")
                self.metrics_calculator = None

            # Data splitter
            DataSplitter = _import_data_splitter()
            if DataSplitter:
                split_config = self.config.get("dataset_splitting", {})
                self.data_splitter = DataSplitter(
                    train_ratio=split_config.get("train_ratio", 0.7),
                    validation_ratio=split_config.get("validation_ratio", 0.2),
                    test_ratio=split_config.get("test_ratio", 0.1),
                    random_seed=split_config.get("random_seed", 42)
                )
            else:
                logger.warning("DataSplitter not available")
                self.data_splitter = None

            # Target format generator
            TargetFormatGenerator = _import_target_format_generator()
            if TargetFormatGenerator:
                self.target_format_generator = TargetFormatGenerator()
            else:
                logger.warning("TargetFormatGenerator not available")
                self.target_format_generator = None

            logger.info("ML pipeline components initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing ML components: {e}")
            raise
    
    def process_segments(self, segments: List[Dict[str, Any]], 
                        enable_topic_modeling: bool = True,
                        enable_semantic_search: bool = True,
                        enable_feature_engineering: bool = True) -> Dict[str, Any]:
        """
        Process segments through the complete ML pipeline.
        
        Args:
            segments: List of text segments
            enable_topic_modeling: Whether to run topic modeling
            enable_semantic_search: Whether to build search index
            enable_feature_engineering: Whether to enhance features
            
        Returns:
            Dictionary with processed results
        """
        if not segments:
            logger.warning("No segments provided for processing")
            return {}
        
        logger.info(f"Processing {len(segments)} segments through ML pipeline")
        
        results = {
            "input_segments": len(segments),
            "processed_segments": [],
            "topic_modeling_results": {},
            "semantic_search_results": {},
            "feature_engineering_results": {},
            "evaluation_results": {},
            "processing_metadata": {
                "timestamp": datetime.now().isoformat(),
                "config": self.config
            }
        }
        
        # Step 1: Feature Engineering
        if enable_feature_engineering and self.feature_engineering:
            logger.info("Running feature engineering...")
            enhanced_segments = self.feature_engineering.process_segments(segments)
            results["processed_segments"] = enhanced_segments
            results["feature_engineering_results"] = {
                "enhanced_segments": len(enhanced_segments),
                "features_added": True
            }
        else:
            if enable_feature_engineering and not self.feature_engineering:
                logger.warning("Feature engineering requested but not available")
            results["processed_segments"] = segments
        
        # Step 2: Topic Modeling
        if enable_topic_modeling and self.topic_modeling:
            logger.info("Running topic modeling...")
            topic_segments = self.topic_modeling.process_segments(results["processed_segments"])
            results["processed_segments"] = topic_segments

            # Get topic information
            topic_info = self.topic_modeling.get_topic_info()
            viz_data = self.topic_modeling.get_topic_visualization_data()

            results["topic_modeling_results"] = {
                "topics": topic_info,
                "visualization_data": viz_data,
                "segments_with_topics": len(topic_segments)
            }
        elif enable_topic_modeling and not self.topic_modeling:
            logger.warning("Topic modeling requested but not available")
        
        # Step 3: Semantic Search Index
        if enable_semantic_search and self.semantic_search:
            logger.info("Building semantic search index...")
            search_built = self.semantic_search.build_index(results["processed_segments"])

            if search_built:
                index_stats = self.semantic_search.get_index_stats()
                results["semantic_search_results"] = {
                    "index_built": True,
                    "index_stats": index_stats
                }
            else:
                results["semantic_search_results"] = {"index_built": False}
        elif enable_semantic_search and not self.semantic_search:
            logger.warning("Semantic search requested but not available")
        
        # Step 4: Evaluation
        if self.metrics_calculator:
            logger.info("Running evaluation...")
            evaluation_results = self.metrics_calculator.generate_evaluation_report(
                results["processed_segments"],
                results.get("topic_modeling_results"),
                None  # No specific search results for evaluation yet
            )
            results["evaluation_results"] = evaluation_results
        else:
            logger.warning("Metrics calculator not available")
        
        logger.info("ML pipeline processing completed")
        return results
    
    def search_segments(self, query: str, k: int = 5,
                       filter_criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant segments using semantic search.

        Args:
            query: Search query
            k: Number of results to return
            filter_criteria: Optional filtering criteria

        Returns:
            List of search results
        """
        if not self.semantic_search or not self.semantic_search.index:
            logger.error("Semantic search index not built. Run process_segments first.")
            return []

        return self.semantic_search.search(query, k, filter_criteria)

    def generate_target_format(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate output in the exact data.json format.

        Args:
            segments: List of segments with enhanced features

        Returns:
            Dictionary in exact data.json format
        """
        if not self.target_format_generator:
            logger.error("TargetFormatGenerator not available")
            return {"entries": []}

        return self.target_format_generator.generate_target_format(segments)
    
    def split_dataset(self, segments: List[Dict[str, Any]],
                     stratify_by: Optional[str] = None,
                     by_document: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """
        Split segments into train/validation/test sets.

        Args:
            segments: List of segments to split
            stratify_by: Optional field to stratify by
            by_document: Whether to split by document to prevent leakage

        Returns:
            Dictionary with train/validation/test splits
        """
        if not self.data_splitter:
            logger.error("DataSplitter not available")
            return {"train": [], "validation": [], "test": []}

        if by_document:
            return self.data_splitter.split_by_document(segments)
        else:
            return self.data_splitter.split_segments(segments, stratify_by)
    
    def save_results(self, results: Dict[str, Any], output_dir: str,
                    prefix: str = "ml_results",
                    generate_target_format: bool = True) -> Dict[str, str]:
        """
        Save ML pipeline results to files.

        Args:
            results: Results from process_segments
            output_dir: Directory to save files
            prefix: Prefix for filenames
            generate_target_format: Whether to generate target format output

        Returns:
            Dictionary with paths to saved files
        """
        os.makedirs(output_dir, exist_ok=True)

        saved_files = {}

        # Save main results
        results_path = os.path.join(output_dir, f"{prefix}_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        saved_files["results"] = results_path

        # Generate and save target format if requested
        if generate_target_format and results.get("processed_segments"):
            target_format_data = self.generate_target_format(results["processed_segments"])
            if target_format_data.get("entries"):
                target_path = os.path.join(output_dir, f"{prefix}_target_format.json")
                with open(target_path, 'w', encoding='utf-8') as f:
                    json.dump(target_format_data, f, ensure_ascii=False, indent=2)
                saved_files["target_format"] = target_path
                logger.info(f"Target format saved: {target_path}")
        
        # Save processed segments separately
        if results.get("processed_segments"):
            segments_path = os.path.join(output_dir, f"{prefix}_processed_segments.json")
            with open(segments_path, 'w', encoding='utf-8') as f:
                json.dump(results["processed_segments"], f, ensure_ascii=False, indent=2)
            saved_files["processed_segments"] = segments_path
        
        # Save topic modeling results
        if results.get("topic_modeling_results"):
            topics_path = os.path.join(output_dir, f"{prefix}_topics.json")
            with open(topics_path, 'w', encoding='utf-8') as f:
                json.dump(results["topic_modeling_results"], f, ensure_ascii=False, indent=2)
            saved_files["topics"] = topics_path
        
        # Save evaluation report
        if results.get("evaluation_results"):
            eval_path = os.path.join(output_dir, f"{prefix}_evaluation.json")
            with open(eval_path, 'w', encoding='utf-8') as f:
                json.dump(results["evaluation_results"], f, ensure_ascii=False, indent=2)
            saved_files["evaluation"] = eval_path
        
        # Save search index if available
        if (self.semantic_search and self.semantic_search.index and
            results.get("semantic_search_results", {}).get("index_built")):
            index_path = os.path.join(output_dir, f"{prefix}_search_index")
            if self.semantic_search.save_index(index_path):
                saved_files["search_index"] = f"{index_path}.faiss"
                saved_files["search_metadata"] = f"{index_path}_metadata.pkl"

        # Save Excel report
        ExcelExporter = _import_excel_exporter()
        if ExcelExporter:
            excel_path = os.path.join(output_dir, f"{prefix}_comprehensive_report.xlsx")
            exporter = ExcelExporter()
            if exporter.export_ml_results(results, excel_path):
                saved_files["excel_report"] = excel_path
                logger.info(f"Excel comprehensive report saved: {excel_path}")
            else:
                logger.warning("Failed to save Excel report")
        else:
            logger.warning("Excel exporter not available")

        logger.info(f"Saved ML results to {len(saved_files)} files in {output_dir}")
        return saved_files
    
    def load_search_index(self, index_path: str) -> bool:
        """
        Load a pre-built search index.
        
        Args:
            index_path: Path to the saved index (without extension)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.semantic_search:
            logger.error("Semantic search component not initialized")
            return False
        
        return self.semantic_search.load_index(index_path)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get status of all pipeline components.
        
        Returns:
            Dictionary with component status
        """
        status = {
            "topic_modeling": {
                "initialized": self.topic_modeling is not None,
                "model_trained": (self.topic_modeling and 
                                self.topic_modeling.topic_model is not None)
            },
            "semantic_search": {
                "initialized": self.semantic_search is not None,
                "index_built": (self.semantic_search and 
                              self.semantic_search.index is not None)
            },
            "feature_engineering": {
                "initialized": self.feature_engineering is not None
            },
            "metrics_calculator": {
                "initialized": self.metrics_calculator is not None
            },
            "data_splitter": {
                "initialized": self.data_splitter is not None
            },
            "target_format_generator": {
                "initialized": self.target_format_generator is not None
            }
        }

        return status
