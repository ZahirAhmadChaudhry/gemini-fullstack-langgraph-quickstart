"""
Data Preparation Module for ML Models Implementation
====================================================

This module handles data extraction, feature engineering, and preparation
for tension detection and thematic classification models.

Author: ML Engineering Team
Date: 2025-06-12
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreparationPipeline:
    """
    Comprehensive data preparation pipeline for ML models.
    
    Handles:
    - Data extraction from 8 tables
    - Feature engineering for tension detection and thematic classification
    - Train/validation/test splits with stratification
    - Class imbalance handling
    """
    
    def __init__(self, data_dir: str = "data_from_Data_Engineering"):
        """
        Initialize the data preparation pipeline.
        
        Args:
            data_dir: Path to the data engineering output directory
        """
        # Resolve the data directory to an absolute path for robustness. If the provided
        # path does not exist relative to the current working directory, attempt to
        # locate it relative to the project root (two levels up from this file).
        provided_path = Path(data_dir)

        if provided_path.exists():
            # Path is valid as-is
            self.data_dir = provided_path
        else:
            # Fallback: assume the data directory is located next to the project root
            fallback_path = Path(__file__).resolve().parent.parent / data_dir

            if fallback_path.exists():
                self.data_dir = fallback_path
                logger.warning(
                    "Provided data_dir '%s' not found. Falling back to '%s'.",
                    data_dir, self.data_dir
                )
            else:
                # Neither path exists – raise a clear error message to help debugging
                raise FileNotFoundError(
                    f"Data directory '{data_dir}' not found. Tried '{provided_path.resolve()}' and "
                    f"'{fallback_path}'. Please verify the path or pass the correct absolute path."
                )
        self.target_format_dir = self.data_dir / "target_format_data"
        self.ml_ready_dir = self.data_dir / "ml_ready_data"
        
        # Data containers
        self.raw_data = []
        self.ml_features = []
        self.tension_data = None
        self.thematic_data = None
        
        # Encoders and scalers
        self.tension_encoder = LabelEncoder()
        self.theme_encoder = LabelEncoder()
        self.feature_scaler = StandardScaler()
        
        logger.info(f"Initialized DataPreparationPipeline with data_dir: {data_dir}")

    def clean_text_for_matching(self, text: str) -> str:
        """
        Clean and normalize text for better matching.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove special characters but keep accented characters
        text = re.sub(r'[^\w\sàâäéèêëïîôöùûüÿç]', ' ', text)

        return text

    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using TF-IDF and cosine similarity.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        # Clean texts
        clean_text1 = self.clean_text_for_matching(text1)
        clean_text2 = self.clean_text_for_matching(text2)

        if not clean_text1 or not clean_text2:
            return 0.0

        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        try:
            tfidf_matrix = vectorizer.fit_transform([clean_text1, clean_text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            # Fallback to simple word overlap if TF-IDF fails
            words1 = set(clean_text1.split())
            words2 = set(clean_text2.split())
            if not words1 or not words2:
                return 0.0
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            return intersection / union if union > 0 else 0.0

    def find_best_feature_match(self, label_text: str, feature_candidates: pd.DataFrame,
                               similarity_threshold: float = 0.05) -> Tuple[int, float]:
        """
        Find the best matching feature for a given label text.

        Args:
            label_text: Text from the label/target data
            feature_candidates: DataFrame of potential feature matches
            similarity_threshold: Minimum similarity score to consider a match

        Returns:
            Tuple of (best_index, similarity_score) or (-1, 0.0) if no good match
        """
        if feature_candidates.empty:
            return -1, 0.0

        best_index = -1
        best_similarity = 0.0

        for idx, row in feature_candidates.iterrows():
            feature_text = row.get('text', '')
            similarity = self.calculate_text_similarity(label_text, feature_text)

            if similarity > best_similarity:
                best_similarity = similarity
                best_index = idx

        # If we found any match above threshold, return it
        if best_similarity >= similarity_threshold:
            return best_index, best_similarity

        # Fallback: if no good similarity match, use the first available feature
        # This ensures we don't lose data due to text matching issues
        if not feature_candidates.empty and best_index == -1:
            logger.warning(f"No good text similarity match found (best: {best_similarity:.3f}), using first available feature")
            return feature_candidates.index[0], 0.0

        return best_index, best_similarity

    def _extract_tension_type(self, code_spe: str) -> str:
        """
        Extract meaningful tension type from Code spé field.

        Args:
            code_spe: The Code spé string from target data

        Returns:
            Standardized tension type
        """
        # Map common tension patterns to standardized types
        tension_mappings = {
            'alloc.travail.richesse.temps': 'accumulation_partage',
            'diff.croiss.dévelpmt': 'croissance_decroissance',
            'respons.indiv.coll.etatique': 'individuel_collectif',
            'local.global': 'local_global',
            'court.long.terme': 'court_terme_long_terme'
        }

        # Check for known patterns
        for pattern, tension_type in tension_mappings.items():
            if pattern in code_spe:
                return tension_type

        # Fallback: extract last meaningful part
        parts = code_spe.split('.')
        if len(parts) >= 2:
            # Try to get a meaningful combination of last parts
            last_parts = parts[-2:]
            return '_'.join(last_parts).replace(' ', '_').lower()

        return 'unknown'

    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and combine data from all 8 tables.
        
        Returns:
            Tuple of (target_format_df, ml_ready_df)
        """
        logger.info("Loading data from all 8 tables...")
        
        target_data = []
        ml_data = []
        
        # Load target format data (ground truth labels)
        for table_file in self.target_format_dir.glob("Table_*_target_format.json"):
            logger.info(f"Loading target format: {table_file.name}")
            with open(table_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for entry in data['entries']:
                    entry['source_table'] = table_file.stem.replace('_target_format', '')
                    target_data.append(entry)
        
        # Load ML-ready data (features)
        for table_file in self.ml_ready_dir.glob("Table_*_ml_ready.json"):
            logger.info(f"Loading ML-ready data: {table_file.name}")
            with open(table_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for segment in data['segments']:
                    segment['source_table'] = table_file.stem.replace('_ml_ready', '')
                    ml_data.append(segment)
        
        # Convert to DataFrames
        target_df = pd.DataFrame(target_data)
        ml_df = pd.DataFrame(ml_data)
        
        logger.info(f"Loaded {len(target_df)} target entries and {len(ml_df)} ML segments")

        return target_df, ml_df



    def load_single_table_data(self, table_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load data from a single table for quick testing.

        Args:
            table_name: Name of the table to load (e.g., "Table_A")

        Returns:
            Tuple of (target_df, ml_df) for the single table
        """
        logger.info(f"Loading single table data: {table_name}")

        # Load target format
        target_file = self.target_format_dir / f"{table_name}_target_format.json"
        if not target_file.exists():
            raise FileNotFoundError(f"Target format file not found: {target_file}")

        with open(target_file, 'r', encoding='utf-8') as f:
            target_data = json.load(f)

        # Extract entries from the nested structure and add source_table
        if isinstance(target_data, dict) and 'entries' in target_data:
            entries = target_data['entries']
            # Add source table information to each entry
            for entry in entries:
                entry['source_table'] = table_name
            target_df = pd.DataFrame(entries)
        else:
            # Fallback for different data structure
            target_df = pd.DataFrame(target_data)
            # Add source_table column if it doesn't exist
            if 'source_table' not in target_df.columns:
                target_df['source_table'] = table_name

        logger.info(f"Loaded target format: {target_file.name}")

        # Load ML-ready data
        ml_file = self.ml_ready_dir / f"{table_name}_ml_ready.json"
        if not ml_file.exists():
            raise FileNotFoundError(f"ML-ready file not found: {ml_file}")

        with open(ml_file, 'r', encoding='utf-8') as f:
            ml_data = json.load(f)

        # Extract segments from the nested structure
        if isinstance(ml_data, dict) and 'segments' in ml_data:
            segments = ml_data['segments']
            # Add source table information to each segment
            for segment in segments:
                segment['source_table'] = table_name
            ml_df = pd.DataFrame(segments)
        else:
            # Fallback for different data structure
            ml_df = pd.DataFrame(ml_data)
            # Add source_table column if it doesn't exist
            if 'source_table' not in ml_df.columns:
                ml_df['source_table'] = table_name

        logger.info(f"Loaded ML-ready data: {ml_file.name}")

        logger.info(f"Single table loaded - Target: {len(target_df)}, ML: {len(ml_df)} entries")

        return target_df, ml_df

    def extract_tension_features(self, ml_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and engineer features for tension detection.
        
        Args:
            ml_df: DataFrame with ML-ready segments
            
        Returns:
            DataFrame with tension detection features
        """
        logger.info("Extracting tension detection features...")
        
        features_list = []
        
        for _, row in ml_df.iterrows():
            features = row['features']
            ml_features = features.get('ml_features', {})
            
            # Base features
            feature_dict = {
                'segment_id': row['id'],
                'source_table': row['source_table'],
                'text': row['text'],
                'word_count': features.get('word_count', 0),
                'sentence_count': features.get('sentence_count', 0),
                'temporal_period': ml_features.get('temporal_period', 2035.0),
                'conceptual_complexity': ml_features.get('conceptual_complexity', 0.0),
            }
            
            # Tension pattern features
            tension_patterns = features.get('tension_patterns', {})
            for tension_type, pattern_data in tension_patterns.items():
                feature_dict[f'tension_{tension_type}_strength'] = pattern_data.get('tension_strength', 0)
                feature_dict[f'tension_{tension_type}_total'] = pattern_data.get('total_indicators', 0)
            
            # Discourse markers
            discourse_markers = features.get('discourse_markers', [])
            feature_dict['discourse_priority'] = 1 if 'priority' in discourse_markers else 0
            feature_dict['discourse_context'] = 1 if 'context' in discourse_markers else 0
            
            # Conceptual markers
            conceptual_markers = features.get('conceptual_markers', [])
            feature_dict['concept_socio_eco'] = 1 if 'MODELES_SOCIO_ECONOMIQUES' in conceptual_markers else 0
            feature_dict['concept_environmental'] = 1 if 'MODELES_ENVIRONNEMENTAUX' in conceptual_markers else 0
            feature_dict['concept_organizational'] = 1 if 'MODELES_ORGANISATIONNELS' in conceptual_markers else 0
            
            # Thematic indicators
            thematic = features.get('thematic_indicators', {})
            feature_dict['performance_density'] = thematic.get('performance_density', 0.0)
            feature_dict['legitimacy_density'] = thematic.get('legitimacy_density', 0.0)
            
            features_list.append(feature_dict)
        
        tension_features_df = pd.DataFrame(features_list)
        logger.info(f"Extracted {len(tension_features_df)} tension feature vectors")
        
        return tension_features_df
    
    def extract_thematic_features(self, ml_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract and engineer features for thematic classification.
        
        Args:
            ml_df: DataFrame with ML-ready segments
            
        Returns:
            DataFrame with thematic classification features
        """
        logger.info("Extracting thematic classification features...")
        
        features_list = []
        
        for _, row in ml_df.iterrows():
            features = row['features']
            ml_features = features.get('ml_features', {})
            
            # Base features for thematic classification
            feature_dict = {
                'segment_id': row['id'],
                'source_table': row['source_table'],
                'text': row['text'],
                'word_count': features.get('word_count', 0),
                'sentence_count': features.get('sentence_count', 0),
                'performance_score': ml_features.get('performance_score', 0.0),
                'legitimacy_score': ml_features.get('legitimacy_score', 0.0),
                'temporal_period': ml_features.get('temporal_period', 2035.0),
            }
            
            # Thematic indicators (key features)
            thematic = features.get('thematic_indicators', {})
            feature_dict['performance_density'] = thematic.get('performance_density', 0.0)
            feature_dict['legitimacy_density'] = thematic.get('legitimacy_density', 0.0)
            feature_dict['performance_indicators'] = thematic.get('performance_indicators', 0)
            feature_dict['legitimacy_indicators'] = thematic.get('legitimacy_indicators', 0)
            
            # Discourse and conceptual context
            discourse_markers = features.get('discourse_markers', [])
            feature_dict['discourse_priority'] = 1 if 'priority' in discourse_markers else 0
            feature_dict['discourse_context'] = 1 if 'context' in discourse_markers else 0
            
            conceptual_markers = features.get('conceptual_markers', [])
            feature_dict['concept_count'] = len(conceptual_markers)
            
            # Noun phrases count (linguistic complexity)
            noun_phrases = features.get('noun_phrases', [])
            feature_dict['noun_phrase_count'] = len(noun_phrases)
            
            features_list.append(feature_dict)
        
        thematic_features_df = pd.DataFrame(features_list)
        logger.info(f"Extracted {len(thematic_features_df)} thematic feature vectors")
        
        return thematic_features_df
    
    def prepare_tension_dataset(self, target_df: pd.DataFrame, ml_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare complete dataset for tension detection.
        
        Args:
            target_df: Target format data with labels
            ml_df: ML-ready data with features
            
        Returns:
            Combined DataFrame ready for tension detection training
        """
        logger.info("Preparing tension detection dataset...")
        
        # Extract features
        tension_features = self.extract_tension_features(ml_df)
        
        # Extract tension labels from target data
        tension_labels = []
        for _, row in target_df.iterrows():
            code_spe = row.get('Code spé', '')
            if 'tensions' in code_spe:
                # Extract tension type from code using improved logic
                tension_type = self._extract_tension_type(code_spe)
                tension_labels.append({
                    'source_table': row['source_table'],
                    'tension_type': tension_type,
                    'details': row.get('Détails', ''),
                    'period': row.get('Période', 2035.0),
                    'code_spe': code_spe  # Keep original code for debugging
                })
        
        tension_labels_df = pd.DataFrame(tension_labels)

        logger.info(f"Found {len(tension_labels_df)} tension labels from target data")
        if len(tension_labels_df) > 0:
            tension_type_counts = tension_labels_df['tension_type'].value_counts()
            logger.info(f"Tension type distribution: {tension_type_counts.to_dict()}")

        # Merge features with labels using improved text similarity matching
        combined_data = []
        match_stats = {'total_labels': len(tension_labels_df), 'matched': 0, 'unmatched': 0}

        for _, label_row in tension_labels_df.iterrows():
            # Find candidate features from same table
            table_features = tension_features[
                tension_features['source_table'] == label_row['source_table']
            ]

            if not table_features.empty:
                # Use text similarity to find best match
                label_text = label_row['details']
                best_idx, similarity = self.find_best_feature_match(label_text, table_features)

                if best_idx != -1:
                    # Found a match
                    feature_row = table_features.loc[best_idx].copy()
                    feature_row['tension_label'] = label_row['tension_type']
                    feature_row['label_details'] = label_row['details']
                    feature_row['match_similarity'] = similarity
                    feature_row['original_code_spe'] = label_row['code_spe']
                    combined_data.append(feature_row)
                    match_stats['matched'] += 1
                else:
                    # No match found, skip this label
                    match_stats['unmatched'] += 1
                    logger.warning(f"No feature match found for tension label: {label_row['tension_type']} (code: {label_row['code_spe']})")
            else:
                match_stats['unmatched'] += 1
                logger.warning(f"No features found for table: {label_row['source_table']}")

        logger.info(f"Tension matching stats: {match_stats['matched']} matched, {match_stats['unmatched']} unmatched out of {match_stats['total_labels']} total labels")
        
        tension_dataset = pd.DataFrame(combined_data)
        logger.info(f"Prepared tension dataset with {len(tension_dataset)} samples")

        return tension_dataset

    def prepare_thematic_dataset(self, target_df: pd.DataFrame, ml_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare complete dataset for thematic classification.

        Args:
            target_df: Target format data with labels
            ml_df: ML-ready data with features

        Returns:
            Combined DataFrame ready for thematic classification training
        """
        logger.info("Preparing thematic classification dataset...")

        # Extract features
        thematic_features = self.extract_thematic_features(ml_df)

        # Extract theme labels from target data
        theme_labels = []
        for _, row in target_df.iterrows():
            theme = row.get('Thème', 'Unknown')
            theme_labels.append({
                'source_table': row['source_table'],
                'theme': theme,
                'details': row.get('Détails', ''),
                'period': row.get('Période', 2035.0)
            })

        theme_labels_df = pd.DataFrame(theme_labels)

        # Merge features with labels using improved text similarity matching
        combined_data = []
        match_stats = {'total_labels': len(theme_labels_df), 'matched': 0, 'unmatched': 0}

        for _, label_row in theme_labels_df.iterrows():
            # Find candidate features from same table
            table_features = thematic_features[
                thematic_features['source_table'] == label_row['source_table']
            ]

            if not table_features.empty:
                # Use text similarity to find best match
                label_text = label_row['details']
                best_idx, similarity = self.find_best_feature_match(label_text, table_features)

                if best_idx != -1:
                    # Found a good match
                    feature_row = thematic_features.loc[best_idx].copy()
                    feature_row['theme_label'] = label_row['theme']
                    feature_row['label_details'] = label_row['details']
                    feature_row['match_similarity'] = similarity
                    combined_data.append(feature_row)
                    match_stats['matched'] += 1
                else:
                    # No good match found, skip this label
                    match_stats['unmatched'] += 1
                    logger.warning(f"No good feature match found for theme label: {label_row['theme']}")
            else:
                match_stats['unmatched'] += 1
                logger.warning(f"No features found for table: {label_row['source_table']}")

        logger.info(f"Thematic matching stats: {match_stats['matched']} matched, {match_stats['unmatched']} unmatched out of {match_stats['total_labels']} total labels")

        thematic_dataset = pd.DataFrame(combined_data)
        logger.info(f"Prepared thematic dataset with {len(thematic_dataset)} samples")

        return thematic_dataset

    def create_train_test_splits(self, dataset: pd.DataFrame, target_column: str,
                                test_size: float = 0.2, val_size: float = 0.1,
                                random_state: int = 42) -> Dict[str, Any]:
        """
        Create stratified train/validation/test splits.

        Args:
            dataset: Complete dataset
            target_column: Name of the target column for stratification
            test_size: Proportion for test set (default: 0.2)
            val_size: Proportion for validation set (default: 0.1)
            random_state: Random seed for reproducibility

        Returns:
            Dictionary with train/val/test splits and metadata
        """
        logger.info(f"Creating train/test splits for {target_column}...")

        # Check if dataset is empty
        if len(dataset) == 0:
            raise ValueError(f"Cannot create splits: dataset is empty. No {target_column} data found.")

        # Check if target column exists
        if target_column not in dataset.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset. Available columns: {list(dataset.columns)}")

        # Prepare features and labels
        feature_columns = [col for col in dataset.columns
                          if col not in [target_column, 'segment_id', 'source_table',
                                       'text', 'label_details', 'match_similarity', 'original_code_spe']]

        # Select feature columns and ensure numeric dtype. Coerce non-numeric to NaN,
        # then fill remaining NaNs with zeros to keep downstream models happy.
        X = dataset[feature_columns].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        y = dataset[target_column]

        # Encode labels
        if target_column == 'tension_label':
            y_encoded = self.tension_encoder.fit_transform(y)
        else:  # theme_label
            y_encoded = self.theme_encoder.fit_transform(y)

        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_encoded, test_size=test_size, stratify=y_encoded,
            random_state=random_state
        )

        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, stratify=y_temp,
            random_state=random_state
        )

        # Scale features
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_val_scaled = self.feature_scaler.transform(X_val)
        X_test_scaled = self.feature_scaler.transform(X_test)

        # Create splits dictionary
        splits = {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_columns': feature_columns,
            'train_indices': X_train.index.tolist(),
            'val_indices': X_val.index.tolist(),
            'test_indices': X_test.index.tolist(),
            'class_distribution': {
                'train': np.bincount(y_train),
                'val': np.bincount(y_val),
                'test': np.bincount(y_test)
            }
        }

        logger.info(f"Created splits - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        logger.info(f"Class distribution - Train: {splits['class_distribution']['train']}")

        return splits

    def setup_cross_validation(self, dataset: pd.DataFrame, target_column: str,
                              n_splits: int = 5, random_state: int = 42) -> StratifiedKFold:
        """
        Set up stratified cross-validation.

        Args:
            dataset: Complete dataset
            target_column: Target column for stratification
            n_splits: Number of CV folds
            random_state: Random seed

        Returns:
            StratifiedKFold object
        """
        logger.info(f"Setting up {n_splits}-fold stratified cross-validation...")

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        return skf

    def get_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Calculate class weights for imbalanced datasets.

        Args:
            y: Target labels (encoded)

        Returns:
            Dictionary mapping class indices to weights
        """
        from sklearn.utils.class_weight import compute_class_weight

        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)

        class_weights = dict(zip(classes, weights))
        logger.info(f"Calculated class weights: {class_weights}")

        return class_weights

    def run_complete_preparation(self) -> Dict[str, Any]:
        """
        Run the complete data preparation pipeline.

        Returns:
            Dictionary with prepared datasets and splits for both tasks
        """
        logger.info("Running complete data preparation pipeline...")

        # Load all data
        target_df, ml_df = self.load_all_data()

        # Prepare datasets for both tasks
        tension_dataset = self.prepare_tension_dataset(target_df, ml_df)
        thematic_dataset = self.prepare_thematic_dataset(target_df, ml_df)

        # Create train/test splits
        tension_splits = self.create_train_test_splits(tension_dataset, 'tension_label')
        thematic_splits = self.create_train_test_splits(thematic_dataset, 'theme_label')

        # Set up cross-validation
        tension_cv = self.setup_cross_validation(tension_dataset, 'tension_label')
        thematic_cv = self.setup_cross_validation(thematic_dataset, 'theme_label')

        # Calculate class weights
        tension_weights = self.get_class_weights(tension_splits['y_train'])
        thematic_weights = self.get_class_weights(thematic_splits['y_train'])

        # Compile results
        results = {
            'datasets': {
                'tension': tension_dataset,
                'thematic': thematic_dataset
            },
            'splits': {
                'tension': tension_splits,
                'thematic': thematic_splits
            },
            'cross_validation': {
                'tension': tension_cv,
                'thematic': thematic_cv
            },
            'class_weights': {
                'tension': tension_weights,
                'thematic': thematic_weights
            },
            'encoders': {
                'tension': self.tension_encoder,
                'thematic': self.theme_encoder,
                'features': self.feature_scaler
            }
        }

        logger.info("Data preparation pipeline completed successfully!")

        return results


if __name__ == "__main__":
    # Example usage
    pipeline = DataPreparationPipeline()
    results = pipeline.run_complete_preparation()

    print("Data Preparation Results:")
    print(f"Tension dataset: {len(results['datasets']['tension'])} samples")
    print(f"Thematic dataset: {len(results['datasets']['thematic'])} samples")
    print(f"Tension classes: {list(results['encoders']['tension'].classes_)}")
    print(f"Thematic classes: {list(results['encoders']['thematic'].classes_)}")
