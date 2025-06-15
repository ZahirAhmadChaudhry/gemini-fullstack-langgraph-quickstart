"""
Master Training Script for All ML Models
========================================

This script orchestrates the complete training pipeline for both tension detection
and thematic classification models, then provides comprehensive comparison and
selection of the best models for production deployment.

Author: ML Engineering Team
Date: 2025-06-12
"""

import sys
import os
from pathlib import Path
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import training modules
from train_tension_models import TensionModelTrainer
from train_thematic_models import ThematicModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('master_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MasterModelTrainer:
    """
    Master trainer that orchestrates both tension and thematic model training.
    """
    
    def __init__(self, data_dir: str = "data_from_Data_Engineering", 
                 output_dir: str = "trained_models"):
        """
        Initialize the master trainer.
        
        Args:
            data_dir: Directory containing the data engineering outputs
            output_dir: Directory to save trained models
        """
        self.data_dir = data_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize individual trainers
        self.tension_trainer = TensionModelTrainer(data_dir, output_dir)
        self.thematic_trainer = ThematicModelTrainer(data_dir, output_dir)
        
        logger.info(f"Initialized MasterModelTrainer with data_dir: {data_dir}")
    
    def run_tension_training(self) -> dict:
        """
        Run tension detection model training.
        
        Returns:
            Tension training results
        """
        logger.info("\n" + "🎯" * 30)
        logger.info("PHASE 1: TENSION DETECTION MODEL TRAINING")
        logger.info("🎯" * 30)
        
        start_time = time.time()
        
        try:
            tension_results = self.tension_trainer.run_complete_training()
            
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info(f"✅ Tension detection training completed in {duration:.2f} seconds")
            
            return tension_results
            
        except Exception as e:
            logger.error(f"❌ Tension detection training failed: {str(e)}")
            raise
    
    def run_thematic_training(self) -> dict:
        """
        Run thematic classification model training.
        
        Returns:
            Thematic training results
        """
        logger.info("\n" + "🎨" * 30)
        logger.info("PHASE 2: THEMATIC CLASSIFICATION MODEL TRAINING")
        logger.info("🎨" * 30)
        
        start_time = time.time()
        
        try:
            thematic_results = self.thematic_trainer.run_complete_training()
            
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info(f"✅ Thematic classification training completed in {duration:.2f} seconds")
            
            return thematic_results
            
        except Exception as e:
            logger.error(f"❌ Thematic classification training failed: {str(e)}")
            raise
    
    def create_master_summary(self, tension_results: dict, thematic_results: dict) -> dict:
        """
        Create a comprehensive summary of all model training results.
        
        Args:
            tension_results: Results from tension model training
            thematic_results: Results from thematic model training
            
        Returns:
            Master summary dictionary
        """
        logger.info("\n" + "📊" * 30)
        logger.info("CREATING MASTER SUMMARY REPORT")
        logger.info("📊" * 30)
        
        # Extract best models
        tension_best = tension_results['comparison']['best_model']
        tension_accuracy = tension_results['comparison']['best_accuracy']
        tension_target_achieved = tension_results['comparison']['target_achieved']
        
        thematic_best = thematic_results['comparison']['best_model']
        thematic_accuracy = thematic_results['comparison']['best_accuracy']
        thematic_target_achieved = thematic_results['comparison']['target_achieved']
        
        # Create master summary
        master_summary = {
            'timestamp': datetime.now().isoformat(),
            'training_overview': {
                'total_models_trained': len(tension_results['models']) + len(thematic_results['models']),
                'tension_models': len(tension_results['models']),
                'thematic_models': len(thematic_results['models']),
                'data_samples': {
                    'tension_total': tension_results['summary']['data_statistics']['total_samples'],
                    'thematic_total': thematic_results['summary']['data_statistics']['total_samples']
                }
            },
            'best_models': {
                'tension_detection': {
                    'model': tension_best,
                    'accuracy': tension_accuracy,
                    'target_achieved': tension_target_achieved,
                    'original_baseline': 0.33,
                    'improvement': tension_accuracy - 0.33 if tension_best else 0,
                    'target_range': [0.75, 0.90]
                },
                'thematic_classification': {
                    'model': thematic_best,
                    'accuracy': thematic_accuracy,
                    'target_achieved': thematic_target_achieved,
                    'target_range': [0.85, 0.95]
                }
            },
            'overall_performance': {
                'both_targets_achieved': tension_target_achieved and thematic_target_achieved,
                'total_improvement': (tension_accuracy - 0.33) if tension_best else 0,
                'production_ready': tension_target_achieved and thematic_target_achieved
            },
            'detailed_results': {
                'tension_models': tension_results['summary']['model_performance'],
                'thematic_models': thematic_results['summary']['model_performance']
            }
        }
        
        # Log summary
        logger.info("🏆 MASTER TRAINING SUMMARY:")
        logger.info(f"  Total Models Trained: {master_summary['training_overview']['total_models_trained']}")
        logger.info(f"  Tension Detection Best: {tension_best} ({tension_accuracy:.4f})")
        logger.info(f"  Thematic Classification Best: {thematic_best} ({thematic_accuracy:.4f})")
        logger.info(f"  Both Targets Achieved: {master_summary['overall_performance']['both_targets_achieved']}")
        logger.info(f"  Production Ready: {master_summary['overall_performance']['production_ready']}")
        
        return master_summary
    
    def save_master_results(self, tension_results: dict, thematic_results: dict, master_summary: dict):
        """
        Save all training results and master summary.
        
        Args:
            tension_results: Tension training results
            thematic_results: Thematic training results
            master_summary: Master summary
        """
        logger.info("💾 Saving master training results...")
        
        # Save master summary
        master_summary_path = self.output_dir / "master_training_summary.json"
        with open(master_summary_path, 'w') as f:
            json.dump(master_summary, f, indent=2)
        
        # Create combined comparison DataFrame
        tension_comparison = tension_results['comparison']['comparison_df'].copy()
        tension_comparison['Task'] = 'Tension Detection'
        
        thematic_comparison = thematic_results['comparison']['comparison_df'].copy()
        thematic_comparison['Task'] = 'Thematic Classification'
        
        # Align columns
        all_columns = set(tension_comparison.columns) | set(thematic_comparison.columns)
        for col in all_columns:
            if col not in tension_comparison.columns:
                tension_comparison[col] = 0
            if col not in thematic_comparison.columns:
                thematic_comparison[col] = 0
        
        combined_comparison = pd.concat([tension_comparison, thematic_comparison], ignore_index=True)
        
        # Save combined comparison
        combined_comparison_path = self.output_dir / "all_models_comparison.csv"
        combined_comparison.to_csv(combined_comparison_path, index=False)
        
        # Create production deployment recommendations
        recommendations = {
            'timestamp': datetime.now().isoformat(),
            'production_recommendations': {
                'tension_detection': {
                    'recommended_model': tension_results['comparison']['best_model'],
                    'model_file': f"tension_{tension_results['comparison']['best_model'].lower().replace(' ', '_')}_full.joblib",
                    'accuracy': tension_results['comparison']['best_accuracy'],
                    'deployment_priority': 'HIGH' if tension_results['comparison']['target_achieved'] else 'MEDIUM'
                },
                'thematic_classification': {
                    'recommended_model': thematic_results['comparison']['best_model'],
                    'model_file': f"thematic_{thematic_results['comparison']['best_model'].lower().replace(' ', '_')}_full.joblib",
                    'accuracy': thematic_results['comparison']['best_accuracy'],
                    'deployment_priority': 'HIGH' if thematic_results['comparison']['target_achieved'] else 'MEDIUM'
                }
            },
            'deployment_checklist': [
                'Load recommended models from saved files',
                'Integrate with existing ML pipeline',
                'Set up confidence thresholds',
                'Implement fallback mechanisms',
                'Configure monitoring and logging',
                'Test end-to-end pipeline',
                'Deploy to production environment'
            ]
        }
        
        recommendations_path = self.output_dir / "production_recommendations.json"
        with open(recommendations_path, 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        logger.info(f"📁 Master results saved to:")
        logger.info(f"  Master Summary: {master_summary_path}")
        logger.info(f"  Combined Comparison: {combined_comparison_path}")
        logger.info(f"  Production Recommendations: {recommendations_path}")
    
    def run_complete_training(self):
        """
        Run the complete master training pipeline.
        """
        logger.info("🚀" * 40)
        logger.info("STARTING COMPLETE ML MODEL TRAINING PIPELINE")
        logger.info("🚀" * 40)
        
        start_time = time.time()
        
        try:
            # Phase 1: Train tension detection models
            tension_results = self.run_tension_training()
            
            # Phase 2: Train thematic classification models
            thematic_results = self.run_thematic_training()
            
            # Phase 3: Create master summary
            master_summary = self.create_master_summary(tension_results, thematic_results)
            
            # Phase 4: Save all results
            self.save_master_results(tension_results, thematic_results, master_summary)
            
            end_time = time.time()
            total_duration = end_time - start_time
            
            logger.info("\n" + "🎉" * 40)
            logger.info("COMPLETE ML MODEL TRAINING PIPELINE FINISHED!")
            logger.info("🎉" * 40)
            logger.info(f"⏱️ Total Training Time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
            
            # Final status report
            both_successful = (tension_results['comparison']['target_achieved'] and 
                             thematic_results['comparison']['target_achieved'])
            
            if both_successful:
                logger.info("🏆 SUCCESS: Both tension detection and thematic classification targets achieved!")
                logger.info("✅ Models are ready for production deployment!")
            else:
                logger.info("⚠️ PARTIAL SUCCESS: Some targets not fully achieved, but models trained successfully")
                logger.info("🔧 Consider additional optimization or data augmentation")
            
            return {
                'tension': tension_results,
                'thematic': thematic_results,
                'master_summary': master_summary,
                'total_duration': total_duration,
                'production_ready': both_successful
            }
            
        except Exception as e:
            logger.error(f"❌ Master training pipeline failed: {str(e)}")
            raise


if __name__ == "__main__":
    # Run the complete master training pipeline
    trainer = MasterModelTrainer()
    results = trainer.run_complete_training()
