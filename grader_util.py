"""
Utility class for grading and categorizing responses
Used across step0, step1, and other evaluation scripts
"""

from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm
from drgrpo_grader import r1_zero_reward_fn


class ResultsGrader:
    """Grade responses using the reward function"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the grader
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def grade_result(self, response: str, ground_truth: str, fast: bool = True) -> Dict:
        """
        Grade a single response
        
        Args:
            response: The model-generated response
            ground_truth: The ground truth solution
            fast: Whether to use fast grading
        
        Returns:
            Dictionary with format_reward, answer_reward, and total reward
        """
        try:
            reward_dict = r1_zero_reward_fn(
                response=response,
                ground_truth=ground_truth,
                fast=fast
            )
            return reward_dict
        except Exception as e:
            self.logger.warning(f"Error grading response: {str(e)}")
            return {
                'format_reward': 0.0,
                'answer_reward': 0.0,
                'reward': 0.0
            }
    
    def grade_results(self, results: List[Dict], response_key: str = 'generated_response',
                     ground_truth_key: str = 'ground_truth', fast: bool = True,
                     show_progress: bool = True) -> List[Dict]:
        """
        Grade multiple results
        
        Args:
            results: List of result dictionaries
            response_key: Key in result dict containing the response
            ground_truth_key: Key in result dict containing ground truth
            fast: Whether to use fast grading
            show_progress: Whether to show progress bar
        
        Returns:
            List of results with added reward fields
        """
        self.logger.info("Grading responses...")
        self.logger.info("="*80)
        
        iterator = tqdm(results, desc="Grading") if show_progress else results
        
        for result in iterator:
            if response_key in result and ground_truth_key in result:
                reward_dict = self.grade_result(
                    response=result[response_key],
                    ground_truth=result[ground_truth_key],
                    fast=fast
                )
                result['format_reward'] = reward_dict.get('format_reward', 0.0)
                result['answer_reward'] = reward_dict.get('answer_reward', 0.0)
                result['reward'] = reward_dict.get('reward', 0.0)
        
        self.logger.info("Grading complete!")
        
        return results


class ResultsCategorizer:
    """Categorize graded results into categories"""
    
    # Category definitions
    CATEGORY_1 = "correct"  # format_reward=1, answer_reward=1
    CATEGORY_2 = "wrong_answer"  # format_reward=1, answer_reward=0
    CATEGORY_3 = "bad_format"  # format_reward=0, answer_reward=0
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the categorizer
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def categorize_results(self, results: List[Dict], 
                          format_reward_key: str = 'format_reward',
                          answer_reward_key: str = 'answer_reward') -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Categorize results into three categories
        
        Args:
            results: List of graded results
            format_reward_key: Key in result dict containing format reward
            answer_reward_key: Key in result dict containing answer reward
        
        Returns:
            Tuple of (category_1, category_2, category_3)
            - Category 1: Format=1, Answer=1 (CORRECT)
            - Category 2: Format=1, Answer=0 (WRONG ANSWER)
            - Category 3: Format=0, Answer=0 (BAD FORMAT)
        """
        self.logger.info("Categorizing results...")
        
        category_1 = []  # format_reward=1, answer_reward=1
        category_2 = []  # format_reward=1, answer_reward=0
        category_3 = []  # format_reward=0, answer_reward=0
        
        for result in results:
            format_reward = result.get(format_reward_key, 0.0)
            answer_reward = result.get(answer_reward_key, 0.0)
            
            if format_reward == 1.0 and answer_reward == 1.0:
                category_1.append(result)
            elif format_reward == 1.0 and answer_reward == 0.0:
                category_2.append(result)
            elif format_reward == 0.0 and answer_reward == 0.0:
                category_3.append(result)
        
        self._log_categorization(results, category_1, category_2, category_3)
        
        return category_1, category_2, category_3
    
    def _log_categorization(self, results: List[Dict], category_1: List[Dict],
                           category_2: List[Dict], category_3: List[Dict]):
        """Log categorization results"""
        self.logger.info("="*80)
        self.logger.info("EVALUATION RESULTS BY CATEGORY")
        self.logger.info("="*80)
        self.logger.info(f"Total test examples: {len(results)}")
        self.logger.info(f"Category 1 (Format=1, Answer=1 - CORRECT): {len(category_1)} ({len(category_1)/len(results)*100:.2f}%)")
        self.logger.info(f"Category 2 (Format=1, Answer=0 - WRONG ANSWER): {len(category_2)} ({len(category_2)/len(results)*100:.2f}%)")
        self.logger.info(f"Category 3 (Format=0, Answer=0 - BAD FORMAT): {len(category_3)} ({len(category_3)/len(results)*100:.2f}%)")
        self.logger.info("="*80)
    
    def get_statistics(self, results: List[Dict],
                      format_reward_key: str = 'format_reward',
                      answer_reward_key: str = 'answer_reward') -> Dict:
        """
        Get categorization statistics
        
        Args:
            results: List of graded results
            format_reward_key: Key in result dict containing format reward
            answer_reward_key: Key in result dict containing answer reward
        
        Returns:
            Dictionary with categorization statistics
        """
        category_1, category_2, category_3 = self.categorize_results(
            results, format_reward_key, answer_reward_key
        )
        
        total = len(results)
        
        return {
            'total_results': total,
            'category_1_count': len(category_1),
            'category_2_count': len(category_2),
            'category_3_count': len(category_3),
            'category_1_percentage': len(category_1) / total * 100 if total > 0 else 0,
            'category_2_percentage': len(category_2) / total * 100 if total > 0 else 0,
            'category_3_percentage': len(category_3) / total * 100 if total > 0 else 0,
            'accuracy': len(category_1) / total * 100 if total > 0 else 0,
            'format_accuracy': sum(1 for r in results if r.get(format_reward_key, 0) == 1.0) / total * 100 if total > 0 else 0
        }


class GradingPipeline:
    """Complete pipeline for grading and categorizing results"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the pipeline
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.grader = ResultsGrader(logger)
        self.categorizer = ResultsCategorizer(logger)
    
    def grade_and_categorize(self, results: List[Dict],
                            response_key: str = 'generated_response',
                            ground_truth_key: str = 'ground_truth',
                            fast: bool = True) -> Tuple[List[Dict], List[Dict], List[Dict], Dict]:
        """
        Grade and categorize results in one pipeline
        
        Args:
            results: List of result dictionaries
            response_key: Key containing the response
            ground_truth_key: Key containing ground truth
            fast: Whether to use fast grading
        
        Returns:
            Tuple of (category_1, category_2, category_3, statistics)
        """
        # Grade results
        graded_results = self.grader.grade_results(
            results,
            response_key=response_key,
            ground_truth_key=ground_truth_key,
            fast=fast,
            show_progress=True
        )
        
        # Categorize results
        category_1, category_2, category_3 = self.categorizer.categorize_results(graded_results)
        
        # Get statistics
        stats = self.categorizer.get_statistics(graded_results)
        
        return category_1, category_2, category_3, stats
