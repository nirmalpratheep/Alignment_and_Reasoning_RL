"""
Utility class for loading and managing MATH dataset
Used across step0 and step1 scripts
"""

from datasets import load_dataset
from typing import Dict, List, Tuple, Optional


class MathDatasetLoader:
    """Loader for EleutherAI/hendrycks_math dataset with utility methods"""
    
    # Available subsets in the MATH dataset
    DEFAULT_SUBSETS = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus"
    ]
    
    def __init__(self):
        """Initialize the dataset loader"""
        self.datasets = {}
        self.subsets = []
        self.total_train = 0
        self.total_test = 0
    
    def load_all_subsets(self, subsets: Optional[List[str]] = None) -> Tuple[Dict, List[str], int, int]:
        """
        Load all or specified MATH dataset subsets
        
        Args:
            subsets: List of subset names to load. If None, loads all default subsets.
        
        Returns:
            Tuple of (datasets_dict, subsets_list, total_train_count, total_test_count)
        """
        if subsets is None:
            subsets = self.DEFAULT_SUBSETS
        
        self.subsets = subsets
        self.datasets = {}
        self.total_train = 0
        self.total_test = 0
        
        for subset in subsets:
            self.datasets[subset] = load_dataset("EleutherAI/hendrycks_math", subset)
            
            for split_name, split_data in self.datasets[subset].items():
                num_examples = len(split_data)
                
                if split_name == 'train':
                    self.total_train += num_examples
                elif split_name == 'test':
                    self.total_test += num_examples
        
        return self.datasets, self.subsets, self.total_train, self.total_test
    
    def load_subset(self, subset: str) -> Dict:
        """
        Load a single MATH dataset subset
        
        Args:
            subset: Name of the subset (e.g., 'algebra')
        
        Returns:
            Dictionary with 'train' and 'test' splits
        """
        dataset = load_dataset("EleutherAI/hendrycks_math", subset)
        self.datasets[subset] = dataset
        
        if subset not in self.subsets:
            self.subsets.append(subset)
            for split_name, split_data in dataset.items():
                if split_name == 'train':
                    self.total_train += len(split_data)
                elif split_name == 'test':
                    self.total_test += len(split_data)
        
        return dataset
    
    def get_train_data(self, subset: Optional[str] = None) -> Dict:
        """
        Get training data from a subset or all subsets
        
        Args:
            subset: Specific subset name. If None, returns all training data combined.
        
        Returns:
            Training dataset
        """
        if subset:
            if subset in self.datasets and 'train' in self.datasets[subset]:
                return self.datasets[subset]['train']
            raise ValueError(f"Subset '{subset}' not loaded or has no training data")
        
        # Return combined training data from all subsets
        all_train = []
        for subset_name, dataset in self.datasets.items():
            if 'train' in dataset:
                all_train.extend(dataset['train'])
        return all_train
    
    def get_test_data(self, subset: Optional[str] = None) -> Dict:
        """
        Get test data from a subset or all subsets
        
        Args:
            subset: Specific subset name. If None, returns all test data combined.
        
        Returns:
            Test dataset
        """
        if subset:
            if subset in self.datasets and 'test' in self.datasets[subset]:
                return self.datasets[subset]['test']
            raise ValueError(f"Subset '{subset}' not loaded or has no test data")
        
        # Return combined test data from all subsets
        all_test = []
        for subset_name, dataset in self.datasets.items():
            if 'test' in dataset:
                all_test.extend(dataset['test'])
        return all_test
    
    def collect_test_examples(self, include_metadata: bool = True) -> List[Dict]:
        """
        Collect all test examples from all loaded subsets
        
        Args:
            include_metadata: If True, includes subset, level, and type metadata
        
        Returns:
            List of test examples with metadata
        """
        all_test_examples = []
        
        for subset_name, dataset in self.datasets.items():
            if 'test' in dataset:
                for example in dataset['test']:
                    example_dict = {
                        'problem': example['problem'],
                        'solution': example['solution'],
                    }
                    
                    if include_metadata:
                        example_dict['subset'] = subset_name
                        example_dict['level'] = example.get('level', 'unknown')
                        example_dict['type'] = example.get('type', 'unknown')
                    
                    all_test_examples.append(example_dict)
        
        return all_test_examples
    
    def collect_train_examples(self, include_metadata: bool = True) -> List[Dict]:
        """
        Collect all training examples from all loaded subsets
        
        Args:
            include_metadata: If True, includes subset, level, and type metadata
        
        Returns:
            List of training examples with metadata
        """
        all_train_examples = []
        
        for subset_name, dataset in self.datasets.items():
            if 'train' in dataset:
                for example in dataset['train']:
                    example_dict = {
                        'problem': example['problem'],
                        'solution': example['solution'],
                    }
                    
                    if include_metadata:
                        example_dict['subset'] = subset_name
                        example_dict['level'] = example.get('level', 'unknown')
                        example_dict['type'] = example.get('type', 'unknown')
                    
                    all_train_examples.append(example_dict)
        
        return all_train_examples
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about loaded datasets
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'total_train': self.total_train,
            'total_test': self.total_test,
            'loaded_subsets': self.subsets,
            'subset_stats': {}
        }
        
        for subset_name, dataset in self.datasets.items():
            stats['subset_stats'][subset_name] = {
                'train': len(dataset.get('train', [])),
                'test': len(dataset.get('test', []))
            }
        
        return stats


# Convenience function for backward compatibility
def load_math_datasets(subsets=None):
    """
    Convenience function to load MATH datasets
    
    Args:
        subsets: List of subset names to load. If None, loads all default subsets.
    
    Returns:
        Tuple of (datasets_dict, subsets_list, total_train_count, total_test_count)
    """
    loader = MathDatasetLoader()
    return loader.load_all_subsets(subsets)
