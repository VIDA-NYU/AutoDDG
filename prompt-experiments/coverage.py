import re
import json
from typing import Dict, List
import numpy as np
from collections import Counter

##---- Coverage Score 

'''
Run an LLM that extracts info about the dataset 
This is data to measure coverage in 5 coverage categories: 
- basic_info (15%)
- data_characteristics (25%)
- provenance (20%)
- usage_context (20%)
- quality_and_limitations (20%)


'''

class CoverageScorer:
    def __init__(self):
        # Define what "complete" coverage means
        self.coverage_dimensions = {
            'basic_info': {
                'weight': 0.15,
                'fields': [
                    'dataset_name',
                    'domain_or_field',
                    'primary_purpose'
                ]
            },
            'data_characteristics': {
                'weight': 0.25,
                'fields': [
                    'size_or_scale',
                    'data_format',
                    'data_types',
                    'temporal_coverage',
                    'sample_unit'  # e.g., "images", "documents", "participants"
                ]
            },
            'provenance': {
                'weight': 0.20,
                'fields': [
                    'collection_method',
                    'data_source',
                    'collection_date',
                    'creators_or_curators',
                    'preprocessing_steps'
                ]
            },
            'usage_context': {
                'weight': 0.20,
                'fields': [
                    'typical_applications',
                    'research_questions_addressed',
                    'how_used_in_paper',
                    'benchmark_or_evaluation_role'
                ]
            },
            'quality_and_limitations': {
                'weight': 0.20,
                'fields': [
                    'known_limitations',
                    'biases_or_caveats',
                    'quality_issues',
                    'challenges_in_use'
                ]
            }
        }
    
    def calculate_coverage(self, extraction_result: Dict) -> Dict:
        """Calculate comprehensive coverage score."""
        
        scores = {}
        details = {}
        
        for dimension, config in self.coverage_dimensions.items():
            dim_score = self._score_dimension(
                extraction_result, 
                dimension,
                config['fields']
            )
            scores[dimension] = dim_score
            details[dimension] = self._get_dimension_details(
                extraction_result,
                dimension,
                config['fields']
            )
        
        # Calculate weighted overall score
        overall_score = sum(
            scores[dim] * self.coverage_dimensions[dim]['weight']
            for dim in scores
        )
        
        return {
            'overall_score': overall_score,
            'dimension_scores': scores,
            'details': details,
            'grade': self._get_grade(overall_score),
            'missing_critical': self._identify_missing_critical(details),
            'completeness_by_dimension': self._completeness_report(scores)
        }
    
    def _score_dimension(self, data: Dict, dimension: str, fields: List[str]) -> float:
        """Score a single dimension (0.0 to 1.0)."""
        
        if dimension not in data:
            return 0.0
        
        dim_data = data[dimension]
        filled_fields = 0
        
        for field in fields:
            if self._is_field_filled(dim_data.get(field)):
                filled_fields += 1
        
        return filled_fields / len(fields) if fields else 0.0
    
    def _is_field_filled(self, value) -> bool:
        """Check if a field has meaningful content."""
        
        if value is None:
            return False
        
        if isinstance(value, str):
            # Check for placeholder values
            placeholder_phrases = [
                'not mentioned',
                'not specified',
                'not provided',
                'not available',
                'unknown',
                'n/a',
                'null'
            ]
            value_lower = value.lower().strip()
            
            if not value_lower or value_lower in placeholder_phrases:
                return False
            
            # Require minimum substance (>10 chars)
            return len(value_lower) > 10
        
        elif isinstance(value, list):
            # List must have at least one meaningful item
            return len(value) > 0 and any(
                self._is_field_filled(item) for item in value
            )
        
        elif isinstance(value, dict):
            # Dict must have at least one filled field
            return any(self._is_field_filled(v) for v in value.values())
        
        return True  # Numbers, booleans, etc.
    
    def _get_dimension_details(self, data: Dict, dimension: str, fields: List[str]) -> Dict:
        """Get detailed status of each field in a dimension."""
        
        dim_data = data.get(dimension, {})
        
        return {
            field: {
                'present': self._is_field_filled(dim_data.get(field)),
                'value': dim_data.get(field),
                'quality': self._assess_field_quality(dim_data.get(field))
            }
            for field in fields
        }
    
    def _assess_field_quality(self, value) -> str:
        """Assess quality of field content: 'high', 'medium', 'low', 'missing'."""
        
        if not self._is_field_filled(value):
            return 'missing'
        
        if isinstance(value, str):
            length = len(value)
            if length > 100:
                return 'high'  # Detailed description
            elif length > 30:
                return 'medium'  # Basic info
            else:
                return 'low'  # Minimal info
        
        elif isinstance(value, list):
            if len(value) >= 3:
                return 'high'
            elif len(value) >= 1:
                return 'medium'
            else:
                return 'low'
        
        return 'medium'
    
    def _get_grade(self, score: float) -> str:
        """Convert score to letter grade."""
        if score >= 0.9:
            return 'A'
        elif score >= 0.8:
            return 'B'
        elif score >= 0.7:
            return 'C'
        elif score >= 0.6:
            return 'D'
        else:
            return 'F'
    
    def _identify_missing_critical(self, details: Dict) -> List[str]:
        """Identify missing fields that are considered critical."""
        
        critical_fields = {
            'basic_info': ['dataset_name', 'domain_or_field'],
            'data_characteristics': ['size_or_scale', 'data_format'],
            'provenance': ['collection_method'],
            'usage_context': ['typical_applications'],
            'quality_and_limitations': ['known_limitations']
        }
        
        missing = []
        
        for dimension, fields in critical_fields.items():
            if dimension in details:
                for field in fields:
                    if field in details[dimension]:
                        if not details[dimension][field]['present']:
                            missing.append(f"{dimension}.{field}")
        
        return missing
    
    def _completeness_report(self, scores: Dict) -> str:
        """Generate human-readable completeness report."""
        
        report_lines = []
        
        for dimension, score in scores.items():
            percentage = score * 100
            status = "✓" if score >= 0.7 else "⚠" if score >= 0.4 else "✗"
            report_lines.append(f"{status} {dimension}: {percentage:.0f}%")
        
        return "\n".join(report_lines)


# Enhanced version with confidence weighting
class ConfidenceWeightedCoverage(CoverageScorer):
    """Coverage score that also considers extraction confidence."""
    
    def calculate_coverage(self, extraction_result: Dict) -> Dict:
        """Calculate coverage with confidence weighting."""
        
        base_coverage = super().calculate_coverage(extraction_result)
        
        # Adjust scores based on confidence levels
        confidence_adjusted = {}
        
        for dimension, score in base_coverage['dimension_scores'].items():
            confidence = self._get_dimension_confidence(
                extraction_result, 
                dimension
            )
            
            # Penalize low-confidence extractions
            adjusted_score = score * self._confidence_multiplier(confidence)
            confidence_adjusted[dimension] = {
                'raw_score': score,
                'confidence': confidence,
                'adjusted_score': adjusted_score
            }
        
        # Recalculate overall with confidence adjustment
        overall_adjusted = sum(
            confidence_adjusted[dim]['adjusted_score'] * 
            self.coverage_dimensions[dim]['weight']
            for dim in confidence_adjusted
        )
        
        base_coverage['confidence_adjusted_score'] = overall_adjusted
        base_coverage['dimension_scores_detailed'] = confidence_adjusted
        
        return base_coverage
    
    def _get_dimension_confidence(self, data: Dict, dimension: str) -> str:
        """Extract confidence level if present in data."""
        
        dim_data = data.get(dimension, {})
        
        # Look for confidence field
        if isinstance(dim_data, dict) and 'confidence' in dim_data:
            return dim_data['confidence']
        
        return 'medium'  # Default
    
    def _confidence_multiplier(self, confidence: str) -> float:
        """Convert confidence level to score multiplier."""
        
        multipliers = {
            'high': 1.0,
            'medium': 0.8,
            'low': 0.5
        }
        
        return multipliers.get(confidence.lower(), 0.8)