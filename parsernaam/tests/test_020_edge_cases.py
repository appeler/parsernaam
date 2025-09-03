#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_020_edge_cases
Test edge cases and error handling for name parsing
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch

from parsernaam.parse import ParseNames


class TestEdgeCases(unittest.TestCase):
    """
    Test edge cases and error handling
    """

    def test_empty_dataframe(self):
        """Test parsing empty DataFrame"""
        df = pd.DataFrame({'name': []})
        result = ParseNames.parse(df)
        self.assertTrue(result.empty)
        self.assertIn('parsed_name', result.columns)

    def test_missing_name_column(self):
        """Test DataFrame without 'name' column raises ValueError"""
        df = pd.DataFrame({'other_column': ['John']})
        with self.assertRaises(ValueError) as context:
            ParseNames.parse(df)
        self.assertIn("'name' column", str(context.exception))

    def test_invalid_input_type(self):
        """Test non-DataFrame input raises ValueError"""
        with self.assertRaises(ValueError) as context:
            ParseNames.parse("not a dataframe")
        self.assertIn("pandas DataFrame", str(context.exception))

    def test_empty_strings(self):
        """Test empty and whitespace-only strings"""
        df = pd.DataFrame({'name': ['', '   ', '\t', '\n']})
        result = ParseNames.parse(df)
        
        for _, row in result.iterrows():
            parsed = row['parsed_name']
            self.assertEqual(parsed['type'], 'unknown')
            self.assertEqual(parsed['prob'], 0.0)

    def test_none_values(self):
        """Test None values in name column"""
        df = pd.DataFrame({'name': [None, 'John', None]})
        result = ParseNames.parse(df)
        
        # Check None values are handled
        for i in [0, 2]:  # None value indices
            parsed = result.iloc[i]['parsed_name']
            self.assertEqual(parsed['type'], 'unknown')
            self.assertEqual(parsed['prob'], 0.0)
        
        # Check valid name is still processed
        valid_parsed = result.iloc[1]['parsed_name']
        self.assertEqual(valid_parsed['name'], 'John')
        self.assertIn(valid_parsed['type'], ['first', 'last'])
        self.assertGreater(valid_parsed['prob'], 0.0)

    def test_numeric_values(self):
        """Test numeric values in name column"""
        df = pd.DataFrame({'name': [123, 45.6, np.nan]})
        result = ParseNames.parse(df)
        
        for _, row in result.iterrows():
            parsed = row['parsed_name']
            self.assertEqual(parsed['type'], 'unknown')
            self.assertEqual(parsed['prob'], 0.0)

    def test_very_long_names(self):
        """Test names longer than sequence length"""
        long_name = 'A' * 100  # Much longer than SEQUENCE_LENGTH (30)
        df = pd.DataFrame({'name': [long_name]})
        result = ParseNames.parse(df)
        
        parsed = result.iloc[0]['parsed_name']
        self.assertEqual(parsed['name'], long_name)
        self.assertIn(parsed['type'], ['first', 'last'])
        self.assertGreater(parsed['prob'], 0.0)

    def test_special_characters(self):
        """Test names with special characters"""
        special_names = [
            'José María',
            "O'Connor", 
            'Anne-Marie',
            'João-Pedro',
            'Müller'
        ]
        df = pd.DataFrame({'name': special_names})
        result = ParseNames.parse(df)
        
        for _, row in result.iterrows():
            parsed = row['parsed_name']
            # Special characters might be treated as single names in some cases
            self.assertIn(parsed['type'], ['first', 'last', 'first_last', 'last_first'])
            self.assertGreater(parsed['prob'], 0.0)

    def test_single_character_names(self):
        """Test single character names"""
        df = pd.DataFrame({'name': ['A', 'X', 'Z']})
        result = ParseNames.parse(df)
        
        for _, row in result.iterrows():
            parsed = row['parsed_name']
            self.assertIn(parsed['type'], ['first', 'last'])
            self.assertGreater(parsed['prob'], 0.0)

    def test_many_word_names(self):
        """Test names with many words"""
        long_names = [
            'María José García López Santos',
            'Dr John Michael Smith Jr III',
            'Anna Maria Elisabeth Van Der Berg'
        ]
        df = pd.DataFrame({'name': long_names})
        result = ParseNames.parse(df)
        
        for _, row in result.iterrows():
            parsed = row['parsed_name']
            # Multi-word names should use positional model
            self.assertIn(parsed['type'], ['first_last', 'last_first'])
            self.assertGreater(parsed['prob'], 0.0)

    def test_mixed_case_names(self):
        """Test names with various capitalizations"""
        mixed_names = [
            'JOHN SMITH',
            'john smith', 
            'John SMITH',
            'jOhN sMiTh'
        ]
        df = pd.DataFrame({'name': mixed_names})
        result = ParseNames.parse(df)
        
        for _, row in result.iterrows():
            parsed = row['parsed_name']
            self.assertIn(parsed['type'], ['first_last', 'last_first'])
            self.assertGreater(parsed['prob'], 0.0)

    def test_model_caching(self):
        """Test that models are cached between calls"""
        df1 = pd.DataFrame({'name': ['John Smith']})
        df2 = pd.DataFrame({'name': ['Jane Doe']})
        
        # First call should cache models
        result1 = ParseNames.parse(df1)
        
        # Second call should use cached models
        result2 = ParseNames.parse(df2)
        
        # Both should work correctly
        self.assertEqual(len(result1), 1)
        self.assertEqual(len(result2), 1)
        
        # Check cache exists
        self.assertIsNotNone(ParseNames._vocab_cache)
        self.assertTrue(len(ParseNames._models_cache) > 0)

    def test_dataframe_with_additional_columns(self):
        """Test DataFrame with extra columns beyond 'name'"""
        df = pd.DataFrame({
            'name': ['John Smith', 'Jane Doe'],
            'age': [30, 25],
            'city': ['NYC', 'LA']
        })
        result = ParseNames.parse(df)
        
        # Should preserve all original columns
        self.assertIn('age', result.columns)
        self.assertIn('city', result.columns)
        self.assertIn('parsed_name', result.columns)
        
        # Should parse names correctly
        for _, row in result.iterrows():
            parsed = row['parsed_name']
            self.assertIn(parsed['type'], ['first_last', 'last_first'])


if __name__ == '__main__':
    unittest.main()