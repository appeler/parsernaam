#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_010_name_parser
"""

import unittest
import pandas as pd

from parsernaam.parse import ParseNames

class TestParseNames(unittest.TestCase):
    """
    TestParseNames
    """

    def setUp(self) -> None:
        """
        Set up
        """
        self.df = pd.DataFrame({'name': ['Jan Petersen', 'Piet', 'Janssen']})
        self.expected = ["first", "last", "first", "first"]

    def tearDown(self) -> None:
        return super().tearDown()

    def test_parse(self) -> None:
        """
        Test parse
        """
        df = ParseNames.parse(self.df)
        for parsed_name in df['parsed_name']:
            for name_dict in parsed_name:
                name_type = name_dict['type']
                prob = name_dict['prob']
                expected_type = self.expected.pop(0)
                self.assertEqual(name_type, expected_type)
                self.assertGreater(prob, 0.5)



if __name__ == '__main__':
    unittest.main()