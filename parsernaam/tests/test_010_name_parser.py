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
        self.df = pd.DataFrame({'name': ['Jan', 'Nicholas Turner', 'Petersen', 'Nichols Richard', 'Piet',
                                         'John Smith', 'Janssen', 'Kim Yeon']})
        self.expected = ["first", "first_last", "last", "last_first", "first", "first_last", "first", "last_first"]

    def tearDown(self) -> None:
        return super().tearDown()

    def test_parse(self) -> None:
        """
        Test parse pos
        """
        df = ParseNames.parse(self.df)
        print(df.to_markdown())
        for parsed_name in df['parsed_name']:
            name_type = parsed_name['type']
            prob = parsed_name['prob']
            expected_type = self.expected.pop(0)
            self.assertEqual(name_type, expected_type)
            self.assertGreater(prob, 0.5)


if __name__ == '__main__':
    unittest.main()
