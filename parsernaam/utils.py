"""
To process arguments from the command line.
"""
import argparse
from typing import List


def get_args(argv: List[str], description: str, epilog: str, default_out: str) -> argparse.Namespace:
    """
    Parse command line arguments for the parsernaam CLI tool.
    
    Args:
        argv: List of command line arguments
        description: Description text for the argument parser
        epilog: Example usage text shown after help
        default_out: Default output filename
        
    Returns:
        Parsed command line arguments namespace
        
    Example:
        >>> args = get_args(['input.csv', '-o', 'output.csv'], 
        ...                 'Parse names', 'Example usage', 'out.csv')
        >>> args.input
        'input.csv'
    """
    parser = argparse.ArgumentParser(
        prog="parsernaam",
        description=description,
        epilog=epilog,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('input', default=None, help='Input file')
    parser.add_argument('-o', '--output', default=default_out, help='Output file')
    parser.add_argument('-n', '--names-col', default="name" ,required=True, help='Names column')
    args = parser.parse_args(argv)
    return args
