"""
To process arguments from the command line.
"""
import argparse


def get_args(argv, description: str, epilog: str, default_out: str) -> argparse.Namespace:
    """
    Returns arguments from the command line.
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
