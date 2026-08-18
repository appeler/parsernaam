"""To process arguments from the command line."""

import argparse


def get_args(
    argv: list[str], description: str, epilog: str, default_out: str
) -> argparse.Namespace:
    """Parse command line arguments for the parsernaam CLI tool.

    Args:
        argv: List of command line arguments
        description: Description text for the argument parser
        epilog: Example usage text shown after help
        default_out: Default output filename

    Returns:
        Parsed command line arguments namespace

    Example:
        >>> from parsernaam.utils import get_args
        >>> args = get_args(['input.csv', '-o', 'output.csv', '-n', 'name'],
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
    parser.add_argument("input", help="Input Parquet file")
    parser.add_argument(
        "-o", "--output", default=default_out, help="Output Parquet file"
    )
    parser.add_argument("-n", "--names-col", default="name", help="Names column")
    return parser.parse_args(argv)
