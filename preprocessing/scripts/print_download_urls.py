"""Prints the URLs for lc0 training files for a specific month.

The files can be later downloaded via wget or curl, for example:

$ python print_download_urls.py -y 2024 -m 12 | xargs -n 1 -P 4 wget -c
"""

import calendar
import argparse


def generate_lc0_urls(year, month):
    """
    Generate URLs for lc0 training files for a specific year and month.

    Args:
        year (int): Year (YYYY)
        month (int): Month (1-12)
    """
    # Base URL template
    base_url = "https://storage.lczero.org/files/training_data/test80/training-run1-test80-{date}-{hour}17.tar"

    # Get number of days in the month
    num_days = calendar.monthrange(year, month)[1]

    # Generate URLs for each day and hour
    for day in range(1, num_days + 1):
        for hour in range(24):
            # Format date as YYYYMMDD
            date_str = f"{year}{month:02d}{day:02d}"
            # Format hour as HH
            hour_str = f"{hour:02d}"

            # Create the full URL
            url = base_url.format(date=date_str, hour=hour_str)
            print(url)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Generate URLs for lc0 training files for a specific month.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-y", "--year", type=int, help="Year (YYYY)")

    parser.add_argument("-m", "--month", type=int, help="Month (1-12)")

    # Parse arguments
    args = parser.parse_args()

    generate_lc0_urls(args.year, args.month)
