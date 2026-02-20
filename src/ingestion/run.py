"""Orchestrate full data ingestion pipeline."""

import logging
import argparse

from .polymarket import run_polymarket_ingestion
from .kalshi import run_kalshi_ingestion
from .normalize import run_normalization

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run full data ingestion pipeline")
    parser.add_argument("--force", action="store_true", help="Force re-fetch all data")
    parser.add_argument("--skip-polymarket", action="store_true")
    parser.add_argument("--skip-kalshi", action="store_true")
    parser.add_argument("--normalize-only", action="store_true", help="Only run normalization on cached data")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    if not args.normalize_only:
        if not args.skip_polymarket:
            logger.info("=" * 60)
            logger.info("PHASE 1: Polymarket ingestion")
            logger.info("=" * 60)
            run_polymarket_ingestion(force=args.force)

        if not args.skip_kalshi:
            logger.info("=" * 60)
            logger.info("PHASE 2: Kalshi ingestion")
            logger.info("=" * 60)
            run_kalshi_ingestion(force=args.force)

    logger.info("=" * 60)
    logger.info("PHASE 3: Normalization")
    logger.info("=" * 60)
    run_normalization()

    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
