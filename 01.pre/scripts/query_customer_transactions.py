"""
----------------------
Query Customer Transactions Table.

Behaviour:
  - Connects to PostgreSQL database
  - Executes SELECT query on "Customer Transactions" table
  - Default: SELECT * FROM "Customer Transactions" LIMIT 10
  - Displays results as formatted table

Requirements:
  pip install psycopg2-binary pandas python-dotenv
"""

import os
import sys
import logging
import traceback
import argparse
from datetime import datetime
import pandas as pd
import psycopg2
from dotenv import load_dotenv

# ─── LOGGING SETUP ────────────────────────────────────────────────────────────
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

log_file = os.path.join(LOG_DIR, f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.info("═" * 80)
logger.info("Customer Transactions Query Process Started")
logger.info("═" * 80)
# ──────────────────────────────────────────────────────────────────────────────

# ─── CONFIG — loaded from .env file ───────────────────────────────────────────
# Expected .env keys:
#   PG_HOST      (default: localhost)
#   PG_PORT      (default: 5432)
#   PG_USER      (required)
#   PG_PASSWORD  (required)
#   PG_TARGET_DB    (default: DEMODB)
#   PG_TARGET_TABLE (default: Customer Transactions)

logger.info("Loading configuration from .env file...")
load_dotenv()  # reads .env from the current working directory

PG_HOST      = os.getenv("PG_HOST",      "localhost")
PG_PORT      = int(os.getenv("PG_PORT",  "5432"))
PG_USER      = os.getenv("PG_USER")
PG_PASSWORD  = os.getenv("PG_PASSWORD")

PG_TARGET_DB    = os.getenv("PG_TARGET_DB",    "DEMODB")
PG_TARGET_TABLE = os.getenv("PG_TARGET_TABLE", "Customer Transactions")

logger.debug(f"PostgreSQL Host: {PG_HOST}")
logger.debug(f"PostgreSQL Port: {PG_PORT}")
logger.debug(f"PostgreSQL User: {PG_USER}")
logger.debug(f"Target Database: {PG_TARGET_DB}")
logger.debug(f"Target Table: {PG_TARGET_TABLE}")

if not PG_USER or not PG_PASSWORD:
    logger.critical("PG_USER and PG_PASSWORD are required in .env file!")
    sys.exit("[!] PG_USER and PG_PASSWORD are required in .env file. Please add them and retry.")

logger.info("✓ Configuration loaded successfully")


def query_table(limit: int = 10, custom_query: str = None):
    """Query the Customer Transactions table and display results."""
    try:
        # Build connection params
        conn_params = {
            "host":     PG_HOST,
            "port":     PG_PORT,
            "user":     PG_USER,
            "password": PG_PASSWORD,
            "dbname":   PG_TARGET_DB,
        }

        logger.info(f"Connecting to database '{PG_TARGET_DB}'...")
        conn = psycopg2.connect(**conn_params)
        cur = conn.cursor()
        logger.info("✓ Connected to database successfully")

        # Build query
        if custom_query:
            query = custom_query
            logger.info(f"Executing custom query: {query}")
        else:
            query = f'SELECT * FROM "{PG_TARGET_TABLE}" LIMIT {limit}'
            logger.info(f"Executing default query: SELECT * FROM \"{PG_TARGET_TABLE}\" LIMIT {limit}")

        # Execute query
        cur.execute(query)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]

        logger.info(f"Query returned {len(rows)} rows")

        if rows:
            # Create DataFrame for nice display
            df = pd.DataFrame(rows, columns=columns)
            print("\n" + "=" * 80)
            print(f"QUERY RESULTS ({len(rows)} rows)")
            print("=" * 80)
            print(df.to_string(index=False))
            print("=" * 80)
        else:
            print("No rows returned from query.")

        cur.close()
        conn.close()
        logger.info("Database connection closed")

    except psycopg2.OperationalError as e:
        logger.critical(f"Failed to connect to database: {str(e)}")
        logger.debug(traceback.format_exc())
        raise
    except psycopg2.ProgrammingError as e:
        logger.critical(f"SQL error: {str(e)}")
        logger.debug(traceback.format_exc())
        raise
    except Exception as e:
        logger.critical(f"Unexpected error: {str(e)}")
        logger.debug(traceback.format_exc())
        raise


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Query Customer Transactions table")
        parser.add_argument(
            "--limit",
            type=int,
            default=10,
            help="Number of rows to select (default: 10)"
        )
        parser.add_argument(
            "--query",
            type=str,
            help="Custom SQL query to execute (overrides default SELECT)"
        )
        args = parser.parse_args()

        query_table(limit=args.limit, custom_query=args.query)

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}")
        logger.debug(traceback.format_exc())
        sys.exit(1)