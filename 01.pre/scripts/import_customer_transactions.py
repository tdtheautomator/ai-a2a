"""
----------------------
Imports csv into PostgreSQL.

Behaviour:
  - Creates database  : DEMODB          (if it doesn't exist)
  - Creates table     : "customer_transactions"  (if it doesn't exist)
  - Import modes:
    - overwrite: drops & recreates the table, then bulk-loads the CSV
    - append: adds data to existing table (creates table if it doesn't exist)
  - Columns/types     : auto-detected from the CSV header & sample rows

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
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv

# ─── LOGGING SETUP ────────────────────────────────────────────────────────────
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

log_file = os.path.join(LOG_DIR, f"import_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.info("═" * 80)
logger.info("CSV → PostgreSQL Import Process Started")
logger.info("═" * 80)
# ──────────────────────────────────────────────────────────────────────────────

# ─── CONFIG — loaded from .env file ───────────────────────────────────────────
# Expected .env keys:
#   PG_HOST      (default: localhost)
#   PG_PORT      (default: 5432)
#   PG_USER      (default: postgres)
#   PG_PASSWORD  (required)
#   PG_TARGET_DB    (default: DEMODB)
#   PG_TARGET_TABLE (default: customer_transactions)
#   PG_CSV_PATH     (default: customer_transaction_v1.csv)

logger.info("Loading configuration from .env file...")
load_dotenv()  # reads .env from the current working directory

PG_HOST      = os.getenv("PG_HOST",      "localhost")
PG_PORT      = int(os.getenv("PG_PORT",  "5432"))
PG_USER      = os.getenv("PG_USER",      "postgres")
PG_PASSWORD  = os.getenv("PG_PASSWORD")

PG_TARGET_DB    = os.getenv("PG_TARGET_DB",    "DEMODB")
PG_TARGET_TABLE = os.getenv("PG_TARGET_TABLE", "customer_transactions")
PG_CSV_PATH     = os.getenv("PG_CSV_PATH",     "customer_transaction.csv")

logger.debug(f"PostgreSQL Host: {PG_HOST}")
logger.debug(f"PostgreSQL Port: {PG_PORT}")
logger.debug(f"PostgreSQL User: {PG_USER}")
logger.debug(f"Target Database: {PG_TARGET_DB}")
logger.debug(f"Target Table: {PG_TARGET_TABLE}")
logger.debug(f"CSV Path: {PG_CSV_PATH}")

if not PG_PASSWORD:
    logger.critical("PG_PASSWORD is not set in .env file!")
    sys.exit("[!] PG_PASSWORD is not set in .env file. Please add it and retry.")

logger.info("✓ Configuration loaded successfully")


def get_pg_type(series: pd.Series) -> str:
    """Map a pandas Series dtype to a PostgreSQL column type."""
    try:
        dtype = series.dtype
        if pd.api.types.is_integer_dtype(dtype):
            logger.debug(f"Column '{series.name}' → BIGINT")
            return "BIGINT"
        if pd.api.types.is_float_dtype(dtype):
            logger.debug(f"Column '{series.name}' → DOUBLE PRECISION")
            return "DOUBLE PRECISION"
        if pd.api.types.is_bool_dtype(dtype):
            logger.debug(f"Column '{series.name}' → BOOLEAN")
            return "BOOLEAN"
        # Try to detect date/timestamp columns by name or content
        if pd.api.types.is_datetime64_any_dtype(dtype):
            logger.debug(f"Column '{series.name}' → TIMESTAMP")
            return "TIMESTAMP"
        logger.debug(f"Column '{series.name}' → TEXT (default)")
        return "TEXT"
    except Exception as e:
        logger.error(f"Error determining type for column '{series.name}': {str(e)}")
        logger.debug(traceback.format_exc())
        return "TEXT"


def try_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Attempt to convert object columns that look like dates to datetime."""
    try:
        date_cols = 0
        for col in df.select_dtypes(include="object").columns:
            try:
                converted = pd.to_datetime(df[col], infer_datetime_format=True)
                df[col] = converted
                date_cols += 1
                logger.debug(f"Successfully parsed column '{col}' as datetime")
            except (ValueError, TypeError) as e:
                logger.debug(f"Column '{col}' is not a date (expected): {str(e)[:50]}")
        logger.debug(f"Parsed {date_cols} column(s) as datetime")
        return df
    except Exception as e:
        logger.error(f"Error parsing dates: {str(e)}")
        logger.debug(traceback.format_exc())
        return df


def create_database_if_missing(conn_params: dict, db_name: str):
    """Connect to the default 'postgres' DB and create target DB if absent."""
    try:
        logger.info(f"Attempting to connect to PostgreSQL at {conn_params['host']}:{conn_params['port']}...")
        conn = psycopg2.connect(**conn_params, dbname="postgres")
        logger.debug(f"✓ Connected to PostgreSQL server")
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        cur.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (db_name,)
        )
        exists = cur.fetchone()
        if not exists:
            logger.info(f"Creating database '{db_name}'...")
            cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
            logger.info(f"✓ Database '{db_name}' created successfully")
        else:
            logger.info(f"✓ Database '{db_name}' already exists")

        cur.close()
        conn.close()
        logger.debug("Connection to postgres DB closed")
    except psycopg2.OperationalError as e:
        logger.critical(f"Failed to connect to PostgreSQL: {str(e)}")
        logger.debug(f"Connection params: host={conn_params.get('host')}, port={conn_params.get('port')}, user={conn_params.get('user')}")
        logger.debug(traceback.format_exc())
        raise
    except Exception as e:
        logger.critical(f"Error creating database '{db_name}': {str(e)}")
        logger.debug(traceback.format_exc())
        raise


def build_column_ddl(df: pd.DataFrame) -> str:
    """Return the column definition block for CREATE TABLE."""
    parts = []
    for col in df.columns:
        pg_type = get_pg_type(df[col])
        parts.append(
            f"    {sql.Identifier(col).as_string(None) if False else quoted(col)} {pg_type}"
        )
    return ",\n".join(parts)


def quoted(name: str) -> str:
    """Double-quote an identifier safely."""
    return '"' + name.replace('"', '""') + '"'


def table_exists(cur, table_name: str) -> bool:
    """Check if a table exists in the current database."""
    try:
        cur.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = %s",
            (table_name,)
        )
        return cur.fetchone() is not None
    except Exception as e:
        logger.error(f"Error checking if table exists: {str(e)}")
        return False


def import_csv(csv_path: str, mode: str = 'overwrite'):
    """Import CSV data into PostgreSQL with comprehensive error logging."""
    try:
        if not os.path.isfile(csv_path):
            logger.critical(f"CSV file not found: {csv_path}")
            logger.debug(f"Current working directory: {os.getcwd()}")
            logger.debug(f"Absolute path attempted: {os.path.abspath(csv_path)}")
            sys.exit(f"[!] CSV file not found: {csv_path}")

        logger.info(f"Reading CSV file: {csv_path}")
        try:
            df = pd.read_csv(csv_path, dtype=str)          # read all as str first
            logger.info(f"✓ CSV file loaded successfully")
        except pd.errors.ParserError as e:
            logger.error(f"CSV parsing error: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
        except Exception as e:
            logger.error(f"Unexpected error reading CSV: {str(e)}")
            logger.debug(traceback.format_exc())
            raise

        logger.info(f"Initial CSV shape: {len(df):,} rows × {len(df.columns)} columns")
        logger.debug(f"Columns: {list(df.columns)}")
        
        # Handle null values
        logger.debug("Converting NaN values to None...")
        df = df.where(pd.notnull(df), None)

        # Infer numeric types
        logger.debug("Attempting to infer numeric types...")
        numeric_cols = 0
        for col in df.columns:
            try:
                prev_dtype = df[col].dtype
                df[col] = pd.to_numeric(df[col])
                numeric_cols += 1
                logger.debug(f"  Column '{col}': {prev_dtype} → numeric")
            except (ValueError, TypeError):
                pass
        logger.debug(f"Converted {numeric_cols} column(s) to numeric")

        # Try to parse dates
        logger.debug("Attempting to infer date/datetime types...")
        df = try_parse_dates(df)

        logger.info(f"Final schema: {len(df.columns)} columns, {len(df):,} rows")
        if len(df) == 0:
            logger.warning("⚠ CSV file is empty (no data rows")
        else:
            logger.debug(f"First row preview: {df.iloc[0].to_dict()}")

        # Build connection params
        conn_params = {
            "host":     PG_HOST,
            "port":     PG_PORT,
            "user":     PG_USER,
            "password": PG_PASSWORD,
        }

        # 1. Ensure the target database exists
        logger.info("─" * 80)
        logger.info("PHASE 1: Database Setup")
        logger.info("─" * 80)
        create_database_if_missing(conn_params, PG_TARGET_DB)

        # 2. Connect to DEMODB database
        logger.info("─" * 80)
        logger.info("PHASE 2: Table Creation")
        logger.info("─" * 80)
        try:
            logger.info(f"Connecting to database '{PG_TARGET_DB}'...")
            conn = psycopg2.connect(**conn_params, dbname=PG_TARGET_DB)
            conn.autocommit = False
            cur = conn.cursor()
            logger.debug(f"✓ Connected to database '{PG_TARGET_DB}'")
        except psycopg2.OperationalError as e:
            logger.critical(f"Failed to connect to database '{PG_TARGET_DB}': {str(e)}")
            logger.debug(traceback.format_exc())
            raise

        table_id = sql.Identifier(PG_TARGET_TABLE)

        # 3. Handle table based on mode
        logger.info("─" * 80)
        logger.info("PHASE 2: Table Setup")
        logger.info("─" * 80)
        if mode == 'overwrite':
            logger.info(f"Dropping table '{PG_TARGET_TABLE}' if it exists...")
            try:
                cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(table_id))
                logger.debug(f"✓ Drop table statement executed (table may not have existed)")
            except Exception as e:
                logger.error(f"Error dropping table: {str(e)}")
                logger.debug(traceback.format_exc())
                conn.rollback()
                raise
            create_table = True
        else:  # append
            if table_exists(cur, PG_TARGET_TABLE):
                logger.info(f"✓ Table '{PG_TARGET_TABLE}' exists, appending data...")
                create_table = False
            else:
                logger.info(f"Table '{PG_TARGET_TABLE}' does not exist, creating it...")
                create_table = True

        if create_table:
            # 4. Build CREATE TABLE statement from inferred schema
            logger.info(f"Creating table '{PG_TARGET_TABLE}' with auto-detected schema...")
            try:
                col_defs = ",\n".join(
                    f"    {quoted(col)} {get_pg_type(df[col])}"
                    for col in df.columns
                )
                create_stmt = f'CREATE TABLE {quoted(PG_TARGET_TABLE)} (\n{col_defs}\n);'
                logger.debug(f"Generated CREATE TABLE statement:\n{create_stmt}")
                cur.execute(create_stmt)
                logger.info(f"✓ Table '{PG_TARGET_TABLE}' created successfully")
            except psycopg2.ProgrammingError as e:
                logger.error(f"SQL error creating table: {str(e)}")
                logger.debug(f"SQL statement:\n{create_stmt}")
                logger.debug(traceback.format_exc())
                conn.rollback()
                raise
            except Exception as e:
                logger.error(f"Unexpected error creating table: {str(e)}")
                logger.debug(traceback.format_exc())
                conn.rollback()
                raise

        # 5. Bulk insert using execute_values for performance
        logger.info("─" * 80)
        logger.info("PHASE 3: Data Import")
        logger.info("─" * 80)
        try:
            from psycopg2.extras import execute_values

            col_names = ", ".join(quoted(c) for c in df.columns)
            insert_sql = f'INSERT INTO {quoted(PG_TARGET_TABLE)} ({col_names}) VALUES %s'
            
            rows = [tuple(row) for row in df.itertuples(index=False, name=None)]
            logger.info(f"Preparing to insert {len(rows):,} rows...")
            logger.debug(f"First row sample: {rows[0] if rows else 'N/A'}")
            
            execute_values(cur, insert_sql, rows, page_size=500)
            logger.info(f"✓ All {len(rows):,} rows inserted to database")
        except psycopg2.DataError as e:
            logger.error(f"Data type error during insert: {str(e)}")
            logger.debug(f"This typically indicates a mismatch between CSV data and detected PostgreSQL type")
            logger.debug(traceback.format_exc())
            conn.rollback()
            raise
        except psycopg2.IntegrityError as e:
            logger.error(f"Integrity constraint error: {str(e)}")
            logger.debug(traceback.format_exc())
            conn.rollback()
            raise
        except Exception as e:
            logger.error(f"Unexpected error during insert: {str(e)}")
            logger.debug(traceback.format_exc())
            conn.rollback()
            raise

        # Commit transaction
        try:
            conn.commit()
            logger.info(f"✓ Transaction committed successfully")
        except Exception as e:
            logger.error(f"Error committing transaction: {str(e)}")
            logger.debug(traceback.format_exc())
            conn.rollback()
            raise

        # 6. Quick verification
        logger.info("─" * 80)
        logger.info("PHASE 4: Verification")
        logger.info("─" * 80)
        try:
            cur.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(table_id))
            count = cur.fetchone()[0]
            logger.info(f"✓ Final row count in table: {count:,}")
            
            if count == len(rows):
                logger.info(f"✓ SUCCESS: All {count:,} rows imported correctly!")
            else:
                logger.warning(f"⚠ Row count mismatch: expected {len(rows):,}, got {count:,}")
                
        except Exception as e:
            logger.error(f"Error verifying row count: {str(e)}")
            logger.debug(traceback.format_exc())
            raise

        cur.close()
        conn.close()
        logger.debug("Database connections closed")
        logger.info("=" * 80)
        logger.info(f"IMPORT COMPLETED SUCCESSFULLY")
        logger.info(f"Log file saved to: {log_file}")
        logger.info("=" * 80)

    except Exception as e:
        logger.critical(f"IMPORT FAILED: {str(e)}")
        logger.critical(f"Log file saved to: {log_file}")
        logger.info("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Import CSV into PostgreSQL")
        parser.add_argument(
            "csv_path",
            nargs="?",
            default=PG_CSV_PATH,
            help=f"Path to the CSV file (default: {PG_CSV_PATH})"
        )
        parser.add_argument(
            "--mode",
            choices=["overwrite", "append"],
            default="overwrite",
            help="Import mode: 'overwrite' drops and recreates the table, 'append' adds to existing table (default: overwrite)"
        )
        args = parser.parse_args()

        PG_CSV_PATH = args.csv_path
        mode = args.mode

        logger.info(f"CSV path: {PG_CSV_PATH}")
        logger.info(f"Import mode: {mode}")

        import_csv(PG_CSV_PATH, mode)
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unhandled exception: {str(e)}")
        logger.debug(traceback.format_exc())
        sys.exit(1)