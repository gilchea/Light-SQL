import pandas as pd
import sqlfluff
from sqlalchemy import text, Engine
from src.utils import clean_sql_markdown
import logging

logger = logging.getLogger(__name__)

# Safe import for SQLFluff errors
try:
    from sqlfluff.core.errors import APIParsingError
except ImportError:
    APIParsingError = Exception

def check_sql_syntax(sql: str) -> bool:
    """Checks if the SQL syntax is valid using SQLFluff (ANSI dialect)."""
    if not sql: return False
    try:
        sqlfluff.parse(sql, dialect="ansi")
        return True
    except (APIParsingError, Exception):
        return False

def check_execution_match(sql_gen: str, sql_truth: str, conn) -> tuple[str, bool]:
    """
    Executes both generated and ground truth SQL on the DB.
    Compares the resulting DataFrames.
    Returns: (Status Message, Is Match Boolean)
    """
    if not sql_gen or str(sql_gen).strip() == "":
        return "Empty SQL", False

    try:
        # Set search path to ensure correct schema usage
        conn.execute(text("SET search_path TO unics_cordis, public;"))
        
        df_truth = pd.read_sql(text(sql_truth), conn)
        df_gen = pd.read_sql(text(sql_gen), conn)

        if df_truth.shape != df_gen.shape:
            return "Shape Mismatch", False

        if df_truth.empty and df_gen.empty:
            return "Both Empty", True

        # Sort values to ensure order doesn't affect comparison
        vals_truth = df_truth.sort_values(by=df_truth.columns.tolist()).values.tolist()
        vals_gen = df_gen.sort_values(by=df_gen.columns.tolist()).values.tolist()

        return ("Match", True) if vals_truth == vals_gen else ("Value Mismatch", False)

    except Exception as e:
        conn.rollback()
        err_msg = str(e).splitlines()[0]
        return f"Error: {err_msg}", False

def check_exact_match(generated_sql: str, ground_truth_sql: str) -> bool:
    """
    Performs a normalized Exact Match (EM) comparison on the SQL strings.
    """
    return clean_sql_markdown(generated_sql) == clean_sql_markdown(ground_truth_sql)

class ExperimentEvaluator:
    """Tracks metrics and logs results for the experiment."""
    
    def __init__(self, db_engine: Engine):
        self.engine = db_engine
        self.logs = []

    def log(self, item: dict, sql_gen: str, latency: float, in_tok: int, out_tok: int, k: int, retriever: str):
        sql_truth = item.get('query', '')
        syntax_ok = check_sql_syntax(sql_gen)

        # Execution Check
        with self.engine.connect() as conn:
            status, is_match = check_execution_match(sql_gen, sql_truth, conn)

        self.logs.append({
            'k': k, 
            'retriever': retriever,
            'db_id': item['db_id'], 
            'question': item['question'],
            'generated_sql': sql_gen, 
            'ground_truth': sql_truth,
            'exact_match': check_exact_match(sql_gen, sql_truth),
            'syntax_valid': syntax_ok, 
            'exec_status': status,
            'exec_match': is_match, 
            'latency': latency,
            'tokens': in_tok + out_tok
        })

    def save(self, log_path: str, metrics_path: str):
        """Saves detailed logs and aggregated metrics."""
        df = pd.DataFrame(self.logs)
        df.to_csv(log_path, index=False)

        # Aggregate metrics by k and retriever strategy
        metrics = df.groupby(['k', 'retriever']).agg(
            exec_match=('exec_match', 'mean'),
            syntax_valid=('syntax_valid', 'mean'),
            exact_match=('exact_match', 'mean'),
            latency=('latency', 'mean')
        ).reset_index()
        
        metrics.to_json(metrics_path, orient='records', indent=4)
        return metrics