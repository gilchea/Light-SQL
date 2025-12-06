import os
import logging
import torch
import sys
import warnings
from tqdm import tqdm
import time
import pandas as pd

# Suppress unnecessary warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Reduce logging noise from libraries
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("sqlfluff").setLevel(logging.ERROR)

from src.utils import set_seed, load_json
from src.database.manager import DatabaseManager
from src.datasets.schema import create_schema_dict
from src.models.llm import LLMEngine
from src.retriever.strategies import RandomRetriever, AdvancedRetriever
from src.prompts.builder import PromptBuilder
from src.metrics.evaluator import ExperimentEvaluator

# --- CONFIGURATION ---
CONFIG = {
    "seed": 42,
    "model_id": "microsoft/Phi-3-mini-4k-instruct",
    "retriever_model": "BAAI/bge-small-en-v1.5",
    "data_dir": "/content/drive/MyDrive/nlp/nl2sql_project/data/cordis", # Adjust path as needed
    "sql_source": "/content/drive/MyDrive/nlp/nl2sql_project/data/cordis.sql",
    
    # Define which strategies to run: "random", "qts" (question), "mqs" (masked), "qrs" (query/SQL)
    "strategies": ["qrs"], 
    
    # Test with different numbers of few-shot examples
    "k_values": [1, 5],
    "sample_limit": 100, # Number of samples to evaluate (for speed)
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def main():
    set_seed(CONFIG["seed"])

    # 1. Setup Database
    print("⏳ Initializing Database and Model... (Please wait)")
    db_manager = DatabaseManager()
    db_manager.setup_database()
    try:
        db_manager.restore_data(CONFIG["sql_source"])
    except Exception:
        print("⚠️ Warning: Data restoration might have failed. Check logs.")

    # 2. Load Data & Models
    tables = load_json(os.path.join(CONFIG["data_dir"], "tables.json"))
    synth_data = load_json(os.path.join(CONFIG["data_dir"], "synth.json"))
    
    # Load dev data and slice it for quicker iteration during dev
    dev_data = load_json(os.path.join(CONFIG["data_dir"], "dev.json"))[:CONFIG["sample_limit"]]
    
    schema_map = create_schema_dict(tables)
    llm = LLMEngine(CONFIG["model_id"], CONFIG["device"])
    prompt_builder = PromptBuilder()
    evaluator = ExperimentEvaluator(db_manager.get_engine())

    # 3. Build Retrievers
    retrievers = {}
    print(f"✅ Models loaded. Building indices for {len(CONFIG['strategies'])} strategies...")

    if "random" in CONFIG["strategies"]:
        r = RandomRetriever()
        r.build_index(synth_data)
        retrievers["random"] = r

    if "qts" in CONFIG["strategies"]:
        r = AdvancedRetriever(CONFIG["retriever_model"], mode='qts', device=CONFIG["device"])
        r.build_index(synth_data)
        retrievers["qts"] = r

    if "mqs" in CONFIG["strategies"]:
        r = AdvancedRetriever(CONFIG["retriever_model"], mode='mqs', device=CONFIG["device"])
        r.build_index(synth_data)
        retrievers["mqs"] = r

    if "qrs" in CONFIG["strategies"]:
        r = AdvancedRetriever(CONFIG["retriever_model"], mode='qrs', device=CONFIG["device"])
        r.build_index(synth_data)
        retrievers["qrs"] = r

    # 4. Experiment Loop
    print("\n🚀 STARTING EXPERIMENT")
    print("="*50)

    for strategy_name in CONFIG["strategies"]:
        retriever = retrievers.get(strategy_name)

        for k in CONFIG["k_values"]:
            desc = f"👉 Strategy: {strategy_name.upper()} | k={k}"

            # Iterate through samples
            for item in tqdm(dev_data, desc=desc, ncols=100, unit="sample"):
                schema = schema_map.get(item['db_id'])
                if not schema: continue

                start_total = time.time()
                icl_ex = []

                # --- Retrieval Logic ---
                if k > 0 and retriever:
                    if strategy_name == "qrs":
                        # For Query-to-SQL: Generate a zero-shot draft first
                        zero_shot_prompt = prompt_builder.build(schema, item['question'], [])
                        draft_sql, _, _ = llm.generate(zero_shot_prompt)
                        # Retrieve examples similar to the drafted SQL
                        icl_ex = retriever.retrieve(draft_sql, k)
                    else:
                        # Standard retrieval
                        icl_ex = retriever.retrieve(item['question'], k)

                # --- Generation ---
                prompt = prompt_builder.build(schema, item['question'], icl_ex)
                sql, in_tok, out_tok = llm.generate(prompt)
                latency = time.time() - start_total

                # --- Evaluation ---
                evaluator.log(item, sql, latency, in_tok, out_tok, k, strategy_name)

    # 5. Save & Print Results
    metrics = evaluator.save("results/logs.csv", "results/metrics.json")

    print("\n" + "="*50)
    print("📊 EXPERIMENT RESULTS SUMMARY")
    print("="*50)
    
    # Configure pandas display for better readability
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', '{:.2f}'.format)

    print(metrics[['retriever', 'k', 'exec_match', 'exact_match', 'syntax_valid' ,'latency']])
    print("="*50)
    print(f"✅ Detailed logs saved at: results/logs.csv")

if __name__ == "__main__":
    main()