import os
import shutil
import subprocess
from sqlalchemy import create_engine, Engine
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages PostgreSQL setup, data restoration, and connection creation."""
    
    def __init__(self, db_name="cordis_temporary", user="postgres", password="password"):
        self.db_name = db_name
        self.user = user
        self.password = password
        self.db_url = f'postgresql+psycopg2://{user}:{password}@localhost:5432/{db_name}'

    def setup_database(self):
        """Configures the postgres user and recreates the target database."""
        logger.info("Setting up PostgreSQL database...")
        subprocess.run(f"sudo -u postgres psql -c \"ALTER USER postgres PASSWORD '{self.password}';\"", shell=True)
        
        # Drop if exists to ensure a clean state
        subprocess.run(f"sudo -u postgres psql -c \"DROP DATABASE IF EXISTS {self.db_name};\"", shell=True)
        subprocess.run(f"sudo -u postgres psql -c \"CREATE DATABASE {self.db_name};\"", shell=True)

    def restore_data(self, source_folder: str, local_dest: str = '/content/cordis_data'):
        """
        Copies data from source, fixes paths in the SQL dump, and restores to DB.
        """
        logger.info(f"Restoring data from {source_folder}...")

        if os.path.exists(local_dest): 
            shutil.rmtree(local_dest)
        shutil.copytree(source_folder, local_dest)
        os.system(f"chmod -R 777 {local_dest}")

        sql_file = os.path.join(local_dest, "restore.sql")
        fixed_sql = os.path.join(local_dest, "restore_fixed.sql")

        # Read and fix paths in the SQL dump
        with open(sql_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        content = content.replace("$$PATH$$", local_dest)

        with open(fixed_sql, 'w', encoding='utf-8') as f:
            f.write(content)

        # Execute restore
        cmd = f"sudo -u postgres psql -d {self.db_name} -f '{fixed_sql}'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"Restore failed: {result.stderr}")
            raise Exception("Database Restore Failed")
        
        logger.info("Database restored successfully.")

    def get_engine(self) -> Engine:
        """Returns a SQLAlchemy engine with search_path configured."""
        return create_engine(self.db_url + "?options=-c search_path=unics_cordis,public -c statement_timeout=5000")