import sqlite3
import os
from datetime import datetime, timedelta
import socket

# Determine database path (support persistent storage on Hugging Face)
DATA_DIR = "/data"
if os.path.exists(DATA_DIR) and os.access(DATA_DIR, os.W_OK):
    DB_PATH = os.path.join(DATA_DIR, "users.db")
else:
    DB_PATH = "users.db"

def init_db():
    """Initialize the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # User table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT NOT NULL,
            signup_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Trial info table (singleton for local use)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trial_info (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            start_date TIMESTAMP NOT NULL
        )
    ''')
    
    # Insert trial start date if first run
    cursor.execute("SELECT COUNT(*) FROM trial_info")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO trial_info (id, start_date) VALUES (1, ?)", (datetime.now(),))
    
    conn.commit()
    conn.close()

def get_trial_status():
    """Returns (is_active, days_remaining, start_date)"""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT start_date FROM trial_info WHERE id = 1")
    start_date_str = cursor.fetchone()[0]
    start_date = datetime.fromisoformat(start_date_str.split('.')[0]) # Simplify format if needed
    
    conn.close()
    
    expiry_date = start_date + timedelta(days=7)
    now = datetime.now()
    
    days_remaining = (expiry_date - now).days + 1
    is_active = now <= expiry_date
    
    return is_active, max(0, days_remaining), start_date

def is_user_registered():
    """Checks if any user is registered in this database."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]
    
    conn.close()
    return count > 0

def register_user(username, email):
    """Registers a new user."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("INSERT INTO users (username, email) VALUES (?, ?)", (username, email))
    
    conn.commit()
    conn.close()
    return True

def get_user_count():
    """Total number of registered users."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]
    conn.close()
    return count

if __name__ == "__main__":
    # Test
    init_db()
    active, rem, start = get_trial_status()
    print(f"Trial Active: {active}, Days Remaining: {rem}, Started: {start}")
    print(f"Registered: {is_user_registered()}")
    print(f"User Count: {get_user_count()}")
