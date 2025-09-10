import csv
import os
from datetime import datetime

class CSVLogger:
    def __init__(self, log_dir: str, filename: str, fieldnames: list):
        """
        Simple CSV logger for training & validation metrics.
        Automatically creates the logs folder if it doesn't exist.
        """
        self.log_dir = log_dir
        self.filename = filename
        self.fieldnames = fieldnames

        os.makedirs(log_dir, exist_ok=True)
        self.file_path = os.path.join(log_dir, filename)

        # Create the CSV file if it doesn't exist
        if not os.path.exists(self.file_path):
            with open(self.file_path, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row: dict):
        """Append one row to the CSV file."""
        with open(self.file_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)

    def log_epoch(self, epoch: int, logs: dict):
        """
        Convenience function for training:
        Adds an epoch and timestamp automatically.
        """
        row = {"epoch": epoch, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        row.update(logs)
        self.log(row)