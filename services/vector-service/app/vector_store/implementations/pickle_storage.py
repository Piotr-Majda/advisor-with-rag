import os
import pickle
from datetime import datetime
from ..protocols import StorageService


class PickleStorageService(StorageService):
    def __init__(self, store_path: str, backup_dir: str):
        self.store_path = store_path
        self.backup_dir = backup_dir
        os.makedirs(os.path.dirname(store_path), exist_ok=True)
        os.makedirs(backup_dir, exist_ok=True)

    def save(self, path: str, data: any) -> None:
        """Save data to a pickle file"""
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str) -> any:
        """Load data from a pickle file"""
        with open(path, "rb") as f:
            return pickle.load(f)

    def create_backup(self, source: str) -> str:
        """Create a backup of the source file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(self.backup_dir, f"backup_{timestamp}.pkl")
        if os.path.exists(source):
            with open(source, "rb") as src, open(backup_path, "wb") as dst:
                dst.write(src.read())
        return backup_path

    def delete(self, path: str) -> None:
        """Delete a file"""
        if os.path.exists(path):
            os.remove(path)

    def exists(self, path: str) -> bool:
        """Check if a file exists"""
        return os.path.exists(path)

    def get_store_path(self) -> str:
        """Get the path to the store file"""
        return self.store_path

    def get_backup_path(self) -> str:
        """Get the path to the backup directory"""
        return self.backup_dir
