import os
import pickle
import shutil
from datetime import datetime
from app.vector_store.protocols import StorageService


class FileSystemStorage(StorageService):
    def __init__(self, store_path: str, backup_path: str):
        """Initialize the storage service.

        Args:
            store_path: Path to the store file (not directory)
            backup_path: Path to the backup directory
        """
        self.store_path = store_path
        self.backup_path = backup_path

        # Ensure backup directory exists
        os.makedirs(self.backup_path, exist_ok=True)

    def save(self, path: str, data: any) -> None:
        """Save data to a file, creating parent directories if needed."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str) -> any:
        """Load data from a file."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Store file not found at {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    def create_backup(self, source: str) -> str:
        """Create a backup of the store file.

        Args:
            source: Path to the source file to backup

        Returns:
            str: Path to the backup file (not directory)
        """
        if not os.path.isfile(source):
            raise FileNotFoundError(f"Source file not found at {source}")

        # Create backup filename with timestamp
        backup_name = f"store_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        backup_file = os.path.join(self.backup_path, backup_name)

        # Ensure backup directory exists
        os.makedirs(self.backup_path, exist_ok=True)

        # Copy the file
        shutil.copy2(source, backup_file)
        return backup_file

    def get_store_path(self) -> str:
        return self.store_path

    def get_backup_path(self) -> str:
        return self.backup_path

    def exists(self, path: str) -> bool:
        """Check if a file exists at the given path."""
        return os.path.isfile(path)
    
    def delete(self, path: str) -> None:
        """Delete a file at the given path."""
        if os.path.isfile(path):
            os.remove(path)
