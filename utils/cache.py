import pathlib
import datetime

import torch


DATE = datetime.datetime(2024, 3, 7, tzinfo=datetime.timezone.utc)

class Cache:
    def __init__(self, verbose=True):
        self.cache = {}
        self.verbose = verbose

    def _print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _file_recency(self, path):
        """Check if file at path is from this year."""
        fname = pathlib.Path(path)
        try:
            mtime = datetime.datetime.fromtimestamp(
                fname.stat().st_mtime, tz=datetime.timezone.utc
            )
        except FileNotFoundError:
            self._print(f"Didn't find {path}")
            return None
        return mtime

    def _check(self, path, check_day=True, check_year=True):
        recency = self._file_recency(path)
        if recency is None:
            self._print(f"Count not find {path}.")
            return False

        if check_day and recency < DATE:
            self._print(
                f"Found {path} but file is stale: {recency} vs {DATE}."
            )
            return False
        if check_year and recency.year < DATE.year:
            self._print(f"Found {path} but file is stale: {recency.year} vs {DATE.year}.")
            return False
        return True

    def check(self, path):
        return self._check(path) or self._check(f"{path}-000")

    def _load_file(self, path, check_recency=True):
        if not self._check(path, check_day=False, check_year=check_recency):
            return None

        try:
            r = torch.load(path, map_location="cpu")
            self._print(f"Successfully loaded {path} from disk.")
        except:
            self._print(f"Found {path} but failed to load from disk.")
            return None

        return r

    def load(self, path, max_length=None):
        # Check if in cache
        if path in self.cache:
            self._print(f"Cache hit on {path}.")
            return self.cache[path]

        # Try to load all at once
        self._print(f"Cache miss on {path}, loading from disk...")
        ds = self._load_file(path, check_recency=True)
        if ds:
            self.cache[path] = ds
            return ds

        # Try to load pieces
        idx = 0
        length = 0
        ds = []
        while True:
            part_idx = str(idx).zfill(3)
            part_path = f"{path}-{part_idx}"
            ds_part = self._load_file(part_path)
            if ds_part is None:
                break

            ds.extend(ds_part)
            idx += 1
            length += len(ds_part)
            if max_length and length >= max_length:
                break

        if not ds:
            self._print(f"{path} not found on disk.")
            return None

        self._print(f"Successfully loaded {path} parts from disk.")
        self.cache[path] = ds
        return ds

    def insert(self, path, ds):
        if path in self.cache:
            raise ValueError(f"{path} already in cache!")
        self.cache[path] = ds

    def evict(self, path):
        del self.cache[path]


CACHE = Cache()
