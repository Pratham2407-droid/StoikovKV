from typing import Dict, Optional
from stoikov_kv.cache import StoikovKVCache
import uuid

class CacheManager:
    def __init__(self, cache_factory):
        self.cache_factory = cache_factory
        self.caches: Dict[str, StoikovKVCache] = {}
    
    def get_or_create(self, session_id: Optional[str] = None) -> tuple[str, StoikovKVCache]:
        if session_id is None:
            session_id = str(uuid.uuid4())[:8]
        if session_id not in self.caches:
            self.caches[session_id] = self.cache_factory(session_id)
        return session_id, self.caches[session_id]
    
    def crop_session(self, session_id: str, max_length: int):
        if session_id in self.caches:
            self.caches[session_id].crop(max_length)
    
    def delete_session(self, session_id: str):
        if session_id in self.caches:
            del self.caches[session_id]
    
    def get_telemetry(self, session_id: str) -> dict:
        if session_id in self.caches:
            return self.caches[session_id].get_telemetry_summary()
        return {}

# Create a singleton instance
cache_manager = CacheManager(cache_factory=lambda session_id: None)