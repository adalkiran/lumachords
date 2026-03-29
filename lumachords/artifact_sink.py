"""
Artifact sink system for managing debug and production visualization artifacts.
"""

from dataclasses import dataclass
import asyncio
from typing import Callable, Optional
import numpy as np

from lumachords.image_utils import ImageUtils
from lumachords.runtime_config import AppMode, LogLevel, ProdMode, RuntimeConfig

@dataclass
class ArtifactConfigEntry:
    prod_mode: ProdMode
    min_log_level: LogLevel
    emit_fn: Optional[Callable[[any], None]] = None
    filename: Optional[str] = None
    emit_kwargs: Optional[dict] = None
    enabled: bool = True

    @classmethod
    def with_kwargs(
        cls,
        *,
        prod_mode: ProdMode,
        min_log_level: LogLevel,
        filename: Optional[str] = None,
        emit_fn: Optional[Callable[[any], None]] = None,
        **emit_kwargs: any,
    ) -> "ArtifactConfigEntry":
        return cls(prod_mode=prod_mode, min_log_level=min_log_level, filename=filename, emit_fn=emit_fn, emit_kwargs=emit_kwargs)
    
class ArtifactSink:
    def __init__(self, config: dict[str, ArtifactConfigEntry], runtime_config: RuntimeConfig):
        self.config = config
        self.runtime_config = runtime_config
        self._emit_lock = asyncio.Lock()
    
    def merge_emit_kwargs(self, key: str, config_entry: ArtifactConfigEntry, caption: str, filename: str, kwargs: dict):
        caption = f"{key}: {caption}" if caption else key
        merged_kwargs = {**config_entry.emit_kwargs} if config_entry.emit_kwargs else {}
        if kwargs:
            merged_kwargs.update(kwargs)
        merged_kwargs["caption"] = caption
        if filename:
            merged_kwargs["filename"] = filename
        return merged_kwargs

    def wants(self, key:str):
        config_entry = self.config.get(key, None)
        if config_entry is None:
            raise Exception(f"Artifact key '{key}' couldn't be found in artifact config.")
        if not config_entry.enabled:
            return False
        if self.runtime_config.prod_mode < config_entry.prod_mode:
            return False
        if self.runtime_config.app_mode < AppMode.NOTEBOOK and config_entry.emit_fn is None:
            return False
        if config_entry.min_log_level is not None and self.runtime_config.log_level < config_entry.min_log_level:
            return False
        return True
    
    def emit(self, key:str, data: any, caption: Optional[str] = None, **kwargs):
        if not self.wants(key):
            return
        config_entry = self.config.get(key, None)
        if config_entry.emit_fn is not None:
            config_entry.emit_fn(data)
            if self.runtime_config.app_mode < AppMode.NOTEBOOK:
                return
        merged_kwargs = self.merge_emit_kwargs(key, config_entry, caption, config_entry.filename, kwargs)
        if isinstance(data, np.ndarray):
            ImageUtils.imshow(data, **merged_kwargs)
        elif isinstance(data, str):
            caption = f"{key}: {caption}" if caption else key
            print(caption + ":")
            print(data)
        elif config_entry.emit_fn is None:
            if data is None:
                caption = f"{key}: {caption}" if caption else key
                print(caption + ": None")
            else:
                raise Exception(f"Unsupported artifact type: '{type(data)}'")

    def emit_lazy(self, key:str, make_value: Callable[[], any], return_value: bool=False, **kwargs):
        if not self.wants(key):
            return
        data = make_value()
        self.emit(key, data, **kwargs)
        if return_value:
            return data

    async def emit_lazy_async(self, key:str, make_value: Callable[[], any], return_value: bool=False, **kwargs):
        if not self.wants(key):
            return None
        data = await asyncio.to_thread(make_value)
        async with self._emit_lock:
            self.emit(key, data, **kwargs)
        if return_value:
            return data
