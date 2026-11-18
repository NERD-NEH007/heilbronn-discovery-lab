"""Buffered JSONL writer with transactional support-attempt output."""

from contextlib import contextmanager
import json
from shutil import copyfileobj
from tempfile import SpooledTemporaryFile


class Attempt:
    def __init__(self, spool):
        self.spool = spool
        self.accepted = False

    def emit(self, record):
        self.spool.write(json.dumps(record,separators=(",",":"))+"\n")

    def commit(self):
        self.accepted = True


class Emitter:
    def __init__(self, stream, buffer_bytes=1<<20):
        self.stream = stream
        self.buffer_bytes = buffer_bytes

    def emit(self, record):
        self.stream.write(json.dumps(record,separators=(",",":"))+"\n")

    @contextmanager
    def support_attempt(self):
        with SpooledTemporaryFile(max_size=self.buffer_bytes,mode="w+t",encoding="utf-8",
                                  newline="") as spool:
            attempt = Attempt(spool)
            yield attempt
            if attempt.accepted:
                spool.seek(0)
                copyfileobj(spool,self.stream,length=65536)
