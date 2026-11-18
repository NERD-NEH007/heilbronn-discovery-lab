"""Small emitter and streaming-cover integration tests."""

from io import StringIO
import json
from pathlib import Path
import unittest

from emitter import Emitter
from stream_trace import HERE, verify
from test_cover import fixture
from verify_trace import InvalidTrace


def emit_fixture(records):
    output = StringIO()
    emitter = Emitter(output,buffer_bytes=64)
    emitter.emit(records[0])
    with emitter.support_attempt() as abandoned:
        abandoned.emit({"op":"ABANDONED"})
    with emitter.support_attempt() as accepted:
        for row in records[1:]:
            accepted.emit(row)
        accepted.commit()
    return output.getvalue()


class StreamTests(unittest.TestCase):
    def test_emitter_rollback_and_proof(self):
        records = fixture()
        trace = emit_fixture(records)
        self.assertNotIn("ABANDONED",trace)
        result = verify(StringIO(trace),records[0])
        self.assertEqual(result["cover_closures"],2)
        self.assertEqual(result["root_upper_bound"],"43/1568")

    def test_failed_attempt_does_not_copy(self):
        output = StringIO()
        with self.assertRaises(RuntimeError):
            with Emitter(output).support_attempt() as attempt:
                attempt.emit({"op":"ABANDONED"})
                attempt.commit()
                raise RuntimeError
        self.assertEqual(output.getvalue(),"")

    def test_invalid_cover_rejected_before_pulling_next_record(self):
        records = fixture()
        records[2]["point_assignment"] = [0]*9
        def source():
            yield from map(json.dumps,records[:3])
            raise AssertionError("verifier read beyond an invalid cover")
        with self.assertRaisesRegex(InvalidTrace,"point assignment"):
            verify(source(),records[0])

    def test_partial_frontier_is_delivered_immediately(self):
        records = fixture()
        records[2] = dict(op="FRONTIER_YIELD")
        seen = []
        def source():
            yield from map(json.dumps,records[:3])
            self.assertEqual(len(seen),1)
            yield json.dumps(records[3])
        result = verify(source(),records[0],on_pending=seen.append)
        self.assertEqual(result["pending"],1)
        self.assertIsNone(result["root_upper_bound"])

    def test_truncated_attempt(self):
        records = fixture()
        trace = emit_fixture(records[:-1])
        with self.assertRaisesRegex(InvalidTrace,"incomplete trace"):
            verify(StringIO(trace),records[0])


if __name__=="__main__":
    trace = fixture()
    (HERE/"streaming-fixture.jsonl").write_text(emit_fixture(trace),encoding="utf-8")
    (HERE/"expected-statement.json").write_text(json.dumps(trace[0],indent=2)+"\n",encoding="utf-8")
    unittest.main()
