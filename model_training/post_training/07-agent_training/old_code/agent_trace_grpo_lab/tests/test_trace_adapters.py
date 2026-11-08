"""Contract checks for trace adapters and SWE-Gym inspection."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from interfaces import AgentTrace, SWEGymTraceAdapter, TraceMessage, parse_instance_id, parse_run_id


class TraceAdapterTests(unittest.TestCase):
    def test_swe_gym_preserves_order_calls_and_private_labels(self):
        record = {
            "instance_id": "getmoto__moto-5321",
            "run_id": "claude-3-5-sonnet-20241022_maxiter_50_N_v2.1-no-hint-run_1",
            "resolved": False,
            "test_result": {"report": {"resolved": False}},
            "messages": [
                {"role": "assistant", "tool_calls": [
                    {"id": "a", "function": {"name": "bash", "arguments": '{"command":"ls"}'}},
                    {"id": "b", "function": {"name": "edit", "arguments": "invalid"}},
                ]},
                {"role": "tool", "tool_call_id": "a", "content": "ok"},
            ],
        }
        trace = SWEGymTraceAdapter().parse(record)
        self.assertEqual([m.index for m in trace.messages], [0, 1])
        self.assertEqual(trace.messages[0].tool_calls[0].arguments, {"command": "ls"})
        self.assertEqual(trace.messages[0].tool_calls[1].arguments, "invalid")
        self.assertTrue(trace.messages[0].tool_calls[1].arguments_error)
        self.assertEqual(trace.messages[1].tool_call_id, "a")
        self.assertFalse(trace.labels["resolved"])
        self.assertNotIn("resolved", vars(trace.messages[0]))
        self.assertEqual(trace.metadata["instance"]["issue_number"], 5321)
        self.assertEqual(trace.metadata["run"]["max_iterations"], 50)

    def test_unknown_identifiers_and_empty_messages(self):
        self.assertIsNone(parse_run_id("unknown")["max_iterations"])
        self.assertIsNone(parse_instance_id("unknown")["issue_number"])
        self.assertEqual(SWEGymTraceAdapter().parse({}).messages, [])

    def test_another_format_can_use_shared_types(self):
        class OtherAdapter:
            def parse(self, record):
                return AgentTrace(messages=[TraceMessage(0, "user", record["prompt"], None)])

        trace = OtherAdapter().parse({"prompt": "hello"})
        self.assertEqual(trace.messages[0].content, "hello")
        self.assertEqual(trace.labels, {})


if __name__ == "__main__":
    unittest.main()
