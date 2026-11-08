import copy
import json
import tempfile
import unittest
from pathlib import Path

from interfaces import SWEGymTraceAdapter, normalize_trace, write_sft_dataset


class SFTDataTests(unittest.TestCase):
    def test_normalize_and_preserve_source(self):
        source = {'messages': [
            {'role': 'user', 'content': 'hello', 'reasoning': 'private'},
            {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': 'call-1', 'function': {'name': 'run', 'arguments': '{"command":"bad"}'}}]},
            {'role': 'tool', 'name': 'run', 'tool_call_id': 'call-1', 'content': 'error'},
            {'role': 'assistant', 'content': 'retry'}], 'tools': []}
        original = copy.deepcopy(source)
        result, issues = normalize_trace(SWEGymTraceAdapter().parse(source))
        self.assertEqual(source, original)
        self.assertEqual(issues, [])
        self.assertEqual(result['messages'][1]['tool_calls'][0]['function']['arguments'], {'command': 'bad'})
        self.assertEqual(result['messages'][2]['tool_call_id'], 'call-1')
        self.assertNotIn('reasoning', json.dumps(result))

    def test_invalid_calls_remain_and_are_counted(self):
        row = {'messages': [{'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'c', 'function': {'name': 'run', 'arguments': '{bad'}}]}]}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary = write_sft_dataset([row], SWEGymTraceAdapter(), root / 'out.jsonl', root / 'summary.json', root / 'issues.jsonl')
            self.assertEqual(summary['invalid_records'], 1)
            self.assertEqual(summary['unmatched_tool_calls'], 1)
            self.assertEqual(json.loads((root / 'out.jsonl').read_text())['messages'][0]['tool_calls'][0]['function']['arguments'], '{bad')

    def test_redaction(self):
        row = {'messages': [{'role': 'assistant', 'content': 'email user@example.com; password=abc123'}]}
        result, issues = normalize_trace(SWEGymTraceAdapter().parse(row))
        self.assertEqual(issues, [])
        self.assertNotIn('user@example.com', str(result))
        self.assertNotIn('abc123', str(result))


if __name__ == '__main__':
    unittest.main()
