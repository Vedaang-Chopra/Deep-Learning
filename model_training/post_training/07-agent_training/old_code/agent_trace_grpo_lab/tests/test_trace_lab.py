import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trace_lab import normalize_events, audit_events, group_advantages

class TraceLabTests(unittest.TestCase):
    def test_normalization_pair(self):
        trace = {'messages': [
            {'role':'assistant', 'tool_calls':[{'id':'c1','type':'function', 'function':{'name':'execute_bash','arguments':'{"command":"pytest -q"}'}}]},
            {'role':'tool','tool_call_id':'c1','content':'1 failed'},
            {'role':'assistant','content':'Need to inspect the failing test.'},
        ], 'resolved': True, 'test_result': {'report': {'resolved': True}}}
        events = normalize_events(trace)
        self.assertEqual([e['kind'] for e in events], ['tool_call','tool_result','message'])
        self.assertEqual(events[0]['arguments'], {'command':'pytest -q'})
        self.assertEqual(events[1]['call_id'], 'c1')
        self.assertNotIn('resolved', str(events))

    def test_orphan_result(self):
        result = audit_events([{'index':0,'kind':'tool_result','call_id':'missing','name':None,'arguments':None,'role':'tool','text':'ok'}])
        self.assertIn('orphan_tool_result', result['flags'])
        self.assertEqual(result['first_error_index'], 0)

    def test_grpo_groups(self):
        out = group_advantages([[0,1,0,1],[0,0,0,0]])
        for a,b in zip(out[0],[-1,1,-1,1]): self.assertAlmostEqual(a,b)
        self.assertEqual(out[1], [0,0,0,0])

if __name__ == '__main__': unittest.main()
