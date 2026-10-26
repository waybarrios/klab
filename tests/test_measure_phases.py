import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class MeasurePhaseTests(unittest.TestCase):
    def test_measurement_phase_brackets_only_randomized_timed_replays(self):
        import bench
        replays = []
        phases = []
        rounds_seen = []

        class Stream:
            def wait_stream(self, _):
                pass

            def synchronize(self):
                pass

        class Graph:
            def replay(self):
                replays.append(1)

            def reset(self):
                pass

        class Event:
            def record(self):
                pass

            def synchronize(self):
                pass

            def elapsed_time(self, _):
                return .001

        cuda = SimpleNamespace(Stream=Stream, current_stream=Stream,
                               stream=lambda _: contextlib.nullcontext(),
                               CUDAGraph=Graph, graph=lambda *args, **kw: contextlib.nullcontext(),
                               Event=lambda **kw: Event(), synchronize=lambda: None)
        with patch.object(bench, "torch", SimpleNamespace(cuda=cuda)):
            result = bench.measure({"vendor": lambda: None, "custom": lambda: None}, rounds=31, repeats=1,
                                   phase_callback=lambda phase: phases.append((phase, len(replays))),
                                   round_callback=lambda index: rounds_seen.append((index, len(replays))))
        self.assertEqual(phases, [("capture", 0), ("warmup", 0), ("measurement", 4), ("finished", 66)])
        self.assertEqual(len(result["vendor"]["samples_us"]), 31)
        self.assertEqual(rounds_seen, [(index, 6 + 2 * index) for index in range(31)])


if __name__ == "__main__":
    unittest.main()
