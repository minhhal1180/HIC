import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core_engine.signal_filters import ExponentialMovingAverage

class TestFilters(unittest.TestCase):
    def test_ema(self):
        ema = ExponentialMovingAverage(alpha=0.5)
        
        # First value should be the value itself
        val1 = ema.update(10)
        self.assertEqual(val1, 10)
        
        # Second value: 0.5 * 20 + 0.5 * 10 = 15
        val2 = ema.update(20)
        self.assertEqual(val2, 15)

if __name__ == '__main__':
    unittest.main()
