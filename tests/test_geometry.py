import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core_engine.geometry_utils import map_coordinates

class TestGeometry(unittest.TestCase):
    def test_map_coordinates(self):
        # Test mapping 5 from [0, 10] to [0, 100] -> should be 50
        val = map_coordinates(5, 0, 10, 0, 100)
        self.assertEqual(val, 50)

        # Test clamping/extrapolation (numpy interp clamps by default? No, it clamps)
        # np.interp(11, [0, 10], [0, 100]) -> 100
        val = map_coordinates(11, 0, 10, 0, 100)
        self.assertEqual(val, 100)

if __name__ == '__main__':
    unittest.main()
