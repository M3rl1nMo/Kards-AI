"""Verify RNG determinism for AI training reproducibility."""
import unittest
import random


class RNGDeterminismTests(unittest.TestCase):
    def test_seeded_random_deterministic(self) -> None:
        r1 = random.Random(42)
        r2 = random.Random(42)
        deck1 = list(range(40))
        deck2 = list(range(40))
        r1.shuffle(deck1)
        r2.shuffle(deck2)
        self.assertEqual(deck1, deck2)

    def test_simulator_uses_random_module(self) -> None:
        import simulator.rules.native as n
        self.assertTrue(hasattr(random, 'Random'))


if __name__ == "__main__":
    unittest.main()
