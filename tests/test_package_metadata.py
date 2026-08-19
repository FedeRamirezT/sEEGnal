"""Tests for package-level public metadata."""

import unittest

import sEEGnal


class TestPackageMetadata(unittest.TestCase):
    """Check the deliberately small package-level public API."""

    def test_version_is_exposed(self):
        self.assertIsInstance(sEEGnal.__version__, str)
        self.assertTrue(sEEGnal.__version__)

    def test_version_and_pipeline_runner_are_exported(self):
        self.assertEqual(
            sEEGnal.__all__,
            ["__version__", "run_sEEGnal"],
        )
        self.assertTrue(callable(sEEGnal.run_sEEGnal))


if __name__ == "__main__":
    unittest.main()
