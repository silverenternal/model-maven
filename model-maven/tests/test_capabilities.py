import unittest

from model_maven_worker.runtime.capabilities import doctor_report


class DoctorCapabilitiesTests(unittest.TestCase):
    def test_reports_checkpointed_multi_batch_supervision_and_remaining_limits(self):
        report = doctor_report(
            python="python3",
            python_version="3.14",
            torch_available=True,
            torch_version="2.15",
            platform="linux",
            compatibility={},
        )
        implemented = report["implemented"]
        not_implemented = report["not_implemented"]
        self.assertTrue(
            any("cross-batch sample and violation ID deduplication" in item for item in implemented)
        )
        self.assertTrue(any("opt-in external-loop supervision" in item for item in implemented))
        self.assertTrue(any("supervision validator" in item and "differentiable-penalty" in item for item in implemented))
        self.assertTrue(any("sample ID authenticity" in item and "task-specific label semantics" in item for item in not_implemented))
        self.assertFalse(any("custom supervision plugin protocol" in item for item in not_implemented))
        self.assertFalse(
            any("cross-batch unique-sample coverage" in item for item in not_implemented)
        )


if __name__ == "__main__":
    unittest.main()
