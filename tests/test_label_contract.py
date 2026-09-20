"""Label-contract regressions for the standard-library unittest CI runner."""
import json
import unittest

from conversation_target_tagger.tagging import parse_and_validate


class LabelContractTests(unittest.TestCase):
    def _assert_rejected(self, index, label):
        response = json.dumps([{"gidx": index, "targets": [label]}])
        with self.assertRaises(ValueError):
            parse_and_validate(response, (1,))

    def test_boolean_index_is_rejected(self):
        self._assert_rejected(True, "self")

    def test_float_index_is_rejected(self):
        self._assert_rejected(1.0, "self")

    def test_project_label_requires_a_name(self):
        self._assert_rejected(1, "project:")

    def test_external_person_label_requires_a_name(self):
        self._assert_rejected(1, "person_external:   ")

    def test_self_label_rejects_a_suffix(self):
        self._assert_rejected(1, "self:someone")

    def test_ambiguous_label_rejects_a_suffix(self):
        self._assert_rejected(1, "ambiguous:maybe")


if __name__ == "__main__":
    unittest.main()
