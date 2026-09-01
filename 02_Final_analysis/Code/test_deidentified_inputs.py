"""Run with: python -m unittest discover -s Code -p 'test_*.py'."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('validator', Path(__file__).with_name('00_validate_inputs.py'))
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class InputTests(unittest.TestCase):
    def test_supplied_data(self):
        self.assertEqual(validator.validate()['status'], 'passed')

    def rejects(self, target, mutate):
        original = validator.read_csv_numeric
        def reader(path):
            frame = original(path)
            if path == target: mutate(frame)
            return frame
        with patch.object(validator, 'read_csv_numeric', reader):
            with self.assertRaises((ValueError, KeyError)):
                validator.validate()

    def test_reject_duplicate_id(self):
        self.rejects(validator.BEHAVIOR_INPUT, lambda d: d.__setitem__('studnr', [d.studnr.iloc[0]]*len(d)))

    def test_reject_recall_corruption(self):
        self.rejects(validator.PREDEFINED_ROI_INPUT, lambda d: d.__setitem__('vwrec_delta', d.vwrec_delta+1))

    def test_reject_volume_unit_error(self):
        c='aseg+DKT_BrainSegVol_pre'
        self.rejects(validator.BRAINSEGVOL_INPUT, lambda d: d.__setitem__(c,d[c]/1000))

    def test_reject_demographic_linkage_error(self):
        self.rejects(validator.DEMOGRAPHICS_INPUT, lambda d: d.loc.__setitem__((0,'subject_id'),'XTC999'))


if __name__ == '__main__': unittest.main()
