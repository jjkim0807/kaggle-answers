from pathlib import Path
import unittest

import wandb

from main import load_data


class Test_load_data(unittest.TestCase):
    def test_test_one_example(self):
        wandb.init()
        wandb.config.update({
            "data_path": "tests/test_data/1",
            "is_train": False,
            "is_test": True,
        }, allow_val_change=True)

        train_dl, dev_dl, test_dl = load_data()

        self.assertIsNone(train_dl)
        self.assertIsNone(dev_dl)

        test_data = next(iter(test_dl))
        self.assertEqual(test_data, {
            "label": ["892", ],
            "data": [("3",), ("Kelly, Mr. James",), ("male",), ("34.5",), ("0",), ("0",), ("330911",), ("7.8292",), ("",), ("Q",)]
        })


if __name__ == '__main__':
    unittest.main()
