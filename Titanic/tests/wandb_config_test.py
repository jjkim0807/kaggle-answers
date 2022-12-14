from pathlib import Path
import unittest

import wandb

from main import load_data


class Test_wandb_config(unittest.TestCase):
    def test_config_defaults(self):
        wandb.init()
        self.assertTrue(wandb.config.is_train)
        self.assertTrue(wandb.config.is_test)
        self.assertEqual(wandb.config.data_path, "data")
        self.assertEqual(wandb.config.model_path, "models")
        self.assertEqual(wandb.config.train_ratio, 0.7)
        self.assertEqual(wandb.config.dev_ratio, 0.3)


if __name__ == '__main__':
    unittest.main()
