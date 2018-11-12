import os
import sys
import platform

import numpy as np
import tensorflow as tf

import getpass

from config import get_config, save_config

config = None


def main(_):

    # Create a random state using the random seed given by the config. This
    # should allow reproducible results.
    rng = np.random.RandomState(config.random_seed)
    tf.set_random_seed(config.random_seed)

    # Train / Test
    if config.task == "train":
        # Import trainer module
        from trainer import Trainer

        # Create a trainer object
        task = Trainer(config, rng)
        save_config(config.logdir, config)

    else:
        # Import tester module
        from tester import Tester

        # Create a tester object
        task = Tester(config, rng)

    # Run the task
    task.run()


if __name__ == "__main__":
    config, unparsed = get_config(sys.argv)

    if len(unparsed) > 0:
        raise RuntimeError("Unknown arguments were given! Check the command line!")
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
