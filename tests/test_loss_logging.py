import csv
import tensorflow as tf
import modules.loss_logging as loss_logging


def test_add():
    logger = loss_logging.LossLogger("path")
    assert(logger._step_counter == 0)
    assert(logger._step_loss == 0) 
    logger.add(1.2)
    logger.add(2.3)
    assert(logger._step_counter == 2)
    assert(logger._step_loss == 3.5)

def test_log_average_loss():
    logger = loss_logging.LossLogger("path")
    logger.add(tf.ones([1]) * 2)
    logger.add(tf.ones([1]) * 4)
    assert(logger._step_loss == 6)
    assert(logger._step_counter == 2)
    assert(len(logger._average_losses) == 0)
    logger.log_average_loss()
    assert(logger._step_loss == 0)
    assert(logger._step_counter == 0)
    assert(len(logger._average_losses) == 1)
    assert(logger._average_losses[0] == 3)
