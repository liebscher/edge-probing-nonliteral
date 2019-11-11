import pytest as pt

# import edge probing model.py


class TestPooler:
    """
    Test the Pooler class
    """
    def test_pooling_io_dims(self):
        """
        Test that the proper dimensions are going into and coming out of Pooler.
        """
        assert 1

    def test_pooling_type(self):
        """
        Ensure that each pooling type is doing what we'd expect it to do.
        """
        assert 1


class TestClassifier:
    """
    Test the Classifier class
    """
    def test_classifier_io_dims(self):
        """
        Test that the proper dimensions are going into and coming out of Classifier.
        """
        assert 1

class TestModel:
    """
    Test the Model building and execution
    """

    def test_training_loss_movement(self):
        """
        Test that the model loss changes with a single step during training
        """

        assert 1

    def test_loss_at_zero(self):
        """
        Make sure the loss is never exactly zero
        """

        assert 1