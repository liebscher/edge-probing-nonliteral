import pytest as pt

import torch as tt

from epnl import model


class TestPooler:
    """
    Test the Pooler class
    """
    def test_pooling_params(self):
        pooler = model.Pooler(project=True, inp_dim=512, out_dim=512, pooling_type="mean")

        assert len(list(pooler.parameters())) == 2, "Projection not creating proper learnable parameters"

        pooler = model.Pooler(project=False, inp_dim=512, out_dim=512, pooling_type="mean")

        assert len(list(pooler.parameters())) == 0, "No projection is wrongly adding layers"


    def test_pooling_io_dims(self):
        """
        Test that the proper dimensions are going into and coming out of Pooler.
        """
        pooler = model.Pooler(project=True, inp_dim=512, out_dim=256, pooling_type="mean")

        sizes = [tt.Size([256, 512]), tt.Size([256])]

        for i, param in enumerate(pooler.parameters()):
            assert param.size() == sizes[i], "Incorrect size of learnable parameters"

        inp_test_a = tt.randn(512)
        inp_test_b = tt.randint(0, 2, inp_test_a.size()).to(tt.bool)

        out_test = pooler(inp_test_a, inp_test_b)

        assert out_test.size() == sizes[1]

    def test_pooling_type(self):
        """
        Ensure that each pooling type is doing what we'd expect it to do.
        """

        pooler = model.Pooler(project=True, inp_dim=4, out_dim=4, pooling_type="max")

        inp_test_a = tt.Tensor([1.0, 1.0, 1.0, 1.0])
        inp_test_b = tt.randint(0, 2, inp_test_a.size()).to(tt.bool)

        out_test = pooler(inp_test_a, inp_test_b)

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