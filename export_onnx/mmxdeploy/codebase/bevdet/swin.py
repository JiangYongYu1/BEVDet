from mmxdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.backbones.swin.SwinBlockSequence.forward')
def SwinBlockSequence__forward(ctx, self, x, hw_shape):
    """Rewrite this function to run simple_test directly."""
    for block in self.blocks:
        block.hw_shape = hw_shape
        x = block(x)

    if self.downsample:
        x_down, down_hw_shape = self.downsample(x, hw_shape)
        return x_down, down_hw_shape, x, hw_shape
    else:
        return x, hw_shape, x, hw_shape
