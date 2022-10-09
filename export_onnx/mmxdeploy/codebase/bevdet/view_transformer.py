from mmxdeploy.core import FUNCTION_REWRITER


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.necks.view_transformer.ViewTransformerLiftSplatShoot.lift')
def ViewTransformerLiftSplatShoot__lift(ctx, self, x):
    """Rewrite this function to run simple_test directly."""
    B, N, C, H, W = x.shape
    x = x.view(B * N, C, H, W)
    x = self.depthnet(x)
    depth = self.get_depth_dist(x[:, :self.D])
    img_feat = x[:, self.D:(self.D + self.numC_Trans)]

    # Lift
    volume = depth.unsqueeze(1) * img_feat.unsqueeze(2)
    volume = volume.view(B, N, self.numC_Trans, self.D, H, W)
    volume = volume.permute(0, 1, 3, 4, 5, 2)
    return volume


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.necks.view_transformer.ViewTransformerLiftSplatShoot.splat'
)
def ViewTransformerLiftSplatShoot__splat(ctx, self, volume, rots, trans,
                                         intrins, post_rots, post_trans):
    if self.accelerate: 
        bev_feat = self.voxel_pooling_accelerated(rots, trans, intrins, post_rots, post_trans, volume)
    else:
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        bev_feat = self.voxel_pooling(geom, volume)
  
    return bev_feat