import torch
from mmxdeploy.core import FUNCTION_REWRITER
from mmdet3d.core import bbox3d2result


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.bevdet.BEVDet.forward_test')
def BEVDet__forward_test(ctx, self, img_inputs, img_metas=None, rescale=False):
    """Rewrite this function to run simple_test directly."""
    return self.simple_test(None, img_metas, img_inputs)


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.bevdet.BEVDet.extract_pseudo_cloud')
def BEVDet__extract_pseudo_cloud(ctx,
                                 self,
                                 img_input,
                                 img_meta=None,
                                 rescale=False):
    """Rewrite this function to run simple_test directly."""
    # return self.simple_test(None, img_metas, img_inputs)
    x = self.image_encoder(img_input)
    volume = self.img_view_transformer.lift(x)
    return volume


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.bevdet.BEVDet.pseudo_cloud_splat')
def BEVDet__pseudo_cloud_splat(ctx, self, volume, rots, trans, intrins,
                               post_rots, post_trans):
    """Rewrite this function to run simple_test directly."""
    # return self.simple_test(None, img_metas, img_inputs)
    bev_feat = self.img_view_transformer.splat(volume, rots, trans, intrins,
                                               post_rots, post_trans)
    return bev_feat


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.bevdet.BEVDet.detector')
def BEVDet__detector(ctx, self, x):
    """Rewrite this function to run simple_test directly."""
    x = self.bev_encoder(x)
    outs = self.pts_bbox_head([x])
    if torch.onnx.is_in_onnx_export():
        export_outs = []
        for task_id, preds_dict in enumerate(outs):
            batch_reg = preds_dict[0]['reg']
            batch_hei = preds_dict[0]['height']
            batch_dim = preds_dict[0]['dim']
            batch_rot = preds_dict[0]['rot']
            batch_vel = preds_dict[0]['vel']            
            batch_heatmap = preds_dict[0]['heatmap']
            export_outs.extend([batch_reg, batch_hei, batch_dim, batch_rot,
                               batch_vel, batch_heatmap])
        outs = export_outs
            # 'reg', 'height', 'dim', 'rot', 'vel', 'heatmap'

    return outs


@FUNCTION_REWRITER.register_rewriter(
    'mmdet3d.models.detectors.bevdet.BEVDet.postprocess')
def BEVDet__postprocess(ctx, self, outs, img_metas, rescale=False):
    bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
    bbox_results = [
        bbox3d2result(bboxes, scores, labels)
        for bboxes, scores, labels in bbox_list
    ]
    return bbox_results
