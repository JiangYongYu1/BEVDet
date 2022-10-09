## 
### BEVDet
1. 模型转换
    ```shell
    cd $BEVDET
    pip install -e .
    ln -s data_path ./data/nuscenes
    wget bevdet-r50.pth
    python export_onnx/export_bevdet_onnx.py configs/bevdet/bevdet-r50.py bevdet-r50.pth
    ```

2. 特征提取 [1, 6, 3, 256, 704]
FLOPs = 81.320718864G
Params = 17.955227M
3. BEV检测 [1, 64, 128, 128]
FLOPs = 74.530684928G
Params = 20.47047M
