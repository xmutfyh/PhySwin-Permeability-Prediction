__all__ = [
    'RegHead',
    'CNNRegHead',
    'LinearRegHead1',
    'LinearRegHead2',
    'LinearRegHead3',
    'LinearRegHead1NoGAP',
    'LinearRegHead2NoGAP',
    'LinearRegHead3NoGAP',
    'VisionTransformerRegHead',
    'PhyRegHeadNoTVP',          # ✅ 新增
]

from configs.heads.reg_head import RegHead
from configs.heads.headcnn import CNNRegHead
from configs.heads.linear_reghead1 import LinearRegHead1
from configs.heads.linear_reghead2 import LinearRegHead2
from configs.heads.linear_reghead3 import LinearRegHead3
from configs.heads.linear_reghead1nogap import LinearRegHead1NoGAP
from configs.heads.linear_reghead2nogap import LinearRegHead2NoGAP
from configs.heads.linear_reghead3nogap import LinearRegHead3NoGAP
from configs.heads.vision_transformer_head import VisionTransformerRegHead
from configs.heads.phy_reg_head_notvp import PhyRegHeadNoTVP  # ✅ 新增
