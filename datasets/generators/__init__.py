# 测试数据生成器模块
# 导出所有公开接口，供 generate_all.py 和外部代码调用
from .base import save_json, generate_sample_id, verify_clique, verify_density
from .planted_clique import generate_planted_clique, generate_clique_batch
from .planted_dense import generate_planted_dense, generate_dense_batch
