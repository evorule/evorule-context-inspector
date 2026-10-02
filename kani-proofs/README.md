# kani-proofs

Kani 形式化验证(验收矩阵 #20,**非阻塞**,后续批次)。

计划覆盖的核心算法(实施文档 §7 #20):

- multiset Jaccard 距离(漂移指纹)
- 压力四档 band 分类(左闭右开边界 0.6 / 0.8 / 0.95 恰等归高档)
- word-shingle 包含度(分母为 0 → ratio=0,无 panic 路径)

运行方式(主仓同款,`#[cfg(kani)]` 门控):

```bash
cargo kani -p evorule-llm-context-inspector
```
