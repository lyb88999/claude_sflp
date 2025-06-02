# 比较脚本更新说明

## 概述
已更新 `run_fair_comparison_satfl.py` 脚本以支持各个实验中新增的分类指标，包括F1分数、精确率、召回率等。

## 主要修改

### 1. 图表生成函数 (create_comparison_plots)

#### 新增图表类型：
- **F1 Score (Macro) 对比图**: `f1_macro_comparison.png`
- **精确率 (Macro) 对比图**: `precision_macro_comparison.png`
- **召回率 (Macro) 对比图**: `recall_macro_comparison.png`
- **综合性能指标对比图**: `comprehensive_metrics_comparison.png` (2x2子图显示准确率、F1、精确率、召回率)
- **F1 Score (Weighted) 对比图**: `f1_weighted_comparison.png`
- **精确率 (Weighted) 对比图**: `precision_weighted_comparison.png`
- **召回率 (Weighted) 对比图**: `recall_weighted_comparison.png`
- **加权指标综合对比图**: `weighted_metrics_comparison.png` (2x2子图显示加权指标)

#### 支持的新指标：
- `f1_macros`: F1分数 (Macro平均)
- `precision_macros`: 精确率 (Macro平均)
- `recall_macros`: 召回率 (Macro平均)
- `f1_weighteds`: F1分数 (加权平均)
- `precision_weighteds`: 精确率 (加权平均)
- `recall_weighteds`: 召回率 (加权平均)

### 2. 报告生成函数 (generate_comparison_report)

#### 新增报告内容：
- **分类性能指标部分**:
  - F1分数性能 (Macro)
  - 精确率性能 (Macro)
  - 召回率性能 (Macro)
- **性能对比分析**: 包括F1分数的算法间对比
- **新的效率指标**: 卫星F1效率（每卫星F1分数）
- **新的收敛速度**: F1分数收敛速度
- **更全面的结论**: 综合考虑准确率、F1分数和资源效率

### 3. 关键指标打印函数 (print_key_metrics)

#### 新增打印内容：
- **分类指标显示**:
  - 最高F1分数 (Macro)
  - 最高精确率 (Macro)
  - 最高召回率 (Macro)
- **效率指标**: 每卫星F1分数
- **性能排名**:
  - 准确率排名
  - F1分数排名
  - 资源效率排名
  - 综合性能排名（准确率 + F1分数）
- **最佳表现总结**: 显示各指标的最佳算法

## 兼容性

脚本具有良好的向后兼容性：
- 自动检查新指标是否存在
- 如果某个实验没有提供新指标，不会报错
- 保持原有功能不变

## 使用方法

脚本使用方法不变：

```bash
# 运行完整比较实验
python experiments/run_fair_comparison_satfl.py

# 重新绘制已有数据的图表
python experiments/run_fair_comparison_satfl.py --replot --data-dir path/to/results
```

## 输出文件

现在会生成更多的图表文件：
- 原有的准确率、损失、能耗等图表
- 新增的分类指标图表（共8个新图表）
- 更详细的报告文件
- 扩展的关键指标日志输出

## 注意事项

1. 确保各个实验类正确返回新的指标
2. 新指标应当以百分比形式存储（与准确率一致）
3. 如果某个实验不支持某项指标，该指标将显示为0，不会影响其他功能 