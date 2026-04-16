#!/bin/bash
# 一键执行所有数据提取、分析与绘图任务

set -e  # 遇到任何错误立即退出

echo "=========================================="
echo "  开始执行实验数据分析与绘图流程"
echo "=========================================="

# 1. 检查依赖
echo ""
echo "[1/5] 检查 Python 依赖..."
python -c "import tensorboard, pandas, matplotlib" 2>/dev/null || {
    echo "错误：缺少必要的 Python 包，请先运行: pip install tensorboard pandas matplotlib"
    exit 1
}
echo "      ... 依赖检查通过"

# 2. 提取所有实验数据
echo ""
echo "[2/5] 从 TensorBoard 日志提取实验数据..."
python extract_all_analysis.py
echo "      ... 数据提取完成"

# 3. 绘制第一阶段图表
echo ""
echo "[3/5] 生成第一阶段图表 (Scratch vs Finetune)..."
python plot_phase1.py
echo "      ... 第一阶段图表生成完成"

# 4. 绘制第二阶段图表（消融实验）
echo ""
echo "[4/5] 生成第二阶段图表 (Ablation Studies)..."
python plot_phase2.py
python plot_phase2_time.py
echo "      ... 第二阶段图表生成完成"

# 5. 生成汇总表格
echo ""
echo "[5/5] 生成实验结果汇总表格..."
python summarize_results.py
echo "      ... 汇总表格生成完成"

echo ""
echo "=========================================="
echo "  所有分析任务执行完毕！"
echo "=========================================="
echo ""
echo "输出文件列表："
echo "  - all_experiments_analysis.json"
echo "  - all_convergence_data.csv"
echo "  - phase1_accuracy.png"
echo "  - phase1_convergence.png"
echo "  - phase1_efficiency.png"
echo "  - phase2_ablation_accuracy.png"
echo "  - phase2_ablation_convergence.png"
echo "  - phase2_training_time.png"
echo "  - all_experiments_summary.csv"
echo ""