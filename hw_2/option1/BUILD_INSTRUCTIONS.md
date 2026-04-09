# 构建与运行说明

工作目录：`hw_2/option1`

依赖安装：
```bash
python3 -m pip install -r requirements.txt
```

快速运行示例：
```bash
cd hw_2/option1
python3 compress.py --input ../../imgs/CMakeTools.png --k 5 20 50 --outdir demo_out
```

生成 PDF 报告（可选）：
- 若已安装 `pandoc` 与 TeX 引擎（`xelatex`），执行：
```bash
pandoc report.md -o report.pdf --pdf-engine=xelatex
```
