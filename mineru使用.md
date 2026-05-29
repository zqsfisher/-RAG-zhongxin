单个文档：

$env:MINERU\_MODEL\_SOURCE="modelscope"; python -m mineru.cli.client -p "data/stock\_data/pdf\_reports/【财报】中芯国际：中芯国际2024年年度报告.pdf" -o "output" -l ch -b pipeline

多个文档（批处理）:

$env:MINERU_MODELSOURCE="modelscope"; python -m mineru.cli.client -p "./data/stock_data/pdf_reports/" -o "./data/stock_data/debug_data/" -l ch -b pipeline

### 1. 模型环境初始化

由于你设置了 `$env:MINERU_MODEL_SOURCE="modelscope"`，程序会优先从国内魔搭社区加载所需的 AI 模型（如 `PDF-Extract-Kit`、版面分析模型等）。如果是首次运行，它会完成下载并缓存到本地 C:\Users\翟铨胜.cache\modelscope\hub\models\OpenDataLab；如果已经下载过（正如你刚才日志里显示的），它会直接读取缓存，快速完成初始化。

### 2. 文档流水线解析 (-b pipeline)

程序会使用轻量且稳定的 `pipeline` 后端引擎来处理这份财报。它会自动执行以下操作：

- **版面分析**：识别出文档中的标题层级、正文段落、页眉页脚等。
- **表格与公式提取**：精准定位并提取年报中的财务报表和数学公式。
- **OCR 文字识别**：结合 `-l ch`（指定中文）参数，针对扫描版或图片格式的页面，使用针对中文优化的 OCR 引擎将图片转换为可编辑文本。

### 3. 生成结构化输出文件

解析完成后，程序会在你当前目录下的 `output` 文件夹中自动生成一系列结果文件：

- **Markdown 文件 (.md)**：保留了原文档的标题层级、列表结构、公式与表格引用，可以直接用于大语言模型训练或知识库构建。
- **JSON 文件 (.json)**：按阅读顺序组织的详细结构化数据。
- **资源文件夹**：包括 `tables/`（提取的 HTML 或 CSV 表格）、`figures/` 或 `images/`（截取的图表原图）以及 `formulas/`（公式渲染图）。

输入是原始文档，输出是

output/
└── 文档名称/
└── auto/                      # 自动生成的内容
├── images/               # 提取的图片资源
├── 文档名称.md           # 核心Markdown文档
├── 文档名称\_content\_list.json
├── 文档名称\_content\_list\_v2.json
├── 文档名称\_layout.pdf
├── 文档名称\_middle.json
├── 文档名称\_model.json
├── 文档名称\_origin.pdf
└── 文档名称\_span.pdf
