import os
import sys
from pathlib import Path

os.environ['MINERU_MODEL_SOURCE'] = "modelscope"

def pdf_to_markdown_mineru(pdf_path, output_dir=None):
    from mineru.cli.client import main
    from click.testing import CliRunner
    
    if output_dir is None:
        output_dir = "data/stock_data/debug_data/03_reports_markdown"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "-p", pdf_path,
            "-o", str(output_dir),
            "-l", "ch",
            "-b", "pipeline",
            "-m", "auto"
        ]
    )
    
    if result.exit_code != 0:
        print("解析失败: {}".format(result.output))
        if result.exception:
            import traceback
            print("异常信息: {}".format(traceback.print_exception(type(result.exception), result.exception, result.exception.__traceback__)))
        return None
    
    print("解析完成！输出目录: {}".format(output_dir))
    return str(output_dir)

def pdf_to_markdown_mineru_cli(pdf_path, output_dir=None):
    import subprocess
    
    if output_dir is None:
        output_dir = "data/stock_data/debug_data/03_reports_markdown"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "-m", "mineru.cli.client",
        "-p", pdf_path,
        "-o", output_dir,
        "-l", "ch",
        "-b", "pipeline",
        "-m", "auto"
    ]
    
    print("执行命令: {}".format(" ".join(cmd)))
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("解析失败!")
        print("错误信息: {}".format(result.stderr))
        return None
    
    print("解析完成！输出目录: {}".format(output_dir))
    print("输出信息: {}".format(result.stdout))
    return str(output_dir)

if __name__ == "__main__":
    file_path = 'data/stock_data/pdf_reports/【财报】中芯国际：中芯国际2024年年度报告.pdf'
    pdf_to_markdown_mineru_cli(file_path)