import sys
sys.path.insert(0, 'src')
from pathlib import Path

documents_dir = Path('data/stock_data/databases/chunked_reports')
vector_db_dir = Path('data/stock_data/databases/vector_dbs')

print("正在测试文件加载...")

all_documents_paths = list(documents_dir.glob('*.json'))
print(f'\n1. 找到 {len(all_documents_paths)} 个 JSON 文件')

loaded_docs = []
for document_path in all_documents_paths:
    try:
        print(f'  尝试加载 {document_path.name}')
        import json
        with open(document_path, 'r', encoding='utf-8') as f:
            document = json.load(f)
        print(f'    OK 成功加载')
        print(f'    metainfo: {document.get("metainfo", {})}')
        
        sha1 = document.get('metainfo', {}).get('sha1', None)
        print(f'    sha1: {sha1}')
        
        if sha1:
            faiss_path = vector_db_dir / f"{sha1}.faiss"
            print(f'    faiss_path: {faiss_path}')
            print(f'    faiss exists: {faiss_path.exists()}')
            if faiss_path.exists():
                print('    OK FAISS 匹配成功')
                loaded_docs.append({
                    'document': document,
                    'faiss_path': faiss_path
                })
            else:
                print('    ERR FAISS 匹配失败')
        
    except Exception as e:
        print(f'    ERR 错误: {e}')

print(f'\n2. 成功加载并匹配到 {len(loaded_docs)} 个报告')

print('\n3. 测试公司名称匹配')
target_company = '中芯国际'
for i, item in enumerate(loaded_docs):
    company_name = item['document']['metainfo']['company_name']
    print(f'  报告 {i+1} 公司名: {company_name}')
    if company_name == target_company:
        print(f'  OK 匹配成功！')
