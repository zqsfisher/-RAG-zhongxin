import os
import hashlib

files = os.listdir('data/stock_data/debug_data/03_reports_markdown')
with open('data/stock_data/subset.csv', 'w', encoding='utf-8') as f:
    f.write('sha1,file_name,company_name\n')
    for fn in files:
        if fn.endswith('.md'):
            file_no_ext = fn[:-3]
            sha1 = hashlib.md5(file_no_ext.encode('utf-8')).hexdigest()
            f.write(f'{sha1},{file_no_ext},中芯国际\n')
print('subset.csv 已重新创建')