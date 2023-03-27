import os
from bs4 import BeautifulSoup

#已弃用
#给site目录及子目录下的所有HTML文件添加<meta name="referrer" content="no-referrer" />头

# 遍历目录树，找到所有HTML文件
def find_html_files(root_dir):
    html_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.html') or filename.endswith('.htm'):
                filepath = os.path.join(dirpath, filename)
                html_files.append(filepath)
    return html_files

# 批量添加元标记
def add_meta_tag(html_files):
    for filepath in html_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            html = f.read()
        soup = BeautifulSoup(html, 'html.parser')
        head_tag = soup.find('head')
        if head_tag is None:
            continue
        meta_tag = soup.new_tag('meta')
        meta_tag['name'] = 'referrer'
        meta_tag['content'] = 'no-referrer'
        head_tag.insert(0, meta_tag)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(str(soup))

# 执行批量添加操作
html_files = find_html_files('site')
add_meta_tag(html_files)