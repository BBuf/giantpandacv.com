import os
from bs4 import BeautifulSoup

# 批量更改site目录及子目录下所有html中的img标签，标签内容增加字段 referrerpolicy="no-referrer"??


# 指定site目录路径
site_path = "site"

# 遍历目录及子目录下的所有html文件
for root, dirs, files in os.walk(site_path):
    for file in files:
        if file.endswith(".html"):
            # 打开html文件并解析内容
            with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                html_content = f.read()
                soup = BeautifulSoup(html_content, "html.parser")
            
            # 遍历所有img标签并增加referrerpolicy属性
            for img_tag in soup.find_all("img"):
                img_tag["referrerpolicy"] = "no-referrer"
            
            # 将修改后的内容写回html文件
            with open(os.path.join(root, file), "w", encoding="utf-8") as f:
                f.write(str(soup))
