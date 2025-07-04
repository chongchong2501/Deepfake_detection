import json
import os

# 读取笔记本文件
with open('deepfake-detection.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# 提取所有代码单元格
code_cells = []
for i, cell in enumerate(notebook.get('cells', [])):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        if isinstance(source, list):
            code = ''.join(source)
        else:
            code = source
        
        # 查找段落标识
        lines = code.split('\n')
        title = f"module_{i+1}"
        segment_num = None
        
        for line in lines[:10]:  # 检查前10行
            if '第' in line and '段' in line:
                # 提取段落号
                for char in line:
                    if char.isdigit():
                        segment_num = char
                        break
                title = line.strip().replace('#', '').strip()
                break
            elif '##' in line and any(char.isdigit() for char in line):
                for char in line:
                    if char.isdigit():
                        segment_num = char
                        break
                title = line.strip().replace('#', '').strip()
                break
        
        if segment_num:
            filename = f"module_{segment_num}.py"
        else:
            filename = f"module_{len(code_cells)+1}.py"
        
        code_cells.append({
            'index': i+1,
            'title': title,
            'filename': filename,
            'code': code
        })

# 创建独立的Python文件
for cell in code_cells:
    filename = cell['filename']
    code = cell['code']
    title = cell['title']
    
    # 添加文件头注释
    header = f"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# {title}
# 
# Kaggle Deepfake Detection Module
# This module can be run as a single cell in Kaggle environment
# 
# Usage:
# 1. Create a new code cell in Kaggle
# 2. Copy the entire content of this file to the cell
# 3. Run the cell

"""
    
    full_code = header + code
    
    # 写入文件
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(full_code)
    
    print(f"Created: {filename}")
    print(f"Title: {title}")
    print(f"Code length: {len(code)} characters")
    print()

print(f"\nSuccessfully extracted {len(code_cells)} modules!")
print("\nGenerated files:")
for cell in code_cells:
    print(f"  - {cell['filename']}: {cell['title']}")

print("\nInstructions:")
print("1. Each file can be run as an independent code cell in Kaggle")
print("2. Run files in order: module_1.py -> module_2.py -> ... -> module_9.py")
print("3. Ensure you have GPU resources and datasets in Kaggle environment")