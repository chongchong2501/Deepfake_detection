#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块验证脚本
检查所有9个模块文件的完整性和标题
"""

import os
import re

def extract_module_title(file_path):
    """从模块文件中提取标题"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 查找标题行
        lines = content.split('\n')
        for line in lines[:20]:  # 检查前20行
            if line.strip().startswith('#') and ('第' in line and '段' in line):
                return line.strip().replace('#', '').strip()
            elif line.strip().startswith('#') and any(char.isdigit() for char in line) and ('模块' in line or 'Module' in line):
                return line.strip().replace('#', '').strip()
        
        return "未找到标题"
    except Exception as e:
        return f"读取错误: {e}"

def get_file_size(file_path):
    """获取文件大小"""
    try:
        size = os.path.getsize(file_path)
        if size < 1024:
            return f"{size} B"
        elif size < 1024 * 1024:
            return f"{size/1024:.1f} KB"
        else:
            return f"{size/(1024*1024):.1f} MB"
    except:
        return "未知"

def check_python_syntax(file_path):
    """检查Python语法"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 简单的语法检查
        compile(content, file_path, 'exec')
        return "✅ 语法正确"
    except SyntaxError as e:
        return f"❌ 语法错误: {e}"
    except Exception as e:
        return f"⚠️ 检查失败: {e}"

def main():
    print("🔍 Kaggle深度伪造检测模块验证")
    print("=" * 60)
    
    modules = []
    for i in range(1, 10):
        file_path = f"module_{i}.py"
        if os.path.exists(file_path):
            title = extract_module_title(file_path)
            size = get_file_size(file_path)
            syntax = check_python_syntax(file_path)
            
            modules.append({
                'file': file_path,
                'title': title,
                'size': size,
                'syntax': syntax
            })
        else:
            modules.append({
                'file': file_path,
                'title': "❌ 文件不存在",
                'size': "0 B",
                'syntax': "❌ 文件缺失"
            })
    
    # 打印验证结果
    print(f"{'模块文件':<15} {'标题':<40} {'大小':<10} {'语法检查':<15}")
    print("-" * 85)
    
    total_size = 0
    valid_modules = 0
    
    for module in modules:
        print(f"{module['file']:<15} {module['title']:<40} {module['size']:<10} {module['syntax']:<15}")
        
        if "✅" in module['syntax']:
            valid_modules += 1
        
        # 计算总大小
        if os.path.exists(module['file']):
            total_size += os.path.getsize(module['file'])
    
    print("-" * 85)
    print(f"\n📊 验证统计:")
    print(f"总模块数: 9")
    print(f"有效模块: {valid_modules}")
    print(f"总大小: {total_size/(1024*1024):.2f} MB")
    
    if valid_modules == 9:
        print("\n🎉 所有模块验证通过！")
        print("\n📋 使用说明:")
        print("1. 在Kaggle中按顺序运行模块 (module_1.py → module_9.py)")
        print("2. 确保有GPU加速器和足够的内存")
        print("3. 添加FaceForensics++数据集")
        print("4. 每个模块作为独立的代码单元格运行")
    else:
        print(f"\n⚠️ 发现 {9-valid_modules} 个问题模块，请检查并修复")
    
    print("\n📁 相关文件:")
    print("- README_modules.md: 详细使用说明")
    print("- deepfake-detection.ipynb: 原始笔记本")
    print("- extract_modules.py: 提取脚本")

if __name__ == "__main__":
    main()