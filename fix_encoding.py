#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复模块文件中的编码问题
清理非打印字符
"""

import os
import re

def clean_file(file_path):
    """清理文件中的非打印字符"""
    try:
        # 读取文件
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 记录原始长度
        original_length = len(content)
        
        # 清理非打印字符（保留常见的空白字符）
        # 移除不可见的Unicode字符，但保留正常的空格、制表符、换行符
        cleaned_content = re.sub(r'[\u00A0\u2000-\u200F\u2028-\u202F\u205F-\u206F\uFEFF]', '', content)
        
        # 如果内容有变化，写回文件
        if len(cleaned_content) != original_length:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            print(f"✅ 已修复 {file_path} (移除了 {original_length - len(cleaned_content)} 个字符)")
            return True
        else:
            print(f"✓ {file_path} 无需修复")
            return False
            
    except Exception as e:
        print(f"❌ 修复 {file_path} 时出错: {e}")
        return False

def main():
    print("🔧 开始修复模块文件编码问题...")
    print("=" * 50)
    
    fixed_count = 0
    total_count = 0
    
    # 修复所有模块文件
    for i in range(1, 10):
        file_path = f"module_{i}.py"
        if os.path.exists(file_path):
            total_count += 1
            if clean_file(file_path):
                fixed_count += 1
        else:
            print(f"⚠️ 文件不存在: {file_path}")
    
    print("=" * 50)
    print(f"📊 修复统计: {fixed_count}/{total_count} 个文件已修复")
    
    if fixed_count > 0:
        print("\n🔍 重新验证语法...")
        # 简单的语法验证
        for i in range(1, 10):
            file_path = f"module_{i}.py"
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    compile(content, file_path, 'exec')
                    print(f"✅ {file_path} 语法正确")
                except SyntaxError as e:
                    print(f"❌ {file_path} 仍有语法错误: {e}")
                except Exception as e:
                    print(f"⚠️ {file_path} 检查失败: {e}")
    
    print("\n🎉 编码修复完成！")

if __name__ == "__main__":
    main()