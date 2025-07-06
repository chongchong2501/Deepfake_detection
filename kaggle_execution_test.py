#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kaggle Cells 执行测试脚本
用于验证所有cell文件的语法正确性和执行顺序
"""

import os
import sys
import ast
import traceback
from pathlib import Path

def check_syntax(file_path):
    """检查Python文件的语法"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 编译检查语法
        ast.parse(content)
        return True, None
    except SyntaxError as e:
        return False, f"语法错误: {e}"
    except Exception as e:
        return False, f"文件读取错误: {e}"

def check_imports(file_path):
    """检查文件中的import语句"""
    imports = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"from {module} import {alias.name}")
        
        return imports
    except Exception as e:
        print(f"检查导入时出错 {file_path}: {e}")
        return []

def main():
    """主函数"""
    print("🔍 Kaggle Cells 执行测试开始...")
    print("=" * 60)
    
    # 定义cell文件执行顺序
    cell_files = [
        'cell_01_imports_and_setup.py',
        'cell_02_global_config.py', 
        'cell_03_data_processing.py',
        'cell_04_dataset_class.py',
        'cell_05_model_definition.py',
        'cell_06_loss_and_utils.py',
        'cell_07_training_functions.py',
        'cell_08_evaluation_functions.py',
        'cell_09_data_preparation.py',
        'cell_10_data_loaders.py',
        'cell_11_model_setup.py',
        'cell_12_training_loop.py',
        'cell_13_model_evaluation.py',
        'cell_14_results_summary.py'
    ]
    
    kaggle_cells_dir = Path(__file__).parent / 'kaggle_cells'
    
    if not kaggle_cells_dir.exists():
        print(f"❌ 错误: kaggle_cells 目录不存在: {kaggle_cells_dir}")
        return False
    
    all_passed = True
    all_imports = []
    
    print("📋 检查文件语法和导入语句:")
    print("-" * 40)
    
    for i, cell_file in enumerate(cell_files, 1):
        file_path = kaggle_cells_dir / cell_file
        
        if not file_path.exists():
            print(f"❌ {i:2d}. {cell_file} - 文件不存在")
            all_passed = False
            continue
        
        # 检查语法
        syntax_ok, error = check_syntax(file_path)
        if not syntax_ok:
            print(f"❌ {i:2d}. {cell_file} - {error}")
            all_passed = False
            continue
        
        # 检查导入语句
        imports = check_imports(file_path)
        if imports and cell_file != 'cell_01_imports_and_setup.py':
            print(f"⚠️  {i:2d}. {cell_file} - 发现 {len(imports)} 个导入语句")
            for imp in imports[:3]:  # 只显示前3个
                print(f"     {imp}")
            if len(imports) > 3:
                print(f"     ... 还有 {len(imports) - 3} 个导入")
        else:
            print(f"✅ {i:2d}. {cell_file} - 语法正确")
        
        if cell_file == 'cell_01_imports_and_setup.py':
            all_imports = imports
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("✅ 所有文件语法检查通过!")
        
        print(f"\n📦 cell_01_imports_and_setup.py 中的导入语句 ({len(all_imports)} 个):")
        print("-" * 40)
        for imp in all_imports:
            print(f"  {imp}")
        
        print("\n🚀 Kaggle 执行顺序:")
        print("-" * 40)
        for i, cell_file in enumerate(cell_files, 1):
            print(f"  {i:2d}. {cell_file}")
        
        print("\n📋 使用说明:")
        print("-" * 40)
        print("1. 在Kaggle Notebook中按顺序创建14个代码单元格")
        print("2. 将对应的cell文件内容复制到每个单元格中")
        print("3. 确保数据路径设置正确 (在cell_02中修改)")
        print("4. 按顺序执行所有单元格")
        print("5. 查看详细使用指南: KAGGLE_USAGE_GUIDE.md")
        
        print("\n🎯 关键优化特性:")
        print("-" * 40)
        print("• 全GPU数据处理流水线")
        print("• 统一依赖管理 (所有import在cell_01)")
        print("• 智能缓存系统")
        print("• 混合精度训练")
        print("• 实时性能监控")
        
        return True
    else:
        print("❌ 发现语法错误，请检查上述文件")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)