# download_to_custom_path.py
from modelscope import snapshot_download
import os

# ===== 在这里修改你的下载路径 =====
DOWNLOAD_PATH = r"D:\PyCharm\traffic_prediction_trial\LLMDiff\model_downloaded"  # ✅ 修改为你想要的路径


# =====================================

def download_qwen():
    """下载 Qwen2-0.5B 到指定路径"""

    print("=" * 80)
    print("📥 Qwen2-0.5B 下载工具（自定义路径版）")
    print("=" * 80)

    # 1. 检查路径
    print(f"\n📂 下载目标路径: {DOWNLOAD_PATH}")

    # 创建目录（如果不存在）
    os.makedirs(DOWNLOAD_PATH, exist_ok=True)

    # 验证路径是否可写
    if not os.access(DOWNLOAD_PATH, os.W_OK):
        print(f"❌ 错误: 路径不可写！")
        print(f"   请检查: {DOWNLOAD_PATH}")
        return None

    print(f"✅ 路径有效且可写")

    # 2. 显示磁盘空间
    import shutil
    total, used, free = shutil.disk_usage(DOWNLOAD_PATH)
    free_gb = free / (1024 ** 3)

    print(f"\n💾 磁盘空间:")
    print(f"   剩余: {free_gb:.2f} GB")

    if free_gb < 3:
        print(f"   ⚠️  警告: 空间可能不足（建议至少 3 GB）")
        response = input("\n是否继续? (y/n): ")
        if response.lower() != 'y':
            return None

    # 3. 开始下载
    print("\n" + "=" * 80)
    print("🚀 开始下载...")
    print("=" * 80)
    print(f"模型: qwen/Qwen2-0.5B")
    print(f"大小: ~1.1 GB")
    print(f"保存位置: {DOWNLOAD_PATH}")
    print(f"预计时间: 5-15 分钟\n")

    try:
        local_path = snapshot_download(
            'Qwen/Qwen2-0.5B',
            cache_dir=DOWNLOAD_PATH,  # ✅ 关键参数
            revision='master'
        )

        print("\n" + "=" * 80)
        print("✅ 下载成功！")
        print("=" * 80)
        print(f"\n📂 模型完整路径:")
        print(f"   {local_path}")

        # 计算实际大小
        total_size = 0
        for root, dirs, files in os.walk(local_path):
            for file in files:
                filepath = os.path.join(root, file)
                total_size += os.path.getsize(filepath)

        print(f"\n📊 下载信息:")
        print(f"   总大小: {total_size / (1024 ** 3):.2f} GB")
        print(f"   文件数: {sum([len(files) for r, d, files in os.walk(local_path)])}")

        # 保存路径到文件
        path_file = os.path.join(DOWNLOAD_PATH, "model_path.txt")
        with open(path_file, 'w', encoding='utf-8') as f:
            f.write(f"Qwen2-0.5B 模型路径:\n")
            f.write(f"{local_path}\n\n")
            f.write(f"配置文件中使用:\n")
            f.write(f'qwen_model_name: "{local_path}"\n')

        print(f"\n💾 路径已保存到: {path_file}")

        # 显示下一步操作
        print("\n" + "=" * 80)
        print("🎯 下一步操作:")
        print("=" * 80)
        print("1. 复制模型路径:")
        print(f'   {local_path}')
        print("\n2. 打开配置文件:")
        print("   LLMDiff/configs/abilene_config_localtest.yaml")
        print("\n3. 修改这一行:")
        print(f'   qwen_model_name: "{local_path}"')
        print("\n4. 运行训练:")
        print("   python LLMDiff/train.py --config LLMDiff/configs/abilene_config_localtest.yaml")
        print("=" * 80 + "\n")

        return local_path

    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        print(f"\n常见问题:")
        print(f"1. 网络连接问题 → 检查网络或换热点")
        print(f"2. 路径权限问题 → 选择其他目录")
        print(f"3. 磁盘空间不足 → 清理磁盘或换盘")
        return None


if __name__ == '__main__':
    # 安装 modelscope（如果没有）
    try:
        import modelscope
    except ImportError:
        print("📦 正在安装 ModelScope...")
        import subprocess

        subprocess.check_call(['pip', 'install', 'modelscope', '-q'])
        print("✅ 安装完成\n")

    download_qwen()