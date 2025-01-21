from datasets import load_from_disk

# 修改后的代码
try:
    # 加载数据集
    dataset = load_from_disk(
        r"F:\Repos\expository-text-generation\data\datasets\wiki_cs\content\wiki_cs\train"
    )

    # 查看数据集信息
    print("Dataset info:")
    print(dataset)

    # 查看第一条数据
    print("\nFirst example:")
    print(dataset[0])

    print("\n2nd example:")
    print(dataset[1])

    print("\n3rd example:")
    print(dataset[2])

    print(len(dataset))

except Exception as e:
    print(f"Error loading dataset: {e}")
