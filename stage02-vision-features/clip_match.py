# 1. 导入库
import torch
from pathlib import Path
from PIL import Image
from transformers import CLIPModel,CLIPProcessor

# 2. 定义模型名称和候选文本
MODEL_NAME = "openai/clip-vit-base-patch32"

TEXTS = [
    "a photo of a mouse",
    "a photo of a keyboard",
    "a photo of a cup",
    "a photo of a book",
    "a photo of a phone",
]

# 3. 选择运行设备

# 4. 获取图片路径
def get_image_path():
    image_dir=Path(__file__).parent/"images"                        ##获取图片文件夹路径
    # png_path=image_dir/"test-base.png"                            ##拼接图片路径
    # jpg_path=image_dir/"test-base.jpg"
    # png_path=image_dir/"test-exp1.png"                              ##拼接图片路径
    # jpg_path=image_dir/"test-exp1.jpg"
    png_path=image_dir/"test-exp2.png"                             ##拼接图片路径
    jpg_path=image_dir/"test-exp2.jpg"

    if png_path.exists():
        return png_path
    if jpg_path.exists():
        return jpg_path
    

# 5. 加载 CLIP processor 和 model
def load_CLIP(device):
    processor=CLIPProcessor.from_pretrained(MODEL_NAME)             ##获取相应模型的预处理器
    model=CLIPModel.from_pretrained(MODEL_NAME)                     ##获取CLIP模型
    model=model.to(device)
    model.eval()
    return processor,model

# 6. 图片匹配文本
def match_image_text(image,texts,processor,model,device):
    inputs=processor(
        images=image,
        text=texts,
        return_tensors="pt",
        padding=True
    )
    inputs=inputs.to(device)

    ## 7. 前向传播得到匹配分数
    with torch.no_grad():
        outputs=model(**inputs)
    ## 8. 计算概率并选出最高匹配文本
    logits=outputs.logits_per_image                                 ##利用CILP对象的logits_per_image属性获得每个图片得分
    probabilities=logits.softmax(dim=1)[0]                          ##softmax方法讲得分转换为概率
    best_index=probabilities.argmax().item()                        ##argmax找出最大值，item转换成数值
    return best_index,probabilities

# 10. main() 主函数
def main():
    device="cuda" if torch.cuda.is_available() else "cpu"
    image_path=get_image_path()
    image=Image.open(image_path).convert("RGB")
    processor,model=load_CLIP(device)
    
    best_index,probabilities=match_image_text(image,TEXTS,processor,model,device)

    ##打印输出
    print(f"使用的设备是{device}")
    print(f"这张图片最符合的描述是{TEXTS[best_index]}")
    for text, probability in zip(TEXTS, probabilities):
        print(f"{text}: {probability.item():.4f}")

if __name__=="__main__":
    main()

