import os
import glob
import time
import torch
import numpy as np
from torch.autograd import Variable
import cv2
from tools import get_test_images, get_image, save_images
from utils.img_read_save import image_read_cv2
from utils.Evaluator import Evaluator
from net import net  # 你的模型定义
import warnings

warnings.filterwarnings("ignore")


# ==== 模型加载 ====
def load_model(path):
    model = net()  # 确保模型支持彩色图像输入输出
    model.load_state_dict(torch.load(path))
    total_params = sum([np.prod(p.size()) for p in model.parameters()])
    print('Model {} : params: {:.4f}M'.format(model._get_name(), total_params * 4 / 1e6))
    model.eval()
    model.cuda()
    return model


# ==== 融合图像 ====
def _generate_fusion_image(model, vi, ir):
    return model(vi, ir)


def run_demo(model, vi_path, ir_path, output_path_root, index):
    # 获取输入图像，确保读取为彩色图像
    vi_img = get_test_images(vi_path, height=None, width=None)  # 默认读取为彩色图像
    ir_img = get_test_images(ir_path, height=None, width=None)  # 默认读取为彩色图像

    # 获取原始图像（用于保存时保留原尺寸）
    out = get_image(vi_path, height=None, width=None)

    vi_img = vi_img.cuda()
    ir_img = ir_img.cuda()
    vi_img = Variable(vi_img, requires_grad=False)
    ir_img = Variable(ir_img, requires_grad=False)

    # 融合
    img_fusion = _generate_fusion_image(model, vi_img, ir_img)

    # 构造输出路径
    file_name = f"{index}.png"  # 不再加 fusion_ 前缀
    output_path = os.path.join(output_path_root, file_name)

    # 转为 uint8
    if torch.cuda.is_available():
        img = img_fusion.cpu().clamp(0, 255).numpy()
    else:
        img = img_fusion.clamp(0, 255).numpy()

    # 保证是彩色图像，确保有3个通道
    img = img.astype('uint8')

    # 保存图像
    save_images(output_path, img, out)
    print(f"Saved: {output_path}")


# ==== 评估指标 ====
def evaluate_metrics(ori_img_folder, eval_folder, model_name="X"):
    ir_folder = os.path.join(ori_img_folder, "Test_ir")
    vi_folder = os.path.join(ori_img_folder, "Test_vi")

    metric_result = np.zeros((8))

    ir_files = os.listdir(ir_folder)

    for img_name in ir_files:
        ir_path = os.path.join(ir_folder, img_name)
        vi_path = os.path.join(vi_folder, img_name)
        fi_path = os.path.join(eval_folder, img_name.split('.')[0] + ".png")

        if not (os.path.exists(ir_path) and os.path.exists(vi_path) and os.path.exists(fi_path)):
            print(f"文件缺失: ir={ir_path}, vi={vi_path}, fi={fi_path}")
            continue

        # 读取彩色图像并转换为灰度图像
        ir = image_read_cv2(ir_path, mode='RGB')  # 读取彩色图像
        vi = image_read_cv2(vi_path, mode='RGB')  # 读取彩色图像
        fi = image_read_cv2(fi_path, mode='RGB')  # 读取彩色图像

        # 转换为灰度图像
        ir_gray = cv2.cvtColor(ir, cv2.COLOR_RGB2GRAY)
        vi_gray = cv2.cvtColor(vi, cv2.COLOR_RGB2GRAY)
        fi_gray = cv2.cvtColor(fi, cv2.COLOR_RGB2GRAY)

        if ir_gray is None or vi_gray is None or fi_gray is None:
            print(f"图像读取失败: {img_name}，检查文件内容/格式")
            continue

        current_metrics = np.array([
            Evaluator.EN(fi_gray),
            Evaluator.SD(fi_gray),
            Evaluator.SF(fi_gray),
            Evaluator.MI(fi_gray, ir_gray, vi_gray),
            Evaluator.SCD(fi_gray, ir_gray, vi_gray),
            Evaluator.VIFF(fi_gray, ir_gray, vi_gray),
            Evaluator.Qabf(fi_gray, ir_gray, vi_gray),
            Evaluator.SSIM(fi_gray, ir_gray, vi_gray)
        ])

        metric_result += current_metrics

    if len(ir_files) == 0:
        print("没有找到有效的评估图像")
        return metric_result

    metric_result /= len(ir_files)

    print("\t\t EN\t SD\t SF\t MI\tSCD\tVIF\tQabf\tSSIM")
    print(f"{model_name}\t" + "\t".join([f"{val:.2f}" for val in metric_result]) + "\n")

    return metric_result


# ==== 主函数 ====
def main():
    vi_path = "images/Test_vi/"
    ir_path = "images/Test_ir/"

    output_path = "./outputs/"
    model_path = "./models/Epoch_19_iters_2500.model"
    model_name = "MyModel"

    os.makedirs(output_path, exist_ok=True)

    with torch.no_grad():
        model = load_model(model_path)

        vi_list = sorted(glob.glob(os.path.join(vi_path, "*.jpg")))
        ir_list = sorted(glob.glob(os.path.join(ir_path, "*.jpg")))
        assert len(vi_list) == len(ir_list), "图像对数量不一致！"

        for vi_file, ir_file in zip(vi_list, ir_list):
            base = os.path.splitext(os.path.basename(vi_file))[0]
            start = time.time()
            run_demo(model, vi_file, ir_file, output_path, base)
            end = time.time()
            print(f"{base} 融合时间: {end - start:.3f} 秒")

    print("图像融合完成，开始评估指标...\n")

    evaluate_metrics(
        ori_img_folder="images",  # 需要目录结构 images/ir 和 images/vis
        eval_folder=output_path,
        model_name='TGFuse'
    )


if __name__ == "__main__":
    main()
