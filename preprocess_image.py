import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def cv_imread(file_path):
    """兼容中文路径的读取"""
    try:
        return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None

def cv_imwrite(file_path, img):
    """兼容中文路径的保存"""
    try:
        # 获取文件后缀，例如 .png
        ext = os.path.splitext(file_path)[1]
        if not ext:
            ext = ".png"
        cv2.imencode(ext, img)[1].tofile(file_path)
    except Exception as e:
        print(f"保存失败 {file_path}: {e}")

# ============================================================
# Handwritten Digit Preprocessing
# Version: color-independent ink detection (去横线逻辑已移除)
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def build_color_independent_mask(gray):
    """
    不固定数字颜色的笔画检测。
    使用“局部变暗程度”：只要笔画与附近纸张存在明显局部对比，就有机会被检测。
    """
    # 局部背景估计
    background = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigmaX=max(3.0, min(gray.shape) / 25.0)
    )

    # 任何比局部背景更暗的笔画都会产生响应
    local_dark = cv2.subtract(background, gray)

    local_dark = cv2.normalize(
        local_dark,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Otsu 自动决定阈值，不固定颜色阈值
    _, mask = cv2.threshold(
        local_dark,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 连接断裂笔画
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1
    )

    # 去除很小的噪声
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_kernel,
        iterations=1
    )

    return mask


def remove_small_components(mask):
    """删除明显的小噪声，同时尽量保留碎裂的数字笔画。"""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    result = np.zeros_like(mask)
    H, W = mask.shape
    image_area = H * W

    # 比原代码宽松，避免潦草数字被删掉
    min_area = max(8, int(image_area * 0.00001))

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        ww = stats[i, cv2.CC_STAT_WIDTH]
        hh = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        # 极端大块背景不要
        if ww * hh > image_area * 0.90:
            continue

        result[labels == i] = 255

    return result


def find_best_digit(mask):
    """
    找数字区域。采用“面积 + 位置 + 形状”综合评分。
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    H, W = mask.shape
    image_area = H * W
    candidates = []

    for i in range(1, num_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        if ww < 5 or hh < 5:
            continue

        if area < max(10, image_area * 0.00002):
            continue

        box_area = ww * hh

        if box_area > image_area * 0.80:
            continue

        aspect_ratio = ww / max(hh, 1)

        # 放宽形状限制，允许潦草书写
        if aspect_ratio > 4.5 or aspect_ratio < 0.05:
            continue

        # 数字通常位于图片中部附近
        cx = x + ww / 2.0
        cy = y + hh / 2.0

        center_distance = np.sqrt(
            ((cx - W / 2) / W) ** 2 +
            ((cy - H / 2) / H) ** 2
        )
        center_score = 1.0 - min(center_distance, 1.0)

        area_score = min(area / (image_area * 0.08), 1.0)

        # 数字区域越完整，分数越高
        size_score = min(
            max(ww, hh) / max(H, W) * 8,
            1.0
        )

        score = (
            area_score * 0.45 +
            center_score * 0.30 +
            size_score * 0.25
        )

        candidates.append(
            (score, i, x, y, ww, hh, area)
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return candidates[0]


def crop_with_margin(mask, x, y, w, h, margin_ratio=0.18):
    """给数字区域增加少量边缘，避免笔画被裁掉。"""
    H, W = mask.shape

    mx = max(3, int(w * margin_ratio))
    my = max(3, int(h * margin_ratio))

    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(W, x + w + mx)
    y2 = min(H, y + h + my)

    return mask[y1:y2, x1:x2]


def to_mnist_28x28(roi):
    """把检测到的数字转换成 MNIST 风格 28x28 黑底白字。"""
    ys, xs = np.where(roi > 0)

    if len(xs) == 0:
        raise ValueError("检测到的数字区域为空。")

    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1

    roi = roi[y1:y2, x1:x2]

    h, w = roi.shape

    # 与 MNIST 类似：数字主体最大约 20 像素
    target_size = 20
    scale = target_size / max(w, h)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(
        roi,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros((28, 28), dtype=np.uint8)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    canvas = cv2.GaussianBlur(
        canvas,
        (3, 3),
        0
    )

    return canvas


def preprocess_image(image_path, save_debug=False):
    """
    主预处理函数（已删除去横线逻辑，双路径直接处理原图灰度图）
    """

    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    # ========================================================
    # 1. 读取
    # ========================================================
    img = cv_imread(image_path)
    if img is None:
        raise FileNotFoundError(f"找不到图片（请检查路径是否包含中文或文件是否存在）：{image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    original_gray = gray.copy()
    H, W = gray.shape

    if save_debug:
        print("=" * 65)
        print(f"Processing image: {W} x {H}")

    # ========================================================
    # 2. 双路径检测 (直接作用于 gray)
    # ========================================================

    # -------- A：局部背景减除 + Otsu --------
    color_independent_mask = build_color_independent_mask(gray)

    # -------- B：Adaptive Threshold --------
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    adaptive = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7
    )

    adaptive = cv2.morphologyEx(
        adaptive,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1
    )

    adaptive = cv2.morphologyEx(
        adaptive,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1
    )

    # ========================================================
    # 3. 去除小噪声
    # ========================================================
    mask_a = remove_small_components(color_independent_mask)
    mask_b = remove_small_components(adaptive)

    # ========================================================
    # 4. 分别寻找数字
    # ========================================================
    result_a = find_best_digit(mask_a)
    result_b = find_best_digit(mask_b)

    # ========================================================
    # 5. 选择结果
    # ========================================================
    selected = None
    selected_mask = None
    selected_method = None

    if result_a is not None:
        selected = result_a
        selected_mask = mask_a
        selected_method = "Color-independent local contrast + Otsu"

    if result_b is not None:
        if selected is None:
            selected = result_b
            selected_mask = mask_b
            selected_method = "Adaptive Threshold"
        else:
            score_a = result_a[0]
            score_b = result_b[0]
            area_a = result_a[6]
            area_b = result_b[6]

            if score_b > score_a * 1.10 or area_b > area_a * 1.35:
                selected = result_b
                selected_mask = mask_b
                selected_method = "Adaptive Threshold"

    # ========================================================
    # 6. 如果候选仍然不存在：使用两个 mask 的并集兜底。
    # ========================================================
    if selected is None:
        combined = cv2.bitwise_or(mask_a, mask_b)
        combined = cv2.morphologyEx(
            combined,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            iterations=1
        )
        selected = find_best_digit(combined)
        if selected is not None:
            selected_mask = combined
            selected_method = "Combined fallback"

    # ========================================================
    # 7. 最终仍然没有候选：使用整图中心区域兜底
    # ========================================================
    if selected is None:
        if save_debug:
            print("WARNING: 未稳定检测到数字区域，进入中心区域兜底模式。")

        margin_x = int(W * 0.15)
        margin_y = int(H * 0.15)

        fallback = cv2.bitwise_or(mask_a, mask_b)
        selected_mask = fallback[margin_y:H - margin_y, margin_x:W - margin_x]

        canvas = to_mnist_28x28(selected_mask)
        selected_method = "Center fallback"

        if save_debug:
            cv_imwrite(os.path.join(results_dir, "preprocessed_digit.png"), canvas)

        return canvas

    # ========================================================
    # 8. 裁剪数字
    # ========================================================
    score, label, x, y, ww, hh, area = selected

    if save_debug:
        print()
        print(f"Selected method: {selected_method}")
        print(f"Detected box: {ww} x {hh}")
        print(f"Detected area: {area}")
        print(f"Detection score: {score:.4f}")

    roi = crop_with_margin(selected_mask, x, y, ww, hh, margin_ratio=0.18)

    # ========================================================
    # 9. 转换成 28x28
    # ========================================================
    canvas = to_mnist_28x28(roi)

    # ========================================================
    # 10. 保存调试结果
    # ========================================================
    if save_debug:
        cv_imwrite(os.path.join(results_dir, "preprocessed_digit.png"), canvas)
        cv_imwrite(os.path.join(results_dir, "color_independent_mask.png"), mask_a)
        cv_imwrite(os.path.join(results_dir, "adaptive_mask.png"), mask_b)

        # 检测框
        debug_img = img.copy()
        cv2.rectangle(debug_img, (x, y), (x + ww, y + hh), (0, 255, 0), 3)
        cv2.putText(debug_img, selected_method, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv_imwrite(os.path.join(results_dir, "detected_box.png"), debug_img)

        # 综合调试图 (调整布局以适配被删除的横线图)
        plt.figure(figsize=(16, 10))

        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Original")
        plt.axis("off")

        plt.subplot(2, 3, 2)
        plt.imshow(mask_a, cmap="gray")
        plt.title("Color-independent Mask")
        plt.axis("off")

        plt.subplot(2, 3, 3)
        plt.imshow(mask_b, cmap="gray")
        plt.title("Adaptive Threshold")
        plt.axis("off")

        plt.subplot(2, 3, 4)
        plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        plt.title("Detected Digit")
        plt.axis("off")

        plt.subplot(2, 3, 5)
        plt.imshow(roi, cmap="gray")
        plt.title("Digit ROI")
        plt.axis("off")

        plt.subplot(2, 3, 6)
        plt.imshow(canvas, cmap="gray")
        plt.title("Final 28x28")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "preprocess_demo.png"), dpi=180)
        plt.show()

        print()
        print("=" * 65)
        print("Preprocessing completed.")
        print("Method:", selected_method)
        print("Saved debug images to 'results/' folder.")
        print("=" * 65)

    return canvas


# ============================================================
# 本地测试
# ============================================================
if __name__ == "__main__":
    test_image_path = os.path.join(BASE_DIR, "test_images", "6_2.jpg")
    preprocess_image(test_image_path, save_debug=True)