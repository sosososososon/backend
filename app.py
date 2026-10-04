from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import sys
import uuid
import sqlite3
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
import cv2

# 让 backend 找到项目根目录中的 train_mnist.py / preprocess_image.py / models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PROJECT_DIR)

from train_mnist import CNN
from preprocess_image import preprocess_image

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DB_PATH = os.path.join(BASE_DIR, "history.db")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# 数据库
# ============================================================
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            prediction INTEGER NOT NULL,
            confidence REAL NOT NULL,
            top1 TEXT,
            top2 TEXT,
            top3 TEXT,
            created_at TEXT NOT NULL
        )
    """)

    # 给你原来的 history 表增加新功能所需要的字段。
    # 已经存在的数据库不会被破坏。
    columns = {
        row["name"]
        for row in cursor.execute("PRAGMA table_info(history)").fetchall()
    }

    if "user_correct" not in columns:
        cursor.execute("ALTER TABLE history ADD COLUMN user_correct INTEGER DEFAULT NULL")

    if "corrected_prediction" not in columns:
        cursor.execute("ALTER TABLE history ADD COLUMN corrected_prediction INTEGER DEFAULT NULL")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS experience_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            types TEXT,
            suggestion TEXT,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


init_db()


# ============================================================
# 模型
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = os.path.join(PROJECT_DIR, "models", "mnist_cnn.pth")

model = CNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Model loaded successfully.")
print("Device:", DEVICE)


# ============================================================
# 工具函数
# ============================================================
def get_confidence_level(confidence):
    if confidence >= 0.90:
        return "HIGH"
    if confidence >= 0.70:
        return "MEDIUM"
    return "LOW"


def get_brightness(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return 255.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def build_suggestions(confidence, brightness):
    suggestions = []

    if brightness < 70:
        suggestions.extend([
            "当前图片亮度偏低",
            "增加环境光，避免手部遮挡",
            "建议重新拍摄一张更清晰的图片"
        ])

    if confidence < 0.70:
        suggestions.extend([
            "当前识别置信度较低",
            "建议重新书写，保证数字完整",
            "建议重新拍摄"
        ])
    elif confidence < 0.90:
        suggestions.extend([
            "当前结果基本可靠，但建议人工确认",
            "数字尽量写大一些并保持完整"
        ])
    else:
        suggestions.append("当前识别置信度较高，可以直接确认结果")

    return suggestions


def image_url_from_path(image_path):
    if not image_path:
        return None
    filename = os.path.basename(image_path)
    return f"/uploads/{filename}"


# ============================================================
# 基础接口
# ============================================================
@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "Handwritten Digit Recognition API",
        "status": "running"
    })


# ============================================================
# 图片预测
# ============================================================
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"success": False, "message": "没有上传图片"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"success": False, "message": "文件名为空"}), 400

    # 保留图片格式，支持 JPG / PNG 等，而不是强制全部改成 jpg
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        ext = ".jpg"

    filename = f"{uuid.uuid4().hex}{ext}"
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)

    try:
        brightness = get_brightness(image_path)

        # 这里继续使用你目前已经跑通的 preprocess_image
        image_28 = preprocess_image(image_path, save_debug=False)

        image = image_28.astype(np.float32) / 255.0
        image = (image - 0.1307) / 0.3081

        image = torch.tensor(
            image,
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(image)
            probabilities = F.softmax(output, dim=1)
            top_probs, top_classes = torch.topk(
                probabilities, k=3, dim=1
            )

        top_probs = top_probs[0].cpu().numpy()
        top_classes = top_classes[0].cpu().numpy()

        prediction = int(top_classes[0])
        confidence = float(top_probs[0])
        level = get_confidence_level(confidence)

        top3 = []
        for i in range(3):
            top3.append({
                "digit": int(top_classes[i]),
                "probability": round(float(top_probs[i]), 4)
            })

        top1_str = f"{top3[0]['digit']}:{top3[0]['probability'] * 100:.2f}%"
        top2_str = f"{top3[1]['digit']}:{top3[1]['probability'] * 100:.2f}%"
        top3_str = f"{top3[2]['digit']}:{top3[2]['probability'] * 100:.2f}%"

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO history (
                image_path,
                prediction,
                confidence,
                top1,
                top2,
                top3,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            image_path,
            prediction,
            confidence,
            top1_str,
            top2_str,
            top3_str,
            now_str
        ))

        history_id = cursor.lastrowid

        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "history_id": history_id,
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "confidence_level": level,
            "top3": top3,
            "brightness": round(brightness, 2),
            "suggestions": build_suggestions(confidence, brightness),
            "image_url": image_url_from_path(image_path)
        })

    except ValueError:
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception:
                pass

        return jsonify({
            "success": False,
            "message": "没有检测到有效数字",
            "suggestions": [
                "请保证数字完整性",
                "请让数字尽量写大一些",
                "避免图片过暗或严重模糊",
                "建议重新拍摄"
            ]
        }), 422

    except Exception as e:
        # 预测失败时删除刚刚上传的文件
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
            except Exception:
                pass

        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


# ============================================================
# 图片访问
# ============================================================
@app.route("/uploads/<path:filename>", methods=["GET"])
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# ============================================================
# 历史记录
# ============================================================
@app.route("/history", methods=["GET"])
def get_history():
    conn = get_db_connection()

    records = conn.execute("""
        SELECT *
        FROM history
        ORDER BY id DESC
    """).fetchall()

    conn.close()

    history_list = []

    for record in records:
        item = dict(record)
        item["image_url"] = image_url_from_path(item.get("image_path"))
        history_list.append(item)

    return jsonify({
        "success": True,
        "total": len(history_list),
        "data": history_list
    })


# 删除单条
@app.route("/history/<int:history_id>", methods=["DELETE"])
def delete_history(history_id):
    conn = get_db_connection()

    record = conn.execute(
        "SELECT * FROM history WHERE id = ?",
        (history_id,)
    ).fetchone()

    if record is None:
        conn.close()
        return jsonify({
            "success": False,
            "message": f"找不到 ID 为 {history_id} 的记录"
        }), 404

    image_path = record["image_path"]

    conn.execute(
        "DELETE FROM history WHERE id = ?",
        (history_id,)
    )

    conn.commit()
    conn.close()

    if image_path and os.path.exists(image_path):
        try:
            os.remove(image_path)
        except Exception:
            pass

    return jsonify({
        "success": True,
        "message": f"成功删除 ID 为 {history_id} 的记录"
    })


# 清空全部
@app.route("/history", methods=["DELETE"])
def clear_history():
    conn = get_db_connection()

    records = conn.execute(
        "SELECT image_path FROM history"
    ).fetchall()

    conn.execute("DELETE FROM history")
    conn.commit()
    conn.close()

    deleted_files = 0

    for record in records:
        image_path = record["image_path"]

        if image_path and os.path.exists(image_path):
            try:
                os.remove(image_path)
                deleted_files += 1
            except Exception:
                pass

    return jsonify({
        "success": True,
        "message": "历史记录已全部清空",
        "deleted_files": deleted_files
    })


# ============================================================
# 识别结果反馈
# ============================================================
@app.route("/feedback", methods=["POST"])
def submit_prediction_feedback():
    data = request.get_json(silent=True) or {}

    history_id = data.get("history_id")
    is_correct = data.get("is_correct")
    corrected_digit = data.get("corrected_digit")

    if history_id is None:
        return jsonify({
            "success": False,
            "message": "缺少 history_id"
        }), 400

    if is_correct is None:
        return jsonify({
            "success": False,
            "message": "缺少 is_correct"
        }), 400

    if not is_correct:
        if corrected_digit is None or int(corrected_digit) not in range(10):
            return jsonify({
                "success": False,
                "message": "请选择 0-9 中的正确数字"
            }), 400

        corrected_digit = int(corrected_digit)
    else:
        corrected_digit = None

    conn = get_db_connection()

    record = conn.execute(
        "SELECT id FROM history WHERE id = ?",
        (history_id,)
    ).fetchone()

    if record is None:
        conn.close()
        return jsonify({
            "success": False,
            "message": "找不到对应的识别记录"
        }), 404

    conn.execute("""
        UPDATE history
        SET user_correct = ?,
            corrected_prediction = ?
        WHERE id = ?
    """, (
        1 if bool(is_correct) else 0,
        corrected_digit,
        history_id
    ))

    conn.commit()
    conn.close()

    return jsonify({
        "success": True,
        "message": "感谢你的反馈，系统已经记录本次结果。"
    })


# ============================================================
# 数据统计
# ============================================================
@app.route("/stats", methods=["GET"])
def get_stats():
    conn = get_db_connection()

    total = conn.execute(
        "SELECT COUNT(*) AS count FROM history"
    ).fetchone()["count"]

    correct = conn.execute(
        "SELECT COUNT(*) AS count FROM history WHERE user_correct = 1"
    ).fetchone()["count"]

    pending = conn.execute(
        "SELECT COUNT(*) AS count FROM history WHERE user_correct IS NULL"
    ).fetchone()["count"]

    low_confidence = conn.execute(
        "SELECT COUNT(*) AS count FROM history WHERE confidence < 0.70"
    ).fetchone()["count"]

    digit_rows = conn.execute("""
        SELECT prediction AS digit, COUNT(*) AS count
        FROM history
        GROUP BY prediction
        ORDER BY count DESC
    """).fetchall()

    conn.close()

    digit_counts = [
        {
            "digit": int(row["digit"]),
            "count": int(row["count"])
        }
        for row in digit_rows
    ]

    return jsonify({
        "success": True,
        "data": {
            "total": total,
            "correct": correct,
            "pending": pending,
            "low_confidence": low_confidence,
            "digit_counts": digit_counts
        }
    })


# ============================================================
# 用户体验反馈
# ============================================================
@app.route("/experience-feedback", methods=["POST"])
def experience_feedback():
    data = request.get_json(silent=True) or {}

    types = data.get("types", [])
    suggestion = str(data.get("suggestion", "")).strip()

    if not types and not suggestion:
        return jsonify({
            "success": False,
            "message": "请选择反馈类型或填写建议"
        }), 400

    if not isinstance(types, list):
        types = [str(types)]

    types_text = "、".join(str(x) for x in types)

    conn = get_db_connection()

    conn.execute("""
        INSERT INTO experience_feedback (
            types,
            suggestion,
            created_at
        )
        VALUES (?, ?, ?)
    """, (
        types_text,
        suggestion,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()

    return jsonify({
        "success": True,
        "message": "感谢你的建议！我们已经收到这份反馈。"
    })


# ============================================================
# 启动
# ============================================================
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)