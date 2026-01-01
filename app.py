from flask import Flask, render_template, request
import cv2
import os
import numpy as np

app = Flask(__name__)

FOUND_DIR = "uploads_found"
LOST_DIR = "uploads_lost"

os.makedirs(FOUND_DIR, exist_ok=True)
os.makedirs(LOST_DIR, exist_ok=True)

def image_similarity(img1_path, img2_path):
    img1 = cv2.imread(img1_path, 0)
    img2 = cv2.imread(img2_path, 0)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    return len(matches)

@app.route("/found", methods=["GET", "POST"])
def found():
    if request.method == "POST":
        file = request.files["image"]
        path = os.path.join(FOUND_DIR, file.filename)
        file.save(path)
        return "อัปโหลดของที่พบเรียบร้อย"
    return render_template("upload_found.html")

@app.route("/lost", methods=["GET", "POST"])
def lost():
    if request.method == "POST":
        file = request.files["image"]
        lost_path = os.path.join(LOST_DIR, file.filename)
        file.save(lost_path)

        best_match = None
        best_score = 0

        for found_img in os.listdir(FOUND_DIR):
            found_path = os.path.join(FOUND_DIR, found_img)
            score = image_similarity(lost_path, found_path)

            if score > best_score:
                best_score = score
                best_match = found_img

        if best_score > 20:  # ค่า threshold
            return render_template(
                "result.html",
                found=True,
                image=best_match,
                score=best_score
            )
        else:
            return render_template("result.html", found=False)

    return render_template("upload_lost.html")

if __name__ == "__main__":
    app.run(debug=True)
