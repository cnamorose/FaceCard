import cv2
import numpy as np
import json
from PIL import ImageFont, ImageDraw, Image
import os
import pygame

pygame.mixer.init()
# success_sound = pygame.mixer.Sound("C:/Users/gofla/OneDrive/바탕 화면/project_mp3/3-01MainTheme.mp3")

DB_PATH = "data/people.json"
FONT_PATH = "C:/Windows/Fonts/malgunbd.ttf"

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

def overlay_image(background, overlay, x, y, overlay_size=None):
    bg_h, bg_w = background.shape[:2]
    if overlay_size is not None:
        overlay = cv2.resize(overlay, overlay_size)

    h, w = overlay.shape[:2]

    if x + w > bg_w or y + h > bg_h:
        return background

    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0
        overlay_rgb = overlay[:, :, :3]
    else:
        alpha = np.ones((h, w))
        overlay_rgb = overlay

    for c in range(3):
        background[y:y+h, x:x+w, c] = (1 - alpha) * background[y:y+h, x:x+w, c] + alpha * overlay_rgb[:, :, c]

    return background

def load_database():
    if not os.path.exists(DB_PATH):
        print("데이터베이스 파일 없음")
        return []

    with open(DB_PATH, "r", encoding="utf-8") as f:
        try:
            people = json.load(f)
        except json.JSONDecodeError:
            print("JSON 파싱 오류")
            return []

    db = []
    for person in people:
        try:
            pil_img = Image.open(person["face_img"]).convert("L") 
            face_img = np.array(pil_img)
        except Exception as e:
            print(f"이미지 로딩 실패: {person['face_img']} - {e}")
            continue
        if face_img is None:
            continue
        kp, des = orb.detectAndCompute(face_img, None)
        db.append({"des": des, "info": person})
    return db

def match_face(des, db):
    best_score = 0
    matched_info = None
    for entry in db:
        if entry["des"] is None or des is None:
            continue
        matches = bf.match(des, entry["des"])
        good_matches = [m for m in matches if m.distance < 50]
        score = len(good_matches)
        if score > best_score and score > 10:
            best_score = score
            matched_info = entry["info"]
    return matched_info

def interpolate_color(c1, c2, t):
    c1 = np.array(c1, dtype=np.float32)
    c2 = np.array(c2, dtype=np.float32)
    color = c1 + (c2 - c1) * t
    return tuple(map(int, color))

def blend_three_colors(colors, t):
    if t <= 0.5:
        return interpolate_color(colors[0], colors[1], t * 2)
    else:
        return interpolate_color(colors[1], colors[2], (t - 0.5) * 2)

def get_text_width(font, text):
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0]

def draw_info(frame, x, y, info, letter_spacing=0):
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(FONT_PATH, 20)

    texts = [info['name'], info['email'], info['company']]
    colors = info.get('colors', [[0, 0, 0]] * 3)

    for i, text in enumerate(texts):
        total_width = sum(get_text_width(font, ch) + letter_spacing for ch in text) - letter_spacing
        cur_x = x

        for j, char in enumerate(text):
            char_width = get_text_width(font, char)
            t_pos = (cur_x - x) / total_width
            blended_color = blend_three_colors(colors, t_pos)
            color_rgb = (blended_color[2], blended_color[1], blended_color[0])

            draw.text((cur_x, y - 24 + i * 24), char, font=font, fill=color_rgb)
            cur_x += char_width + letter_spacing

    return np.array(img_pil)

def main():
    db = load_database()
    if not db:
        print("등록된 얼굴이 없습니다.")
        return
    
    animal1 = cv2.imread("C:\image\project_cam_image2.png", cv2.IMREAD_UNCHANGED)

    cap = cv2.VideoCapture(0)

    # success_sound.play()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame_h, frame_w = frame.shape[:2]
        x = 0
        y = frame_h - 100
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        frame = overlay_image(frame, animal1, x=400, y=280, overlay_size=(200, 200))

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            kp, des = orb.detectAndCompute(roi, None)
            matched = match_face(des, db)

            if matched:
                frame = draw_info(frame, x, y, matched)
            else:
                img_pil = Image.fromarray(frame)
                draw = ImageDraw.Draw(img_pil)
                font = ImageFont.truetype(FONT_PATH, 20)
                draw.text((x, y - 30), "등록되지 않음", font=font, fill=(0, 0, 255))
                frame = np.array(img_pil)

        cv2.imshow("Face Card", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()

