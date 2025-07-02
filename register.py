import cv2
import numpy as np
import pytesseract
import os
import json
import re
import itertools
from PIL import ImageFont, ImageDraw, Image
from datetime import datetime
import pygame

pygame.mixer.init()
# success_sound = pygame.mixer.Sound("project_mp3/{your mp3}")

COLOR_NAME_TO_RGB = {
    '빨강': (255, 0, 0),
    '주황': (255, 127, 0),
    '노랑': (255, 255, 0),
    '초록': (0, 255, 0),
    '파랑': (0, 0, 255),
    '보라': (127, 0, 255),
    '핑크': (255, 105, 180),
}

PASTEL_COLOR_NAME_TO_RGB = {
    '빨강': (255, 102, 102),
    '주황': (255, 178, 102),
    '노랑': (255, 255, 153),
    '초록': (153, 255, 153),
    '파랑': (153, 204, 255),
    '보라': (204, 153, 255),
    '핑크': (255, 153, 204),
}

pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
os.makedirs("data/faces", exist_ok=True)
os.makedirs("data/card", exist_ok=True)
if not os.path.isfile("data/people.json"):
    with open("data/people.json", "w", encoding="utf-8") as f:
        json.dump([], f, indent=2, ensure_ascii=False)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def draw_text_korean(img, text, position=(10, 30), font_path="C:/Windows/Fonts/malgun.ttf", font_size=24, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] 
    rect[2] = pts[np.argmax(s)]  
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  
    rect[3] = pts[np.argmax(diff)]  
    return rect

def cluster_coordinates_opencv(coords, cluster_count=2):
    coords = np.array(coords).astype(np.float32).reshape(-1, 1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    flags = cv2.KMEANS_PP_CENTERS

    compactness, labels, centers = cv2.kmeans(
        coords, cluster_count, None, criteria, 10, flags
    )
    centers = np.sort(centers.flatten().astype(int)).tolist()
    return centers

def detect_card_warp(image):
    img = image.copy()
    img = cv2.resize(img, (400, 300))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=30)
    _, thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)

    grad_x = cv2.Sobel(thresh, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.subtract(grad_x, grad_y)
    grad = cv2.convertScaleAbs(grad)

    vertical = grad.copy()
    v_size = max(10, vertical.shape[0] // 20)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))
    vertical = cv2.erode(vertical, v_kernel)
    vertical = cv2.dilate(vertical, v_kernel)

    horizontal = grad.copy()
    h_size = max(10, horizontal.shape[1] // 20)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))
    horizontal = cv2.erode(horizontal, h_kernel)
    horizontal = cv2.dilate(horizontal, h_kernel)

    ys_v, xs_v = np.where(vertical == 255)
    ys_h, xs_h = np.where(horizontal == 255)

    cleaned_xs = cluster_coordinates_opencv(xs_v)
    cleaned_ys = cluster_coordinates_opencv(ys_h)

    if len(cleaned_xs) >= 2 and len(cleaned_ys) >= 2:
        left, right = cleaned_xs[0], cleaned_xs[-1]
        top, bottom = cleaned_ys[0], cleaned_ys[-1]
        
        pts_src = np.array([
            [left, top],   
            [right, top],   
            [right, bottom],
            [left, bottom]  
        ], dtype=np.float32)
        
        w, h = 700, 400
        dst_pts = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(pts_src, dst_pts)
        warped = cv2.warpPerspective(img, M, (w, h))
        return warped

    else:
        print("선이 충분하지 않아 꼭짓점을 만들 수 없습니다.")
        return None

def enhance_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)
    blur = cv2.medianBlur(gray, 3)
    sharpened = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
    thresh = cv2.adaptiveThreshold(
        sharpened, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 15
    )
    return thresh

def extract_info(img):
    processed = enhance_for_ocr(img)
    text = pytesseract.image_to_string(processed, lang='kor+eng')

    lines = [line.strip() for line in text.split('\n') if line.strip()]
    name, company, email = "Unknown", "Unknown", "Unknown"

    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+", text)
    if email_match:
        email = email_match.group(0)

    if lines:
        raw_name = lines[0]
        name = re.sub(r"[^\w가-힣]", "", raw_name).strip()

    exclude_keywords = [name, email]
    filtered_lines = [
        line for line in lines 
        if not any(keyword in line for keyword in exclude_keywords)
    ]

    if filtered_lines:
        company_raw = filtered_lines[0]
        company = re.sub(r"[^\w가-힣A-Za-z\s]", "", company_raw).strip()

    return name, email, company

def preprocess_for_ocr(img):
    img = cv2.resize(img, (800, 600))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=2.0, beta=0)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def confirm_or_edit(field_name, value):
    print(f"{field_name} 추출값: '{value}'")
    new_value = input(f"→ {field_name} 수정 (그대로 두려면 엔터): ")
    return new_value.strip() if new_value.strip() else value

def user_color_pipeline():
    valid_colors = ['빨강', '주황', '노랑', '초록', '파랑', '보라', '핑크']
    
    print("좋아하는 색상 3가지를 선택하세요 (빨강,주황,노랑,초록,파랑,보라,핑크 중에서)")
    selected_colors = []
    while len(selected_colors) < 3:
        color = input(f"{len(selected_colors)+1}번째 색상: ").strip()
        if color not in valid_colors:
            print("다시 입력하세요! 올바른 색상만 입력 가능")
            continue
        if color in selected_colors:
            print("중복 선택 불가! 다른 색을 선택하세요.")
            continue
        selected_colors.append(color)
    
    print("\n파스텔 톤이 좋으신가요? 원색이 좋으신가요? (입력: '파스텔' or '원색')")
    tone = ""
    while tone not in ['파스텔', '원색']:
        tone = input("톤 선택: ").strip()
    
    if tone == '원색':
        rgb_list = [list(COLOR_NAME_TO_RGB[c]) for c in selected_colors]
    else:
        rgb_list = [list(PASTEL_COLOR_NAME_TO_RGB[c]) for c in selected_colors]
    
    print(f"\n선택된 색상: {selected_colors}, {tone}")
    
    return rgb_list

def save_to_json(person):
    import shutil

    path = "data/people.json"
    backup_path = "data/people_backup.json"

    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            data = []
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("people.json 내용이 리스트 형식이 아님")
    except Exception as e:
        print(f"JSON 로딩 오류: {e}")

        if os.path.exists(path):
            shutil.copy(path, backup_path)
            print(f"기존 파일을 백업했습니다: {backup_path}")
        data = []

    data.append(person)

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("JSON 저장 완료")
    except Exception as e:
        print(f"JSON 저장 실패: {e}")


def save_image_unicode(path, image):
    try:
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img_pil.save(path)
        return True
    except Exception as e:
        print(f"이미지 저장 실패: {path}\n오류: {e}")
        return False
    
def show_final_result(face_img, card_img):
    
    target_height = 400
    face_resized = cv2.resize(face_img, (int(face_img.shape[1] * (target_height / face_img.shape[0])), target_height))
    card_resized = cv2.resize(card_img, (int(card_img.shape[1] * (target_height / card_img.shape[0])), target_height))

    stacked = np.hstack([card_resized, face_resized])

    # success_sound.play()

    cv2.imshow("register success!", stacked)
    cv2.waitKey(1600)  
    cv2.destroyAllWindows()

def main():
    colors = user_color_pipeline()

    cap = cv2.VideoCapture(0)
    print("[1단계] 명함을 정면으로 비추고 스페이스바를 눌러주세요")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        display = frame.copy()
        display = draw_text_korean(display, "캡쳐 [스페이스] / 취소 [ESC]", position=(10, 30), font_size=28, color=(0, 255, 255))
        cv2.imshow("Capture Card", display)
        key = cv2.waitKey(1)
        if key == 32:
            card_img = frame.copy()
            print("\U0001f4f8 명함 캡처 완료!")
            break
        elif key == 27:
            print("캡처 취소됨")
            cap.release()
            cv2.destroyAllWindows()
            exit()
    cap.release()
    cv2.destroyAllWindows()

    pts = detect_card_warp(card_img)
    if pts is not None:
        card_img = pts
        print("명함 투시변환 성공!")
    else:
        print("명함 인식 실패 - 원본 이미지로 OCR 시도")

    name, email, company = extract_info(card_img)

    print("\n[2단계] 명함 정보 추출 완료 (수정 가능)")
    name = confirm_or_edit("이름", name)
    email = confirm_or_edit("이메일", email)
    company = confirm_or_edit("회사", company)

    # colors = user_color_pipeline()

    print("\n[2단계] 명함 정보:")
    print(f"  이름: {name}")
    print(f"  이메일: {email}")
    print(f"  회사: {company}")

    cap = cv2.VideoCapture(0)
    print("\n[3단계] 얼굴을 정면으로 비추고 스페이스바를 눌러주세요 (ESC로 취소)")
    face_img = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
        largest_face = None
        max_area = 0
        for (x, y, w, h) in faces:
            area = w * h
            if area > max_area:
                max_area = area
                largest_face = (x, y, w, h)
            cv2.rectangle(display, (x, y), (x + w, y + h), (100, 100, 255), 1)

        if largest_face:
            (x, y, w, h) = largest_face
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        display = draw_text_korean(display, "캡처 [스페이스] / 취소 [ESC]", position=(10, 30), font_size=28, color=(255, 255, 0))
        cv2.imshow("Capture Face", display)

        key = cv2.waitKey(1)
        if key == 32 and largest_face:
            face_img = frame[y:y + h, x:x + w]
            print("얼굴 캡처 완료!")
            break
        elif key == 27:
            print("얼굴 캡처 취소됨")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    cap.release()
    cv2.destroyAllWindows()

    timestamp = int(datetime.now().timestamp())
    safe_name = name.strip().replace(" ", "_")

    face_path = f"data/faces/{safe_name}_{timestamp}.jpg"
    card_path = f"data/card/{safe_name}_{timestamp}.jpg"
    save_image_unicode(face_path, face_img)
    save_image_unicode(card_path, card_img)

    person = {
        "name": name,
        "email": email,
        "company": company,
        "colors": colors,
        "face_img": face_path,
        "card_img": card_path
    }

    save_to_json(person)
    show_final_result(face_img, card_img)
    print(f"\n등록 완료! [{name}] 정보가 저장되었습니다.")
    
if __name__ == "__main__":
    main()
