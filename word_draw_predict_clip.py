import pygame
import sys
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import scipy.ndimage
import clip

# ================= CONFIG =================
WIDTH, HEIGHT = 1000, 500
BRUSH_RADIUS = 10
FONT_SIZE = 48
PRED_Y = HEIGHT - 80

MODEL_PATH = "clip_chars74k_36cls.pth"  # your new saved model
CHAR_MAP = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (80, 110, 255)

DEBUG_DIR = "debug_chars"
os.makedirs(DEBUG_DIR, exist_ok=True)

# ================= MODEL =================
class ClipCharClassifier(nn.Module):
    def __init__(self, clip_model, num_classes=36):
        super().__init__()
        self.clip_model = clip_model
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        with torch.no_grad():
            feats = self.clip_model.encode_image(x)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return self.fc(feats.float())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

model = ClipCharClassifier(clip_model).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ================= PYGAME =================
pygame.init()
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Draw Letters / Words")
font = pygame.font.SysFont(None, FONT_SIZE)

def reset_canvas():
    WINDOW.fill(WHITE)

reset_canvas()
prediction_text = ""
drawing = False

# ================= SEGMENTATION =================
def extract_character_boxes(arr, thresh=240, min_pixels=400):
    mask = (arr < thresh).astype(np.uint8)
    mask = scipy.ndimage.binary_closing(mask, iterations=2).astype(np.uint8)
    mask = scipy.ndimage.binary_opening(mask, iterations=1).astype(np.uint8)

    labeled, num = scipy.ndimage.label(mask)
    boxes = []

    for label_id in range(1, num + 1):
        ys, xs = np.where(labeled == label_id)
        if len(xs) < min_pixels:
            continue
        boxes.append((xs.min(), ys.min(), xs.max(), ys.max()))

    boxes.sort(key=lambda b: b[0])
    return boxes

def crop_and_preprocess_clip(arr, box, idx):
    x_min, y_min, x_max, y_max = box
    char_arr = arr[y_min:y_max + 1, x_min:x_max + 1]

    crop = Image.fromarray(char_arr)
    crop.thumbnail((20, 20), Image.LANCZOS)

    canvas28 = Image.new("L", (28, 28), color=255)
    cx, cy = 14, 14
    w, h = crop.size
    canvas28.paste(crop, (cx - w // 2, cy - h // 2))

    # ---- SAVE DEBUG IMAGE ----
    debug_path = os.path.join(DEBUG_DIR, f"char_{idx:02d}.png")
    canvas28.save(debug_path)

    img_rgb = canvas28.convert("RGB")
    return clip_preprocess(img_rgb).unsqueeze(0).to(device)

# ================= PREDICTION =================
def predict_character(img_tensor):
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        return CHAR_MAP[pred.item()], conf.item()

def predict_word(surface):
    data = pygame.image.tostring(surface, "RGB")
    img = Image.frombytes("RGB", (WIDTH, HEIGHT), data).convert("L")
    arr = np.array(img).astype(np.uint8)

    boxes = extract_character_boxes(arr)
    if not boxes:
        return "No drawing"

    preds = []
    for i, box in enumerate(boxes):
        img_tensor = crop_and_preprocess_clip(arr, box, i + 1)
        char, conf = predict_character(img_tensor)
        preds.append(char)

    return "".join(preds)

# ================= MAIN LOOP =================
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                prediction_text = ""
                reset_canvas()
            elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                word = predict_word(WINDOW)
                prediction_text = f"Prediction: {word}"

    if drawing:
        pygame.draw.circle(WINDOW, BLACK, pygame.mouse.get_pos(), BRUSH_RADIUS)

    if prediction_text:
        pygame.draw.rect(WINDOW, WHITE, (0, PRED_Y - 10, WIDTH, 90))
        pred_surf = font.render(prediction_text, True, BLUE)
        WINDOW.blit(pred_surf, (20, PRED_Y))

    pygame.display.flip()

pygame.quit()
sys.exit()
