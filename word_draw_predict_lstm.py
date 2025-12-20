import pygame
import sys
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.ndimage

# ---------------- CNN definition ----------------

import torch
import torch.nn as nn


class SimpleLSTM(nn.Module):
    def __init__(self, num_classes, input_size=28, hidden_size=128, num_layers=2, bidirectional=True):
        super().__init__()
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.fc = nn.Linear(hidden_size * self.num_directions, num_classes)

    def forward(self, x):
        x = x.squeeze(1)  # (B, 28, 28): seq_len=28, input_size=28
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]  # (B, H * num_directions)
        return self.fc(last_out)


# ---------------- load letter model ----------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

digit_model = SimpleLSTM(num_classes=10).to(device)
digit_model.load_state_dict(torch.load("digit_lstm_10cls.pth", map_location=device))
digit_model.eval()

letter_model = SimpleLSTM(num_classes=26).to(device)
letter_model.load_state_dict(torch.load("letter_lstm_26cls.pth", map_location=device))
letter_model.eval()

# ---------------- Pygame setup ----------------

pygame.init()
WIDTH, HEIGHT = 400, 200  # wider canvas for words
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Draw a WORD (uppercase letters)")
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BRUSH_RADIUS = 4

font = pygame.font.SysFont(None, 32)
prediction_text = ""


def reset_canvas():
    WINDOW.fill(WHITE)


reset_canvas()


# ---------------- segmentation helpers ----------------

def extract_character_boxes(arr, thresh=250, min_pixels=30):
    # arr: 2D grayscale numpy array (H,W)
    mask = (arr < thresh).astype(np.uint8)
    labeled, num = scipy.ndimage.label(mask)
    boxes = []
    for label_id in range(1, num + 1):
        ys, xs = np.where(labeled == label_id)
        if len(xs) < min_pixels:
            continue
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        boxes.append((x_min, y_min, x_max, y_max))
    boxes.sort(key=lambda b: b[0])  # left-to-right
    return boxes


def crop_and_preprocess(arr, box, device):
    x_min, y_min, x_max, y_max = box
    char_arr = arr[y_min:y_max + 1, x_min:x_max + 1]

    crop = Image.fromarray(char_arr)
    crop.thumbnail((20, 20), Image.LANCZOS)

    canvas28 = Image.new("L", (28, 28), color=255)
    cx, cy = canvas28.size[0] // 2, canvas28.size[1] // 2
    w, h = crop.size
    canvas28.paste(crop, (cx - w // 2, cy - h // 2))

    a = np.array(canvas28).astype(np.float32)
    a = 255.0 - a  # invert if training used white bg, black ink
    a /= 255.0
    a = a.reshape(1, 1, 28, 28)
    return torch.from_numpy(a).to(device)


def predict_word(surface, device):
    import pygame

    data = pygame.image.tostring(surface, "RGB")
    img = Image.frombytes("RGB", surface.get_size(), data).convert("L")
    arr = np.array(img).astype(np.uint8)

    boxes = extract_character_boxes(arr)
    if not boxes:
        return "No drawing"

    chars = []
    with torch.no_grad():
        for box in boxes:
            char_tensor = crop_and_preprocess(arr, box, device)  # (1,1,28,28)

            # run both models
            d_logits = digit_model(char_tensor)
            l_logits = letter_model(char_tensor)

            d_probs = torch.softmax(d_logits, dim=1)
            l_probs = torch.softmax(l_logits, dim=1)

            d_conf, d_pred = d_probs.max(dim=1)
            l_conf, l_pred = l_probs.max(dim=1)

            d_conf = d_conf.item()
            l_conf = l_conf.item()
            d_pred = d_pred.item()
            l_pred = l_pred.item()

            if d_conf >= l_conf:
                ch = str(d_pred)  # digit 0–9
            else:
                ch = chr(ord("A") + l_pred)  # letter A–Z

            chars.append(ch)

    return "".join(chars)


# ---------------- main loop ----------------

running = True
drawing = False

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

            if event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                prediction_text = predict_word(WINDOW, device)

    if drawing:
        x, y = pygame.mouse.get_pos()
        pygame.draw.circle(WINDOW, BLACK, (x, y), BRUSH_RADIUS)

    # show prediction at bottom
    if prediction_text:
        text_surface = font.render(f"Prediction: {prediction_text}", True, (105, 125, 255))
        # clear a strip at bottom so text stays readable
        pygame.draw.rect(WINDOW, WHITE, (0, HEIGHT - 40, WIDTH, 40))
        WINDOW.blit(text_surface, (10, HEIGHT - 35))

    pygame.display.flip()

pygame.quit()
sys.exit()
