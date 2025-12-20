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
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ---------------- load letter model ----------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

digit_model = SimpleCNN(num_classes=10).to(device)
digit_model.load_state_dict(torch.load("digit_cnn_10cls.pth", map_location=device))
digit_model.eval()

letter_model = SimpleCNN(num_classes=26).to(device)
letter_model.load_state_dict(torch.load("letter_cnn_26cls.pth", map_location=device))
letter_model.eval()

# ---------------- Pygame setup ----------------

pygame.init()
WIDTH, HEIGHT = 400, 200  # wider canvas for words
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Draw a WORD (digits or uppercase letters)")
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
    """
    arr: 2D grayscale numpy array (H,W)
    returns list of (x_min, y_min, x_max, y_max) for each connected component, left-to-right.
    """
    # binary mask: 1 = ink, 0 = background
    mask = (arr < thresh).astype(np.uint8)

    labeled, num = scipy.ndimage.label(mask)
    boxes = []

    for label_id in range(1, num + 1):
        ys, xs = np.where(labeled == label_id)
        if len(xs) < min_pixels:
            continue  # skip tiny noise
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        boxes.append((x_min, y_min, x_max, y_max))

    # sort left-to-right
    boxes.sort(key=lambda b: b[0])
    return boxes


def crop_and_preprocess(arr, box):
    """
    arr: 2D grayscale numpy array
    box: (x_min, y_min, x_max, y_max)
    returns tensor of shape (1,1,28,28)
    """
    x_min, y_min, x_max, y_max = box
    char_arr = arr[y_min:y_max + 1, x_min:x_max + 1]

    crop = Image.fromarray(char_arr)
    crop.thumbnail((20, 20), Image.LANCZOS)

    canvas28 = Image.new("L", (28, 28), color=255)
    cx, cy = canvas28.size[0] // 2, canvas28.size[1] // 2
    w, h = crop.size
    canvas28.paste(crop, (cx - w // 2, cy - h // 2))

    a = np.array(canvas28).astype(np.float32)
    a = 255.0 - a  # invert to match training; remove if needed
    a /= 255.0
    a = a.reshape(1, 1, 28, 28)
    return torch.from_numpy(a).to(device)


def predict_word(surface):
    """
    Takes the whole canvas, splits it into characters, predicts each one
    using digit_model and letter_model, and returns the predicted string.
    """
    data = pygame.image.tostring(surface, "RGB")
    img = Image.frombytes("RGB", (WIDTH, HEIGHT), data).convert("L")
    arr = np.array(img).astype(np.uint8)

    boxes = extract_character_boxes(arr)
    if not boxes:
        return "No drawing"

    chars = []
    with torch.no_grad():
        for box in boxes:
            char_tensor = crop_and_preprocess(arr, box)  # (1,1,28,28)

            # run both models, like in your digit/letter script
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
                ch = str(d_pred)  # 0–9
            else:
                ch = chr(ord("A") + l_pred)  # A–Z

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
                prediction_text = predict_word(WINDOW)

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
