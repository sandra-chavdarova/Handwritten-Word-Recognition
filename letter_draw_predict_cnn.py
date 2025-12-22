import pygame
import sys
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1   = nn.Linear(128 * 7 * 7, 256)
        self.fc2   = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

digit_model = SimpleCNN(num_classes=10).to(device)
digit_state = torch.load("digit_cnn_10cls.pth", map_location=device)
digit_model.load_state_dict(digit_state)
digit_model.eval()

letter_model = SimpleCNN(num_classes=26).to(device)
letter_state = torch.load("letter_cnn_26cls.pth", map_location=device)
letter_model.load_state_dict(letter_state)
letter_model.eval()


# pygame
pygame.init()
WIDTH, HEIGHT = 280, 280  # 10x 28x28
WINDOW = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Draw a digit or letter")
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BRUSH_RADIUS = 4

font = pygame.font.SysFont(None, 40)
prediction_text = ""

def reset_canvas():
    WINDOW.fill(WHITE)

reset_canvas()



def preprocess_surface(surface):
    data = pygame.image.tostring(surface, "RGB")
    img = Image.frombytes("RGB", (WIDTH, HEIGHT), data).convert("L")

    arr = np.array(img).astype(np.uint8)

    # Find bounding box of drawing
    thresh = 250
    ys, xs = np.where(arr < thresh)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    arr = arr[y_min:y_max+1, x_min:x_max+1]

    crop = Image.fromarray(arr)
    crop.thumbnail((20, 20), Image.LANCZOS)

    canvas28 = Image.new("L", (28, 28), color=255)
    cx, cy = canvas28.size[0] // 2, canvas28.size[1] // 2
    w, h = crop.size
    canvas28.paste(crop, (cx - w // 2, cy - h // 2))

    arr = np.array(canvas28).astype(np.float32)
    arr = 255.0 - arr    # if this gives bad results, remove this line
    arr /= 255.0

    arr = arr.reshape(1, 1, 28, 28)
    tensor = torch.from_numpy(arr).to(device)
    return tensor



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
                img_tensor = preprocess_surface(WINDOW)
                if img_tensor is None:
                    prediction_text = "No drawing"
                else:
                    with torch.no_grad():
                        d_logits = digit_model(img_tensor)
                        l_logits = letter_model(img_tensor)

                        d_probs = torch.softmax(d_logits, dim=1)
                        l_probs = torch.softmax(l_logits, dim=1)

                        d_conf, d_pred = d_probs.max(dim=1)
                        l_conf, l_pred = l_probs.max(dim=1)

                    d_conf = d_conf.item()
                    l_conf = l_conf.item()
                    d_pred = d_pred.item()
                    l_pred = l_pred.item()

                    if d_conf >= l_conf:
                        prediction_text = f"{d_pred}"
                    else:
                        ch = chr(ord("A") + l_pred)
                        prediction_text = f"{ch}"

    if drawing:
        x, y = pygame.mouse.get_pos()
        pygame.draw.circle(WINDOW, BLACK, (x, y), BRUSH_RADIUS)

    if prediction_text:
        text_surface = font.render(f"Prediction: {prediction_text}", True, (105, 125, 255))
        WINDOW.blit(text_surface, (10, 10))

    pygame.display.flip()

pygame.quit()
sys.exit()