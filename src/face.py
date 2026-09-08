import math
import random
import threading
import time

import pygame

COLORS = {
    "neutral": (90, 210, 230), "listening": (80, 180, 255), "thinking": (190, 140, 255),
    "speaking": (120, 240, 170), "happy": (255, 215, 95), "confused": (255, 150, 120),
    "curious": (120, 220, 255), "surprised": (255, 245, 150), "concerned": (255, 160, 130),
    "sad": (110, 160, 230),
}


class Face:
    def __init__(self):
        self.state = "neutral"
        self.gaze = [0.0, 0.0]
        self.running = False
        self.lock = threading.Lock()

    def set_state(self, state):
        with self.lock:
            self.state = state if state in COLORS else "neutral"

    def set_gaze(self, x, y):
        with self.lock:
            if abs(x) < 0.08: x = 0.0
            if abs(y) < 0.08: y = 0.0
            self.gaze = [max(-1, min(1, x)), max(-1, min(1, y))]

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        clock = pygame.time.Clock()
        self.running = True
        blink_until, next_blink = 0, time.time() + random.uniform(2, 5)
        while self.running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    self.running = False
            now = time.time()
            if now > next_blink:
                blink_until = now + 0.12
                next_blink = now + random.uniform(2.5, 6.5)
            with self.lock:
                state, gaze = self.state, tuple(self.gaze)
            w, h = screen.get_size()
            screen.fill((7, 10, 14))
            color = COLORS.get(state, COLORS["neutral"])
            idle = math.sin(now * 1.4) * h * 0.01
            self._draw_eye(screen, (w * 0.35, h * 0.48 + idle), min(w, h) * 0.16, gaze, color, now < blink_until, state)
            self._draw_eye(screen, (w * 0.65, h * 0.48 - idle), min(w, h) * 0.16, gaze, color, now < blink_until, state)
            pygame.display.flip()
            clock.tick(60)
        pygame.quit()

    def _draw_eye(self, s, c, r, gaze, color, blink, state):
        x, y = c
        if blink:
            pygame.draw.line(s, color, (x - r, y), (x + r, y), max(3, int(r * 0.08)))
            return
        squint = 0.65 if state in {"happy", "sad", "concerned"} else 1.0
        rect = pygame.Rect(0, 0, r * 2, r * 2 * squint)
        rect.center = (x, y)
        pygame.draw.ellipse(s, color, rect)
        px = x + gaze[0] * r * 0.35
        py = y + gaze[1] * r * 0.22
        pygame.draw.circle(s, (5, 8, 12), (int(px), int(py)), int(r * (0.34 if state != "surprised" else 0.24)))
