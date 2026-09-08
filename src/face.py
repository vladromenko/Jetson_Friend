import math
import random
import threading
import time

import pygame


VALID_STATES = {
    "neutral",
    "listening",
    "thinking",
    "speaking",
    "happy",
    "confused",
    "curious",
    "surprised",
    "concerned",
    "sad",
}


class Face:
    """Fullscreen retro pixel-art cat face drawn entirely in pygame."""

    def __init__(self):
        self.state = "neutral"
        self.gaze = [0.0, 0.0]
        self.running = False
        self.lock = threading.Lock()

    def set_state(self, state):
        with self.lock:
            self.state = state if state in VALID_STATES else "neutral"

    def set_gaze(self, x, y):
        with self.lock:
            if abs(x) < 0.03:
                x = 0.0
            if abs(y) < 0.03:
                y = 0.0
            self.gaze = [
                max(-1.0, min(1.0, float(x))),
                max(-1.0, min(1.0, float(y))),
            ]

    def stop(self):
        self.running = False

    def run(self):
        pygame.init()
        pygame.mouse.set_visible(False)

        try:
            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        except pygame.error:
            screen = pygame.display.set_mode((960, 540))

        pygame.display.set_caption("Hugh")
        clock = pygame.time.Clock()

        # We draw on a tiny canvas and scale with nearest-neighbor.
        # That gives a true retro-game pixel-art look at any monitor size.
        canvas = pygame.Surface((160, 90))

        self.running = True
        now = time.monotonic()
        next_blink = now + random.uniform(2.5, 5.0)
        blink_until = 0.0

        while self.running:
            for event in pygame.event.get():
                close_requested = event.type == pygame.QUIT
                escape_requested = (
                    event.type == pygame.KEYDOWN
                    and event.key == pygame.K_ESCAPE
                )
                if close_requested or escape_requested:
                    self.running = False

            now = time.monotonic()

            if now >= next_blink:
                blink_until = now + 0.10
                next_blink = now + random.uniform(2.5, 6.0)

            with self.lock:
                state = self.state
                gaze = tuple(self.gaze)

            blink = now < blink_until and state not in {
                "surprised",
                "thinking",
            }

            self._draw_scene(
                canvas,
                state,
                gaze,
                blink,
                now,
            )

            screen_size = screen.get_size()
            scaled = pygame.transform.scale(
                canvas,
                screen_size,
            )
            screen.blit(scaled, (0, 0))
            pygame.display.flip()
            clock.tick(30)

        pygame.mouse.set_visible(True)
        pygame.quit()

    def _draw_scene(self, surface, state, gaze, blink, now):
        palette = {
            "bg": (12, 15, 24),
            "bg2": (18, 23, 35),
            "fur": (225, 158, 72),
            "fur_dark": (154, 91, 51),
            "fur_light": (246, 190, 97),
            "cream": (247, 226, 184),
            "eye": (82, 222, 210),
            "eye_dark": (14, 45, 52),
            "ink": (24, 24, 31),
            "pink": (232, 121, 132),
            "white": (249, 245, 228),
            "blue": (91, 175, 235),
            "purple": (181, 130, 225),
            "yellow": (248, 213, 94),
            "red": (226, 92, 88),
        }

        accent = {
            "neutral": palette["eye"],
            "listening": palette["blue"],
            "thinking": palette["purple"],
            "speaking": (100, 220, 145),
            "happy": palette["yellow"],
            "confused": (241, 156, 91),
            "curious": palette["eye"],
            "surprised": palette["yellow"],
            "concerned": palette["red"],
            "sad": (105, 155, 220),
        }.get(state, palette["eye"])

        surface.fill(palette["bg"])

        # Simple scanline/grid ambience.
        for y in range(0, 90, 6):
            pygame.draw.line(
                surface,
                palette["bg2"],
                (0, y),
                (159, y),
                1,
            )

        bob = int(round(math.sin(now * 2.0) * 1.2))
        ox = 0
        oy = bob

        # Ears.
        left_ear = [(39 + ox, 31 + oy), (48 + ox, 12 + oy), (61 + ox, 28 + oy)]
        right_ear = [(99 + ox, 28 + oy), (112 + ox, 12 + oy), (121 + ox, 31 + oy)]
        pygame.draw.polygon(surface, palette["fur_dark"], left_ear)
        pygame.draw.polygon(surface, palette["fur_dark"], right_ear)

        left_inner = [(47, 26 + oy), (50, 18 + oy), (56, 27 + oy)]
        right_inner = [(104, 27 + oy), (110, 18 + oy), (113, 26 + oy)]
        pygame.draw.polygon(surface, palette["pink"], left_inner)
        pygame.draw.polygon(surface, palette["pink"], right_inner)

        # Head: blocky/pixel silhouette.
        head = pygame.Rect(38, 25 + oy, 84, 48)
        pygame.draw.rect(surface, palette["fur"], head)
        pygame.draw.rect(surface, palette["fur_dark"], (38, 35 + oy, 5, 25))
        pygame.draw.rect(surface, palette["fur_dark"], (117, 35 + oy, 5, 25))
        pygame.draw.rect(surface, palette["fur_light"], (45, 29 + oy, 18, 5))
        pygame.draw.rect(surface, palette["fur_light"], (97, 29 + oy, 18, 5))

        # Forehead stripes.
        pygame.draw.rect(surface, palette["fur_dark"], (71, 25 + oy, 5, 12))
        pygame.draw.rect(surface, palette["fur_dark"], (79, 25 + oy, 4, 9))
        pygame.draw.rect(surface, palette["fur_dark"], (86, 25 + oy, 5, 12))

        # Cheeks / muzzle.
        pygame.draw.rect(surface, palette["cream"], (49, 53 + oy, 62, 18))
        pygame.draw.rect(surface, palette["cream"], (44, 58 + oy, 72, 10))

        self._draw_eyes(
            surface,
            state,
            gaze,
            blink,
            accent,
            palette,
            oy,
            now,
        )
        self._draw_mouth(
            surface,
            state,
            palette,
            oy,
            now,
        )
        self._draw_state_effect(
            surface,
            state,
            accent,
            palette,
            now,
        )

    def _draw_eyes(
        self,
        surface,
        state,
        gaze,
        blink,
        accent,
        palette,
        oy,
        now,
    ):
        left = (62, 47 + oy)
        right = (98, 47 + oy)

        if state == "happy":
            self._pixel_arc(surface, left, accent, upward=True)
            self._pixel_arc(surface, right, accent, upward=True)
            return

        if state == "sad":
            self._pixel_arc(surface, left, accent, upward=False)
            self._pixel_arc(surface, right, accent, upward=False)
            return

        if state == "concerned":
            pygame.draw.line(surface, accent, (54, 41 + oy), (68, 44 + oy), 2)
            pygame.draw.line(surface, accent, (92, 44 + oy), (106, 41 + oy), 2)

        if blink:
            pygame.draw.rect(surface, accent, (54, 47 + oy, 16, 2))
            pygame.draw.rect(surface, accent, (90, 47 + oy, 16, 2))
            return

        eye_w = 14
        eye_h = 13

        if state == "surprised":
            eye_w = 16
            eye_h = 17

        if state == "thinking":
            # One eye slightly squinted.
            pygame.draw.rect(surface, accent, (54, 46 + oy, 16, 3))
            self._open_eye(
                surface,
                right,
                eye_w,
                eye_h,
                gaze,
                accent,
                palette,
                oy,
            )
            return

        if state == "confused":
            pygame.draw.line(surface, accent, (53, 41 + oy), (68, 39 + oy), 2)
            pygame.draw.line(surface, accent, (91, 39 + oy), (107, 42 + oy), 2)

        self._open_eye(
            surface,
            left,
            eye_w,
            eye_h,
            gaze,
            accent,
            palette,
            oy,
        )
        self._open_eye(
            surface,
            right,
            eye_w,
            eye_h,
            gaze,
            accent,
            palette,
            oy,
        )

    def _open_eye(
        self,
        surface,
        center,
        width,
        height,
        gaze,
        accent,
        palette,
        oy,
    ):
        cx, cy = center
        rect = pygame.Rect(
            cx - width // 2,
            cy - height // 2,
            width,
            height,
        )
        pygame.draw.rect(surface, accent, rect)

        pupil_x = int(round(gaze[0] * 6))
        pupil_y = int(round(gaze[1] * 4))
        pygame.draw.rect(
            surface,
            palette["eye_dark"],
            (
                cx - 2 + pupil_x,
                cy - 3 + pupil_y,
                5,
                7,
            ),
        )
        pygame.draw.rect(
            surface,
            palette["white"],
            (
                cx - 1 + pupil_x,
                cy - 2 + pupil_y,
                1,
                2,
            ),
        )

    def _pixel_arc(self, surface, center, color, upward):
        cx, cy = center
        if upward:
            points = [
                (cx - 8, cy + 2),
                (cx - 4, cy - 2),
                (cx, cy - 4),
                (cx + 4, cy - 2),
                (cx + 8, cy + 2),
            ]
        else:
            points = [
                (cx - 8, cy - 2),
                (cx - 4, cy + 2),
                (cx, cy + 4),
                (cx + 4, cy + 2),
                (cx + 8, cy - 2),
            ]
        pygame.draw.lines(surface, color, False, points, 3)

    def _draw_mouth(self, surface, state, palette, oy, now):
        nose_y = 58 + oy
        pygame.draw.rect(surface, palette["pink"], (77, nose_y, 6, 4))
        pygame.draw.rect(surface, palette["ink"], (79, nose_y + 4, 2, 3))

        if state == "speaking":
            frame = int(now * 7) % 2
            mouth_h = 7 if frame == 0 else 4
            pygame.draw.rect(
                surface,
                palette["ink"],
                (74, nose_y + 7, 12, mouth_h),
            )
            pygame.draw.rect(
                surface,
                palette["pink"],
                (77, nose_y + 10, 6, max(1, mouth_h - 3)),
            )
        elif state == "happy":
            pygame.draw.line(
                surface,
                palette["ink"],
                (79, nose_y + 7),
                (74, nose_y + 10),
                2,
            )
            pygame.draw.line(
                surface,
                palette["ink"],
                (81, nose_y + 7),
                (86, nose_y + 10),
                2,
            )
        elif state == "surprised":
            pygame.draw.rect(
                surface,
                palette["ink"],
                (77, nose_y + 7, 6, 7),
            )
        elif state == "sad":
            pygame.draw.line(
                surface,
                palette["ink"],
                (75, nose_y + 12),
                (80, nose_y + 8),
                2,
            )
            pygame.draw.line(
                surface,
                palette["ink"],
                (80, nose_y + 8),
                (85, nose_y + 12),
                2,
            )
        elif state == "confused":
            pygame.draw.line(
                surface,
                palette["ink"],
                (75, nose_y + 10),
                (85, nose_y + 8),
                2,
            )
        else:
            pygame.draw.line(
                surface,
                palette["ink"],
                (75, nose_y + 8),
                (80, nose_y + 10),
                2,
            )
            pygame.draw.line(
                surface,
                palette["ink"],
                (80, nose_y + 10),
                (85, nose_y + 8),
                2,
            )

    def _draw_state_effect(
        self,
        surface,
        state,
        accent,
        palette,
        now,
    ):
        if state == "listening":
            level = int(now * 6) % 4
            for index in range(3):
                height = 3 + ((index + level) % 4) * 2
                pygame.draw.rect(
                    surface,
                    accent,
                    (126 + index * 5, 45 - height // 2, 3, height),
                )

        elif state == "thinking":
            phase = int(now * 3) % 3
            for index in range(3):
                color = accent if index == phase else palette["bg2"]
                pygame.draw.rect(
                    surface,
                    color,
                    (128 + index * 6, 31, 4, 4),
                )

        elif state == "curious":
            pygame.draw.rect(surface, accent, (130, 28, 4, 4))
            pygame.draw.rect(surface, accent, (134, 24, 4, 4))
            pygame.draw.rect(surface, accent, (138, 28, 4, 8))
            pygame.draw.rect(surface, accent, (136, 39, 4, 4))

        elif state == "surprised":
            pygame.draw.rect(surface, accent, (132, 24, 4, 13))
            pygame.draw.rect(surface, accent, (132, 40, 4, 4))

        elif state == "concerned":
            pygame.draw.rect(surface, accent, (130, 28, 4, 8))
            pygame.draw.rect(surface, accent, (134, 24, 4, 5))

        elif state == "sad":
            drop_y = 52 + int((now * 8) % 8)
            pygame.draw.rect(surface, accent, (108, drop_y, 2, 4))
