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
    """Cute fullscreen 2D pixel-art cat face for MILO."""

    def __init__(self):
        self.state = "neutral"
        self.gaze = [0.0, 0.0]
        self.gaze_target = [0.0, 0.0]
        self.running = False
        self.speaking_active = False
        self.lock = threading.RLock()

    def set_state(self, state):
        with self.lock:
            self.state = state if state in VALID_STATES else "neutral"

    def set_speaking_active(self, active):
        with self.lock:
            self.speaking_active = bool(active)

    def set_gaze(self, x, y):
        with self.lock:
            if abs(x) < 0.03:
                x = 0.0
            if abs(y) < 0.03:
                y = 0.0

            self.gaze_target = [
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

        pygame.display.set_caption("MILO")
        clock = pygame.time.Clock()
        canvas = pygame.Surface((160, 90))

        self.running = True
        now = time.monotonic()
        last_frame = now

        next_blink = now + random.uniform(2.4, 4.8)
        blink_until = 0.0

        next_ear_twitch = now + random.uniform(5.0, 9.0)
        ear_twitch_until = 0.0

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
            blend = 1.0 - math.exp(-min(0.1, now - last_frame) * 12.0)
            last_frame = now

            if now >= next_blink:
                blink_until = now + 0.11
                next_blink = now + random.uniform(2.3, 5.2)

            if now >= next_ear_twitch:
                ear_twitch_until = now + 0.22
                next_ear_twitch = now + random.uniform(5.0, 10.0)

            with self.lock:
                state = self.state
                for axis in (0, 1):
                    self.gaze[axis] += (self.gaze_target[axis] - self.gaze[axis]) * blend
                gaze = tuple(self.gaze)
                speaking_active = self.speaking_active

            blink = (
                now < blink_until
                and state not in {"surprised", "thinking"}
            )

            ear_twitch = now < ear_twitch_until

            self._draw_scene(
                canvas,
                state,
                gaze,
                blink,
                ear_twitch,
                speaking_active,
                now,
            )

            scaled = pygame.transform.scale(
                canvas,
                screen.get_size(),
            )

            screen.blit(scaled, (0, 0))
            pygame.display.flip()

            clock.tick(30)

        pygame.mouse.set_visible(True)
        pygame.quit()

    def _draw_scene(
        self,
        surface,
        state,
        gaze,
        blink,
        ear_twitch,
        speaking_active,
        now,
    ):
        palette = {
            "bg": (12, 15, 24),
            "bg2": (18, 23, 35),
            "fur": (229, 161, 78),
            "fur_dark": (150, 89, 50),
            "fur_light": (250, 196, 105),
            "cream": (249, 231, 194),
            "eye": (95, 231, 218),
            "eye_dark": (13, 39, 47),
            "ink": (28, 25, 31),
            "pink": (241, 139, 151),
            "blush": (239, 151, 156),
            "white": (253, 250, 238),
            "blue": (103, 183, 241),
            "purple": (188, 139, 233),
            "yellow": (250, 218, 105),
            "red": (230, 102, 99),
            "green": (116, 224, 154),
        }

        accent = {
            "neutral": palette["eye"],
            "listening": palette["blue"],
            "thinking": palette["purple"],
            "speaking": palette["green"],
            "happy": palette["yellow"],
            "confused": (242, 164, 100),
            "curious": palette["eye"],
            "surprised": palette["yellow"],
            "concerned": palette["red"],
            "sad": (116, 165, 224),
        }.get(
            state,
            palette["eye"],
        )

        surface.fill(
            palette["bg"]
        )

        for y in range(0, 90, 6):
            pygame.draw.line(
                surface,
                palette["bg2"],
                (0, y),
                (159, y),
                1,
            )

        idle_bob = math.sin(
            now * 1.8
        ) * 0.8

        if speaking_active:
            idle_bob += math.sin(
                now * 6.0
            ) * 0.35

        oy = int(
            round(
                idle_bob
            )
        )

        left_tip_y = 11 + oy
        right_tip_y = 11 + oy

        if ear_twitch:
            left_tip_y -= 2
            right_tip_y += 1

        left_ear = [
            (37, 31 + oy),
            (48, left_tip_y),
            (62, 29 + oy),
        ]

        right_ear = [
            (98, 29 + oy),
            (112, right_tip_y),
            (123, 31 + oy),
        ]

        pygame.draw.polygon(
            surface,
            palette["fur_dark"],
            left_ear,
        )

        pygame.draw.polygon(
            surface,
            palette["fur_dark"],
            right_ear,
        )

        pygame.draw.polygon(
            surface,
            palette["pink"],
            [
                (46, 27 + oy),
                (50, 17 + oy),
                (57, 28 + oy),
            ],
        )

        pygame.draw.polygon(
            surface,
            palette["pink"],
            [
                (103, 28 + oy),
                (110, 17 + oy),
                (115, 27 + oy),
            ],
        )

        pygame.draw.rect(
            surface,
            palette["fur_dark"],
            (36, 34 + oy, 88, 31),
        )

        pygame.draw.rect(
            surface,
            palette["fur"],
            (39, 26 + oy, 82, 44),
        )

        pygame.draw.rect(
            surface,
            palette["fur"],
            (44, 22 + oy, 72, 48),
        )

        pygame.draw.rect(
            surface,
            palette["fur_light"],
            (47, 27 + oy, 18, 4),
        )

        pygame.draw.rect(
            surface,
            palette["fur_light"],
            (95, 27 + oy, 18, 4),
        )

        pygame.draw.rect(
            surface,
            palette["fur_dark"],
            (71, 23 + oy, 5, 11),
        )

        pygame.draw.rect(
            surface,
            palette["fur_dark"],
            (79, 23 + oy, 3, 8),
        )

        pygame.draw.rect(
            surface,
            palette["fur_dark"],
            (85, 23 + oy, 5, 11),
        )

        pygame.draw.rect(
            surface,
            palette["cream"],
            (47, 53 + oy, 66, 17),
        )

        pygame.draw.rect(
            surface,
            palette["cream"],
            (43, 58 + oy, 74, 9),
        )

        self._draw_blush(
            surface,
            state,
            palette,
            oy,
            now,
        )

        self._draw_eyes(
            surface,
            state,
            gaze,
            blink,
            accent,
            palette,
            oy,
        )

        self._draw_mouth(
            surface,
            state,
            speaking_active,
            palette,
            oy,
            now,
        )

        self._draw_whiskers(
            surface,
            palette,
            oy,
        )

        self._draw_state_effect(
            surface,
            state,
            accent,
            palette,
            now,
        )

    def _draw_blush(
        self,
        surface,
        state,
        palette,
        oy,
        now,
    ):
        if state not in {
            "happy",
            "speaking",
            "curious",
            "neutral",
        }:
            return

        strength = 1

        if state == "happy":
            strength = 2

        if state == "speaking":
            strength = 1 + int(
                (
                    math.sin(
                        now * 4.0
                    )
                    + 1.0
                )
                > 1.35
            )

        for offset in range(
            strength
        ):
            pygame.draw.rect(
                surface,
                palette["blush"],
                (
                    48 + offset * 3,
                    57 + oy,
                    2,
                    2,
                ),
            )

            pygame.draw.rect(
                surface,
                palette["blush"],
                (
                    110 - offset * 3,
                    57 + oy,
                    2,
                    2,
                ),
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
    ):
        left = (
            62,
            46 + oy,
        )

        right = (
            98,
            46 + oy,
        )

        if state == "happy":
            self._cute_closed_eye(
                surface,
                left,
                accent,
            )

            self._cute_closed_eye(
                surface,
                right,
                accent,
            )

            return

        if state == "sad":
            self._sad_eye(
                surface,
                left,
                accent,
            )

            self._sad_eye(
                surface,
                right,
                accent,
            )

            return

        if state == "concerned":
            pygame.draw.line(
                surface,
                accent,
                (53, 39 + oy),
                (68, 42 + oy),
                2,
            )

            pygame.draw.line(
                surface,
                accent,
                (92, 42 + oy),
                (107, 39 + oy),
                2,
            )

        if blink:
            pygame.draw.rect(
                surface,
                accent,
                (53, 46 + oy, 18, 2),
            )

            pygame.draw.rect(
                surface,
                accent,
                (89, 46 + oy, 18, 2),
            )

            return

        eye_w = 17
        eye_h = 16

        if state == "surprised":
            eye_w = 18
            eye_h = 19

        if state == "thinking":
            pygame.draw.rect(
                surface,
                accent,
                (53, 45 + oy, 18, 3),
            )

            self._open_eye(
                surface,
                right,
                eye_w,
                eye_h,
                gaze,
                accent,
                palette,
            )

            return

        if state == "confused":
            pygame.draw.line(
                surface,
                accent,
                (52, 39 + oy),
                (69, 37 + oy),
                2,
            )

            pygame.draw.line(
                surface,
                accent,
                (91, 37 + oy),
                (108, 40 + oy),
                2,
            )

        self._open_eye(
            surface,
            left,
            eye_w,
            eye_h,
            gaze,
            accent,
            palette,
        )

        self._open_eye(
            surface,
            right,
            eye_w,
            eye_h,
            gaze,
            accent,
            palette,
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
    ):
        cx, cy = center

        eye_rect = pygame.Rect(
            cx - width // 2,
            cy - height // 2,
            width,
            height,
        )

        pygame.draw.rect(
            surface,
            accent,
            eye_rect,
        )

        pygame.draw.rect(
            surface,
            palette["white"],
            (
                cx - width // 2 + 2,
                cy - height // 2 + 2,
                width - 4,
                height - 4,
            ),
        )

        pupil_x = int(
            round(
                gaze[0] * 5
            )
        )

        pupil_y = int(
            round(
                gaze[1] * 3
            )
        )

        pygame.draw.rect(
            surface,
            palette["eye_dark"],
            (
                cx - 3 + pupil_x,
                cy - 4 + pupil_y,
                7,
                9,
            ),
        )

        pygame.draw.rect(
            surface,
            palette["white"],
            (
                cx - 1 + pupil_x,
                cy - 3 + pupil_y,
                2,
                2,
            ),
        )

        pygame.draw.rect(
            surface,
            palette["white"],
            (
                cx + 2 + pupil_x,
                cy + 1 + pupil_y,
                1,
                1,
            ),
        )

    def _cute_closed_eye(
        self,
        surface,
        center,
        color,
    ):
        cx, cy = center

        points = [
            (cx - 8, cy + 2),
            (cx - 4, cy - 2),
            (cx, cy - 4),
            (cx + 4, cy - 2),
            (cx + 8, cy + 2),
        ]

        pygame.draw.lines(
            surface,
            color,
            False,
            points,
            3,
        )

    def _sad_eye(
        self,
        surface,
        center,
        color,
    ):
        cx, cy = center

        points = [
            (cx - 8, cy - 2),
            (cx - 4, cy + 1),
            (cx, cy + 3),
            (cx + 4, cy + 1),
            (cx + 8, cy - 2),
        ]

        pygame.draw.lines(
            surface,
            color,
            False,
            points,
            3,
        )

    def _draw_mouth(
        self,
        surface,
        state,
        speaking_active,
        palette,
        oy,
        now,
    ):
        nose_y = 57 + oy

        pygame.draw.rect(
            surface,
            palette["pink"],
            (77, nose_y, 6, 3),
        )

        pygame.draw.rect(
            surface,
            palette["pink"],
            (79, nose_y + 3, 2, 1),
        )

        pygame.draw.rect(
            surface,
            palette["ink"],
            (79, nose_y + 4, 2, 2),
        )

        if (
            state == "speaking"
            and speaking_active
        ):
            phase = int(
                now * 10.0
            ) % 4

            heights = (
                3,
                6,
                8,
                5,
            )

            mouth_h = heights[
                phase
            ]

            pygame.draw.rect(
                surface,
                palette["ink"],
                (
                    74,
                    nose_y + 7,
                    12,
                    mouth_h,
                ),
            )

            if mouth_h >= 5:
                pygame.draw.rect(
                    surface,
                    palette["pink"],
                    (
                        77,
                        nose_y + 10,
                        6,
                        max(
                            1,
                            mouth_h - 4,
                        ),
                    ),
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

            pygame.draw.rect(
                surface,
                palette["pink"],
                (78, nose_y + 11, 4, 2),
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

    @staticmethod
    def _draw_whiskers(
        surface,
        palette,
        oy,
    ):
        pygame.draw.line(
            surface,
            palette["cream"],
            (45, 61 + oy),
            (31, 58 + oy),
            1,
        )

        pygame.draw.line(
            surface,
            palette["cream"],
            (45, 64 + oy),
            (29, 64 + oy),
            1,
        )

        pygame.draw.line(
            surface,
            palette["cream"],
            (115, 61 + oy),
            (129, 58 + oy),
            1,
        )

        pygame.draw.line(
            surface,
            palette["cream"],
            (115, 64 + oy),
            (131, 64 + oy),
            1,
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
            level = int(
                now * 7
            ) % 4

            for index in range(3):
                height = (
                    3
                    + (
                        (
                            index
                            + level
                        )
                        % 4
                    )
                    * 2
                )

                pygame.draw.rect(
                    surface,
                    accent,
                    (
                        128 + index * 5,
                        45 - height // 2,
                        3,
                        height,
                    ),
                )

        elif state == "thinking":
            phase = int(
                now * 3
            ) % 3

            for index in range(3):
                color = (
                    accent
                    if index == phase
                    else palette["bg2"]
                )

                pygame.draw.rect(
                    surface,
                    color,
                    (
                        128 + index * 6,
                        31,
                        4,
                        4,
                    ),
                )

        elif state == "curious":
            pygame.draw.rect(
                surface,
                accent,
                (131, 28, 4, 4),
            )

            pygame.draw.rect(
                surface,
                accent,
                (135, 24, 4, 4),
            )

            pygame.draw.rect(
                surface,
                accent,
                (139, 28, 4, 8),
            )

            pygame.draw.rect(
                surface,
                accent,
                (137, 39, 4, 4),
            )

        elif state == "surprised":
            pygame.draw.rect(
                surface,
                accent,
                (133, 24, 4, 13),
            )

            pygame.draw.rect(
                surface,
                accent,
                (133, 40, 4, 4),
            )

        elif state == "concerned":
            pygame.draw.rect(
                surface,
                accent,
                (131, 28, 4, 8),
            )

            pygame.draw.rect(
                surface,
                accent,
                (135, 24, 4, 5),
            )

        elif state == "sad":
            drop_y = (
                52
                + int(
                    (
                        now * 8
                    )
                    % 8
                )
            )

            pygame.draw.rect(
                surface,
                accent,
                (109, drop_y, 2, 4),
            )
