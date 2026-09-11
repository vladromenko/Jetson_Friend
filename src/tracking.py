"""Small detector-rate IoU associator. Ambiguous crossings start fresh tracks.

This does not infer identity or claim continuous optical-flow tracking.
"""


def iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / area if area > 0 else 0.0


class FaceTracks:
    def __init__(self, threshold=0.35, max_gap=1.0):
        self.threshold = threshold
        self.max_gap = max_gap
        self.tracks = {}
        self.next_id = 1

    def update(self, detections, now):
        old = {
            tid: item
            for tid, item in self.tracks.items()
            if now - item["seen"] <= self.max_gap
        }
        candidates = [
            [
                tid
                for tid, item in old.items()
                if iou(d["box"], item["box"]) >= self.threshold
            ]
            for d in detections
        ]
        result = []
        for detection, matches in zip(detections, candidates):
            unique = len(matches) == 1 and sum(matches[0] in m for m in candidates) == 1
            if unique:
                tid = matches[0]
            else:
                tid = self.next_id
                self.next_id += 1
            result.append({**detection, "track_id": tid, "detected_at": now})
        # Never associate across a missed detection. Return recognition must verify again.
        self.tracks = {d["track_id"]: {"box": d["box"], "seen": now} for d in result}
        return result
