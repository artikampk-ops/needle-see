"""Synthetic typed-decision suite with exact labels.

Tasks are procedural renders on a 256x256 canvas: colored geometry, UI dialogs
and a small receipt card.  Every example carries its exact truth (computed at
render time, not by re-analysis) in a format the trainer consumes directly and
the runtime reuses through :mod:`needlesee.questions`.

    .venv/bin/python -m needlesee.data.gen --out out/data --count 4000 --seed 1

Writes ``<out>/<split>.jsonl`` plus ``<out>/<split>/images/*.jpg``.  Every task
is sampled round-robin, and each example's RNG is derived from ``(seed, index)``
so a split is reproducible.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..questions import ChoiceQuestion, JudgeQuestion, Question, RateQuestion, SYSTEM, TOOLS

SIZE = 256
FORMAT = "needle-see/suite/v1"

COLORS = {
    "red": (211, 47, 47),
    "green": (46, 160, 67),
    "blue": (25, 101, 194),
    "yellow": (240, 196, 25),
    "purple": (136, 77, 178),
    "orange": (230, 126, 34),
}
COLOR_NAMES = tuple(COLORS)
SHAPES = ("circle", "square", "triangle")
BACKGROUNDS = ((246, 247, 251), (255, 250, 235), (232, 245, 235),
               (240, 235, 250), (250, 232, 234), (38, 42, 56))
DESKTOP_BG = ((62, 82, 120), (92, 112, 148), (48, 58, 82), (108, 92, 130))
DIALOG_KINDS = {
    "Error": {"bar": (198, 40, 40), "icon": (211, 47, 47), "glyph": "x",
              "messages": ("Something went wrong.", "Could not save your file.",
                           "The connection failed.", "The file is not valid.")},
    "Warning": {"bar": (230, 145, 20), "icon": (245, 170, 30), "glyph": "!",
                "messages": ("Your changes are unsaved.", "This cannot be undone.",
                             "The battery is running low.", "You are offline.")},
    "Success": {"bar": (52, 152, 90), "icon": (60, 175, 100), "glyph": "v",
                "messages": ("Your file was saved.", "Upload complete.",
                             "The payment was received.", "Everything is in sync.")},
    "Information": {"bar": (40, 100, 190), "icon": (55, 120, 200), "glyph": "i",
                    "messages": ("A new version is available.", "Your session ends soon.",
                                 "Saved to the cloud.", "Updates are ready to install.")},
}
BUTTON_NAMES = ("OK", "Cancel", "Retry", "Save")
ITEM_NAMES = ("Coffee", "Bagel", "Juice", "Salad", "Muffin", "Latte", "Tea", "Soup")

FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
)

_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}


def load_font(size: int) -> ImageFont.ImageFont:
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    for path in FONT_PATHS:
        try:
            font = ImageFont.truetype(path, size)
            _FONT_CACHE[size] = font
            return font
        except OSError:
            continue
    font = ImageFont.load_default(size) if hasattr(ImageFont, "load_default") else ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


# --------------------------------------------------------------------------- #
# drawing helpers
# --------------------------------------------------------------------------- #
def scatter_positions(rng, n: int, grid: int = 4, margin: int = 40, jitter: int = 14):
    cells = [(gx, gy) for gy in range(grid) for gx in range(grid)]
    rng.shuffle(cells)
    span = SIZE - 2 * margin
    out = []
    for gx, gy in cells[:n]:
        cx = margin + gx * span // (grid - 1) + int(rng.integers(-jitter, jitter + 1))
        cy = margin + gy * span // (grid - 1) + int(rng.integers(-jitter, jitter + 1))
        out.append((int(np.clip(cx, 30, SIZE - 30)), int(np.clip(cy, 30, SIZE - 30))))
    return out


def draw_shape(draw: ImageDraw.ImageDraw, kind: str, color, cx: int, cy: int, size: int):
    half = size // 2
    if kind == "circle":
        draw.ellipse((cx - half, cy - half, cx + half, cy + half), fill=color,
                     outline=tuple(int(c * 0.75) for c in color), width=2)
    elif kind == "square":
        draw.rounded_rectangle((cx - half, cy - half, cx + half, cy + half), radius=6,
                               fill=color,
                               outline=tuple(int(c * 0.75) for c in color), width=2)
    else:  # triangle
        points = [(cx, cy - half - 2), (cx - half - 2, cy + half), (cx + half + 2, cy + half)]
        draw.polygon(points, fill=color, outline=tuple(int(c * 0.75) for c in color))


def new_canvas(rng, desktop: bool = False):
    background = DESKTOP_BG[rng.integers(len(DESKTOP_BG))] if desktop \
        else BACKGROUNDS[rng.integers(len(BACKGROUNDS))]
    image = Image.new("RGB", (SIZE, SIZE), tuple(int(v) for v in background))
    draw = ImageDraw.Draw(image)
    if desktop:  # a faint taskbar strip so dialogs look like dialogs
        draw.rectangle((0, SIZE - 18, SIZE, SIZE), fill=(28, 32, 44))
    return image, draw


def wrap_text(text: str, every: int) -> str:
    words, lines, line = text.split(), [], ""
    for word in words:
        if line and len(line) + 1 + len(word) > every:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return "\n".join(lines)


def draw_dialog(image: Image.Image, draw: ImageDraw.ImageDraw, kind: str,
                message: str, buttons: list[str]):
    spec = DIALOG_KINDS[kind]
    card = (24, 44, 232, 208)
    draw.rectangle((card[0] + 4, card[1] + 5, card[2] + 4, card[3] + 5), fill=(30, 30, 40))
    draw.rounded_rectangle(card, radius=10, fill=(252, 252, 253))
    draw.rounded_rectangle((card[0], card[1], card[2], card[1] + 30), radius=10,
                           fill=spec["bar"])
    draw.rectangle((card[0], card[1] + 20, card[2], card[1] + 30), fill=spec["bar"])
    font_title = load_font(17)
    draw.text((card[0] + 12, card[1] + 6), kind, font=font_title, fill=(255, 255, 255))
    icon_c = (card[0] + 34, card[1] + 68)
    draw.ellipse((icon_c[0] - 17, icon_c[1] - 17, icon_c[0] + 17, icon_c[1] + 17),
                 fill=spec["icon"])
    font_glyph = load_font(20)
    draw.text((icon_c[0], icon_c[1]), spec["glyph"], font=font_glyph,
              fill=(255, 255, 255), anchor="mm")
    font_msg = load_font(15)
    draw.multiline_text((card[0] + 58, card[1] + 48), wrap_text(message, 22),
                        font=font_msg, fill=(40, 42, 48), spacing=4)
    font_btn = load_font(14)
    x_right = card[2] - 12
    for label in reversed(buttons):
        width = int(draw.textlength(label, font=font_btn)) + 22
        x0 = x_right - width
        draw.rounded_rectangle((x0, card[3] - 34, x_right, card[3] - 10), radius=6,
                               fill=(238, 240, 245), outline=(190, 196, 206), width=1)
        draw.text(((x0 + x_right) / 2, card[3] - 22), label, font=font_btn,
                  fill=(40, 42, 48), anchor="mm")
        x_right = x0 - 8
    return image


# --------------------------------------------------------------------------- #
# rendering realism controls (set by the CLI or by callers via set_render_config)
NOISE_MODE = "none"
ROTATE_RANGE = 0.0


def set_render_config(noise="none", rotate=0.0):
    global NOISE_MODE, ROTATE_RANGE
    NOISE_MODE, ROTATE_RANGE = noise, float(rotate)


def humanize_image(image, rng, mode=None, rotate_range=None):
    mode = mode if mode is not None else NOISE_MODE
    rr = ROTATE_RANGE if rotate_range is None else rotate_range
    if mode == "none" and rr == 0:
        return image
    if rr:
        fill = BACKGROUNDS[int(rng.integers(len(BACKGROUNDS)))]
        image = image.rotate(float(rng.uniform(-rr, rr)), resample=Image.BICUBIC,
                             expand=False, fillcolor=tuple(int(v) for v in fill))
    if mode == "none":
        return image
    arr = np.asarray(image, np.int16)
    if mode in ("light", "scan"):
        scale = 0.5 if mode == "light" else 1.0
        arr = arr + rng.integers(-int(10 * scale) - 1, int(10 * scale) + 2,
                                 arr.shape, dtype=np.int16)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        image = Image.fromarray(arr)
    if mode == "scan" and rng.random() < 0.7:
        arr = np.array(image, dtype=np.uint8)
        density = 0.0006
        n = max(1, int(arr.shape[0] * arr.shape[1] * density))
        ys = rng.integers(0, arr.shape[0], size=n)
        xs = rng.integers(0, arr.shape[1], size=n)
        arr[ys, xs] = 0
        image = Image.fromarray(arr)
    return image


# --------------------------------------------------------------------------- #
# text rendering primitives
WORDS = ("the", "of", "and", "to", "in", "a", "is", "for", "on", "with", "that",
         "it", "by", "from", "as", "this", "be", "at", "which", "or", "party",
         "owner", "agent", "agreement", "property", "effective", "date", "lease",
         "rent", "term", "state", "landlord", "tenant")
COLORS_DISTINCT = ("red", "green", "blue")


def render_text_page(rng, lines, font_size=20, color=(36, 38, 44)):
    image, draw = new_canvas(rng)
    font = load_font(font_size)
    y = 34
    for line in lines:
        draw.text((28, y), line, font=font, fill=color)
        y += font_size + 6
    draw.line((22, 26, 234, 26), fill=(150, 150, 160), width=1)
    return image, draw


# --------------------------------------------------------------------------- #
# tasks
@dataclass
class Example:
    image: Image.Image
    question: Question
    truth_index: int
    meta: dict


def _sample_shapes(rng, n: int, size_lo: int = 44, size_hi: int = 58,
                   kinds=None, colors=None) -> list[dict]:
    positions = scatter_positions(rng, n)
    out = []
    for (cx, cy) in positions:
        out.append({
            "kind": str(rng.choice(kinds if kinds is not None else SHAPES)),
            "color": str(rng.choice(colors if colors is not None else COLOR_NAMES)),
            "cx": cx, "cy": cy, "size": int(rng.integers(size_lo, size_hi)),
        })
    return out


def task_color_of_shape(rng) -> Example:
    shape = _sample_shapes(rng, 1, 78, 104)[0]
    image, draw = new_canvas(rng)
    draw_shape(draw, shape["kind"], COLORS[shape["color"]], shape["cx"], shape["cy"], shape["size"])
    question = ChoiceQuestion(f"What color is the {shape['kind']}?", COLOR_NAMES, task="color_of_shape")
    return Example(image, question, COLOR_NAMES.index(shape["color"]),
                   {"color": shape["color"], "shape": shape["kind"]})


def task_shape_of_color(rng) -> Example:
    shape = _sample_shapes(rng, 1, 78, 104)[0]
    image, draw = new_canvas(rng)
    draw_shape(draw, shape["kind"], COLORS[shape["color"]], shape["cx"], shape["cy"], shape["size"])
    question = ChoiceQuestion(f"What shape is the {shape['color']} object?", SHAPES, task="shape_of_color")
    return Example(image, question, SHAPES.index(shape["kind"]),
                   {"color": shape["color"], "shape": shape["kind"]})


def task_count_color(rng) -> Example:
    target = COLOR_NAMES[int(rng.integers(len(COLOR_NAMES)))]
    n_target = int(rng.integers(0, 4))
    others = [c for c in COLOR_NAMES if c != target]
    n_other = int(rng.integers(0, 3 - min(n_target, 2) + 1))
    shapes = _sample_shapes(rng, n_target, colors=[target])
    shapes += _sample_shapes(rng, n_other, colors=others)
    rng.shuffle(shapes)
    image, draw = new_canvas(rng)
    for s in shapes:
        draw_shape(draw, s["kind"], COLORS[s["color"]], s["cx"], s["cy"], s["size"])
    options = ("0", "1", "2", "3")
    question = ChoiceQuestion(f"How many {target} shapes are in the image?", options,
                              task="count_color")
    return Example(image, question, options.index(str(n_target)),
                   {"target_color": target, "count": n_target, "total": len(shapes)})


def task_has_color_shape(rng) -> Example:
    n = int(rng.integers(1, 4))
    shapes = _sample_shapes(rng, n)
    present = {(s["color"], s["kind"]) for s in shapes}
    if rng.random() < 0.5 and present:
        color, kind = rng.choice(sorted(present))
    else:
        absent = [(c, k) for c in COLOR_NAMES for k in SHAPES if (c, k) not in present]
        color, kind = absent[int(rng.integers(len(absent)))]
    truth = 1 if (color, kind) in present else 0
    image, draw = new_canvas(rng)
    for s in shapes:
        draw_shape(draw, s["kind"], COLORS[s["color"]], s["cx"], s["cy"], s["size"])
    question = JudgeQuestion(f"Is there a {color} {kind} in the image?", task="has_color_shape")
    return Example(image, question, truth,
                   {"asked": [color, kind], "present": sorted([list(p) for p in present])})


def task_count_shapes(rng) -> Example:
    n = int(rng.integers(0, 4))
    shapes = _sample_shapes(rng, n)
    image, draw = new_canvas(rng)
    for s in shapes:
        draw_shape(draw, s["kind"], COLORS[s["color"]], s["cx"], s["cy"], s["size"])
    levels = ("none", "one shape", "two shapes", "three or more shapes")
    question = RateQuestion("How busy is the image?", levels, task="count_shapes")
    return Example(image, question, min(n, 3), {"count": n})


def task_dialog_type(rng) -> Example:
    kinds = tuple(DIALOG_KINDS)
    kind = kinds[int(rng.integers(len(kinds)))]
    spec = DIALOG_KINDS[kind]
    message = str(rng.choice(spec["messages"]))
    button_sets = (("OK",), ("Cancel",), ("OK", "Cancel"), ("Save", "Cancel"),
                   ("Retry", "Cancel"))
    buttons = list(button_sets[int(rng.integers(len(button_sets)))])
    image, draw = new_canvas(rng, desktop=True)
    draw_dialog(image, draw, kind, message, buttons)
    question = ChoiceQuestion("What kind of dialog is shown?", kinds, task="dialog_type")
    return Example(image, question, kinds.index(kind),
                   {"kind": kind, "message": message, "buttons": buttons})


def task_dialog_button(rng) -> Example:
    kind = tuple(DIALOG_KINDS)[int(rng.integers(len(DIALOG_KINDS)))]
    spec = DIALOG_KINDS[kind]
    target = str(rng.choice(BUTTON_NAMES))
    others = [b for b in BUTTON_NAMES if b != target]
    present = bool(rng.integers(2))
    if present:
        buttons = [target] if rng.random() < 0.4 else [str(rng.choice(others)), target]
    else:
        buttons = [str(rng.choice(others))]
        if rng.random() < 0.5:
            second = str(rng.choice([b for b in others if b != buttons[0]]))
            buttons.append(second)
    image, draw = new_canvas(rng, desktop=True)
    draw_dialog(image, draw, kind, str(rng.choice(spec["messages"])), buttons)
    question = JudgeQuestion(f"Does the dialog have a {target} button?", task="dialog_button")
    return Example(image, question, 1 if target in buttons else 0,
                   {"target": target, "buttons": buttons})


def task_receipt_total(rng) -> Example:
    n_items = int(rng.integers(1, 3))
    names = [str(x) for x in rng.choice(ITEM_NAMES, size=n_items, replace=False)]
    prices = np.round(rng.uniform(2.0, 8.0, n_items), 2)
    total = round(float(prices.sum()), 2)
    total_text = f"${total:.2f}"

    distractors = []
    while len(distractors) < 3:
        delta = float(rng.uniform(0.5, 3.5)) * (1 if rng.random() < 0.5 else -1)
        candidate = f"${max(0.5, total + delta):.2f}"
        if candidate != total_text and candidate not in distractors:
            distractors.append(candidate)
    options = tuple([total_text] + distractors)
    order = list(rng.permutation(len(options)))
    options = tuple(options[i] for i in order)

    image, draw = new_canvas(rng)
    card = (28, 22, 228, 234)
    draw.rectangle((card[0] + 3, card[1] + 4, card[2] + 3, card[3] + 4), fill=(60, 60, 70))
    draw.rectangle(card, fill=(253, 252, 247))
    font_title = load_font(19)
    draw.text(((card[0] + card[2]) / 2, card[1] + 14), "RECEIPT", font=font_title,
              fill=(35, 35, 40), anchor="mm")
    font_row = load_font(21)
    y = card[1] + 48
    for name, price in zip(names, prices):
        draw.text((card[0] + 16, y), name, font=font_row, fill=(45, 45, 50))
        draw.text((card[2] - 16, y), f"${price:.2f}", font=font_row,
                  fill=(45, 45, 50), anchor="ra")
        y += 34
    draw.line((card[0] + 14, y + 4, card[2] - 14, y + 4), fill=(150, 150, 158), width=2)
    font_total = load_font(26)
    draw.text((card[0] + 16, y + 18), "TOTAL", font=font_total, fill=(20, 20, 25))
    draw.text((card[2] - 16, y + 18), total_text, font=font_total,
              fill=(20, 20, 25), anchor="ra")
    question = ChoiceQuestion("What is the total amount on the receipt?", options,
                              task="receipt_total")
    return Example(image, question, options.index(total_text),
                   {"items": names, "prices": [float(p) for p in prices], "total": total,
                    "options": list(options)})


def task_contains_text(rng) -> Example:
    has_text = bool(rng.integers(2))
    if has_text:
        lines = [" ".join(str(x) for x in rng.choice(WORDS, size=int(rng.integers(5, 9)), replace=True))
                 for _ in range(int(rng.integers(2, 5)))]
        image, _ = render_text_page(rng, lines, font_size=int(rng.integers(15, 21)))
    else:
        if rng.random() < 0.5:
            image, draw = new_canvas(rng)
            for _ in range(int(rng.integers(1, 4))):
                draw_shape(draw, str(rng.choice(SHAPES)), COLORS[str(rng.choice(COLOR_NAMES))],
                           int(rng.integers(40, SIZE - 40)), int(rng.integers(40, SIZE - 40)),
                           int(rng.integers(40, 70)))
        else:
            image = Image.new("RGB", (SIZE, SIZE), BACKGROUNDS[int(rng.integers(len(BACKGROUNDS)))])
    question = JudgeQuestion("Does the page contain readable text (words or sentences)?", task="contains_text")
    return Example(image, question, 1 if has_text else 0, {"has_text": has_text})


def task_is_blank(rng) -> Example:
    if rng.random() < 0.5:
        image = Image.new("RGB", (SIZE, SIZE), (248, 248, 250))
        truth = 1
    else:
        image, draw = new_canvas(rng)
        for _ in range(int(rng.integers(1, 5))):
            draw_shape(draw, str(rng.choice(SHAPES)), COLORS[str(rng.choice(COLOR_NAMES))],
                       int(rng.integers(40, SIZE - 40)), int(rng.integers(40, SIZE - 40)),
                       int(rng.integers(30, 60)))
        truth = 0
    question = JudgeQuestion("Is the image completely blank (no shapes, text, or dialog)?", task="is_blank")
    return Example(image, question, truth, {"blank": bool(truth)})


def task_contains_word(rng) -> Example:
    target = WORDS[int(rng.integers(len(WORDS)))]
    n_occurrences = 0 if rng.random() < 0.45 else int(rng.integers(1, 4))
    lines = []
    for _ in range(int(rng.integers(2, 4))):
        words = [str(x) for x in rng.choice(WORDS, size=7, replace=True)]
        lines.append(" ".join(words))
    # inject occurrences on random lines to test reading
    for _ in range(n_occurrences):
        li = int(rng.integers(len(lines)))
        words = lines[li].split()
        pos = int(rng.integers(0, len(words)))
        words[pos] = target
        lines[li] = " ".join(words)
    image, _ = render_text_page(rng, lines, font_size=int(rng.integers(16, 22)))
    truth = 1 if n_occurrences > 0 else 0
    question = JudgeQuestion(f"Does the text contain the word '{target}' anywhere?", task="contains_word")
    return Example(image, question, truth, {"target": target, "occurrences": n_occurrences})


def task_count_word(rng) -> Example:
    target = WORDS[int(rng.integers(len(WORDS)))]
    n = int(rng.integers(0, 4))
    lines = []
    for _ in range(int(rng.integers(2, 5))):
        words = [str(x) for x in rng.choice(WORDS, size=6, replace=True)]
        lines.append(" ".join(words))
    for _ in range(n):
        li = int(rng.integers(len(lines)))
        words = lines[li].split()
        pos = int(rng.integers(0, len(words)))
        words[pos] = target
        lines[li] = " ".join(words)
    image, _ = render_text_page(rng, lines, font_size=int(rng.integers(15, 20)))
    question = RateQuestion(f"How many times does the word '{target}' appear on the page?",
                            ("none", "once", "twice", "three or more"), task="count_word")
    return Example(image, question, min(n, 3), {"target": target, "count": n})


def task_document_topic(rng) -> Example:
    topic_words = {
        "lease": ("rent", "tenant", "landlord", "lease", "property", "term", "month"),
        "invoice": ("invoice", "pay", "amount", "due", "receipt", "total", "date"),
        "notice": ("notice", "expiry", "warning", "expiry", "deadline", "action"),
    }
    topic = str(rng.choice(tuple(topic_words)))
    lines = []
    for _ in range(int(rng.integers(3, 6))):
        pool = topic_words[topic] + tuple(str(x) for x in rng.choice(WORDS, size=4))
        words = [str(x) for x in rng.choice(pool, size=8, replace=True)]
        lines.append(" ".join(words))
    image, _ = render_text_page(rng, lines, font_size=int(rng.integers(16, 21)))
    options = tuple(topic_words)
    question = ChoiceQuestion("What kind of document is this?", options, task="document_topic")
    return Example(image, question, tuple(topic_words).index(topic), {"topic": topic})


def task_line_count(rng) -> Example:
    n = int(rng.integers(1, 5))
    lines = []
    for _ in range(n):
        words = [str(x) for x in rng.choice(WORDS, size=rng.integers(4, 9), replace=True)]
        lines.append(" ".join(words))
    image, _ = render_text_page(rng, lines, font_size=int(rng.integers(16, 22)))
    question = RateQuestion("How many lines of text are on the page?",
                            ("none", "one", "two", "three or more"),
                            task="line_count")
    return Example(image, question, min(n, 3), {"lines": n})


TASK_FUNCS = (
    task_color_of_shape,
    task_shape_of_color,
    task_count_color,
    task_has_color_shape,
    task_count_shapes,
    task_dialog_type,
    task_dialog_button,
    task_receipt_total,
    task_contains_word,
    task_count_word,
    task_document_topic,
    task_line_count,
    task_contains_text,
    task_is_blank,
)
TASK_NAMES = tuple(f.__name__.removeprefix("task_") for f in TASK_FUNCS)


# --------------------------------------------------------------------------- #
# data writing
# --------------------------------------------------------------------------- #
def build_example(seed: int, index: int, task: str | None = None) -> Example:
    name = task or TASK_NAMES[index % len(TASK_NAMES)]
    func = TASK_FUNCS[TASK_NAMES.index(name)]
    rng = np.random.default_rng([int(seed), int(index)])
    return func(rng)


def generate(out_dir: str | Path, count: int, seed: int, split: str = "train",
             size: int = SIZE, quality: int = 92, noise: str = NOISE_MODE,
             rotate: float = ROTATE_RANGE) -> Path:
    set_render_config(noise, rotate)
    out_dir = Path(out_dir)
    images_dir = out_dir / split / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"{split}.jsonl"
    manifest = {"format": FORMAT, "split": split, "seed": seed, "count": count,
                "tasks": list(TASK_NAMES), "image_size": size, "noise": noise,
                "rotate": rotate}
    with jsonl_path.open("w") as handle:
        for index in range(count):
            example = build_example(seed, index)
            image = example.image
            image = humanize_image(image, np.random.default_rng([int(seed), int(index), 101]),
                                    mode=noise, rotate_range=rotate)
            if image.size != (size, size):
                image = image.resize((size, size), Image.LANCZOS)
            name = f"{index:06d}.jpg"
            image.save(images_dir / name, quality=quality)
            line = {
                "id": f"{split}-{index:06d}",
                "task": example.question.task,
                "image": f"{split}/images/{name}",
                "system": SYSTEM,
                "query": example.question.query(),
                "tools": TOOLS,
                "answers": example.question.answers(example.truth_index),
                "truth_index": example.truth_index,
                "candidates": example.question.candidate_texts(),
                "meta": example.meta,
            }
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
    (out_dir / f"{split}.manifest.json").write_text(json.dumps(manifest, indent=2))
    return jsonl_path


def contact_sheet(seed: int, out_path: str | Path, count: int = 8, size: int = 220):
    """A quick visual grid of one example per task (for eyeballing)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(count):
        example = build_example(seed, i)
        image = example.image.resize((size, size), Image.LANCZOS)
        rows.append(image)
    sheet = Image.new("RGB", (size * len(rows), size), (255, 255, 255))
    for i, image in enumerate(rows):
        sheet.paste(image, (i * size, 0))
    sheet.save(out_path)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="generate the synthetic suite")
    parser.add_argument("--out", default="out/data")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--size", type=int, default=SIZE)
    parser.add_argument("--noise", default="none", choices=("none", "light", "scan"))
    parser.add_argument("--rotate", type=float, default=0.0)
    parser.add_argument("--sheet", default=None,
                        help="write a contact sheet instead of a dataset")
    args = parser.parse_args()
    if args.sheet:
        print(contact_sheet(args.seed, args.sheet))
        return 0
    path = generate(args.out, args.count, args.seed, args.split, args.size,
                    noise=args.noise, rotate=args.rotate)
    print(f"{path}  {args.count} examples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
