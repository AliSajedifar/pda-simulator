import os
import sys
import json
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
import ctypes
from ctypes import wintypes

import requests

from PyQt6.QtCore import (
    Qt, QTimer, QVariantAnimation, QPointF,
    QSequentialAnimationGroup, QThread, pyqtSignal
)
from PyQt6.QtGui import (
    QPolygonF, QBrush, QPen, QFont, QColor, QPainter
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout,
    QLabel, QLineEdit, QPushButton,
    QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsSimpleTextItem,
    QGraphicsPolygonItem,
    QMessageBox, QFrame, QSplitter,
    QHeaderView, QTextBrowser, QTableView
)
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtWidgets import QStyledItemDelegate
from PyQt6.QtCore import QModelIndex


# -------------------------------------------------------------------------
# If environment variable GEMINI_API_KEYS doesn't set properly in your IDE,
# paste keys here temporarily (unsafe for sharing, ok for local use).
# Format: "KEY1,KEY2,KEY3"
# -------------------------------------------------------------------------
RAW_KEYS = ""


# ----------------------------- Utilities -----------------------------

EPS_TOKENS = {
    "ε", "\u03b5", "\\u03b5",
    "eps", "epsilon",
    "λ", "\u03bb", "lambda",
    "ϵ", "\u03f5"
}

def normalize_eps(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        if s == "":
            return None
        low = s.lower()
        if s in EPS_TOKENS or low in EPS_TOKENS:
            return None
        if low in {"<eps>", "<epsilon>", "(eps)", "(epsilon)"}:
            return None
        return s
    return str(value)


def html_escape(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;"))


def one_line(s: str) -> str:
    s = (s or "").replace("\r", " ").replace("\n", " ").strip()
    while "  " in s:
        s = s.replace("  ", " ")
    return s


# ----------------------------- PDA Data Model -----------------------------

@dataclass
class PDATransition:
    from_state: str
    input_sym: Optional[str]   # None means epsilon
    stack_top: str
    to_state: str
    action: str                # push|pop|noop|accept
    push_symbol: Optional[str] = None


@dataclass
class PDAConfig:
    state: str
    i: int
    stack: List[str]
    last_transition: str = ""
    last_rule_idx: int = -1
    last_action: str = ""


class PDACore:
    """
    Deterministic PDA simulator.

    IMPORTANT:
    - Epsilon transitions are only allowed when input is finished (sym is None).
    - Epsilon transitions can have push/pop/noop/accept (Option 2).
    """
    def __init__(self):
        self.language_text = "Language: L = { a^n b^n | n ≥ 0 }   |   Alphabet: {a, b}"
        self.alphabet = ["a", "b"]

        self.start_state = "q0"
        self.accept_state = "qacc"
        self.reject_state = "qrej"

        self.bottom = "$"
        self.ai_error = ""

        self.transitions: List[PDATransition] = [
            PDATransition("q0", "a", "$", "q0", "push", "A"),
            PDATransition("q0", "a", "A", "q0", "push", "A"),
            PDATransition("q0", "b", "A", "q1", "pop", None),
            PDATransition("q1", "b", "A", "q1", "pop", None),
            PDATransition("q0", None, "$", "qacc", "accept", None),
            PDATransition("q1", None, "$", "qacc", "accept", None),
        ]
        self._rebuild_delta()

    def _rebuild_delta(self):
        self.delta: Dict[Tuple[str, Optional[str], str], Tuple[PDATransition, int]] = {}
        for idx, tr in enumerate(self.transitions):
            k = (tr.from_state, tr.input_sym, tr.stack_top)
            if k in self.delta:
                raise ValueError(f"Non-deterministic / duplicate transition for {k}.")
            self.delta[k] = (tr, idx)

    def load_from_spec(self, spec: Dict[str, Any]):
        self.ai_error = str(spec.get("error", "")).strip()

        lang = one_line(str(spec.get("language_text", "")).strip())
        alphabet = spec.get("alphabet", [])

        if not lang or not isinstance(alphabet, list) or len(alphabet) == 0:
            raise ValueError("Invalid AI spec: missing language_text or alphabet.")

        if len(lang) > 140:
            lang = lang[:140].rstrip()

        self.language_text = f"Language: {lang}   |   Alphabet: {{{', '.join([str(x) for x in alphabet])}}}"
        self.alphabet = [str(x) for x in alphabet]

        self.start_state = str(spec.get("start_state", "q0"))
        self.accept_state = str(spec.get("accept_state", "qacc"))
        self.reject_state = str(spec.get("reject_state", "qrej"))
        self.bottom = str(spec.get("stack_bottom", "$"))

        transitions = spec.get("transitions", [])
        if not isinstance(transitions, list):
            raise ValueError("Invalid AI spec: transitions must be a list.")

        if len(transitions) == 0:
            self.transitions = []
            self._rebuild_delta()
            return

        parsed: List[PDATransition] = []
        for t in transitions:
            fs = t.get("from")
            ins = t.get("input")
            top = t.get("stack_top")
            ts = t.get("to")
            act = t.get("action")

            ins_parsed = normalize_eps(ins)
            if not all([fs, top, ts, act]):
                raise ValueError("Invalid AI spec: each transition must include from, stack_top, to, action.")

            act = str(act).lower().strip()
            push_sym = t.get("push_symbol", None)

            if act == "push":
                if not push_sym or str(push_sym).strip() == "":
                    raise ValueError("Invalid AI spec: push transition missing push_symbol.")
                push_sym = str(push_sym).strip()
                if len(push_sym) != 1:
                    raise ValueError("Invalid AI spec: push_symbol must be single-character.")
            else:
                push_sym = None

            if act not in ("push", "pop", "noop", "accept"):
                raise ValueError(f"Invalid AI spec: unsupported action '{act}'.")

            parsed.append(PDATransition(str(fs), ins_parsed, str(top), str(ts), act, push_sym))

        self.transitions = parsed
        self._rebuild_delta()

    def initial_config(self) -> PDAConfig:
        return PDAConfig(state=self.start_state, i=0, stack=[self.bottom])

    def is_done(self, tape: str, cfg: PDAConfig) -> bool:
        if cfg.state == self.accept_state and cfg.i == len(tape):
            return True
        return cfg.state in (self.accept_state, self.reject_state)

    def step(self, tape: str, cfg: PDAConfig) -> PDAConfig:
        if self.is_done(tape, cfg):
            return cfg

        sym = tape[cfg.i] if cfg.i < len(tape) else None
        top = cfg.stack[-1] if cfg.stack else self.bottom
        shown_sym = sym if sym is not None else "ε"

        # 1) Try normal (consuming) transition first: δ(q, sym, top)
        key = (cfg.state, sym, top)
        if key in self.delta:
            tr, ridx = self.delta[key]
            return self._apply_transition(tape, cfg, tr, ridx, sym, top, shown_sym)

        # 2) Then try epsilon transition even if input is NOT finished: δ(q, ε, top)
        key_eps = (cfg.state, None, top)
        if key_eps in self.delta:
            tr, ridx = self.delta[key_eps]

            # Critical rule:
            # ε-ACCEPT is allowed ONLY when input is fully consumed.
            if tr.action == "accept" and cfg.i != len(tape):
                # Do NOT accept early; fall through to reject
                pass
            else:
                # Apply ε transition properly (including push/pop/noop effects)
                before_state = cfg.state
                before_i = cfg.i
                before_stack = cfg.stack.copy()

                new_cfg = self._apply_transition(tape, cfg, tr, ridx, None, top, "ε")

                # Safety guard: prevent infinite ε-noop loops that change nothing
                if (new_cfg.state == before_state and
                        new_cfg.i == before_i and
                        new_cfg.stack == before_stack and
                        tr.action in ("noop",)):
                    return PDAConfig(
                        state=self.reject_state,
                        i=cfg.i,
                        stack=cfg.stack.copy(),
                        last_transition=f"ε-loop detected at state {cfg.state} → reject",
                        last_rule_idx=-1,
                        last_action="reject"
                    )

                return new_cfg

        # 3) No valid move -> reject
        return PDAConfig(
            state=self.reject_state,
            i=cfg.i,
            stack=cfg.stack.copy(),
            last_transition=f"δ({cfg.state}, {shown_sym}, {top}) = undefined → reject",
            last_rule_idx=-1,
            last_action="reject"
        )

    def _apply_transition(self, tape: str, cfg: PDAConfig, tr: PDATransition, ridx: int,
                          sym: Optional[str], top: str, shown_sym: str) -> PDAConfig:
        new_state = tr.to_state
        new_stack = cfg.stack.copy()
        new_i = cfg.i

        if tr.action == "push":
            new_stack.append(tr.push_symbol)
            if sym is not None:
                new_i += 1
            txt = f"δ({cfg.state}, {shown_sym}, {top}) → ({new_state}, push {tr.push_symbol})"

        elif tr.action == "pop":
            if not new_stack or new_stack[-1] != tr.stack_top:
                return PDAConfig(
                    state=self.reject_state,
                    i=cfg.i,
                    stack=cfg.stack.copy(),
                    last_transition="pop failed → reject",
                    last_rule_idx=-1,
                    last_action="reject"
                )
            new_stack.pop()
            if sym is not None:
                new_i += 1
            txt = f"δ({cfg.state}, {shown_sym}, {top}) → ({new_state}, pop)"

        elif tr.action == "noop":
            if sym is not None:
                new_i += 1
            txt = f"δ({cfg.state}, {shown_sym}, {top}) → ({new_state}, noop)"

        elif tr.action == "accept":
            # accept does not consume input
            txt = f"δ({cfg.state}, ε, {top}) → ({new_state}, accept)"

        else:
            txt = f"δ({cfg.state}, {shown_sym}, {top}) → ({new_state}, {tr.action})"

        return PDAConfig(
            state=new_state,
            i=new_i,
            stack=new_stack,
            last_transition=txt,
            last_rule_idx=ridx,
            last_action=tr.action
        )


# ----------------------------- Transition Table Delegate -----------------------------

class TransitionDelegate(QStyledItemDelegate):
    def __init__(self, parent, palette):
        super().__init__(parent)
        self.p = palette

    def paint(self, painter: QPainter, option, index: QModelIndex):
        view = self.parent()
        active_row = getattr(view, "_active_row", -1)
        active_color = getattr(view, "_active_color", QColor(self.p["active_border"]))

        super().paint(painter, option, index)

        painter.save()

        sep = QColor(self.p["border"])
        sep.setAlpha(160)
        pen_sep = QPen(sep)
        pen_sep.setWidth(1)
        painter.setPen(pen_sep)
        r = option.rect
        painter.drawLine(r.bottomLeft(), r.bottomRight())

        if index.row() == active_row:
            col = index.column()
            cols = index.model().columnCount()

            border = QColor(active_color)
            border.setAlpha(245)
            pen = QPen(border)
            pen.setWidth(2)
            painter.setPen(pen)

            painter.drawLine(r.topLeft(), r.topRight())
            painter.drawLine(r.bottomLeft(), r.bottomRight())

            if col == 0:
                painter.drawLine(r.topLeft(), r.bottomLeft())
            if col == cols - 1:
                painter.drawLine(r.topRight(), r.bottomRight())

        painter.restore()


# ----------------------------- Tape View -----------------------------

class TapeView(QGraphicsView):
    def __init__(self, palette):
        super().__init__()
        self.p = palette

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setFrameShape(QFrame.Shape.NoFrame)

        self.cell_w = 40
        self.cell_h = 40
        self.cell_gap = 2
        self.base_x = 10
        self.base_y = 18

        self.rects: List[QGraphicsRectItem] = []
        self.texts: List[QGraphicsSimpleTextItem] = []
        self.arrow_item: Optional[QGraphicsPolygonItem] = None

        self.arrow_anim: Optional[QVariantAnimation] = None
        self.flash_anim: Optional[QVariantAnimation] = None
        self.tape_str = ""

        self.font_cell = QFont(palette["mono_font"], 12)
        self.font_cell.setBold(True)

        self.pen_cell = QPen(QColor(self.p["border"]))
        self.pen_cell.setWidth(2)

    def stop_anims(self):
        for anim in (self.arrow_anim, self.flash_anim):
            if anim and anim.state() == QVariantAnimation.State.Running:
                anim.stop()
        self.arrow_anim = None
        self.flash_anim = None

    def clear(self):
        self.stop_anims()
        self.scene.clear()
        self.rects.clear()
        self.texts.clear()
        self.arrow_item = None
        self.tape_str = ""

    def load_tape(self, s: str):
        self.clear()
        self.tape_str = s

        symbols = list(s) + ["⊔"]
        for idx, ch in enumerate(symbols):
            x = self.base_x + idx * (self.cell_w + self.cell_gap)
            y = self.base_y

            rect = QGraphicsRectItem(x, y, self.cell_w, self.cell_h)
            rect.setPen(self.pen_cell)
            rect.setBrush(QBrush(QColor(self.p["card_bg"])))
            self.scene.addItem(rect)

            text = QGraphicsSimpleTextItem(ch)
            text.setFont(self.font_cell)
            text.setBrush(QBrush(QColor(self.p["text"])))
            br = text.boundingRect()
            text.setPos(x + (self.cell_w - br.width()) / 2, y + (self.cell_h - br.height()) / 2)
            self.scene.addItem(text)

            self.rects.append(rect)
            self.texts.append(text)

        self._create_arrow()
        self._set_arrow_pos(0)
        self.arrow_item.setVisible(True)

        self.update_visual(read_upto=0, head_idx=0, done=False, accepted=False)

        width = max(720, self.base_x + len(symbols) * (self.cell_w + self.cell_gap) + 20)
        self.scene.setSceneRect(0, 0, width, 95)

        self.setMinimumHeight(105)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def _cell_center_x(self, idx: int) -> float:
        x = self.base_x + idx * (self.cell_w + self.cell_gap)
        return x + self.cell_w / 2

    def _cell_rect_item(self, idx: int) -> Optional[QGraphicsRectItem]:
        if not self.rects:
            return None
        idx = max(0, min(idx, len(self.rects) - 1))
        return self.rects[idx]

    def _create_arrow(self):
        poly = QPolygonF([QPointF(0, 0), QPointF(-10, -18), QPointF(10, -18)])
        arrow = QGraphicsPolygonItem(poly)
        arrow.setBrush(QBrush(QColor(self.p["text"])))
        arrow.setPen(QPen(QColor(self.p["text"])))
        self.scene.addItem(arrow)
        self.arrow_item = arrow

    def _set_arrow_pos(self, idx: int):
        if self.arrow_item is None:
            return
        idx = max(0, min(idx, len(self.rects) - 1))
        x = self._cell_center_x(idx)
        y = self.base_y - 6
        self.arrow_item.setPos(QPointF(x, y))

    def flash_current_cell(self, idx: int) -> Optional[QVariantAnimation]:
        rect = self._cell_rect_item(idx)
        if rect is None:
            return None

        start = QColor(self.p["card_bg"])
        mid = QColor("#2b2a14")
        anim = QVariantAnimation(self)
        anim.setDuration(140)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)

        def on_val(v):
            t = float(v)
            k = t * 2.0 if t <= 0.5 else (1.0 - t) * 2.0
            col = QColor(
                int(start.red() * (1 - k) + mid.red() * k),
                int(start.green() * (1 - k) + mid.green() * k),
                int(start.blue() * (1 - k) + mid.blue() * k)
            )
            rect.setBrush(QBrush(col))

        self.flash_anim = anim
        anim.valueChanged.connect(on_val)
        return anim

    def make_head_animation(self, from_idx: int, to_idx: int) -> Optional[QVariantAnimation]:
        if self.arrow_item is None or not self.rects:
            return None

        from_idx = max(0, min(from_idx, len(self.rects) - 1))
        to_idx = max(0, min(to_idx, len(self.rects) - 1))
        if from_idx == to_idx:
            return None

        start_x = self._cell_center_x(from_idx)
        end_x = self._cell_center_x(to_idx)
        y = self.base_y - 6

        anim = QVariantAnimation(self)
        anim.setDuration(190)
        anim.setStartValue(start_x)
        anim.setEndValue(end_x)

        def on_val(v):
            if self.arrow_item is None:
                return
            self.arrow_item.setPos(QPointF(float(v), y))

        anim.valueChanged.connect(on_val)
        self.arrow_anim = anim
        return anim

    def ensure_head_visible(self, idx: int):
        rect = self._cell_rect_item(idx)
        if rect is None:
            return
        self.ensureVisible(rect, 80, 20)

    def update_visual(self, read_upto: int, head_idx: int, done: bool, accepted: bool):
        for rect in self.rects:
            rect.setBrush(QBrush(QColor(self.p["card_bg"])))
            rect.setPen(self.pen_cell)

        for i in range(len(self.texts)):
            if i < len(self.texts) - 1 and i < read_upto:
                self.texts[i].setBrush(QBrush(QColor(self.p["muted"])))
            else:
                self.texts[i].setBrush(QBrush(QColor(self.p["text"])))

        if not self.rects:
            return

        head_idx = max(0, min(head_idx, len(self.rects) - 1))

        if done:
            if accepted:
                fill = QColor(self.p["ok_fill"])
                border = QColor(self.p["ok_border"])
            else:
                fill = QColor(self.p["bad_fill"])
                border = QColor(self.p["bad_border"])
        else:
            fill = QColor(self.p["active_fill"])
            border = QColor(self.p["active_border"])

        self.rects[head_idx].setBrush(QBrush(fill))
        pen = QPen(border)
        pen.setWidth(3)
        self.rects[head_idx].setPen(pen)


# ----------------------------- Stack View -----------------------------

class StackView(QGraphicsView):
    def __init__(self, palette):
        super().__init__()
        self.p = palette

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setFrameShape(QFrame.Shape.NoFrame)

        self.cell_w = 92
        self.cell_h = 28
        self.cell_gap = 6
        self.base_x = 10
        self.margin = 12

        self.font_cell = QFont(palette["mono_font"], 11)
        self.font_cell.setBold(True)

        self.base_pen = QPen(QColor(self.p["stack_border"]))
        self.base_pen.setWidth(1)

        self.glow_pen = QPen(QColor(self.p["active_border"]))
        self.glow_pen.setWidth(2)

        self._anim: Optional[QVariantAnimation] = None
        self._top_pair: Optional[Tuple[QGraphicsRectItem, QGraphicsSimpleTextItem]] = None

        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def stop_anims(self):
        if self._anim and self._anim.state() == QVariantAnimation.State.Running:
            self._anim.stop()
        self._anim = None

    def _scene_height_for(self, n: int) -> int:
        content = self.margin * 2 + max(1, n) * self.cell_h + max(0, n - 1) * self.cell_gap
        return max(260, content)

    def _draw_stack(self, stack: List[str]):
        self.stop_anims()
        self.scene.clear()
        self._top_pair = None

        n = len(stack)
        h = self._scene_height_for(n)
        self.scene.setSceneRect(0, 0, 120, h)

        bottom_y = h - self.margin - self.cell_h

        for idx, sym in enumerate(stack):
            x = self.base_x
            y = bottom_y - idx * (self.cell_h + self.cell_gap)

            rect = QGraphicsRectItem(x, y, self.cell_w, self.cell_h)
            rect.setPen(self.base_pen)
            rect.setBrush(QBrush(QColor(self.p["card_bg"])))
            self.scene.addItem(rect)

            text = QGraphicsSimpleTextItem(sym)
            text.setFont(self.font_cell)
            text.setBrush(QBrush(QColor(self.p["text"])))
            br = text.boundingRect()
            text.setPos(x + (self.cell_w - br.width()) / 2, y + (self.cell_h - br.height()) / 2)
            self.scene.addItem(text)

            if sym == "$":
                rect.setBrush(QBrush(QColor(self.p["card_bg_2"])))

            if idx == n - 1:
                self._top_pair = (rect, text)

        self.setMinimumWidth(120)
        QTimer.singleShot(0, lambda: self.verticalScrollBar().setValue(self.verticalScrollBar().maximum()))

    def make_push_animation(self, new_stack: List[str]) -> Optional[QVariantAnimation]:
        self._draw_stack(new_stack)
        if self._top_pair is None:
            return None

        rect, text = self._top_pair
        rect.setOpacity(0.0)
        text.setOpacity(0.0)
        rect.setPen(self.glow_pen)

        anim = QVariantAnimation(self)
        anim.setDuration(520)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)

        def on_val(v):
            if rect.scene() is None:
                return
            op = float(v)
            rect.setOpacity(op)
            text.setOpacity(op)
            rect.setPen(self.glow_pen)

        def on_done():
            if rect.scene() is not None:
                rect.setPen(self.base_pen)

        anim.valueChanged.connect(on_val)
        anim.finished.connect(on_done)
        self._anim = anim
        return anim

    def make_pop_animation(self, old_stack: List[str], new_stack: List[str]) -> Optional[QVariantAnimation]:
        self._draw_stack(old_stack)
        if self._top_pair is None:
            self._draw_stack(new_stack)
            return None

        rect, text = self._top_pair
        rect.setPen(self.glow_pen)

        anim = QVariantAnimation(self)
        anim.setDuration(520)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)

        def on_val(v):
            if rect.scene() is None:
                return
            op = float(v)
            rect.setOpacity(op)
            text.setOpacity(op)
            rect.setPen(self.glow_pen)

        def on_done():
            self._draw_stack(new_stack)

        anim.valueChanged.connect(on_val)
        anim.finished.connect(on_done)
        self._anim = anim
        return anim

    def render_static(self, stack: List[str]):
        self._draw_stack(stack)


# ----------------------------- Gemini Worker -----------------------------

class GeminiWorker(QThread):
    ok = pyqtSignal(dict)
    fail = pyqtSignal(str)

    def __init__(self, keys: List[str], key_index_ref: Dict[str, int], user_text: str, parent=None):
        super().__init__(parent)
        self.keys = keys
        self.key_index_ref = key_index_ref
        self.user_text = user_text.strip()

    def _pick_key_round_robin(self) -> str:
        if not self.keys:
            raise RuntimeError("No API keys provided.")
        i = self.key_index_ref.get("i", 0) % len(self.keys)
        self.key_index_ref["i"] = (i + 1) % len(self.keys)
        return self.keys[i].strip()

    def _prompt(self) -> str:
        return f"""
You are an expert in Theory of Computation and deterministic Pushdown Automata (DPDA).

OUTPUT RULES (STRICT):
- Output MUST be a single valid JSON object ONLY.
- Do NOT wrap in markdown (no ```json).
- Do NOT include any extra text before/after the JSON.

TASK:
Given the user request, construct a DPDA specification suitable for step-by-step simulation in a transition table.

DETERMINISM CONSTRAINT:
Do NOT create two transitions with the same triple (from, input, stack_top).
If the language is inherently non-deterministic for DPDA design (e.g., palindromes),
or you cannot confidently construct a DPDA, then return a valid JSON with:
- transitions: []
- language_text: very short (max 10 words)
- error: short explanation (max 20 words)

EPSILON:
Use input = "ε" for epsilon transitions ONLY (exactly "ε").
Epsilon transitions may be used for push/pop/noop/accept, but ONLY when input is finished.

STACK SYMBOLS:
Use single-character stack symbols. You may use helper symbols like "X" if needed.
Avoid multi-character symbols like "Z1", "Z0".

ACCEPTANCE:
Acceptance must be represented by reaching accept_state (qacc).
Use action="accept" only for an epsilon transition into accept_state.

language_text:
Must be one-line, student-friendly, and short (max 120 characters).

JSON schema (exact):
{{
  "language_text": "L = {{ ... }}",
  "alphabet": ["a","b"],
  "stack_bottom": "$",
  "start_state": "q0",
  "accept_state": "qacc",
  "reject_state": "qrej",
  "error": "",
  "transitions": [
    {{
      "from": "q0",
      "input": "a",
      "stack_top": "$",
      "to": "q0",
      "action": "push",
      "push_symbol": "X"
    }}
  ]
}}

User request:
{self.user_text}
""".strip()

    def run(self):
        try:
            if not self.user_text:
                self.fail.emit("AI: Please type a language description first.")
                return

            key = self._pick_key_round_robin()
            model = "gemini-2.5-flash"
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"

            payload = {
                "contents": [{
                    "parts": [{"text": self._prompt()}]
                }],
                "generationConfig": {
                    "response_mime_type": "application/json"
                }
            }

            r = requests.post(url, json=payload, timeout=60)
            if r.status_code != 200:
                self.fail.emit(f"AI: request failed (HTTP {r.status_code}).")
                return

            data = r.json()
            try:
                text = data["candidates"][0]["content"]["parts"][0]["text"]
            except Exception:
                self.fail.emit("AI: unexpected response format.")
                return

            try:
                spec = json.loads(text)
            except Exception:
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        spec = json.loads(text[start:end+1])
                    except Exception:
                        self.fail.emit("AI: invalid JSON output.")
                        return
                else:
                    self.fail.emit("AI: invalid JSON output.")
                    return

            self.ok.emit(spec)

        except requests.exceptions.Timeout:
            self.fail.emit("AI: timeout. Please try again.")
        except requests.exceptions.RequestException:
            self.fail.emit("AI: network error / cannot reach Gemini.")
        except Exception:
            self.fail.emit("AI: unknown error.")


# ----------------------------- Main Window -----------------------------

class PDAWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.p = {
            "bg": "#0b1220",
            "card_bg": "#121a2a",
            "card_bg_2": "#0f1726",
            "border": "#27324a",
            "text": "#e5e7eb",
            "muted": "#9aa3b2",
            "mono_font": "Cascadia Mono",
            "active_fill": "#0b2a4a",
            "active_border": "#3b82f6",
            "ok_fill": "#0b3a2a",
            "ok_border": "#10b981",
            "bad_fill": "#3a0b12",
            "bad_border": "#ef4444",
            "stack_border": "#475569",
            "push": "#3b82f6",
            "pop": "#f59e0b",
            "accept": "#10b981",
            "rej": "#ef4444",
            "btn_primary": "#3B82F6",
            "btn_secondary": "#1f2937",
            "btn_secondary_border": "#334155",
            "btn_stop": "#F59E0B",
            "btn_reset_bg": "#2a0f14",
            "btn_reset_border": "#7f1d1d",
            "btn_reset_text": "#fecaca",
        }

        self.setWindowTitle("PDA Simulator — Theory of Computation")
        self.resize(920, 620)

        self.core = PDACore()
        self.tape = ""
        self.cfg = self.core.initial_config()
        self.history: List[PDAConfig] = []

        self.running = False
        self._stepping = False
        self._animating = False
        self._group: Optional[QSequentialAnimationGroup] = None

        self.run_timer = QTimer(self)
        self.run_timer.setInterval(250)
        self.run_timer.timeout.connect(self.on_step)

        self.gemini_keys = self._load_gemini_keys()
        self.key_rr = {"i": 0}
        self.ai_worker: Optional[GeminiWorker] = None

        self._build_ui()
        self._sync_ui(initial=True)

        if os.name == 'nt':
            self.apply_dark_title_bar(self.winId())

    def apply_dark_title_bar(self, hwnd):
        try:
            hwnd_int = int(hwnd)
            dwmapi = ctypes.WinDLL("dwmapi")

            def try_attr(attr: int) -> bool:
                val = ctypes.c_int(1)
                res = dwmapi.DwmSetWindowAttribute(
                    wintypes.HWND(hwnd_int),
                    attr,
                    ctypes.byref(val),
                    ctypes.sizeof(val)
                )
                return res == 0

            if not try_attr(20):
                try_attr(19)

        except Exception as e:
            print(f"DWM Dark Mode not supported: {e}")

    def _load_gemini_keys(self) -> List[str]:
        raw = os.getenv("GEMINI_API_KEYS", "").strip()
        if not raw:
            raw = (RAW_KEYS or "").strip()
        if not raw:
            return []
        return [k.strip() for k in raw.split(",") if k.strip()]

    def _card(self, title: str, content: QWidget, tight: bool = False) -> QWidget:
        box = QWidget()
        box.setObjectName("Card")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(8, 8, 8, 8) if tight else lay.setContentsMargins(10, 8, 10, 10)
        lay.setSpacing(6)

        lbl = QLabel(title)
        lbl.setObjectName("CardTitle")
        lay.addWidget(lbl)
        lay.addWidget(content)
        return box

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        main = QVBoxLayout(root)
        main.setContentsMargins(12, 12, 12, 12)
        main.setSpacing(10)

        header_row = QWidget()
        hr = QHBoxLayout(header_row)
        hr.setContentsMargins(0, 0, 0, 0)
        hr.setSpacing(10)

        left_box = QWidget()
        ll = QVBoxLayout(left_box)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(2)

        title = QLabel("Pushdown Automaton Simulator")
        title.setObjectName("HeaderTitle")

        self.sub = QLabel(self.core.language_text)
        self.sub.setObjectName("HeaderSub")
        self.sub.setWordWrap(True)
        self.sub.setMaximumWidth(820)

        ll.addWidget(title)
        ll.addWidget(self.sub)

        right_box = QWidget()
        rl = QHBoxLayout(right_box)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)

        self.ai_inp = QLineEdit()
        self.ai_inp.setObjectName("AIBox")
        self.ai_inp.setPlaceholderText("Ask AI (describe the language)")
        self.ai_inp.setText("")

        self.ai_btn = QPushButton("✨")
        self.ai_btn.setObjectName("AIMagic")
        self.ai_btn.setFixedWidth(44)

        rl.addWidget(self.ai_inp)
        rl.addWidget(self.ai_btn)

        hr.addWidget(left_box, stretch=1)
        hr.addWidget(right_box, stretch=0)
        main.addWidget(header_row)

        self.tape_view = TapeView(self.p)
        main.addWidget(self._card("Tape", self.tape_view))

        self.status_strip = QLabel("State: q0 | Stack top: $ | Remaining: ε | Result: —")
        self.status_strip.setObjectName("StatusStrip")
        self.status_strip.setTextFormat(Qt.TextFormat.RichText)
        main.addWidget(self.status_strip)

        controls = QWidget()
        cl = QHBoxLayout(controls)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(8)

        self.inp = QLineEdit()
        self.inp.setPlaceholderText("Input your string here")

        self.btn_load = QPushButton("Load")
        self.btn_next = QPushButton("Next")
        self.btn_back = QPushButton("Back")
        self.btn_run = QPushButton("AutoRun")
        self.btn_reset = QPushButton("Reset")

        for b in (self.btn_load, self.btn_next, self.btn_back, self.btn_run, self.btn_reset):
            b.setFixedWidth(100)

        cl.addWidget(QLabel("Input:"))
        cl.addWidget(self.inp, stretch=1)
        cl.addWidget(self.btn_load)
        cl.addWidget(self.btn_next)
        cl.addWidget(self.btn_back)
        cl.addWidget(self.btn_run)
        cl.addWidget(self.btn_reset)
        main.addWidget(controls)

        mid_split = QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(mid_split, stretch=1)

        self.stack_view = StackView(self.p)
        stack_card = self._card("Stack", self.stack_view)
        stack_card.setMinimumWidth(160)
        mid_split.addWidget(stack_card)

        self.trans_view = QTableView()
        self.trans_view.setObjectName("TransView")
        self.trans_model = QStandardItemModel()
        self.trans_model.setColumnCount(5)
        self.trans_model.setHorizontalHeaderLabels(["q", "input", "stack top", "q'", "action"])
        self.trans_view.setModel(self.trans_model)

        self.trans_view.verticalHeader().setVisible(False)
        self.trans_view.setSelectionMode(QTableView.SelectionMode.NoSelection)
        self.trans_view.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)

        self.trans_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.trans_view.setShowGrid(False)

        self.trans_view._active_row = -1
        self.trans_view._active_color = QColor(self.p["active_border"])
        self.trans_view.setItemDelegate(TransitionDelegate(self.trans_view, self.p))

        self._fill_transition_model()
        self._configure_transition_view()

        self.trans_card = self._card("Transition function δ", self.trans_view, tight=True)
        self.trans_card.setFixedWidth(430)
        mid_split.addWidget(self.trans_card)

        self.trace = QTextBrowser()
        self.trace.setReadOnly(True)
        self.trace.setObjectName("TraceBox")
        trace_card = self._card("Trace", self.trace)
        trace_card.setMinimumWidth(240)
        mid_split.addWidget(trace_card)

        mid_split.setStretchFactor(0, 0)
        mid_split.setStretchFactor(1, 0)
        mid_split.setStretchFactor(2, 1)
        mid_split.setSizes([180, 430, 260])

        self.setStyleSheet(f"""
            QWidget {{
                background: {self.p["bg"]};
                color: {self.p["text"]};
                font-family: "Segoe UI";
            }}
            QLabel#HeaderTitle {{
                font-size: 16px;
                font-weight: 800;
                color: {self.p["text"]};
            }}
            QLabel#HeaderSub {{
                font-size: 12px;
                color: {self.p["muted"]};
            }}
            QLabel#StatusStrip {{
                background: {self.p["card_bg_2"]};
                border: 1px solid {self.p["border"]};
                border-radius: 10px;
                padding: 8px 10px;
                font-size: 12px;
            }}
            QWidget#Card {{
                background: {self.p["card_bg"]};
                border: 1px solid {self.p["border"]};
                border-radius: 12px;
            }}
            QLabel#CardTitle {{
                font-size: 12px;
                font-weight: 800;
                color: {self.p["text"]};
            }}
            QLineEdit {{
                background: {self.p["card_bg_2"]};
                border: 1px solid {self.p["border"]};
                border-radius: 8px;
                padding: 7px 10px;
                font-size: 12px;
                color: {self.p["text"]};
            }}
            QLineEdit#AIBox {{
                min-width: 420px;
            }}
            QPushButton#AIMagic {{
                background: {self.p["btn_secondary"]};
                border: 1px solid {self.p["btn_secondary_border"]};
                border-radius: 8px;
                font-size: 16px;
            }}
            QPushButton {{
                border-radius: 7px;
                padding: 8px 10px;
                font-size: 12px;
                font-weight: 600;
            }}
            QPushButton[text="Next"] {{
                background: {self.p["btn_primary"]};
                color: white;
                border: 1px solid #1d4ed8;
            }}
            QPushButton[text="Load"],
            QPushButton[text="Back"],
            QPushButton[text="AutoRun"] {{
                background: {self.p["btn_secondary"]};
                color: white;
                border: 1px solid {self.p["btn_secondary_border"]};
            }}
            QPushButton[text="Stop"] {{
                background: {self.p["btn_stop"]};
                color: #0b1220;
                border: 1px solid #b45309;
            }}
            QPushButton[text="Reset"] {{
                background: {self.p["btn_reset_bg"]};
                color: {self.p["btn_reset_text"]};
                border: 1px solid {self.p["btn_reset_border"]};
            }}
            QTableView#TransView {{
                background: {self.p["card_bg"]};
                border: 0px;
                font-family: "{self.p["mono_font"]}";
                font-size: 11px;
                outline: none;
            }}
            QHeaderView::section {{
                background: {self.p["card_bg_2"]};
                color: {self.p["muted"]};
                padding: 6px;
                border: 0px;
                font-weight: 700;
            }}
            QTextBrowser#TraceBox {{
                background: {self.p["card_bg"]};
                border: 0px;
                font-family: "{self.p["mono_font"]}";
                font-size: 15px;
            }}
        """)

        self.btn_load.clicked.connect(self.on_load)
        self.btn_next.clicked.connect(self.on_step)
        self.btn_back.clicked.connect(self.on_back)
        self.btn_run.clicked.connect(self.on_run_toggle)
        self.btn_reset.clicked.connect(self.on_reset)
        self.ai_btn.clicked.connect(self.on_ai_request)

    def _fill_transition_model(self):
        self.trans_model.setRowCount(0)
        for tr in self.core.transitions:
            inp_str = "ε" if tr.input_sym is None else tr.input_sym

            row = []
            for v in (tr.from_state, inp_str, tr.stack_top, tr.to_state, tr.action):
                it = QStandardItem(str(v))
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                row.append(it)

            if tr.action == "push":
                row[4].setForeground(QBrush(QColor(self.p["push"])))
            elif tr.action == "pop":
                row[4].setForeground(QBrush(QColor(self.p["pop"])))
            elif tr.action == "accept":
                row[4].setForeground(QBrush(QColor(self.p["accept"])))

            self.trans_model.appendRow(row)

    def _configure_transition_view(self):
        hv = self.trans_view.horizontalHeader()
        hv.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hv.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hv.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        hv.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        hv.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)

        self.trans_view.setColumnWidth(4, 85)
        self.trans_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        for r in range(self.trans_model.rowCount()):
            self.trans_view.setRowHeight(r, 30)

    def _trace_html(self, line: str) -> str:
        esc = html_escape(line)

        if esc.strip() in ("ACCEPT.", "REJECT."):
            if esc.strip() == "ACCEPT.":
                return (
                    f"<div style='white-space:pre; margin:8px 0 4px 0;"
                    f"font-size:18px; font-weight:900; color:{self.p['accept']};'>ACCEPT</div>"
                )
            else:
                return (
                    f"<div style='white-space:pre; margin:8px 0 4px 0;"
                    f"font-size:18px; font-weight:900; color:{self.p['rej']};'>REJECT</div>"
                )

        esc = esc.replace("δ(", f"<span style='color:{self.p['accept']};font-weight:800;'>δ(</span>")
        esc = esc.replace(" → ", "<span style='color:#93c5fd;font-weight:900;'> → </span>")
        esc = esc.replace("push", f"<span style='color:{self.p['push']};font-weight:900;'>push</span>")
        esc = esc.replace("pop", f"<span style='color:{self.p['pop']};font-weight:900;'>pop</span>")
        esc = esc.replace("noop", f"<span style='color:{self.p['muted']};font-weight:900;'>noop</span>")
        esc = esc.replace("accept", f"<span style='color:{self.p['accept']};font-weight:900;'>accept</span>")
        esc = esc.replace("reject", f"<span style='color:{self.p['rej']};font-weight:900;'>reject</span>")
        esc = esc.replace("undefined", f"<span style='color:{self.p['rej']};font-weight:900;'>undefined</span>")
        esc = esc.replace("thinking...", f"<span style='color:{self.p['muted']};font-weight:800;'>thinking...</span>")

        return f"<div style='white-space:pre; margin:0; padding:0;'>{esc}</div>"

    def _append_trace(self, text: str):
        self.trace.append(self._trace_html(text))

    def _set_active_transition_row(self, rule_idx: int, action: str):
        self.trans_view._active_row = rule_idx

        color = QColor(self.p["active_border"])
        a = (action or "").lower()
        if a == "push":
            color = QColor(self.p["push"])
        elif a == "pop":
            color = QColor(self.p["pop"])
        elif a == "accept":
            color = QColor(self.p["accept"])
        elif a in ("reject", "undefined"):
            color = QColor(self.p["rej"])

        self.trans_view._active_color = color

        if 0 <= rule_idx < self.trans_model.rowCount():
            self.trans_view.scrollTo(self.trans_model.index(rule_idx, 0), QTableView.ScrollHint.PositionAtCenter)

        self.trans_view.viewport().update()

    def _stop_run(self):
        if self.running:
            self.running = False
            self.run_timer.stop()
            self.btn_run.setText("AutoRun")

    def _sync_ui(self, initial: bool = False):
        self.sub.setText(self.core.language_text)

        if self.tape != "":
            if self.tape_view.tape_str != self.tape:
                self.tape_view.load_tape(self.tape)

            head_idx = min(self.cfg.i, len(self.tape))
            self.tape_view.ensure_head_visible(head_idx)

            done = self.core.is_done(self.tape, self.cfg)
            accepted = (self.cfg.state == self.core.accept_state)
            self.tape_view.update_visual(self.cfg.i, head_idx, done, accepted)

        if not self._animating:
            self.stack_view.render_static(self.cfg.stack)

        rem = self.tape[self.cfg.i:] if self.cfg.i < len(self.tape) else "ε"
        top = self.cfg.stack[-1] if self.cfg.stack else self.core.bottom

        if self.cfg.state == self.core.accept_state:
            res_html = f"<span style='color:{self.p['accept']}; font-weight:900;'>ACCEPT</span>"
        elif self.cfg.state == self.core.reject_state:
            res_html = f"<span style='color:{self.p['rej']}; font-weight:900;'>REJECT</span>"
        else:
            res_html = "—"

        self.status_strip.setText(
            f"State: {html_escape(self.cfg.state)}  |  "
            f"Stack top: {html_escape(top)}  |  "
            f"Remaining: {html_escape(rem)}  |  "
            f"Result: {res_html}"
        )

        self._set_active_transition_row(self.cfg.last_rule_idx, self.cfg.last_action)

        done = self.core.is_done(self.tape, self.cfg)
        has_tape = (self.tape != "")

        busy = self.running or self._animating or (self.ai_worker is not None and self.ai_worker.isRunning())

        self.btn_run.setEnabled(has_tape and not done and not busy)
        self.btn_reset.setEnabled(True)
        self.ai_btn.setEnabled(not busy)
        self.ai_inp.setEnabled(not busy)

        if busy:
            self.btn_next.setEnabled(False)
            self.btn_back.setEnabled(False)
            self.btn_load.setEnabled(False)
            self.inp.setEnabled(False)
        else:
            self.btn_next.setEnabled(has_tape and not done)
            self.btn_back.setEnabled(len(self.history) > 0)
            self.btn_load.setEnabled(True)
            self.inp.setEnabled(True)

    def on_load(self):
        s = self.inp.text().strip()

        if any(ch not in set(self.core.alphabet) for ch in s):
            QMessageBox.warning(self, "Invalid input",
                                f"Alphabet must be only {{{', '.join(self.core.alphabet)}}}.")
            return

        self._stop_run()
        self._animating = False
        if self._group:
            self._group.stop()
            self._group = None

        self.tape = s
        self.cfg = self.core.initial_config()
        self.history.clear()
        self.trace.clear()

        self.tape_view.load_tape(self.tape)
        self.stack_view.render_static(self.cfg.stack)

        self._append_trace("Loaded. Ready.")
        self._sync_ui(initial=True)

    def on_reset(self):
        self._stop_run()
        self._animating = False
        if self._group:
            self._group.stop()
            self._group = None

        self.tape = ""
        self.cfg = self.core.initial_config()
        self.history.clear()
        self.trace.clear()

        self.tape_view.clear()
        self.stack_view.render_static(self.cfg.stack)

        self.status_strip.setText("State: q0 | Stack top: $ | Remaining: ε | Result: —")
        self._set_active_transition_row(-1, "")
        self._sync_ui(initial=True)

    def on_back(self):
        if self.running or self._animating:
            return
        self._stop_run()
        if not self.history:
            return
        self.cfg = self.history.pop()
        self._append_trace("Back.")
        self._sync_ui()

    def on_run_toggle(self):
        if self.tape == "" or self.core.is_done(self.tape, self.cfg):
            return

        self.running = not self.running
        if self.running:
            self.btn_run.setText("Stop")
            self.run_timer.start()
        else:
            self._stop_run()

        self._sync_ui()

    def on_step(self):
        if self._stepping or self._animating:
            return

        try:
            self._stepping = True

            if self.tape == "" or self.core.is_done(self.tape, self.cfg):
                self._stop_run()
                return

            old_cfg = self.cfg
            old_stack = old_cfg.stack.copy()
            old_i = old_cfg.i

            new_cfg = self.core.step(self.tape, self.cfg)

            self.history.append(PDAConfig(old_cfg.state, old_cfg.i, old_cfg.stack.copy(),
                                          old_cfg.last_transition, old_cfg.last_rule_idx, old_cfg.last_action))

            self._set_active_transition_row(new_cfg.last_rule_idx, new_cfg.last_action)
            self._append_trace(new_cfg.last_transition)

            self._animating = True
            if self._group:
                self._group.stop()
                self._group = None

            grp = QSequentialAnimationGroup(self)
            self._group = grp

            from_head = min(old_i, len(self.tape))
            to_head = min(new_cfg.i, len(self.tape))

            flash = self.tape_view.flash_current_cell(from_head)
            if flash is not None:
                grp.addAnimation(flash)

            head_anim = self.tape_view.make_head_animation(from_head, to_head)
            if head_anim is not None:
                grp.addAnimation(head_anim)

            stack_anim = None
            if new_cfg.last_action == "push":
                stack_anim = self.stack_view.make_push_animation(new_cfg.stack.copy())
            elif new_cfg.last_action == "pop":
                stack_anim = self.stack_view.make_pop_animation(old_stack, new_cfg.stack.copy())
            else:
                self.stack_view.render_static(new_cfg.stack.copy())

            if stack_anim is not None:
                grp.addAnimation(stack_anim)

            def on_anim_done():
                self.cfg = new_cfg
                self._animating = False

                if self.core.is_done(self.tape, self.cfg):
                    self._stop_run()
                    self._append_trace("ACCEPT." if self.cfg.state == self.core.accept_state else "REJECT.")

                self._sync_ui()

            grp.finished.connect(on_anim_done)
            self._sync_ui()
            grp.start()

        except Exception as e:
            self._stop_run()
            self._animating = False
            QMessageBox.critical(self, "Runtime error", f"Unexpected error:\n{type(e).__name__}: {e}")
        finally:
            self._stepping = False

    # ----------------------------- AI -----------------------------

    def on_ai_request(self):
        if self.ai_worker is not None and self.ai_worker.isRunning():
            return

        if not self.gemini_keys:
            self._append_trace("AI: No API keys. Set GEMINI_API_KEYS or RAW_KEYS.")
            return

        user_text = self.ai_inp.text().strip()
        if not user_text:
            self._append_trace("AI: please type a language description.")
            return

        self._append_trace("thinking...")
        self._sync_ui()

        self.ai_worker = GeminiWorker(self.gemini_keys, self.key_rr, user_text, parent=self)
        self.ai_worker.ok.connect(self._on_ai_ok)
        self.ai_worker.fail.connect(self._on_ai_fail)
        self.ai_worker.finished.connect(lambda: self._sync_ui())
        self.ai_worker.start()

    def _on_ai_fail(self, msg: str):
        self._append_trace(msg)

    def _on_ai_ok(self, spec: dict):
        try:
            self.core.load_from_spec(spec)

            self._fill_transition_model()
            self._configure_transition_view()
            self._set_active_transition_row(-1, "")

            self._stop_run()
            self._animating = False
            if self._group:
                self._group.stop()
                self._group = None

            self.tape = ""
            self.cfg = self.core.initial_config()
            self.history.clear()
            self.trace.clear()

            self.tape_view.clear()
            self.stack_view.render_static(self.cfg.stack)

            if len(self.core.transitions) == 0:
                msg = self.core.ai_error.strip() or "Could not build a deterministic PDA."
                self._append_trace(f"AI: {msg}")
            else:
                self._append_trace("AI: PDA updated successfully.")

            self._sync_ui(initial=True)

        except Exception as e:
            self._append_trace(f"AI: failed to apply PDA ({type(e).__name__}).")
            self._sync_ui(initial=True)


def main():
    app = QApplication(sys.argv)
    w = PDAWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
