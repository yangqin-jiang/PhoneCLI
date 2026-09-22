"""PhoneCLI agent as an AndroidWorld EnvironmentInteractingAgent.

This is a thin wrapper that adapts PhoneCLI's agent logic (VLM prompts,
action parsing, macro routing via app maps) to AndroidWorld's environment
interface. It does NOT modify any existing PhoneCLI code — it only imports
and delegates.

Integration:
    from aworld_agent.agent import PhoneCLIAndroidWorldAgent
    agent = PhoneCLIAndroidWorldAgent(
        env=env,
        agent_model=OpenAIAgent(...),
        app_map=AppMap("app_maps/clock_android.yaml"),
        llm_config={...},
    )

Architecture:
    AndroidWorld framework
    └── PhoneCLIAndroidWorldAgent.step(goal)
        ├── Round 1: app map macro routing (LLM maps task → operation → ADB macro)
        │   ├── [if OP]: macro replay → VLM verify → finish
        │   └── [if MACRO_VLM]: macro replay → hand off to VLM
        └── Round 2+: cloud-style VLM (screenshot + SoM labels → actions)
"""

import base64
import dataclasses
import io
import logging
import math
import os
import re
import tempfile
import time
from typing import Any, Optional

import numpy as np
from PIL import Image

# cv2 is only needed for optional SoM annotation (not used by default).
try:
    import cv2  # noqa: F401
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from android_world.agents import base_agent
from android_world.env import adb_utils
from android_world.env import interface
from android_world.env import json_action as ja
from android_world.env import representation_utils

from aworld_agent.prompts import MACRO_PLAN_PROMPT, MACRO_VERIFY_PROMPT, VLM_VERIFY_PROMPT
from aworld_agent.llm_client import text_completion, vision_completion
from aworld_agent.android_screenshot_template import SYSTEM_PROMPT_ANDROID_MLLM_CLOUD_V0

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SoM (Set-of-Mark) annotation — draw bounding boxes with indices on screenshot
# ---------------------------------------------------------------------------

# Screen edges to ignore (status bar, nav bar). Elements fully in these zones
# are typically system UI, not app content.
_STATUS_BAR_BOTTOM = 100   # top status bar
_NAV_BAR_TOP_RATIO = 0.92   # bottom nav bar (92% of screen height)
_TOGGLE_SETTLE_SECONDS = 2.0


def _is_actionable(elem: representation_utils.UIElement) -> bool:
    """Check if a UI element is something the agent should interact with."""
    # Anonymous RecyclerView/ScrollView containers often cover most of the
    # screen.  Drawing a numeric tap target on them makes the VLM click the
    # container center instead of the labeled child row it intended.  Page
    # scrolling remains available through the reserved swipe index 0.
    if (
        elem.is_scrollable
        and not elem.is_clickable
        and not elem.is_editable
        and not elem.is_long_clickable
        and not elem.is_checkable
        and not elem.text
        and not elem.content_description
    ):
        return False
    return (
        elem.is_clickable
        or elem.is_editable
        or elem.is_long_clickable
        or elem.is_scrollable
        or elem.is_focusable
        or elem.is_checkable
        # Accessibility trees often put the visible row label on a
        # non-clickable child TextView while the clickable parent is anonymous.
        # Marking the label lets a tap at its center bubble to the parent and
        # gives the VLM a grounded target such as "Connection preferences".
        or bool(elem.text)
        or bool(elem.content_description)
    )


def _is_continuous_control(elem: representation_utils.UIElement) -> bool:
    """Return whether an accessibility element behaves like a slider."""
    # Do not classify ordinary text such as "Dark theme, font size,
    # brightness" as a slider.  Only widget/resource metadata is reliable
    # enough for execution-side coordinate rewriting.
    descriptor = " ".join(
        str(value or "")
        for value in (
            elem.class_name,
            elem.resource_name,
            elem.resource_id,
        )
    ).lower()
    return any(token in descriptor for token in ("seekbar", "slider"))


def _annotate_screenshot(
    screenshot: np.ndarray,
    ui_elements: list[representation_utils.UIElement],
    screen_h: int,
) -> tuple[np.ndarray, list[int]]:
    """Draw green bounding boxes with numeric indices on a screenshot.

    Returns (labeled_image, valid_indices) where valid_indices is the list of
    element indices that were actually drawn (1-indexed, matching prompt).
    """
    labeled = screenshot.copy()
    valid_indices = []
    nav_bar_y = int(screen_h * _NAV_BAR_TOP_RATIO)

    for i, elem in enumerate(ui_elements):
        if not _is_actionable(elem):
            continue
        if elem.bbox_pixels is None:
            continue

        x1, x2, y1, y2 = (
            elem.bbox_pixels.x_min,
            elem.bbox_pixels.x_max,
            elem.bbox_pixels.y_min,
            elem.bbox_pixels.y_max,
        )

        # Skip elements centered in the status bar.  Some icon boxes straddle
        # y=100 by a few pixels (e.g. [24,104]); checking only y2 let those
        # system icons become misleading SoM tap targets.
        if (y1 + y2) / 2 < _STATUS_BAR_BOTTOM:
            continue
        if y1 > nav_bar_y:
            continue

        # Skip zero-size elements
        if x2 <= x1 or y2 <= y1:
            continue

        index = len(valid_indices) + 1  # 1-indexed for VLM
        valid_indices.append(i)

        # Scale for drawing thickness
        iso_scale = math.sqrt(screenshot.shape[1] / 1080.0)

        # Draw green bounding box
        cv2.rectangle(
            labeled,
            (x1, y1),
            (x2, y2),
            color=(0, 200, 0),
            thickness=max(1, int(2 * iso_scale)),
        )
        # Draw white background + black text for index label
        label_w = int(28 * iso_scale)
        label_h = int(22 * iso_scale)
        cv2.rectangle(
            labeled,
            (x1, max(0, y1 - label_h)),
            (x1 + label_w, y1),
            (255, 255, 255),
            -1,
        )
        cv2.putText(
            labeled,
            str(index),
            (x1 + int(3 * iso_scale), y1 - int(4 * iso_scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55 * iso_scale,
            (0, 0, 0),
            thickness=max(1, int(2 * iso_scale)),
        )

    return labeled, valid_indices


def _build_ui_elements_list(
    ui_elements: list[representation_utils.UIElement],
    valid_indices: list[int],
) -> str:
    """Build a text description of actionable UI elements for the VLM prompt."""
    # Index 0 is synthetic and intentionally has no box on the image.  It lets
    # the model request a global page swipe without selecting a giant anonymous
    # scroll container as a tappable element.
    lines = ["  [0] full-screen [scrollable; swipe only, never tap]"]
    if not valid_indices:
        return "\n".join(lines)
    for idx_1, orig_idx in enumerate(valid_indices):
        elem = ui_elements[orig_idx]
        parts = [f"  [{idx_1 + 1}]"]
        if elem.text:
            parts.append(f'text="{elem.text}"')
        if elem.content_description:
            parts.append(f'desc="{elem.content_description}"')
        if elem.hint_text:
            parts.append(f'hint="{elem.hint_text}"')
        if elem.resource_id:
            rid = elem.resource_id.split("/")[-1] if "/" in elem.resource_id else elem.resource_id
            parts.append(f"id={rid}")

        # Type tags
        tags = []
        if elem.is_editable:
            tags.append("EDITABLE")
        if elem.is_clickable:
            tags.append("clickable")
        if elem.is_long_clickable:
            tags.append("long-clickable")
        if elem.is_scrollable:
            tags.append("scrollable")
        if elem.is_checked:
            tags.append("CHECKED")
        if elem.is_checkable:
            tags.append("checkable")
        if _is_continuous_control(elem):
            tags.append("CONTINUOUS")
        parts.append(f"[{', '.join(tags)}]")
        lines.append(" ".join(parts))

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Action translation: PhoneCLI parsed action → AndroidWorld JSONAction
# ---------------------------------------------------------------------------
# PhoneCLI’s parse_action() returns dicts like:
#   {"name": "tap", "args": [x, y]}           — physical pixel coords
#   {"name": "text", "args": ["hello"]}
#   {"name": "swipe", "args": [x1,y1,x2,y2]}
#   {"name": "back", "args": []}
#   {"name": "home", "args": []}
#   {"name": "launch", "args": ["Settings"]}
#   {"name": "finish", "args": ["done"]}
#   {"name": "wait", "args": [3]}
#   {"name": "long_press", "args": [x, y]}


def _resolve_center(
    index: int,
    ui_elements: list[representation_utils.UIElement],
    valid_indices: list[int],
) -> Optional[tuple[int, int]]:
    """Resolve an element index (1-based, as shown in SoM) to pixel center."""
    arr_idx = index - 1
    if arr_idx < 0 or arr_idx >= len(valid_indices):
        return None
    elem_idx = valid_indices[arr_idx]
    if elem_idx >= len(ui_elements):
        return None
    elem = ui_elements[elem_idx]
    if elem.bbox_pixels is None:
        return None
    cx = (elem.bbox_pixels.x_min + elem.bbox_pixels.x_max) // 2
    cy = (elem.bbox_pixels.y_min + elem.bbox_pixels.y_max) // 2
    return cx, cy


def _resolve_element(
    index: int,
    ui_elements: list[representation_utils.UIElement],
    valid_indices: list[int],
) -> Optional[representation_utils.UIElement]:
    """Resolve a 1-based SoM index to its original accessibility element."""
    arr_idx = index - 1
    if arr_idx < 0 or arr_idx >= len(valid_indices):
        return None
    elem_idx = valid_indices[arr_idx]
    if elem_idx < 0 or elem_idx >= len(ui_elements):
        return None
    return ui_elements[elem_idx]


def _find_continuous_control_index(
    ui_elements: list[representation_utils.UIElement],
    valid_indices: list[int],
) -> Optional[int]:
    """Find the best SoM target for an on-screen continuous control."""
    candidates = []
    for som_index, original_index in enumerate(valid_indices, 1):
        if original_index >= len(ui_elements):
            continue
        elem = ui_elements[original_index]
        if not _is_continuous_control(elem) or elem.bbox_pixels is None:
            continue
        class_name = (elem.class_name or "").lower()
        score = 100 if "seekbar" in class_name or "slider" in class_name else 0
        if elem.bbox_pixels.width > 3 * max(1, elem.bbox_pixels.height):
            score += 10
        candidates.append((score, som_index))
    return max(candidates, default=(0, None))[1]


def _find_brightness_level_index(
    ui_elements: list[representation_utils.UIElement],
    valid_indices: list[int],
) -> Optional[int]:
    """Find the Display-page row that opens Android's brightness slider."""
    for som_index, original_index in enumerate(valid_indices, 1):
        if original_index >= len(ui_elements):
            continue
        elem = ui_elements[original_index]
        descriptor = " ".join(
            str(value or "")
            for value in (elem.text, elem.content_description, elem.hint_text)
        ).lower()
        if "brightness level" in descriptor:
            return som_index
    return None


def _normalize_continuous_control_action(
    goal: str,
    pa: dict,
    ui_elements: list[representation_utils.UIElement],
    valid_indices: list[int],
) -> tuple[dict, bool]:
    """Replace imprecise endpoint taps with a full slider swipe.

    AndroidWorld's brightness evaluator requires the exact endpoint.  A visual
    coordinate tap can land at 94--99%, while dragging the accessibility
    slider from edge to edge reliably reaches its minimum or maximum.
    """
    goal_lower = goal.lower()
    if not any(word in goal_lower for word in ("brightness", "volume", "slider")):
        return pa, False

    if any(word in goal_lower for word in ("maximum", " max", "highest")):
        direction = "right"
    elif any(word in goal_lower for word in ("minimum", " min", "lowest")):
        direction = "left"
    else:
        return pa, False

    slider_index = _find_continuous_control_index(ui_elements, valid_indices)
    if slider_index is None:
        # Pixel Settings exposes a normal "Brightness level" row first.  The
        # actual SystemUI SeekBar appears only after that row is tapped.
        brightness_row = _find_brightness_level_index(ui_elements, valid_indices)
        args = pa.get("args", [])
        if (
            brightness_row is not None
            and pa.get("name") == "swipe"
            and len(args) >= 2
            and str(args[1]).lower() in ("left", "right")
        ):
            return {"name": "tap", "args": [float(brightness_row)]}, True
        return pa, False

    should_replace = pa.get("name") == "tap" and len(pa.get("args", [])) == 2
    if pa.get("name") == "tap" and len(pa.get("args", [])) == 1:
        try:
            target = _resolve_element(
                int(pa["args"][0]), ui_elements, valid_indices
            )
        except (TypeError, ValueError):
            target = None
        should_replace = bool(target and _is_continuous_control(target))
    if pa.get("name") == "swipe" and len(pa.get("args", [])) >= 2:
        should_replace = str(pa["args"][1]).lower() in ("left", "right")
    if not should_replace:
        return pa, False

    return {
        "name": "swipe",
        "args": [float(slider_index), direction, "long"],
    }, True


def _execute_precise_continuous_swipe(
    pa: dict,
    ui_elements: list[representation_utils.UIElement],
    valid_indices: list[int],
    controller,
) -> bool:
    """Execute a slider swipe with its exact accessibility bounds via ADB."""
    args = pa.get("args", [])
    if pa.get("name") != "swipe" or len(args) < 2:
        return False
    try:
        target = _resolve_element(int(args[0]), ui_elements, valid_indices)
    except (TypeError, ValueError):
        return False
    if target is None or target.bbox_pixels is None or not _is_continuous_control(target):
        return False

    direction = str(args[1]).lower()
    if direction not in ("left", "right", "up", "down"):
        return False
    bounds = target.bbox_pixels
    x_min, x_max = int(bounds.x_min), int(bounds.x_max)
    y_min, y_max = int(bounds.y_min), int(bounds.y_max)
    mid_x = (x_min + x_max) // 2
    mid_y = (y_min + y_max) // 2
    if direction == "right":
        start_x, start_y, end_x, end_y = x_min, mid_y, x_max, mid_y
    elif direction == "left":
        # Material's brightness slider includes a trailing icon/thumb inset
        # inside its accessibility bounds.  Starting at x_max misses the
        # thumb when it is at 100%; its centre is one half-control-height in.
        thumb_inset = min((y_max - y_min) // 2, (x_max - x_min) // 4)
        start_x, start_y, end_x, end_y = (
            x_max - thumb_inset, mid_y, x_min, mid_y
        )
    elif direction == "down":
        start_x, start_y, end_x, end_y = mid_x, y_min, mid_x, y_max
    else:
        start_x, start_y, end_x, end_y = mid_x, y_max, mid_x, y_min

    distance = str(args[2]).lower() if len(args) > 2 else "medium"
    duration = {"short": 250, "medium": 500, "long": 800}.get(distance, 500)
    command = adb_utils.generate_swipe_command(
        start_x, start_y, end_x, end_y, duration
    )
    adb_utils.issue_generic_request(command, controller)
    return True


def _toggle_signature(elem: representation_utils.UIElement) -> tuple:
    """Build a stable-enough identifier for a toggle across observations."""
    bounds = elem.bbox_pixels
    center = None
    if bounds is not None:
        center = (round(bounds.center[0] / 20), round(bounds.center[1] / 20))
    return (
        elem.resource_name or elem.resource_id or "",
        elem.text or "",
        elem.content_description or "",
        elem.class_name or "",
        center,
    )


def _find_toggle_state(
    signature: tuple,
    ui_elements: list[representation_utils.UIElement],
) -> Optional[bool]:
    for elem in ui_elements:
        if elem.is_checkable and _toggle_signature(elem) == signature:
            return bool(elem.is_checked)
    return None


def _desired_toggle_state(goal: str) -> Optional[bool]:
    goal_lower = goal.lower()
    if re.search(r"\b(off|disable|disabled)\b", goal_lower):
        return False
    if re.search(r"\b(on|enable|enabled)\b", goal_lower):
        return True
    return None


def phonecli_to_json_action(
    pa: dict,
    screen_w: int,
    screen_h: int,
    ui_elements: Optional[list[representation_utils.UIElement]] = None,
    valid_indices: Optional[list[int]] = None,
) -> ja.JSONAction:
    """Translate a PhoneCLI parsed action dict to an AndroidWorld JSONAction.

    Supports both index-based (SoM mode: tap(5), long_press(12)) and
    coordinate-based (raw mode: tap(540, 960)) actions.
    """
    name = pa.get("name", "")
    args = pa.get("args", [])
    ui_elems = ui_elements or []
    v_idx = valid_indices or []

    # ---- Index-based actions (SoM mode) ----
    if name == "tap" and len(args) == 1:
        center = _resolve_center(int(args[0]), ui_elems, v_idx)
        if center:
            return ja.JSONAction(action_type=ja.CLICK, x=center[0], y=center[1])
        logger.warning("Ignoring tap on invalid SoM index %s", args[0])
        return ja.JSONAction(action_type=ja.WAIT)

    elif name == "long_press" and len(args) == 1:
        center = _resolve_center(int(args[0]), ui_elems, v_idx)
        if center:
            return ja.JSONAction(action_type=ja.LONG_PRESS, x=center[0], y=center[1])
        logger.warning("Ignoring long_press on invalid SoM index %s", args[0])
        return ja.JSONAction(action_type=ja.WAIT)

    elif name == "swipe" and len(args) >= 2 and isinstance(args[1], str):
        # PhoneCLI describes the physical finger gesture, while AndroidWorld's
        # SCROLL action describes the direction that content should move.  Use
        # SCROLL so a downward gesture starts in the screen center instead of
        # at y=0 (which can accidentally open the notification shade).
        gesture_direction = args[1].lower()
        if gesture_direction not in ("up", "down", "left", "right"):
            gesture_direction = "down"
        scroll_direction = {
            "up": "down",
            "down": "up",
            "left": "right",
            "right": "left",
        }[gesture_direction]
        requested_index = int(args[0])
        element_index = None
        if requested_index > 0:
            array_index = requested_index - 1
            if array_index < len(v_idx):
                element_index = v_idx[array_index]
        return ja.JSONAction(
            action_type=ja.SCROLL,
            index=element_index,
            direction=scroll_direction,
        )

    elif name == "input_text" and len(args) == 2:
        # input_text(index, text): tap field + type + Enter in one step
        # (M3A-style). Resolve the 1-based SoM index to the original
        # accessibility element index for AndroidWorld's INPUT_TEXT.
        arr_idx = int(args[0]) - 1
        if 0 <= arr_idx < len(v_idx):
            return ja.JSONAction(
                action_type=ja.INPUT_TEXT,
                index=v_idx[arr_idx],
                text=str(args[1]),
            )
        logger.warning("Ignoring input_text on invalid SoM index %s", args[0])
        return ja.JSONAction(action_type=ja.WAIT)

    # ---- Coordinate-based actions (raw mode, fallback) ----
    if name == "tap":
        x, y = int(args[0]), int(args[1])
        return ja.JSONAction(action_type=ja.CLICK, x=x, y=y)

    elif name in ("text", "type"):
        return ja.JSONAction(action_type=ja.INPUT_TEXT, text=str(args[0]) if args else "")

    elif name == "swipe":
        x1, y1, x2, y2 = int(args[0]), int(args[1]), int(args[2]), int(args[3])
        dx, dy = x2 - x1, y2 - y1
        if abs(dy) >= abs(dx):
            gesture_direction = "up" if dy < 0 else "down"
        else:
            gesture_direction = "left" if dx < 0 else "right"
        scroll_direction = {
            "up": "down",
            "down": "up",
            "left": "right",
            "right": "left",
        }[gesture_direction]
        return ja.JSONAction(action_type=ja.SCROLL, direction=scroll_direction)

    elif name == "back":
        return ja.JSONAction(action_type=ja.NAVIGATE_BACK)

    elif name == "home":
        return ja.JSONAction(action_type=ja.NAVIGATE_HOME)

    elif name == "launch":
        app = str(args[0]) if args else ""
        return ja.JSONAction(action_type=ja.OPEN_APP, app_name=app)

    elif name == "answer":
        # M3A-style: answer a query without ending the episode.
        # The VLM must call finish() afterwards to terminate.
        return ja.JSONAction(action_type=ja.ANSWER, text=str(args[0]) if args else "")

    elif name == "finish":
        msg = str(args[0]) if args else "Task completed"
        if msg and msg != "Task completed":
            return ja.JSONAction(action_type=ja.ANSWER, text=msg)
        return ja.JSONAction(action_type=ja.STATUS, goal_status="complete")

    elif name == "wait":
        secs = int(args[0]) if args else 2
        return ja.JSONAction(action_type=ja.WAIT)

    elif name == "long_press":
        x, y = int(args[0]), int(args[1])
        return ja.JSONAction(action_type=ja.LONG_PRESS, x=x, y=y)

    elif name == "double_tap" and len(args) == 1:
        center = _resolve_center(int(args[0]), ui_elems, v_idx)
        if center:
            return ja.JSONAction(action_type=ja.DOUBLE_TAP, x=center[0], y=center[1])
        return ja.JSONAction(action_type=ja.WAIT)

    elif name == "keyboard_enter":
        return ja.JSONAction(action_type=ja.KEYBOARD_ENTER)

    else:
        logger.warning("Unknown PhoneCLI action %%s, using no-op", name)
        return ja.JSONAction(action_type=ja.WAIT)
def _parse_call(code: str) -> Optional[dict]:
    """Parse a function-call string like tap(100, 200) into an action dict."""
    m = re.match(r"(\w+)\s*\(\s*(.*?)\s*\)\s*$", code.strip(), re.DOTALL)
    if not m:
        return None
    name = m.group(1).lower()
    raw = m.group(2).strip()

    args = []
    quoted = []
    if raw:
        # Handle quoted string args (text / finish)
        parts = []
        current = ""
        in_string = False
        quote_char = None
        for ch in raw:
            if ch in ('"', "'") and not in_string:
                in_string = True
                quote_char = ch
                current = ""  # discard any whitespace before the quote
            elif ch == quote_char and in_string:
                in_string = False
                quote_char = None
                parts.append(current)
                quoted.append(True)
                current = ""
            elif in_string:
                current += ch
            elif ch == ",":
                if current.strip():
                    parts.append(current.strip())
                    quoted.append(False)
                current = ""
            else:
                current += ch
        if current.strip():
            parts.append(current.strip())
            quoted.append(False)
        if not parts and not in_string:
            pass  # no args
        else:
            args = parts

    # Normalize unquoted numeric args (e.g. tap indices) to float.
    # Quoted string args (e.g. text("+14632272737")) must stay strings —
    # float("+14632272737") would silently corrupt phone numbers.
    result_args = []
    for a, is_quoted in zip(args, quoted):
        if is_quoted:
            result_args.append(a)
            continue
        try:
            result_args.append(float(a))
        except ValueError:
            result_args.append(a)

    return {"name": name, "args": result_args}


def parse_action(text: str) -> Optional[dict]:
    """Parse a PhoneCLI action from model response text."""
    if not text:
        return None

    patterns = [
        r"Action:\s*(.*?)(?=\n|$)",
        r"<CALLED_FUNCTION>\s*(.*?)\s*</CALLED_FUNCTION>",
        r"```(?:\w+)?\s*\n(.*?)\n\s*```",
        r"```\s*(.*?)\s*```",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if m:
            code = m.group(1).strip()
            if code:
                result = _parse_call(code)
                if result:
                    return result

    fallback = [
        r"(tap\([^)]+\))",
        r"(answer\([^)]*\))",
        r"(input_text\([^)]+\))",
        r"(text\([^)]+\))",
        r"(type\([^)]+\))",
        r"(swipe\([^)]+\))",
        r"(back\(\))",
        r"(home\(\))",
        r"(launch\([^)]+\))",
        r"(finish\([^)]*\))",
        r"(wait\([^)]*\))",
        r"(long_press\([^)]+\))",
        r"(scroll\([^)]+\))",
        r"(double_tap\([^)]+\))",
        r"(keyboard_enter\(\))",
    ]
    for pat in fallback:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return _parse_call(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Macro execution helpers — translate app map macro steps → AndroidWorld actions
# ---------------------------------------------------------------------------

def _macro_step_to_json_action(step: dict, app_name: str = "") -> ja.JSONAction:
    """Translate a single macro step dict to a JSONAction for execution."""
    action = step.get("action", "")

    if action == "force_stop":
        # AndroidWorld has no force_stop — go home, then re-launch the app
        # in the next step to get a clean state.
        return ja.JSONAction(action_type=ja.NAVIGATE_HOME)
    elif action == "launch":
        # Use app name (e.g. "Settings") not package name (e.g. "com.android.settings").
        # AndroidWorld's launch_app() matches friendly names via regex patterns.
        name = app_name or step.get("package", "")
        return ja.JSONAction(action_type=ja.OPEN_APP, app_name=name)
    elif action == "tap":
        return ja.JSONAction(
            action_type=ja.CLICK,
            x=int(step["x"]),
            y=int(step["y"]),
        )
    elif action == "swipe":
        x1, y1 = int(step["x1"]), int(step["y1"])
        x2, y2 = int(step["x2"]), int(step["y2"])
        dx, dy = x2 - x1, y2 - y1
        if abs(dy) >= abs(dx):
            direction = "up" if dy < 0 else "down"
        else:
            direction = "left" if dx < 0 else "right"
        return ja.JSONAction(
            action_type=ja.SWIPE,
            direction=direction,
        )
    elif action == "back":
        return ja.JSONAction(action_type=ja.NAVIGATE_BACK)
    elif action == "home":
        return ja.JSONAction(action_type=ja.NAVIGATE_HOME)
    elif action == "wait":
        return ja.JSONAction(action_type=ja.WAIT)
    else:
        logger.warning("Unknown macro action %s", action)
        return ja.JSONAction(action_type=ja.WAIT)


def _make_scroll_json_action(direction: str) -> ja.JSONAction:
    """Create a scroll action into the page-fraction coordinates AndroidWorld expects."""
    d = direction.lower()
    # For a global page scroll, AndroidWorld's actuation layer uses touch positions.
    # SWIPE action type with x,y represent the start point of the swipe in
    # screen-fraction coordinates. We use 0.5 as center and add direction movement.
    return ja.JSONAction(action_type=ja.SWIPE, x=0.5, y=0.5, direction=d)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# VLM prompt: simplified Reason/Action format (aligned with M3A).
# Uses the core action definitions from CLOUD_V0 but with a lighter output
# format that saves tokens for actual screen reading.
_SYSTEM_PROMPT_VLM = """\
You are an intelligent agent tasked with completing specific operations on a smartphone by interacting with its user interface (UI). You will be provided with two screenshots: a raw screenshot and the same screenshot with interactive UI elements labeled with numeric tags starting from 1.

You may only use the following functions with their EXACT parameter formats:

1. tap(index: int)  OR  tap(x: int, y: int)
   # Tap the element with the given index number. Example: tap(5)
   # OR tap raw pixel coordinates: tap(540, 960)
   # Use coordinates for canvas/drawing areas where elements are not indexed
   # individually (e.g. drawing apps, game boards).

2. text(input_str: str)
   # Enter text into the currently focused input field. Example: text("Hello")
   # NOTE: Before using text(), you must first tap() the input field to focus it

3. long_press(index: int)
   # Long press the element with the given index number. Example: long_press(5)

4. double_tap(index: int)
   # Double-tap the element with the given index number. Example: double_tap(5)

5. swipe(index: int, direction: str, dist: str)
   # Swipe on the element with given index, direction and distance
   # direction: MUST be one of ["up", "down", "left", "right"]
   # dist: MUST be one of ["short", "medium", "long"]
   # Example: swipe(21, "up", "medium")
   # Index 0 = full-screen scroll (swipe only, never tap)

6. launch(app_name: str)
   # Open an app by its visible name. Example: launch("Simple Gallery")
   # Use this instead of finding app icons on the home screen.

7. back()
   # Simulate the back button

8. home()
   # Simulate the home button

9. wait(interval: int)
   # Wait for interval seconds (default 5). Example: wait(3)

10. keyboard_enter()
    # Press the enter/return key on the keyboard. Example: keyboard_enter()

11. finish(message: str)
    # Call when task is complete. Example: finish("Task completed")
    # For query tasks: finish("the answer")

IMPORTANT RULES:
- One action per step.
- All index parameters MUST be numbers, matching visible labeled elements.
- NEVER combine parameters from different functions.
- If you can SEE the answer to a query on screen, use finish(answer) immediately.
- If you are stuck, try a different approach (back, search, scroll).
- Review history before acting: if you tapped the same elements repeatedly without progress, STOP and try something completely different.
- For sliders: use swipe(slider_index, 'right', 'long') for max, 'left' for min.

GUIDANCE:
- Usually there are multiple ways to complete a task — pick the easiest one.
  If something does not work (visible from history), SWITCH to other solutions.
- Use launch(app_name) whenever you want to open an app; do not use the app
  drawer to find app icons unless all other ways have failed.
- Use text() whenever you want to type something, instead of clicking keyboard
  characters one by one. If the text field has default text, delete it first
  (long-press backspace accelerates deletion).
- For tap, long_press and double_tap, the index must be VISIBLE in the labeled
  screenshot AND in the UI Elements list.
- Explore screens with swipe(0, direction, dist) to reveal hidden content.
  swipe direction is the GESTURE direction (finger movement): swipe up to see
  content at the bottom. If one direction does not work, try the opposite.
- To select text: long-press on the text area to enter selection mode, adjust
  the range with the two pointers if needed, or tap "select all" in the bar.
- To copy: select the text, then tap "copy" in the selection bar.
- To paste: long-press the text box, then tap "paste" in the bar.
- To delete text: place the cursor and use backspace (long-press to accelerate),
  or select the text then press backspace.
- When typing into an enum field, an auto-complete dropdown may appear —
  select the best match from the list.
- If the desired state is already achieved (e.g. Wi-Fi already on), call
  finish() directly.
- For question tasks, answer explicitly via finish(your answer).

OUTPUT FORMAT (STRICT):
Reason: <one sentence about what you see and why this action>
Action: <single function call>

Example:
Reason: The Settings main page shows "Connected devices" at index 7, which contains Bluetooth settings. I need to tap it.
Action: tap(7)
"""


_QUERY_TASK_RE = re.compile(
    r"\b(what|how many|how much|does|do i|when|is|are|which|list|count|"
    r"how long|answer|find|show me)\b", re.IGNORECASE,
)

# M3A-aligned operating knowledge — FULL translation of M3A's GUIDANCE
# (m3a.py), injected into every VLM step prompt. M3A puts GUIDANCE in the
# action-selection user prompt (not the system prompt); VLM follows
# instructions near the task much more reliably. The action names are
# adapted to this agent's action space (tap/text/input_text/swipe/launch/
# answer/finish) but the operational knowledge is kept complete.
_OPERATING_KNOWLEDGE = (
    "Operating Knowledge:\n"
    "General:\n"
    "- Usually there will be multiple ways to complete a task, pick the "
    "easiest one. Also when something does not work as expected (due to "
    "various reasons), sometimes a simple retry can solve the problem, but "
    "if it doesn't (you can see that from the history), SWITCH to other "
    "solutions.\n"
    '- Sometimes you may need to navigate the phone to gather information '
    'needed to complete the task, for example if user asks "what is my '
    'schedule tomorrow", then you may want to open the calendar app (using '
    'the launch action), look up information there, answer user\'s question '
    "(using the answer action) and finish.\n"
    "- For requests that are questions (or chat messages), remember to use "
    "the answer action to reply to user explicitly before finish! Merely "
    "displaying the answer on screen is NOT sufficient (unless the goal is "
    'something like "show me ...").\n'
    "- If the desired state is already achieved (e.g., enabling Wi-Fi when "
    "it's already on), you can just finish the task.\n"
    "Action Related:\n"
    "- Use the launch action whenever you want to open an app (nothing will "
    "happen if the app is not installed), do not use the app drawer to open "
    "an app unless all other ways have failed.\n"
    "- Use input_text(index, text) whenever you want to type something "
    "(including password) instead of clicking characters on the keyboard "
    "one by one. Sometimes there is some default text in the text field you "
    "want to type in, remember to delete them before typing.\n"
    "- For tap, long_press and input_text, the index parameter you pick "
    "must be VISIBLE in the screenshot and also in the UI element list "
    "given to you (some elements in the list may NOT be visible on the "
    "screen so you can not interact with them).\n"
    "- Consider exploring the screen by using the swipe action with "
    "different directions to reveal additional content.\n"
    "- The direction parameter for the swipe action can be confusing "
    "sometimes as it's opposite to scroll, for example, to view content at "
    "the bottom, swipe UP. It has been observed that you have difficulties "
    "in choosing the correct direction, so if one does not work, try the "
    "opposite as well.\n"
    "Text Related Operations:\n"
    "- Normally to select certain text on the screen: (i) Enter text "
    "selection mode by long pressing the area where the text is, then some "
    "of the words near the long press point will be selected (highlighted "
    "with two pointers indicating the range) and usually a text selection "
    "bar will also appear with options like copy, paste, select all, etc. "
    "(ii) Select the exact text you need. Usually the text selected from "
    "the previous step is NOT the one you want, you need to adjust the "
    "range by dragging the two pointers. If you want to select all text in "
    "the text field, simply tap the select all button in the bar.\n"
    "- At this point, you don't have the ability to drag something around "
    "the screen, so in general you can not select arbitrary text.\n"
    "- To delete some text: the most traditional way is to place the cursor "
    "at the right place and use the backspace button in the keyboard to "
    "delete the characters one by one (can long press the backspace to "
    "accelerate if there are many to delete). Another approach is to first "
    "select the text you want to delete, then tap the backspace button in "
    "the keyboard.\n"
    "- To copy some text: first select the exact text you want to copy, "
    "which usually also brings up the text selection bar, then tap the "
    "copy button in the bar.\n"
    "- To paste text into a text box, first long press the text box, then "
    "usually the text selection bar will appear with a paste button in it.\n"
    "- When typing into a text field, sometimes an auto-complete dropdown "
    "list will appear. This usually indicating this is a enum field and "
    "you should try to select the best match by tapping the corresponding "
    "one in the list.\n"
)


def _cloud_vlm_prompt(
    goal: str,
    history_text: str,
    ui_elements_text: str,
    screen_w: int = 1080,
    screen_h: int = 2400,
    vlm_fallback: bool = False,
) -> str:
    """Build the VLM step prompt (M3A-aligned: simple, focused on the task)."""
    parts = [f"Task: {goal}"]
    if history_text:
        parts.append(f"History:\n{history_text}")
    if ui_elements_text:
        parts.append(f"UI Elements:\n{ui_elements_text}")
    parts.append(
        "Note: index 0 is full-screen scroll (swipe only, never tap). "
        "Elements tagged CONTINUOUS are slider targets. For exact min/max, "
        "use swipe(index,'right','long') for max or swipe(index,'left','long') "
        "for min. If no CONTINUOUS element is visible, tap the setting row "
        "first to reveal the slider."
    )
    if vlm_fallback:
        parts.append(
            "This is an information retrieval task. CRITICAL: the data you "
            "see now is usually an unfiltered DEFAULT view \u2014 it is almost "
            "never the complete answer. You MUST first navigate to the view "
            "matching the task's filter criteria (specific date, priority, "
            "week, named item, etc.), confirm the visible data satisfies ALL "
            "conditions in the task, and only then call finish(your answer). "
            "Finishing from the default view is the most common failure mode."
        )
    parts.append("Two screenshots are provided: raw (unlabeled) and labeled with UI element indices.")
    parts.append(_OPERATING_KNOWLEDGE)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LLM map-task response parser (mirrors evaluation/macro_agent.py)
# ---------------------------------------------------------------------------

def _parse_llm_map_response(response: str) -> dict:
    response = response.strip()
    for line in response.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.upper().startswith("OP:"):
            return {"type": "op", "op_id": line[3:].strip()}
        elif line.upper().startswith("MACRO_VLM:"):
            return {"type": "macro_vlm", "op_id": line[10:].strip()}
        elif line.upper().startswith("NEED_VLM:"):
            return {"type": "need_vlm", "reason": line[9:].strip()}
        elif line.upper().startswith("FINISH:"):
            return {"type": "finish", "answer": line[7:].strip()}
    return {"type": "need_vlm", "reason": "Unparseable response"}


# ---------------------------------------------------------------------------
# Main agent class
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class _StepData:
    """Per-step metadata recorded for history."""
    summary: str = ""
    action_json: Optional[ja.JSONAction] = None
    raw_response: str = ""


class PhoneCLIAndroidWorldAgent(base_agent.EnvironmentInteractingAgent):
    """PhoneCLI hybrid agent wrapped as an AndroidWorld EnvironmentInteractingAgent.

    Round 1 (optional, with app_map):
        LLM maps the task to a pre-defined operation → replays the ADB macro →
        VLM verifies completion. If MACRO_VLM, hands off to VLM for interaction.

    Round 2+:
        Pure VLM — takes a screenshot, produces a coordinate-based action.

    Without an app_map, all rounds are VLM-based from the start.
    """

    def __init__(
        self,
        env: interface.AsyncEnv,
        name: str = "phonecli",
        app_map=None,
        app_maps: Optional[dict] = None,
        llm_config: Optional[dict] = None,
        api_key: str = "EMPTY",
        api_base: str = "http://localhost:8002/v1",
        model: str = "Qwen/Qwen2.5-3B-Instruct",
        vlm_model: Optional[str] = None,
    ):
        """Initialize the agent.

        Args:
            env: AndroidWorld AsyncEnv.
            name: Agent display name.
            app_map: Optional AppMap instance for macro routing (Round 1).
            app_maps: Optional package-to-AppMap lookup.  When supplied, the
                AndroidWorld runner calls ``prepare_for_task`` before each
                episode so the map follows the task's declared app.
            llm_config: Dict with api_key, api_base, model for text LLM calls.
            api_key: VLM API key (fallback if not in llm_config).
            api_base: VLM API base URL (fallback).
            model: VLM model name (fallback).
            vlm_model: Override VLM model. If None, uses llm_config["model"] or `model`.
        """
        super().__init__(env, name, transition_pause=None)
        self.app_map = None
        self.app_maps = dict(app_maps or {})
        if app_map is not None and getattr(app_map, "package", None):
            self.app_maps.setdefault(app_map.package, app_map)
        self.llm_config = llm_config or {}

        # Resolve API credentials
        self.vlm_api_key = (
            self.llm_config.get("api_key") or api_key
        )
        self.vlm_api_base = (
            self.llm_config.get("api_base") or api_base
        )
        self.vlm_model = (
            vlm_model or self.llm_config.get("model") or model
        )

        # Macro phase state
        self._round = 0
        self._macro_phase_done = False
        self._fruitless_rounds = 0
        self._ops_catalog: dict = {}

        # History of VLM steps
        self._history: list[_StepData] = []
        self._pending_toggle: Optional[dict] = None
        self.task_usage_records: list[dict] = []
        self._task_usage_snapshot: Optional[dict] = None

        self.set_app_map(app_map)

    def set_app_map(self, app_map) -> None:
        """Switch the macro map and rebuild its operation catalog."""
        self.app_map = app_map
        self._ops_catalog = {}
        if app_map is None:
            return
        try:
            self._ops_catalog = app_map.build_operations()
            logger.info(
                "App map selected: %s (%s), %d operations",
                app_map.app_name,
                app_map.package,
                len(self._ops_catalog),
            )
        except Exception:
            logger.warning("Failed to build operations catalog from app map")

    def prepare_for_task(self, task) -> None:
        """Select the first available map declared by an AndroidWorld task.

        AndroidWorld exposes the real app list on each instantiated task,
        including dynamically generated information-retrieval tasks.  Prefer
        a concrete ``app_name`` parameter (notably OpenAppTaskEval), then fall
        back to the declared order for multi-app tasks.
        """
        from aworld_agent.token_usage import token_usage

        self._task_usage_snapshot = token_usage.snapshot()
        self._pending_toggle = None
        candidate_names = []
        params = getattr(task, "params", {}) or {}
        parameter_app = params.get("app_name")
        if isinstance(parameter_app, str) and parameter_app:
            candidate_names.append(parameter_app)
        candidate_names.extend(getattr(task, "app_names", ()) or ())

        selected = None
        selected_name = None
        seen = set()
        for app_name in candidate_names:
            if not app_name or app_name in seen:
                continue
            seen.add(app_name)
            activity = adb_utils.get_adb_activity(app_name)
            package = activity.split("/", 1)[0] if activity else None
            if package and package in self.app_maps:
                selected = self.app_maps[package]
                selected_name = app_name
                break

        self.set_app_map(selected)
        if selected is None:
            logger.info(
                "Task %s has no matching map for apps=%s; using pure VLM",
                getattr(task, "name", type(task).__name__),
                tuple(candidate_names),
            )
        else:
            logger.info(
                "Task %s uses map %s (declared app=%s)",
                getattr(task, "name", type(task).__name__),
                selected.app_name,
                selected_name,
            )

    def on_task_end(self, task, result) -> None:
        """Record the exact token delta for one completed episode."""
        if self._task_usage_snapshot is None:
            return
        from aworld_agent.token_usage import token_usage

        record = token_usage.diff(self._task_usage_snapshot)
        if isinstance(result, dict):
            metadata = {
                "is_successful": result.get("is_successful"),
                "episode_length": result.get("episode_length"),
                "run_time": result.get("run_time"),
                "has_error": bool(result.get("exception_info")),
            }
        else:
            # AndroidWorld invokes this lifecycle hook with EpisodeResult
            # before it computes the benchmark score and runtime.
            step_data = getattr(result, "step_data", {}) or {}
            step_numbers = step_data.get("step_number", [])
            metadata = {
                "is_successful": None,
                "episode_length": len(step_numbers),
                "run_time": None,
                "has_error": False,
                "agent_done": bool(getattr(result, "done", False)),
            }
        record.update({
            "task": getattr(task, "name", type(task).__name__),
            **metadata,
        })
        self.task_usage_records.append(record)
        self._task_usage_snapshot = None

    # ------------------------------------------------------------------
    # AndroidWorld interface
    # ------------------------------------------------------------------

    def reset(self, go_home: bool = False):
        super().reset(go_home)
        self._round = 0
        self._macro_phase_done = False
        self._fruitless_rounds = 0
        self._history = []
        self._pending_toggle = None

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        self._round += 1
        logger.info("=== PhoneCLI step %d ===", self._round)

        # ---- Round 1: try macro routing ----
        if self._round == 1 and not self._macro_phase_done and self.app_map:
            result = self._round_macro(goal)
            if result is not None:
                return result

        # ---- Round 2+: cloud-style VLM ----
        return self._round_vlm(goal)

    # ------------------------------------------------------------------
    # Round 1: macro routing
    # ------------------------------------------------------------------

    def _round_macro(self, goal: str) -> Optional[base_agent.AgentInteractionResult]:
        """Execute macro routing for the first round.

        Returns an AgentInteractionResult if the task was completed or the
        macro was executed (to hand off to VLM). Returns None if macro routing
        failed and VLM should be used instead.
        """
        map_result = self._llm_map_task(goal)
        logger.info("Macro map result: %s", map_result["type"])

        # FINISH: query answerable without action
        if map_result["type"] == "finish":
            self._macro_phase_done = True
            return base_agent.AgentInteractionResult(
                done=True,
                data={"macro_result": map_result, "action": "finish"},
            )

        # OP / MACRO_VLM: execute the mapped macro
        if map_result["type"] in ("op", "macro_vlm"):
            op = self._ops_catalog.get(map_result["op_id"]) if self._ops_catalog else None
            if op and op.macro:
                success = self._execute_macro(op)
                if success and map_result["type"] == "op":
                    # OP path: verify completion
                    if self._vlm_verify(goal):
                        self._macro_phase_done = True
                        return base_agent.AgentInteractionResult(
                            done=True,
                            data={"macro_result": map_result, "action": "macro_finish"},
                        )

                    # Macro executed but task not complete — fall to VLM
                    logger.info("Macro ok but task not complete, switching to VLM")

                if success:
                    # MACRO_VLM path: macro navigated, hand off to VLM
                    self._macro_phase_done = True
                    logger.info("Macro navigated, switching to VLM")

                    # Verify landing: check detected screen matches expected target.
                    # If mismatch, kill/relaunch app to avoid misleading the VLM.
                    screen_context, detected_screen_id = self._detect_screen_context()
                    if self._landing_mismatch(
                        op, screen_context, detected_screen_id
                    ):
                        logger.info("Landing mismatch — resetting app for VLM fallback")
                        if self.app_map and self.app_map.package:
                            try:
                                self.env.execute_action(
                                    ja.JSONAction(action_type=ja.NAVIGATE_HOME))
                                time.sleep(0.3)
                                self.env.execute_action(
                                    ja.JSONAction(action_type=ja.OPEN_APP,
                                                   app_name=self.app_map.app_name))
                                time.sleep(2.0)
                            except Exception:
                                pass
                        # Let step() proceed to _round_vlm from clean state
                        return None

                    # Inject macro navigation context into VLM history so the VLM
                    # starts with awareness of where it is and what to look for.
                    # Include the target screen's expected interactive elements
                    # (mirrors evaluation/macro_agent.py) so the VLM knows what
                    # to look for instead of exploring blindly.
                    _parts = [
                        f"Macro completed: navigated to target screen via {op.id}.",
                        f"Task: {goal}.",
                    ]
                    if screen_context:
                        _parts.append(screen_context.strip())
                    target_screen = getattr(op, "target_screen", "")
                    if target_screen:
                        try:
                            _screen = self.app_map.get_screen(target_screen)
                            if _screen and _screen.elements:
                                _interactive = [
                                    e for e in _screen.elements
                                    if e.semantic_type or e.leads_to
                                ]
                                if _interactive:
                                    _hints = [
                                        f"{e.text}[{e.semantic_type or 'nav'}]"
                                        for e in _interactive[:8]
                                    ]
                                    _parts.append(
                                        "Expected elements: "
                                        + ", ".join(_hints) + "."
                                    )
                        except Exception:
                            pass
                    _parts.append(
                        "Look at the current screen and complete the task step "
                        "by step. The macro has only navigated here \u2014 you must "
                        "still perform any interactions (tap, type, swipe, "
                        "etc.) that the task requires."
                    )
                    state_assessment = " ".join(_parts)
                    self._history.append(_StepData(
                        summary=state_assessment[:400],
                        raw_response=f"MACRO_VLM: {op.id}",
                    ))
                    return None  # let step() proceed to _round_vlm

        # No match or macro failed — fall back to pure VLM.
        # No macro was executed, so the app is already in its initial state
        # from task setup.  Killing + relaunching would waste time; just hand
        # control to the VLM (mirrors evaluation/macro_agent.py).
        logger.info("No matching macro or macro failed, falling back to VLM")
        self._macro_phase_done = True
        return None

    def _llm_map_task(self, goal: str) -> dict:
        """LLM maps the task to an operation in the app map catalog.

        Phase 1: LLM task→operation mapping.
        Phase 2: LLM verifies the candidate operation's target page matches the task.
        """
        if not self._ops_catalog:
            return {"type": "need_vlm", "reason": "No operations catalog"}

        ops_str = self.app_map.format_ops_catalog(self._ops_catalog, nav_only=True)
        if not ops_str.strip():
            return {"type": "need_vlm", "reason": "Empty operations catalog"}

        system_prompt = MACRO_PLAN_PROMPT.format(
            app_name=self.app_map.app_name,
            operations_catalog=ops_str,
            memory_context="",
        )

        try:
            llm_api_key = self.llm_config.get("api_key") or self.vlm_api_key
            llm_api_base = self.llm_config.get("api_base") or self.vlm_api_base
            llm_model = self.llm_config.get("model") or self.vlm_model
            response = text_completion(
                system_prompt=system_prompt,
                user_prompt=f"Task: {goal}",
                api_key=llm_api_key,
                api_base=llm_api_base,
                model=llm_model,
                label="aw_map_task",
            )
        except Exception as e:
            logger.warning("LLM map-task failed: %s", e)
            return {"type": "need_vlm", "reason": str(e)}

        result = _parse_llm_map_response(response)
        logger.info("Phase 1 → %s %s", result["type"], result.get("op_id", ""))

        # Phase 2: verify candidate operation against target screen description
        if result["type"] in ("op", "macro_vlm") and result.get("op_id"):
            if not self._verify_operation(goal, result):
                logger.info("Phase 2 → REJECTED, falling back to VLM")
                return {"type": "need_vlm",
                        "reason": "Target page unmatched — fallback to VLM"}

        return result

    def _verify_operation(self, goal: str, phase1_result: dict) -> bool:
        """Phase 2: confirm the selected operation navigates to a useful page."""
        op_id = phase1_result.get("op_id")
        if not op_id:
            return True

        op = self._ops_catalog.get(op_id)
        if not op:
            return True

        screen_desc = self.app_map.get_target_screen_info(
            self._ops_catalog, op_id)
        if not screen_desc:
            return True  # No description available — trust Phase 1

        verify_prompt = MACRO_VERIFY_PROMPT.format(
            task=goal,
            op_description=op.description,
            screen_description=screen_desc,
        )

        try:
            llm_api_key = self.llm_config.get("api_key") or self.vlm_api_key
            llm_api_base = self.llm_config.get("api_base") or self.vlm_api_base
            llm_model = self.llm_config.get("model") or self.vlm_model
            response = text_completion(
                system_prompt=verify_prompt,
                user_prompt="YES / NO / UNCLEAR?",
                api_key=llm_api_key,
                api_base=llm_api_base,
                model=llm_model,
                max_tokens=16,
                temperature=0.0,
                label="aw_map_verify",
            )
        except Exception as e:
            logger.warning("Phase 2 failed: %s", e)
            return True  # On error, trust Phase 1

        verdict = response.strip().upper()
        logger.info("Phase 2 screen='%s...' → %s", screen_desc[:80], verdict)
        return verdict.startswith("YES") or verdict.startswith("UNCLEAR")

    def _execute_macro(self, op) -> bool:
        """Replay a macro operation's steps using AndroidWorld env actions."""
        if not op or not op.macro:
            return False

        logger.info("Replaying macro %s (%d steps)", op.id, len(op.macro))
        for i, step in enumerate(op.macro):
            wait_seconds = step.get("wait", 0.5)
            action_str = step.get("action", "")
            try:
                if action_str == "swipe":
                    # Map crawling records exact swipe coordinates.  Replaying
                    # through JSONAction would expand this to a full-screen
                    # fling, changing which row remains at the recorded tap
                    # coordinate.  Send the recorded gesture through the same
                    # ADB path used by AndroidWorld instead.
                    command = adb_utils.generate_swipe_command(
                        int(step["x1"]), int(step["y1"]),
                        int(step["x2"]), int(step["y2"]),
                        int(step.get("duration", 400)),
                    )
                    adb_utils.issue_generic_request(command, self.env.controller)
                else:
                    aw_action = _macro_step_to_json_action(
                        step,
                        app_name=self.app_map.app_name if self.app_map else "",
                    )
                    self.env.execute_action(aw_action)
                logger.debug("  [%d/%d] %s", i + 1, len(op.macro), action_str)
            except Exception as e:
                logger.warning("  Macro step [%d] failed: %s — %s", i, action_str, e)
                return False

            if wait_seconds:
                time.sleep(wait_seconds)

        logger.info("Macro replay complete")
        return True

    # ------------------------------------------------------------------
    # Screen detection & landing verification
    # ------------------------------------------------------------------

    @staticmethod
    def _build_pseudo_xml(ui_elements: list[representation_utils.UIElement]) -> str:
        """Build a minimal XML-like string from UI elements for screen matching.

        The existing AppMap.identify_current_screen() parses <node> tags with
        text/content-desc attributes. We reconstruct a compatible format here
        so we don't need raw ADB uiautomator dump.
        """
        lines = []
        for elem in ui_elements:
            attrs = []
            if elem.text:
                attrs.append(f'text="{elem.text}"')
            if elem.content_description:
                attrs.append(f'content-desc="{elem.content_description}"')
            if elem.resource_id:
                attrs.append(f'resource-id="{elem.resource_id}"')
            if attrs:
                lines.append(f'<node {" ".join(attrs)}/>')
        return "\n".join(lines)

    def _detect_screen_context(self) -> tuple[str, Optional[str]]:
        """Identify current screen after macro replay using the app map.

        Builds a pseudo-XML from the current UI elements and matches against
        the app map's known screens.
        """
        if not self.app_map:
            return "", None
        try:
            state = self.env.get_state(wait_to_stabilize=True)
            xml_str = self._build_pseudo_xml(state.ui_elements)
            if not xml_str:
                return "", None

            sid, confidence = self.app_map.identify_current_screen(xml_str)
            if sid and confidence >= 0.4:
                hint = self.app_map.build_enriched_screen_hint(sid)
                logger.info("Post-macro screen: %s (%.0f%%)", sid, confidence * 100)
                return f"Current screen context: {hint}. ", sid
        except Exception as e:
            logger.warning("Screen detection: %s", e)
        return "", None

    def _landing_mismatch(
        self,
        op,
        screen_context: str,
        detected_screen_id: Optional[str] = None,
    ) -> bool:
        """Check if the detected screen matches the macro operation's target.

        Returns True if the landing page is clearly wrong.
        """
        if not screen_context or not op:
            return False

        # The map's screen matcher is the authoritative landing check.  Text
        # overlap between an operation description and an unenriched screen
        # hint is only a fallback: e.g. an operation called "Dark theme, font
        # size, brightness" can correctly land on screen_9 whose basic hint
        # contains only its screen id and a fixed "Navigate up" element.
        target_screen = getattr(op, "target_screen", "")
        if target_screen and detected_screen_id:
            mismatch = detected_screen_id != target_screen
            if mismatch:
                logger.info(
                    "Landing mismatch: expected screen %s, detected %s",
                    target_screen,
                    detected_screen_id,
                )
            return mismatch

        target_words = set()
        for part in op.description.split(" → "):
            for w in part.lower().split():
                if len(w) > 2:
                    target_words.add(w)

        if not target_words:
            return False

        context_lower = screen_context.lower()
        matches = sum(1 for w in target_words if w in context_lower)
        mismatch = matches < min(2, len(target_words))
        if mismatch:
            logger.info("Landing mismatch: expected %s, got '%s...'",
                        target_words, screen_context[:100])
        return mismatch

    # ------------------------------------------------------------------
    # VLM verify (after macro)
    # ------------------------------------------------------------------

    def _vlm_verify(self, goal: str) -> bool:
        """Use VLM to check if the task is complete after macro replay."""
        state = self.env.get_state(wait_to_stabilize=True)
        pixels = state.pixels
        if pixels is None:
            return True  # can't verify, assume success

        fd, path = tempfile.mkstemp(suffix=".png", prefix="aw_verify_")
        os.close(fd)
        try:
            img = Image.fromarray(pixels.astype(np.uint8))
            img.save(path)

            prompt = VLM_VERIFY_PROMPT.replace("{task}", goal)
            response = vision_completion(
                system_prompt=prompt,
                user_text=f"Task: {goal}\nIs this task complete?",
                image_paths=[path],
                api_key=self.vlm_api_key,
                api_base=self.vlm_api_base,
                model=self.vlm_model,
                label="aw_verify",
            )
            is_complete = "COMPLETE" in response.upper()
            logger.info("VLM verify: %s", "COMPLETE" if is_complete else "INCOMPLETE")
            return is_complete
        except Exception as e:
            logger.warning("VLM verify failed: %s", e)
            return True
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Round 2+: cloud-style VLM
    # ------------------------------------------------------------------

    def _round_vlm(self, goal: str) -> base_agent.AgentInteractionResult:
        """VLM step with SoM-labeled screenshot and index-based actions.

        Same design as Android Lab macro agent: SoM-annotated screenshot with
        numbered UI elements, VLM outputs tap(index) / swipe(index, dir, dist),
        and the agent resolves indices to pixel coordinates internally.
        """
        state = self.env.get_state(wait_to_stabilize=True)
        screen_w, screen_h = self.env.device_screen_size
        pixels = state.pixels
        ui_elements = state.ui_elements

        # A toggle can update visually before its accessibility CHECKED state,
        # especially when disabling Wi-Fi.  Reserve the next agent step for a
        # fresh observation instead of asking the VLM to click the same switch
        # again from contradictory evidence.
        pending_toggle = getattr(self, "_pending_toggle", None)
        if pending_toggle is not None:
            observed = _find_toggle_state(
                pending_toggle["signature"], ui_elements
            )
            before = pending_toggle.get("before")
            desired = pending_toggle.get("desired")
            if observed is None:
                observation = "the toggle is temporarily absent from accessibility"
            elif observed != before:
                observation = f"accessibility changed to {'ON' if observed else 'OFF'}"
            else:
                observation = (
                    f"accessibility still reports {'ON' if observed else 'OFF'}; "
                    "the screenshot may already show the new state"
                )
            if desired is not None and observed == desired:
                observation += " (the requested state)"
            summary = (
                "[SYSTEM OBSERVATION: After the previous toggle, "
                f"{observation}. Waited for another stable observation rather "
                "than immediately toggling it again.]"
            )
            wait_action = ja.JSONAction(action_type=ja.WAIT)
            try:
                self.env.execute_action(wait_action)
            except Exception as exc:
                summary += f" [Wait action failed: {exc}]"
            self._history.append(_StepData(
                summary=summary,
                action_json=wait_action,
                raw_response="TOGGLE_SETTLE_GUARD",
            ))
            self._pending_toggle = None
            return base_agent.AgentInteractionResult(
                done=False,
                data={
                    "phonecli_action": {"name": "wait", "args": [1]},
                    "androidworld_action": repr(wait_action),
                    "n_ui_elements": 0,
                    "raw_response": "TOGGLE_SETTLE_GUARD",
                    "toggle_observed": observed,
                },
            )

        # Annotate screenshot with SoM bounding boxes + indices
        labeled_pixels = None
        valid_indices: list[int] = []
        if pixels is not None and _HAS_CV2:
            labeled_pixels, valid_indices = _annotate_screenshot(
                pixels, ui_elements, screen_h)
        elif pixels is not None and not _HAS_CV2:
            # cv2 not available — fall back to raw screenshot, all UI elements
            # are treated as valid (index-based actions won't resolve)
            labeled_pixels = pixels

        # Build UI element text list
        ui_text = _build_ui_elements_list(ui_elements, valid_indices)

        fd_raw, raw_path = tempfile.mkstemp(suffix=".png", prefix="aw_raw_")
        os.close(fd_raw)
        fd, som_path = tempfile.mkstemp(suffix=".png", prefix="aw_som_")
        os.close(fd)
        try:
            if pixels is not None:
                img_raw = Image.fromarray(pixels.astype(np.uint8))
                img_raw.save(raw_path)
            if labeled_pixels is not None:
                img = Image.fromarray(labeled_pixels.astype(np.uint8))
                img.save(som_path)

            # Build structured history from step summaries (M3A format:
            # "Step N- <summary>")
            history_text = ""
            if self._history:
                valid = [h for h in self._history if h is not None]
                if valid:
                    history_text = "\n".join(
                        f"Step {i}- {h.summary}"
                        for i, h in enumerate(valid, 1)
                    )
            if not history_text:
                history_text = "You just started, no action has been performed yet."

            prompt = _cloud_vlm_prompt(
                goal, history_text, ui_text, screen_w, screen_h,
                vlm_fallback=_QUERY_TASK_RE.search(goal) is not None,
            )

            try:
                response = vision_completion(
                    system_prompt=_SYSTEM_PROMPT_VLM,
                    user_text=prompt,
                    image_paths=[raw_path, som_path] if pixels is not None else [som_path],
                    max_tokens=1500,
                    api_key=self.vlm_api_key,
                    api_base=self.vlm_api_base,
                    model=self.vlm_model,
                    label="aw_vlm",
                )
            except Exception as e:
                logger.error("VLM call failed: %s", e)
                return base_agent.AgentInteractionResult(
                    done=True,
                    data={"error": str(e), "action": "vlm_error"},
                )

            pa = parse_action(response)
            if pa is None:
                logger.warning(
                    "Could not parse action from VLM response: %s", response[:200]
                )
                step_data = _StepData(
                    summary=f"Unparseable response: {response[:100]}",
                    raw_response=response,
                )
                self._history.append(step_data)
                return base_agent.AgentInteractionResult(
                    done=False,
                    data={"raw_response": response, "action": "parse_error"},
                )

            logger.info(
                "Parsed action: %s(%s)  |  %d UI elements",
                pa["name"], pa.get("args", []), len(valid_indices),
            )

            summary = self._extract_summary(response)

            pa, slider_action_replaced = _normalize_continuous_control_action(
                goal, pa, ui_elements, valid_indices
            )
            if slider_action_replaced:
                summary = (
                    f"{summary} [SYSTEM ACTION CORRECTION: Applied the required "
                    "brightness-control step: open the Brightness level row "
                    "before the slider exists, or use a full edge-to-edge "
                    "swipe once its SeekBar is visible.]"
                )
                logger.info("Corrected continuous-control action to %s", pa)

            # Detect fruitless rounds: consecutive swipes → inject search hint
            called_fn = pa.get("name", "")
            if called_fn in ("swipe",):
                self._fruitless_rounds += 1
            else:
                self._fruitless_rounds = 0

            if self._fruitless_rounds >= 3:
                summary = (
                    f"{summary} "
                    f"[SYSTEM HINT: After several scroll attempts without finding "
                    f"the target, try using the search bar at the top of the screen "
                    f"to search for the relevant setting directly.]"
                )

            done = pa.get("name") == "finish"

            # Translate to AndroidWorld action — resolve index to pixel coordinates
            aw_action = phonecli_to_json_action(
                pa, screen_w, screen_h, ui_elements, valid_indices,
            )
            toggle_target = None
            if pa.get("name") == "tap" and len(pa.get("args", [])) == 1:
                try:
                    candidate = _resolve_element(
                        int(pa["args"][0]), ui_elements, valid_indices
                    )
                except (TypeError, ValueError):
                    candidate = None
                if candidate is not None and candidate.is_checkable:
                    toggle_target = candidate
            if (
                pa.get("name") in ("tap", "long_press")
                and aw_action.action_type == ja.WAIT
            ):
                summary = (
                    f"{summary} [SYSTEM HINT: The selected SoM index is invalid. "
                    f"Index 0 is swipe-only; choose one of the visible numbered "
                    f"text elements from 1 to {len(valid_indices)}.]"
                )

            action_succeeded = False
            precise_slider_swipe = False
            try:
                precise_slider_swipe = _execute_precise_continuous_swipe(
                    pa, ui_elements, valid_indices, self.env.controller
                )
                if not precise_slider_swipe:
                    self.env.execute_action(aw_action)
                action_succeeded = True
            except Exception as e:
                logger.error("Action execution failed: %s — %s", aw_action, e)
                summary = (
                    f"{summary} "
                    f"[SYSTEM HINT: The previous action failed with error: "
                    f"'{e}'. Choose a different element or approach.]"
                )

            if action_succeeded and precise_slider_swipe:
                # The corrected gesture targets the exact min/max endpoint and
                # ADB waits for the swipe to finish.  End now: asking the VLM
                # to visually re-confirm a value it cannot read causes it to
                # repeat the gesture until AndroidWorld's step limit.
                done = True
                summary = (
                    f"{summary} [SYSTEM COMPLETION: The exact endpoint swipe "
                    "completed successfully; finish this episode without "
                    "repeating the gesture.]"
                )

            if action_succeeded and toggle_target is not None:
                before_checked = bool(toggle_target.is_checked)
                time.sleep(_TOGGLE_SETTLE_SECONDS)
                try:
                    refreshed_state = self.env.get_state(wait_to_stabilize=True)
                    after_checked = _find_toggle_state(
                        _toggle_signature(toggle_target),
                        refreshed_state.ui_elements,
                    )
                except Exception as exc:
                    logger.warning("Toggle refresh observation failed: %s", exc)
                    after_checked = None
                self._pending_toggle = {
                    "signature": _toggle_signature(toggle_target),
                    "before": before_checked,
                    "after": after_checked,
                    "desired": _desired_toggle_state(goal),
                }
                summary = (
                    f"{summary} [SYSTEM OBSERVATION: Waited "
                    f"{_TOGGLE_SETTLE_SECONDS:.0f}s after toggling. The next "
                    "step must observe the stabilized state and must not "
                    "immediately toggle the same switch again.]"
                )

            # M3A-aligned: post-hoc summary with before/after UI elements + reason
            if action_succeeded and not done:
                try:
                    # Fixed settle time + single fast read (M3A-style), instead
                    # of wait_to_stabilize=True which polls up to 6s per call.
                    time.sleep(1.5)
                    after_state = self.env.get_state(wait_to_stabilize=False)
                    after_pixels = after_state.pixels
                    if after_pixels is not None and pixels is not None:
                        fd_after, after_path = tempfile.mkstemp(
                            suffix=".png", prefix="aw_after_")
                        os.close(fd_after)
                        try:
                            img_after = Image.fromarray(
                                after_pixels.astype(np.uint8))
                            img_after.save(after_path)

                            # Build after-action UI element list
                            after_ui = after_state.ui_elements
                            after_ui_text = _build_ui_elements_list(
                                after_ui, list(range(len(after_ui)))
                            ) if after_ui else ""

                            action_reason = self._extract_reason(response)
                            # M3A-aligned summary template: explicitly asks
                            # the model to be CRITICAL about whether the
                            # action/reason was right (this drives the
                            # self-correcting behavior seen in M3A traces).
                            summary_prompt = (
                                f"The (overall) user goal/request is: {goal}\n"
                                f"Now I want you to summarize the latest step.\n"
                                f"You will be given the screenshot before you "
                                f"performed the action (raw), the action you "
                                f"chose (together with the reason) and the "
                                f"screenshot after the action was performed.\n"
                                f"Also here is the list of detailed information "
                                f"for some UI elements in the before "
                                f"screenshot:\n{ui_text}\n"
                                f"Here is the list for the after screenshot:\n"
                                f"{after_ui_text}\n"
                                f"This is the action you picked: "
                                f"{pa.get('name', '?')}({pa.get('args', [])})\n"
                                f"Based on the reason: {action_reason}\n\n"
                                f"By comparing the two screenshots (plus the "
                                f"UI element lists) and the action performed, "
                                f"give a brief summary of this step. This "
                                f"summary will be added to action history and "
                                f"used in future action selection, so try to "
                                f"include essential information you think that "
                                f"will be most useful for future action "
                                f"selections like what you intended to do, why, "
                                f"if it worked as expected, if not what might "
                                f"be the reason (BE CRITICAL, the action/reason "
                                f"might be wrong), what should/should not be "
                                f"done next and so on. Some more rules/tips: "
                                f"- Keep it short (better less than 50 words) "
                                f"and in a single line\n"
                                f"- Some actions (like answer, wait) don't "
                                f"involve screen change, you can just assume "
                                f"they work as expected.\n\n"
                                f"Summary of this step: "
                            )
                            summary_response = vision_completion(
                                system_prompt=_SYSTEM_PROMPT_VLM,
                                user_text=summary_prompt,
                                image_paths=[raw_path, after_path],
                                api_key=self.vlm_api_key,
                                api_base=self.vlm_api_base,
                                model=self.vlm_model,
                                max_tokens=200,
                                label="aw_summary",
                            )
                            summary = summary_response.strip()[:200]
                        except Exception:
                            pass
                        finally:
                            try:
                                os.remove(after_path)
                            except OSError:
                                pass
                except Exception:
                    pass

            step_data = _StepData(
                summary=summary,
                action_json=aw_action,
                raw_response=response,
            )
            self._history.append(step_data)

            return base_agent.AgentInteractionResult(
                done=done,
                data={
                    "phonecli_action": pa,
                    "androidworld_action": repr(aw_action),
                    "execution": (
                        "precise_continuous_swipe"
                        if precise_slider_swipe
                        else "androidworld"
                    ),
                    "n_ui_elements": len(valid_indices),
                    "raw_response": response,
                },
            )
        finally:
            try:
                os.remove(raw_path)
            except OSError:
                pass
            try:
                os.remove(raw_path)
            except OSError:
                pass
            try:
                os.remove(som_path)
            except OSError:
                pass

    def _extract_reason(self, response: str) -> str:
        """Extract the Reason: line from a VLM response."""
        m = re.search(r"Reason:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return ""

    def _extract_summary(self, response: str) -> str:
        """Extract a structured summary from the VLM response for history."""
        m = re.search(r"Reason:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if m:
            return m.group(1).strip()[:200]
        for line in response.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("Action:") and not stripped.startswith("```"):
                return stripped[:200]
        return response[:200]
