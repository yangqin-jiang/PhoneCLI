"""M3A agent + PhoneCLI app-map macro routing.

Round 1: LLM maps the task to an app-map operation; if a macro matches,
replay it (navigate to the target screen) and hand off to the vanilla M3A
loop with a landing hint.  If no macro matches, fall straight through to
M3A — no forced relaunch, no prompt rewriting.  This keeps M3A's proven
prompt / action / SoM machinery 100% intact (the source of its 75/116)
while preserving the app-map's step-saving navigation value.
"""

import base64
import io
import json
import logging
import re
import time
from typing import Optional

import numpy as np
from PIL import Image

from android_world.agents import base_agent
from android_world.agents import m3a
from android_world.env import adb_utils
from android_world.env import json_action as ja

from aworld_agent.llm_client import text_completion, vision_completion
from aworld_agent.prompts import (
    MACRO_PLAN_PROMPT,
    MACRO_VERIFY_PROMPT,
    VLM_VERIFY_PROMPT,
)

logger = logging.getLogger(__name__)

_QUERY_TASK_RE = re.compile(
    r"\b(what|how many|how much|does|do i|when|is|are|which|list|count|"
    r"how long|answer|find|show me)\b", re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# OpenRouter-backed multimodal LLM wrapper (same as run_m3a.py)
# ---------------------------------------------------------------------------

class OpenRouterGpt4Wrapper:
    """Drop-in replacement for android_world.agents.infer.Gpt4Wrapper.

    Uses the same predict_mm() interface but sends requests to a configurable
    API base (e.g. https://openrouter.ai/api/v1) instead of api.openai.com.
    """

    RETRY_WAITING_SECONDS = 20

    def __init__(self, model_name, api_key, api_base, max_retry=3, temperature=0.0):
        self.model = model_name
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.max_retry = min(max(max_retry, 1), 5)
        self.temperature = temperature

    def predict(self, text_prompt):
        return self.predict_mm(text_prompt, [])

    def predict_mm(self, text_prompt, images):
        import requests
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
        }
        payload = {
            'model': self.model,
            'temperature': self.temperature,
            'messages': [{
                'role': 'user',
                'content': [{'type': 'text', 'text': text_prompt}],
            }],
            'max_tokens': 1000,
        }
        if (
            'openrouter.ai' in self.api_base.lower()
            and self.model.lower() in ('qwen/qwen3.7-plus', 'z-ai/glm-4.6v')
        ):
            # M3A needs a strict Reason:/Action: answer; disable reasoning.
            payload['reasoning'] = {'effort': 'none'}
        for image in images:
            payload['messages'][0]['content'].append({
                'type': 'image_url',
                'image_url': {
                    'url': f'data:image/jpeg;base64,{self._encode(image)}',
                },
            })

        url = f'{self.api_base}/chat/completions'
        counter = self.max_retry
        while counter > 0:
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=120)
                body = r.json()
                if r.ok and 'choices' in body:
                    usage = body.get('usage') or {}
                    prompt_details = usage.get('prompt_tokens_details') or {}
                    from aworld_agent.token_usage import token_usage
                    token_usage.add(
                        prompt_tokens=int(usage.get('prompt_tokens') or 0),
                        completion_tokens=int(usage.get('completion_tokens') or 0),
                        cache_read_tokens=int(
                            prompt_details.get('cached_tokens') or 0
                        ),
                        label='m3a_vlm',
                    )
                    content = body['choices'][0]['message'].get('content')
                    if isinstance(content, str) and content.strip():
                        return content, None, r
                    print('  LLM error: successful response had no final content')
                    time.sleep(self.RETRY_WAITING_SECONDS)
                    counter -= 1
                    continue
                msg = r.json().get('error', {}).get('message', str(r.status_code))
                print(f'  LLM error: {msg}')
                time.sleep(self.RETRY_WAITING_SECONDS)
                counter -= 1
            except Exception as e:
                print(f'  Request error: {e}')
                time.sleep(self.RETRY_WAITING_SECONDS)
                counter -= 1
        return None, False, None

    @staticmethod
    def _encode(image):
        img = Image.fromarray(image.astype('uint8'))
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
        return base64.b64encode(buf.getvalue()).decode('utf-8')


# ---------------------------------------------------------------------------
# M3A + map routing
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


class M3AMapAgent(m3a.M3A):
    """M3A with a round-1 app-map macro-routing phase."""

    def __init__(
        self,
        env,
        llm,
        app_maps: Optional[dict] = None,
        llm_config: Optional[dict] = None,
        name: str = "m3a_map",
    ):
        super().__init__(env, llm, name=name)
        self.app_maps = dict(app_maps or {})
        self.llm_config = llm_config or {}
        self.app_map = None
        self._ops_catalog: dict = {}
        self._round = 0
        self._macro_phase_done = False
        self._task_usage_snapshot = None
        self.task_usage_records: list[dict] = []

        # Full PhoneCLI map-layer options (mirror evaluation/macro_agent.py).
        self.force_macro_vlm = self.llm_config.get("force_macro_vlm", False)
        self.skip_landing_check = self.llm_config.get("skip_landing_check", False)
        self.landing_hint_level = self.llm_config.get("landing_hint_level", 2)
        self.cold_read_warmup = self.llm_config.get("cold_read_warmup", False)

    # -- lifecycle ----------------------------------------------------------

    def prepare_for_task(self, task) -> None:
        """Snapshot tokens and pick the app map for this task (if any)."""
        from aworld_agent.token_usage import token_usage

        self._task_usage_snapshot = token_usage.snapshot()
        self.app_map = None
        self._ops_catalog = {}
        self._macro_phase_done = False
        self._round = 0

        candidate_names = []
        params = getattr(task, "params", {}) or {}
        parameter_app = params.get("app_name")
        if isinstance(parameter_app, str) and parameter_app:
            candidate_names.append(parameter_app)
        candidate_names.extend(getattr(task, "app_names", ()) or ())

        selected = None
        for app_name in candidate_names:
            if not app_name:
                continue
            try:
                activity = adb_utils.get_adb_activity(app_name)
            except Exception:
                activity = None
            package = activity.split("/", 1)[0] if activity else None
            if package and package in self.app_maps:
                selected = self.app_maps[package]
                break

        if selected is not None:
            self.app_map = selected
            try:
                self._ops_catalog = selected.build_operations()
                logger.info(
                    "Task %s uses map %s (%d ops)",
                    getattr(task, "name", type(task).__name__),
                    selected.app_name,
                    len(self._ops_catalog),
                )
            except Exception:
                self._ops_catalog = {}
                logger.warning("Failed to build ops for %s", selected.app_name)
        else:
            logger.info(
                "Task %s has no matching map (apps=%s); pure M3A",
                getattr(task, "name", type(task).__name__),
                tuple(candidate_names),
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
            step_data = getattr(result, "step_data", {}) or {}
            metadata = {
                "is_successful": None,
                "episode_length": len(step_data.get("step_number", [])),
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

    def reset(self, go_home_on_reset: bool = False):
        super().reset(go_home_on_reset)
        self._round = 0
        self._macro_phase_done = False

    # -- main step ----------------------------------------------------------

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        self._round += 1
        if self._round == 1 and not self._macro_phase_done and self.app_map:
            result = self._round_macro(goal)
            if result is not None:
                return result
        return super().step(goal)

    # -- map routing (adapted from PhoneCLIAndroidWorldAgent) ---------------

    def _round_macro(self, goal: str) -> Optional[base_agent.AgentInteractionResult]:
        map_result = self._llm_map_task(goal)
        logger.info("Macro map result: %s", map_result["type"])

        # Full PhoneCLI option: some models misclassify interactive tasks as
        # OP; downgrade OP → MACRO_VLM so the VLM always gets to interact.
        if self.force_macro_vlm and map_result["type"] == "op":
            map_result["type"] = "macro_vlm"

        if map_result["type"] == "finish":
            self._macro_phase_done = True
            return base_agent.AgentInteractionResult(
                True, {"macro_result": map_result, "action": "finish"})

        if map_result["type"] in ("op", "macro_vlm"):
            op = self._ops_catalog.get(map_result["op_id"]) if self._ops_catalog else None
            if op and op.macro:
                success = self._execute_macro(op)
                if success and map_result["type"] == "op":
                    if self._vlm_verify(goal):
                        self._macro_phase_done = True
                        return base_agent.AgentInteractionResult(
                            True, {"macro_result": map_result, "action": "macro_finish"})
                    logger.info("Macro ok but task not complete, switching to M3A")

                if success:
                    self._macro_phase_done = True
                    logger.info("Macro navigated, switching to M3A")

                    screen_context = self._detect_screen_context()  # (hint_str, sid)
                    if (
                        not self.skip_landing_check
                        and self._landing_mismatch(op, screen_context)
                    ):
                        logger.info("Landing mismatch — resetting app for M3A fallback")
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
                        return None

                    # Inject macro context into M3A history (only 'summary' is
                    # read by M3A's action-selection prompt). landing_hint_level
                    # mirrors evaluation/macro_agent.py: 0=bare, 1=+screen desc,
                    # 2=+expected elements + cold read.
                    parts = [
                        f"Macro completed: navigated to target screen via {op.id}.",
                        f"Task: {goal}.",
                    ]
                    if self.landing_hint_level >= 1:
                        if screen_context and screen_context[0]:
                            parts.append(screen_context[0].strip())
                        else:
                            target_desc = self.app_map.get_target_screen_info(
                                self._ops_catalog, op.id)
                            if target_desc:
                                parts.append(f"Current screen: {target_desc}.")
                    if self.landing_hint_level >= 2:
                        target_screen = getattr(op, "target_screen", "")
                        if target_screen:
                            try:
                                screen = self.app_map.get_screen(target_screen)
                                if screen and screen.elements:
                                    interactive = [
                                        e for e in screen.elements
                                        if e.semantic_type or e.leads_to
                                    ]
                                    if interactive:
                                        hints = [
                                            f"{e.text}[{e.semantic_type or 'nav'}]"
                                            for e in interactive[:8]
                                        ]
                                        parts.append(
                                            "Expected elements: "
                                            + ", ".join(hints) + ".")
                            except Exception:
                                pass
                        # Cold-read warmup: one-shot VLM describes the live
                        # screen so the M3A VLM doesn't flounder on a
                        # teleported page.
                        if self.cold_read_warmup:
                            cold_desc = self._cold_read_screen()
                            if cold_desc:
                                parts.append(
                                    f"Current screen state: {cold_desc}")
                    parts.append(
                        "The macro has only navigated here — you must still "
                        "perform any interactions the task requires.")
                    self.history.append({"summary": " ".join(parts)[:400]})
                    return None

        # No match or macro failed — pure M3A from the current state.
        logger.info("No matching macro or macro failed, falling back to M3A")
        self._macro_phase_done = True
        return None

    def _llm_map_task(self, goal: str) -> dict:
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
            response = text_completion(
                system_prompt=system_prompt,
                user_prompt=f"Task: {goal}",
                api_key=self.llm_config.get("api_key") or self.llm.api_key,
                api_base=self.llm_config.get("api_base") or self.llm.api_base,
                model=self.llm_config.get("model") or self.llm.model,
                label="aw_map_task",
            )
        except Exception as e:
            logger.warning("LLM map-task failed: %s", e)
            return {"type": "need_vlm", "reason": str(e)}

        result = _parse_llm_map_response(response)
        logger.info("Phase 1 → %s %s", result["type"], result.get("op_id", ""))
        if result["type"] in ("op", "macro_vlm") and result.get("op_id"):
            if not self._verify_operation(goal, result):
                logger.info("Phase 2 → REJECTED, falling back to M3A")
                return {"type": "need_vlm", "reason": "Target page unmatched"}
        return result

    def _verify_operation(self, goal: str, phase1_result: dict) -> bool:
        op_id = phase1_result.get("op_id")
        if not op_id:
            return True
        op = self._ops_catalog.get(op_id)
        if not op:
            return True
        screen_desc = self.app_map.get_target_screen_info(self._ops_catalog, op_id)
        if not screen_desc:
            return True
        verify_prompt = MACRO_VERIFY_PROMPT.format(
            task=goal,
            op_description=op.description,
            screen_description=screen_desc,
        )
        try:
            response = text_completion(
                system_prompt=verify_prompt,
                user_prompt="YES / NO / UNCLEAR?",
                api_key=self.llm_config.get("api_key") or self.llm.api_key,
                api_base=self.llm_config.get("api_base") or self.llm.api_base,
                model=self.llm_config.get("model") or self.llm.model,
                max_tokens=16,
                temperature=0.0,
                label="aw_map_verify",
            )
        except Exception as e:
            logger.warning("Phase 2 failed: %s", e)
            return True
        verdict = response.strip().upper()
        logger.info("Phase 2 screen='%s...' → %s", screen_desc[:80], verdict)
        return verdict.startswith("YES") or verdict.startswith("UNCLEAR")

    def _execute_macro(self, op) -> bool:
        if not op or not op.macro:
            return False
        logger.info("Replaying macro %s (%d steps)", op.id, len(op.macro))
        for i, step in enumerate(op.macro):
            wait_seconds = step.get("wait", 0.5)
            action_str = step.get("action", "")
            try:
                if action_str == "swipe":
                    command = adb_utils.generate_swipe_command(
                        int(step["x1"]), int(step["y1"]),
                        int(step["x2"]), int(step["y2"]),
                        int(step.get("duration", 400)),
                    )
                    adb_utils.issue_generic_request(command, self.env.controller)
                else:
                    from aworld_agent.agent import _macro_step_to_json_action
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

    @staticmethod
    def _build_pseudo_xml(ui_elements) -> str:
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

    def _detect_screen_context(self) -> tuple:
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

    def _cold_read_screen(self) -> str:
        """One-shot VLM read of the current screen state after macro landing.

        Weak VLMs struggle to cold-read a page they were teleported to.  A
        dedicated call describes the visible state and the description is
        injected into the M3A history so the main loop knows where it is.
        """
        import os
        import tempfile
        try:
            state = self.env.get_state(wait_to_stabilize=True)
            pixels = state.pixels
            if pixels is None:
                return ""
            fd, path = tempfile.mkstemp(suffix=".png", prefix="coldread_")
            os.close(fd)
            try:
                Image.fromarray(pixels.astype(np.uint8)).save(path)
                response = vision_completion(
                    system_prompt=(
                        "You are a screen reader. Describe the current mobile "
                        "screen state precisely: which page this is, which input "
                        "fields/toggles/sliders are visible, what values are "
                        "already set, and which element is currently focused. "
                        "Be concise (under 150 words)."
                    ),
                    user_text="Describe this screen state.",
                    image_paths=[path],
                    api_key=self.llm_config.get("api_key") or self.llm.api_key,
                    api_base=self.llm_config.get("api_base") or self.llm.api_base,
                    model=self.llm_config.get("model") or self.llm.model,
                    label="aw_cold_read",
                )
                desc = (response or "").strip()
                if desc:
                    logger.info("Cold read: %s", desc[:120])
                return desc
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass
        except Exception as e:
            logger.warning("Cold read failed: %s", e)
            return ""

    def _landing_mismatch(self, op, screen_context: str) -> bool:
        if not screen_context or not op:
            return False
        target_screen = getattr(op, "target_screen", "")
        detected_screen_id = screen_context[1] if isinstance(screen_context, tuple) else None
        if target_screen and detected_screen_id:
            if detected_screen_id != target_screen:
                logger.info(
                    "Landing mismatch: expected %s, detected %s",
                    target_screen, detected_screen_id,
                )
                return True
            return False
        # Fallback: text overlap on the hint (no screen id available).
        context_lower = screen_context[0].lower()
        target_words = set()
        for part in op.description.split(" → "):
            for w in part.lower().split():
                if len(w) > 2:
                    target_words.add(w)
        if not target_words:
            return False
        matches = sum(1 for w in target_words if w in context_lower)
        mismatch = matches < min(2, len(target_words))
        if mismatch:
            logger.info(
                "Landing mismatch (text): expected %s, got '%s...'",
                target_words, context_lower[:100],
            )
        return mismatch

    def _vlm_verify(self, goal: str) -> bool:
        """Use VLM to check if the task is complete after macro replay."""
        state = self.env.get_state(wait_to_stabilize=True)
        pixels = state.pixels
        if pixels is None:
            return True
        import os
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".png", prefix="m3amap_verify_")
        os.close(fd)
        try:
            Image.fromarray(pixels.astype(np.uint8)).save(path)
            prompt = VLM_VERIFY_PROMPT.replace("{task}", goal)
            response = vision_completion(
                system_prompt=prompt,
                user_text=f"Task: {goal}\nIs this task complete?",
                image_paths=[path],
                api_key=self.llm_config.get("api_key") or self.llm.api_key,
                api_base=self.llm_config.get("api_base") or self.llm.api_base,
                model=self.llm_config.get("model") or self.llm.model,
                label="aw_verify",
            )
            return "COMPLETE" in response.upper()
        except Exception as e:
            logger.warning("VLM verify failed: %s", e)
            return True
        finally:
            try:
                os.remove(path)
            except OSError:
                pass
