import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml
import numpy as np

from android_world.env import representation_utils

from aworld_agent.agent import (
    PhoneCLIAndroidWorldAgent,
    _annotate_screenshot,
    _build_ui_elements_list,
    _execute_precise_continuous_swipe,
    _is_continuous_control,
    _normalize_continuous_control_action,
    _toggle_signature,
    _is_actionable,
    _macro_step_to_json_action,
    parse_action,
    phonecli_to_json_action,
)
from aworld_agent.app_map import AppMap
from aworld_agent.build_map_core import build_app_map
from aworld_agent.llm_client import vision_completion
from aworld_agent.run_m3a import OpenRouterGpt4Wrapper


def _xml(text: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<hierarchy rotation="0">'
        f'<node text="{text}" class="android.widget.TextView" '
        'clickable="true" focusable="true" bounds="[100,800][900,1000]" />'
        '</hierarchy>'
    )


class _ScrollingController:
    """Small deterministic UI used to exercise multi-page BFS replay."""

    def __init__(self):
        self.screen = "root"
        self.page = 0
        self.commands = []

    def get_device_size(self):
        return 1000, 2000

    def kill_package(self, _package):
        pass

    def launch_app(self, _package):
        self.screen = "root"
        self.page = 0

    def get_xml(self, prefix, save_dir):
        labels = {
            ("root", 0): "TopAction",
            ("root", 1): "Deep",
            ("child", 0): "ChildTopAction",
            ("child", 1): "ChildDeep",
            ("grandchild", 0): "Done",
        }
        text = labels.get((self.screen, self.page), labels[(self.screen, 0)])
        Path(save_dir, prefix + ".xml").write_text(_xml(text))

    def run_command(self, command):
        self.commands.append(command)
        if "input swipe" in command:
            max_page = 1 if self.screen in ("root", "child") else 0
            self.page = min(self.page + 1, max_page)

    def tap(self, _x, _y):
        if self.screen == "root" and self.page == 1:
            self.screen, self.page = "child", 0
        elif self.screen == "child" and self.page == 1:
            self.screen, self.page = "grandchild", 0


class MapScrollRegressionTest(unittest.TestCase):

    @mock.patch("aworld_agent.build_map_core.time.sleep", return_value=None)
    def test_scroll_is_recorded_replayed_and_used_on_child_screens(self, _sleep):
        controller = _ScrollingController()
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "map.yaml")
            build_app_map(
                controller=controller,
                package="example.app",
                app_name="Example",
                output_path=output,
                max_screens=5,
                max_depth=2,
                scroll_pages=2,
                classify=False,
                enrich=False,
            )
            data = yaml.safe_load(Path(output).read_text())

            # Temporary captures are namespaced by output file and removed
            # after success, making concurrent builds in one directory safe.
            self.assertFalse(Path(tmp, ".build_tmp", "map.yaml").exists())

            root = data["screens"][0]
            deep = next(e for e in root["elements"] if e["text"] == "Deep")
            self.assertEqual(deep["found_at_scroll"], 1)
            child_id = deep["leads_to"]

            child = next(s for s in data["screens"] if s["id"] == child_id)
            child_deep = next(
                e for e in child["elements"] if e["text"] == "ChildDeep"
            )
            self.assertEqual(child_deep["found_at_scroll"], 1)
            grandchild_id = child_deep["leads_to"]

            child_macro = data["screen_macros"][child_id]
            self.assertEqual(
                [s["action"] for s in child_macro],
                ["force_stop", "launch", "swipe", "tap"],
            )
            self.assertEqual(
                (child_macro[2]["y1"], child_macro[2]["y2"]),
                (1400, 400),
            )

            grandchild_macro = data["screen_macros"][grandchild_id]
            self.assertEqual(
                [s["action"] for s in grandchild_macro],
                ["force_stop", "launch", "swipe", "tap", "swipe", "tap"],
            )

            # Operation generation must use the same scroll geometry as crawl.
            operations = AppMap(output).build_operations()
            deep_op = operations["deep"]
            swipe = next(s for s in deep_op.macro if s["action"] == "swipe")
            self.assertEqual((swipe["y1"], swipe["y2"]), (1400, 400))


class AgentActionRegressionTest(unittest.TestCase):

    def test_task_end_accepts_androidworld_episode_result(self):
        from aworld_agent.token_usage import token_usage

        token_usage.reset()
        agent = PhoneCLIAndroidWorldAgent.__new__(PhoneCLIAndroidWorldAgent)
        agent.task_usage_records = []
        agent._task_usage_snapshot = token_usage.snapshot()
        token_usage.add(prompt_tokens=12, completion_tokens=3, label="test")
        episode = SimpleNamespace(
            done=True,
            step_data={"step_number": [0, 1]},
        )

        agent.on_task_end(SimpleNamespace(name="ExampleTask"), episode)

        self.assertEqual(agent.task_usage_records[0]["total_tokens"], 15)
        self.assertEqual(agent.task_usage_records[0]["episode_length"], 2)
        self.assertTrue(agent.task_usage_records[0]["agent_done"])
        token_usage.reset()

    @mock.patch("aworld_agent.agent.adb_utils.get_adb_activity")
    def test_task_selects_its_own_app_map(self, get_activity):
        get_activity.side_effect = lambda name: {
            "markor": "net.gsantner.markor/.MainActivity",
            "simple gallery pro": "com.example.gallery/.MainActivity",
        }.get(name)
        markor = SimpleNamespace(
            app_name="Markor",
            package="net.gsantner.markor",
            build_operations=lambda: {"new_note": object()},
        )
        gallery = SimpleNamespace(
            app_name="Gallery",
            package="com.example.gallery",
            build_operations=lambda: {"open": object()},
        )
        agent = PhoneCLIAndroidWorldAgent.__new__(PhoneCLIAndroidWorldAgent)
        agent.app_maps = {
            markor.package: markor,
            gallery.package: gallery,
        }
        agent._ops_catalog = {}
        agent.prepare_for_task(SimpleNamespace(
            name="MultiAppTask",
            params={},
            app_names=("markor", "simple gallery pro"),
        ))
        self.assertIs(agent.app_map, markor)
        self.assertEqual(set(agent._ops_catalog), {"new_note"})

    @mock.patch("aworld_agent.agent.adb_utils.get_adb_activity")
    def test_concrete_app_parameter_overrides_broad_app_list(self, get_activity):
        get_activity.side_effect = lambda name: {
            "contacts": "com.google.android.contacts/.PeopleActivity",
            "camera": "com.android.camera2/.CameraLauncher",
        }.get(name)
        contacts = SimpleNamespace(
            app_name="Contacts",
            package="com.google.android.contacts",
            build_operations=lambda: {},
        )
        camera = SimpleNamespace(
            app_name="Camera",
            package="com.android.camera2",
            build_operations=lambda: {},
        )
        agent = PhoneCLIAndroidWorldAgent.__new__(PhoneCLIAndroidWorldAgent)
        agent.app_maps = {
            contacts.package: contacts,
            camera.package: camera,
        }
        agent._ops_catalog = {}
        agent.prepare_for_task(SimpleNamespace(
            name="OpenAppTaskEval",
            params={"app_name": "contacts"},
            app_names=("camera", "contacts"),
        ))
        self.assertIs(agent.app_map, contacts)

    def test_index_swipe_preserves_direction(self):
        action = phonecli_to_json_action(
            {"name": "swipe", "args": [0.0, "up", "medium"]},
            1080,
            2400,
        )
        self.assertEqual(action.action_type, "scroll")
        self.assertEqual(action.direction, "down")

    def test_coordinate_swipe_infers_direction(self):
        action = phonecli_to_json_action(
            {"name": "swipe", "args": [500.0, 1600.0, 500.0, 500.0]},
            1080,
            2400,
        )
        self.assertEqual(action.action_type, "scroll")
        self.assertEqual(action.direction, "down")

    def test_index_swipe_targets_original_ui_element(self):
        elements = [
            representation_utils.UIElement(text="first"),
            representation_utils.UIElement(text="slider"),
        ]
        action = phonecli_to_json_action(
            {"name": "swipe", "args": [1.0, "right", "long"]},
            1080,
            2400,
            elements,
            [1],
        )
        self.assertEqual(action.action_type, "scroll")
        self.assertEqual(action.index, 1)
        self.assertEqual(action.direction, "left")

    def test_swipe_parser_does_not_insert_empty_quoted_arguments(self):
        action = parse_action(
            '<CALLED_FUNCTION>swipe(1, "right", "long")</CALLED_FUNCTION>'
        )
        self.assertEqual(action, {
            "name": "swipe",
            "args": [1.0, "right", "long"],
        })

    @mock.patch("aworld_agent.agent.adb_utils.issue_generic_request")
    @mock.patch("aworld_agent.agent.adb_utils.generate_swipe_command")
    def test_brightness_tap_becomes_exact_edge_to_edge_swipe(
        self, generate_swipe, issue_request
    ):
        slider = representation_utils.UIElement(
            text="Display brightness",
            class_name="android.widget.SeekBar",
            bbox_pixels=representation_utils.BoundingBox(40, 1040, 180, 260),
            is_focusable=True,
        )
        action, replaced = _normalize_continuous_control_action(
            "Turn brightness to the max value.",
            {"name": "tap", "args": [1000.0, 220.0]},
            [slider],
            [0],
        )
        self.assertTrue(replaced)
        self.assertEqual(action, {
            "name": "swipe",
            "args": [1.0, "right", "long"],
        })

        generate_swipe.return_value = ["shell", "input", "swipe"]
        controller = object()
        self.assertTrue(_execute_precise_continuous_swipe(
            action, [slider], [0], controller
        ))
        generate_swipe.assert_called_once_with(40, 220, 1040, 220, 800)
        issue_request.assert_called_once_with(
            ["shell", "input", "swipe"], controller
        )

    def test_brightness_summary_text_is_not_mistaken_for_slider(self):
        display_summary = representation_utils.UIElement(
            text="Dark theme, font size, brightness",
            class_name="android.widget.TextView",
            resource_name="android:id/summary",
        )
        self.assertFalse(_is_continuous_control(display_summary))

    @mock.patch("aworld_agent.agent.adb_utils.issue_generic_request")
    @mock.patch("aworld_agent.agent.adb_utils.generate_swipe_command")
    def test_brightness_min_swipe_starts_at_material_thumb_center(
        self, generate_swipe, issue_request
    ):
        slider = representation_utils.UIElement(
            text="Display brightness",
            class_name="android.widget.SeekBar",
            bbox_pixels=representation_utils.BoundingBox(42, 1038, 149, 275),
            is_focusable=True,
        )
        action = {"name": "swipe", "args": [1.0, "left", "long"]}
        generate_swipe.return_value = ["shell", "input", "swipe"]
        controller = object()

        self.assertTrue(_execute_precise_continuous_swipe(
            action, [slider], [0], controller
        ))
        generate_swipe.assert_called_once_with(975, 212, 42, 212, 800)
        issue_request.assert_called_once_with(
            ["shell", "input", "swipe"], controller
        )

    @mock.patch("aworld_agent.agent.adb_utils.issue_generic_request")
    @mock.patch("aworld_agent.agent.adb_utils.generate_swipe_command")
    @mock.patch("aworld_agent.agent.vision_completion")
    def test_exact_brightness_swipe_finishes_episode(
        self, vision, generate_swipe, issue_request
    ):
        slider = representation_utils.UIElement(
            text="Display brightness",
            class_name="android.widget.SeekBar",
            bbox_pixels=representation_utils.BoundingBox(42, 1038, 149, 275),
            is_focusable=True,
        )
        state = SimpleNamespace(
            pixels=np.zeros((2400, 1080, 3), dtype=np.uint8),
            ui_elements=[slider],
        )
        agent = PhoneCLIAndroidWorldAgent.__new__(PhoneCLIAndroidWorldAgent)
        agent.env = SimpleNamespace(
            controller=object(),
            device_screen_size=(1080, 2400),
            execute_action=mock.Mock(),
            get_state=mock.Mock(return_value=state),
        )
        agent.vlm_api_key = "test"
        agent.vlm_api_base = "https://example.test"
        agent.vlm_model = "test-model"
        agent._history = []
        agent._pending_toggle = None
        agent._fruitless_rounds = 0
        vision.return_value = (
            "<STATE_ASSESSMENT>Slider is visible.</STATE_ASSESSMENT>"
            "<CALLED_FUNCTION>swipe(1, \"right\", \"long\")</CALLED_FUNCTION>"
        )
        generate_swipe.return_value = ["shell", "input", "swipe"]

        result = agent._round_vlm("Turn brightness to the max value.")

        self.assertTrue(result.done)
        self.assertEqual(result.data["execution"], "precise_continuous_swipe")
        agent.env.execute_action.assert_not_called()
        issue_request.assert_called_once()

    def test_brightness_row_is_opened_before_slider_swipe(self):
        section = representation_utils.UIElement(
            text="Brightness",
            class_name="android.widget.TextView",
            bbox_pixels=representation_utils.BoundingBox(63, 1038, 661, 712),
        )
        row = representation_utils.UIElement(
            text="Brightness level",
            class_name="android.widget.TextView",
            bbox_pixels=representation_utils.BoundingBox(63, 1038, 775, 900),
        )
        action, replaced = _normalize_continuous_control_action(
            "Turn brightness to the max value.",
            {"name": "swipe", "args": [1.0, "right", "long"]},
            [section, row],
            [0, 1],
        )
        self.assertTrue(replaced)
        self.assertEqual(action, {"name": "tap", "args": [2.0]})

    def test_toggle_guard_waits_instead_of_immediate_second_tap(self):
        toggle = representation_utils.UIElement(
            text="Use Wi-Fi",
            class_name="android.widget.Switch",
            bbox_pixels=representation_utils.BoundingBox(900, 1030, 820, 920),
            is_checkable=True,
            is_checked=False,
        )
        state = SimpleNamespace(pixels=None, ui_elements=[toggle])
        agent = PhoneCLIAndroidWorldAgent.__new__(PhoneCLIAndroidWorldAgent)
        agent.env = SimpleNamespace(
            get_state=mock.Mock(return_value=state),
            device_screen_size=(1080, 2400),
            execute_action=mock.Mock(),
        )
        agent._history = []
        agent._pending_toggle = {
            "signature": _toggle_signature(toggle),
            "before": True,
            "after": False,
            "desired": False,
        }

        result = agent._round_vlm("Turn wifi off.")

        self.assertFalse(result.done)
        executed = agent.env.execute_action.call_args.args[0]
        self.assertEqual(executed.action_type, "wait")
        self.assertIsNone(agent._pending_toggle)
        self.assertIn("rather than immediately toggling", agent._history[-1].summary)

    def test_map_macro_swipe_infers_direction(self):
        action = _macro_step_to_json_action({
            "action": "swipe",
            "x1": 540,
            "y1": 1680,
            "x2": 540,
            "y2": 480,
            "duration": 400,
        })
        self.assertEqual(action.action_type, "swipe")
        self.assertEqual(action.direction, "up")

    @mock.patch("aworld_agent.agent.time.sleep", return_value=None)
    @mock.patch("aworld_agent.agent.adb_utils.issue_generic_request")
    @mock.patch("aworld_agent.agent.adb_utils.generate_swipe_command")
    def test_macro_replay_uses_exact_recorded_swipe(
        self, generate_swipe, issue_request, _sleep
    ):
        generate_swipe.return_value = ["shell", "input", "swipe"]
        agent = PhoneCLIAndroidWorldAgent.__new__(PhoneCLIAndroidWorldAgent)
        agent.env = SimpleNamespace(controller=object())
        agent.app_map = SimpleNamespace(app_name="Settings")
        op = SimpleNamespace(
            id="display",
            macro=[{
                "action": "swipe",
                "x1": 540,
                "y1": 1680,
                "x2": 540,
                "y2": 480,
                "duration": 400,
                "wait": 0,
            }],
        )

        self.assertTrue(agent._execute_macro(op))
        generate_swipe.assert_called_once_with(540, 1680, 540, 480, 400)
        issue_request.assert_called_once_with(
            ["shell", "input", "swipe"], agent.env.controller
        )

    def test_landing_check_trusts_matching_map_screen_id(self):
        agent = PhoneCLIAndroidWorldAgent.__new__(PhoneCLIAndroidWorldAgent)
        op = SimpleNamespace(
            description="Dark theme, font size, brightness",
            target_screen="screen_9",
        )
        context = (
            'Current screen context: You are on "screen_9". '
            "Reference elements: Navigate up."
        )
        self.assertFalse(agent._landing_mismatch(op, context, "screen_9"))
        self.assertTrue(agent._landing_mismatch(op, context, "screen_3"))

    def test_anonymous_scroll_container_is_not_a_tap_mark(self):
        container = representation_utils.UIElement(
            is_scrollable=True,
            is_focusable=True,
            is_clickable=False,
        )
        child = representation_utils.UIElement(
            text="Connection preferences",
            is_clickable=False,
        )
        self.assertFalse(_is_actionable(container))
        self.assertTrue(_is_actionable(child))
        self.assertIn("[0] full-screen", _build_ui_elements_list([], []))

    def test_status_bar_icon_straddling_boundary_is_not_marked(self):
        icon = representation_utils.UIElement(
            content_description="Battery",
            is_clickable=True,
            bbox_pixels=representation_utils.BoundingBox(950, 1040, 24, 104),
        )
        _, valid_indices = _annotate_screenshot(
            np.zeros((2400, 1080, 3), dtype=np.uint8),
            [icon],
            2400,
        )
        self.assertEqual(valid_indices, [])

    def test_tap_zero_is_a_safe_noop(self):
        action = phonecli_to_json_action(
            {"name": "tap", "args": [0.0]},
            1080,
            2400,
        )
        self.assertEqual(action.action_type, "wait")


class _Response:

    def __init__(self, content, prompt_tokens=0, completion_tokens=0):
        self.ok = True
        self.status_code = 200
        self._content = content
        self._prompt_tokens = prompt_tokens
        self._completion_tokens = completion_tokens

    def json(self):
        return {
            "choices": [{"message": {"content": self._content}}],
            "usage": {
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
            },
        }


class _OpenAIResponse:

    def __init__(self, content):
        self.choices = [SimpleNamespace(
            message=SimpleNamespace(content=content),
        )]
        self.usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0)


class M3AResponseRegressionTest(unittest.TestCase):

    def setUp(self):
        from aworld_agent.token_usage import token_usage
        token_usage.reset()

    def tearDown(self):
        from aworld_agent.token_usage import token_usage
        token_usage.reset()

    @mock.patch("time.sleep", return_value=None)
    @mock.patch("aworld_agent.run_m3a.requests.post")
    def test_empty_content_is_retried_with_reasoning_disabled(self, post, _sleep):
        post.side_effect = [
            _Response(None, 10, 2),
            _Response("Reason: ok\nAction: {}", 20, 3),
        ]
        wrapper = OpenRouterGpt4Wrapper(
            "qwen/qwen3.7-plus",
            "test-key",
            "https://openrouter.ai/api/v1",
        )
        content, is_safe, response = wrapper.predict_mm("prompt", [])
        self.assertEqual(content, "Reason: ok\nAction: {}")
        self.assertIsNone(is_safe)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(
            post.call_args_list[0].kwargs["json"]["reasoning"],
            {"effort": "none"},
        )
        from aworld_agent.token_usage import token_usage
        self.assertEqual(token_usage.total_calls, 2)
        self.assertEqual(token_usage.total_tokens, 35)
        self.assertEqual(token_usage.per_label["m3a_vlm"]["calls"], 2)

    @mock.patch("aworld_agent.llm_client._get_client")
    def test_macro_vision_retries_empty_content_and_disables_reasoning(
        self, get_client
    ):
        create = mock.Mock(side_effect=[
            _OpenAIResponse(None),
            _OpenAIResponse("Action: tap(1)"),
        ])
        get_client.return_value = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create),
            )
        )
        with tempfile.NamedTemporaryFile(suffix=".png") as image:
            result = vision_completion(
                "system",
                "user",
                [image.name],
                api_key="test-key",
                api_base="https://openrouter.ai/api/v1",
                model="qwen/qwen3.7-plus",
            )

        self.assertEqual(result, "Action: tap(1)")
        self.assertEqual(create.call_count, 2)
        self.assertEqual(
            create.call_args_list[0].kwargs["extra_body"],
            {"reasoning": {"effort": "none"}},
        )


if __name__ == "__main__":
    unittest.main()
