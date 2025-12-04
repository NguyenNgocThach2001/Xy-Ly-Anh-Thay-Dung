from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class AppState:
    running: bool = True
    active_tab: int = 0
    ui: Any = None
    debug_enabled: bool = True
    ui_flip_view: bool = False

    num_available_cameras: int = 0
    available_cameras: List[int] = field(default_factory=list)
    selected_camera_index: int = 0
    camera_ready: bool = False
    camera_status: str = ""

    replay_moves: List[Any] = field(default_factory=list)
    replay_index: int = 0
    replay_auto: bool = False

    record_filename: str = ""
    recording: bool = False
    record_status: str = "Idle"

    fps_dropdown_open: bool = False
    fps_current: int = 15
    frame_grab_dropdown_open: bool = False
    frame_grab_every_current: int = 1
    camera_dropdown_open: bool = False
    piece_conf_dropdown_open: bool = False
    piece_conf_current: float = 0.6

    rotate_steps: int = 0
    segment_requested: Any = False
    test_piece_once_requested: bool = False

    rect_board_area: Any = None
    rect_right_panel: Any = None
    rect_tabs: Dict[str, Any] = field(default_factory=dict)
    rect_dropdowns: Dict[str, Any] = field(default_factory=dict)
    rect_buttons: Dict[str, Any] = field(default_factory=dict)
    rect_lists: Dict[str, Any] = field(default_factory=dict)
    rect_playback: Dict[str, Any] = field(default_factory=dict)
    rect_top_tabs: Dict[str, Any] = field(default_factory=dict)

    manual_enabled: bool = False
    manual_swap_axes: bool = False
    manual_river_extra: float = 64.0
    manual_scale_x: float = 1.0
    manual_scale_y: float = 1.0
    manual_offset_x: float = 0.0
    manual_offset_y: float = 0.0

    # Replay panel state
    replay_files: List[str] = field(default_factory=list)
    replay_selected_name: str = ""
    replay_step: int = 0
    replay_speed_ms: int = 800
    replay_last_tick: int = 0
    replay_rects: Dict[str, Any] = field(default_factory=dict)