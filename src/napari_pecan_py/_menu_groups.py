"""Semantic grouping for napari-pecan-py widget entries in the Plugins menu.

Napari builds the plugin submenu at ``napari/plugins/<plugin-name>``. Manifest
``menus`` entries for that path are ignored, so we patch widget menu registration
and can refresh existing menu items after the plugin is registered.

Group order (pipeline): color → masks → edge → workflow → ML.
"""

from __future__ import annotations

_PLUGIN_NAME = "napari-pecan-py"
_MENU_ID = f"napari/plugins/{_PLUGIN_NAME}"

# Widget display_name -> (menu group id, order within group)
WIDGET_MENU_GROUPS: dict[str, tuple[str, int]] = {
    "Color Thresholding": ("1_color", 1),
    "Adjustments": ("1_color", 2),
    "Mask Retouching": ("2_masks", 1),
    "Mask Ops": ("2_masks", 2),
    "Pecan Ellipse": ("2_masks", 3),
    "Edge Detection": ("3_edge", 1),
    "Pipeline Recorder": ("4_pipeline", 1),
    "Batch Pipeline": ("4_pipeline", 2),
    "YOLO Segmentation": ("5_ml", 1),
    "SAM 2 Segmentation": ("5_ml", 2),
    "Contrastive Coding": ("5_ml", 3),
}

_build_patch_installed = False
_register_hook_installed = False
_napari_hook_installed = False


def install_widget_menu_groups() -> None:
    """Patch napari widget menu registration once (no-op without Qt)."""
    global _build_patch_installed
    if _build_patch_installed:
        return
    try:
        import napari._qt._qplugins._qnpe2 as qnpe2
    except ImportError:
        return

    orig = qnpe2._build_widgets_submenu_actions

    def _grouped_build(mf):
        submenu, actions = orig(mf)
        if mf.name != _PLUGIN_NAME:
            return submenu, actions

        from app_model.types import MenuRule

        grouped = []
        for action in actions:
            widget_name = action.id.split(":", 1)[-1]
            group, order = WIDGET_MENU_GROUPS.get(
                widget_name, ("3_plugin_contributions", 0)
            )
            grouped.append(
                action.model_copy(
                    update={
                        "menus": [
                            MenuRule(id=_MENU_ID, group=group, order=order)
                        ]
                    }
                )
            )
        return submenu, grouped

    qnpe2._build_widgets_submenu_actions = _grouped_build
    _build_patch_installed = True


def _refresh_pecan_plugin_menu_groups() -> None:
    """Re-tag existing plugin submenu items with semantic groups (post-register fix)."""
    try:
        from napari._app_model import get_app_model
        from app_model.types import MenuItem
    except ImportError:
        return

    app = get_app_model()
    menu_dict = app.menus._menu_items.get(_MENU_ID)
    if not menu_dict:
        return

    prefix = f"{_PLUGIN_NAME}:"
    to_fix = [
        item
        for item in list(menu_dict)
        if getattr(item, "command", None) is not None
        and item.command.id.startswith(prefix)
    ]
    if not to_fix:
        return

    changed = False
    for item in to_fix:
        widget_name = item.command.id.split(":", 1)[-1]
        group, order = WIDGET_MENU_GROUPS.get(
            widget_name, ("3_plugin_contributions", 0)
        )
        if item.group == group and item.order == order:
            continue
        menu_dict.pop(item, None)
        menu_dict[
            MenuItem(
                command=item.command,
                when=item.when,
                group=group,
                order=order,
            )
        ] = None
        changed = True

    if changed:
        app.menus.menus_changed.emit({_MENU_ID})


def _ensure_npe2_register_hook() -> None:
    """Install patches before napari registers plugins (idempotent)."""
    global _register_hook_installed, _napari_hook_installed

    install_widget_menu_groups()

    if not _register_hook_installed:
        try:
            from npe2 import plugin_manager as pm
        except ImportError:
            pm = None
        if pm is not None:
            orig_register = pm.register

            def register(manifest_or_package, warn_disabled=True):
                install_widget_menu_groups()
                result = orig_register(manifest_or_package, warn_disabled)
                try:
                    name = (
                        manifest_or_package.name
                        if hasattr(manifest_or_package, "name")
                        else str(manifest_or_package)
                    )
                except Exception:
                    name = None
                if name == _PLUGIN_NAME:
                    _refresh_pecan_plugin_menu_groups()
                return result

            pm.register = register
            _register_hook_installed = True

    if not _napari_hook_installed:
        try:
            import napari.plugins._npe2 as napari_npe2
        except ImportError:
            napari_npe2 = None
        if napari_npe2 is not None and not getattr(
            napari_npe2, "_pecan_menu_hook", False
        ):
            orig_on_registered = napari_npe2.on_plugins_registered

            def on_plugins_registered(manifests):
                install_widget_menu_groups()
                result = orig_on_registered(manifests)
                if any(getattr(m, "name", None) == _PLUGIN_NAME for m in manifests):
                    _refresh_pecan_plugin_menu_groups()
                return result

            napari_npe2.on_plugins_registered = on_plugins_registered
            napari_npe2._pecan_menu_hook = True
            _napari_hook_installed = True

    _refresh_pecan_plugin_menu_groups()


def on_plugin_activate(_ctx) -> None:
    """Manifest ``on_activate``: ensure grouped menus after first plugin use."""
    _ensure_npe2_register_hook()
