bl_info = {
    "name": "Mesh Topo AI",
    "author": "HevenHeHe / Hermes CTO",
    "version": (0, 1, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > Mesh Topo AI",
    "description": "AI-driven retopology & native UV generation via local inference",
    "category": "Mesh",
    "support": "COMMUNITY",
}

import bpy
import threading
import requests
import os
import tempfile
from bpy.props import StringProperty, BoolProperty, EnumProperty, FloatProperty
from bpy.types import Panel, Operator, PropertyGroup

# ---------------------------------------------------------------------------
#  Properties
# ---------------------------------------------------------------------------
class MeshTopoAIProps(PropertyGroup):
    server_url: StringProperty(
        name="Server URL",
        default="http://127.0.0.1:8000",
        description="FastAPI backend endpoint"
    )
    export_format: EnumProperty(
        name="Export Format",
        items=[("OBJ", "OBJ", "Wavefront OBJ"),
               ("PLY", "PLY", "Stanford PLY")],
        default="OBJ"
    )
    auto_hide_source: BoolProperty(
        name="Auto-hide Source",
        default=True,
        description="Hide the original high-poly mesh after importing result"
    )
    merge_distance: FloatProperty(
        name="Merge Distance",
        default=0.0001,
        min=0.0,
        description="Merge by distance threshold for UV seam vertices"
    )

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def ensure_single_selected_mesh(context):
    objs = [o for o in context.selected_objects if o.type == 'MESH']
    if not objs:
        return None, "No mesh object selected"
    if len(objs) > 1:
        return None, "Please select exactly one mesh object"
    return objs[0], None

def export_temp(obj, fmt):
    tmpdir = tempfile.mkdtemp(prefix="mesh_topo_ai_")
    ext = ".obj" if fmt == "OBJ" else ".ply"
    path = os.path.join(tmpdir, f"export{ext}")
    if fmt == "OBJ":
        bpy.ops.wm.obj_export(filepath=path, export_selected_objects=True)
    else:
        bpy.ops.wm.ply_export(filepath=path, export_selected_objects=True)
    return path

def import_result(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=path)
    else:
        return None
    # Return the newly active object
    return bpy.context.active_object

def apply_post_process(obj, merge_dist):
    """Merge by distance to fix UV seam duplicate vertices."""
    if merge_dist <= 0:
        return
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=merge_dist)
    bpy.ops.object.mode_set(mode='OBJECT')

# ---------------------------------------------------------------------------
#  Async Request Helper
# ---------------------------------------------------------------------------
class InferenceThread(threading.Thread):
    def __init__(self, filepath, server_url, callback):
        super().__init__(daemon=True)
        self.filepath = filepath
        self.server_url = server_url.rstrip("/")
        self.callback = callback
        self.result = None
        self.error = None

    def run(self):
        try:
            url = f"{self.server_url}/infer"
            with open(self.filepath, "rb") as f:
                files = {"file": (os.path.basename(self.filepath), f, "application/octet-stream")}
                resp = requests.post(url, files=files, timeout=300)
            if resp.status_code == 200:
                data = resp.json()
                out_path = os.path.join(os.path.dirname(self.filepath), "result.ply")
                # If backend returns base64 content, decode it; otherwise assume URL
                import base64
                content = base64.b64decode(data["mesh_b64"])
                with open(out_path, "wb") as wf:
                    wf.write(content)
                self.result = out_path
            else:
                self.error = f"Server error {resp.status_code}: {resp.text}"
        except Exception as e:
            self.error = str(e)

# ---------------------------------------------------------------------------
#  Operators
# ---------------------------------------------------------------------------
class MESHTOPOAI_OT_send_inference(Operator):
    bl_idname = "meshtopoai.send_inference"
    bl_label = "Generate Low-Poly + UV"
    bl_description = "Export selected mesh, send to backend, import result"
    bl_options = {"REGISTER", "UNDO"}

    _timer = None
    _thread = None

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        if self._thread.is_alive():
            # Still running
            context.workspace.status_text_set(f"MeshTopoAI: Inferencing...")
            return {"RUNNING_MODAL"}

        # Thread finished
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        context.workspace.status_text_set(None)

        props = context.scene.mesh_topo_ai_props
        if self._thread.error:
            self.report({"ERROR"}, self._thread.error)
            return {"CANCELLED"}

        result_path = self._thread.result
        source_obj = self.source_obj

        # Import result
        imported = import_result(result_path)
        if imported is None:
            self.report({"ERROR"}, "Failed to import result mesh")
            return {"CANCELLED"}

        # Match transform
        imported.location = source_obj.location
        imported.rotation_euler = source_obj.rotation_euler
        imported.scale = source_obj.scale

        # Post-process: merge by distance for UV seams
        apply_post_process(imported, props.merge_distance)

        if props.auto_hide_source:
            source_obj.hide_set(True)

        self.report({"INFO"}, f"Imported result: {imported.name}")
        return {"FINISHED"}

    def execute(self, context):
        obj, err = ensure_single_selected_mesh(context)
        if err:
            self.report({"ERROR"}, err)
            return {"CANCELLED"}

        props = context.scene.mesh_topo_ai_props
        filepath = export_temp(obj, props.export_format)

        self.source_obj = obj
        self._thread = InferenceThread(filepath, props.server_url, None)
        self._thread.start()

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.5, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

class MESHTOPOAI_OT_test_mock(Operator):
    """Operator to test UI without a running backend."""
    bl_idname = "meshtopoai.test_mock"
    bl_label = "Mock: Import Test Cylinder"
    bl_description = "Import a simple cylinder to test post-processing pipeline"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=1, depth=2)
        obj = context.active_object
        apply_post_process(obj, context.scene.mesh_topo_ai_props.merge_distance)
        self.report({"INFO"}, "Mock cylinder imported")
        return {"FINISHED"}

# ---------------------------------------------------------------------------
#  Panel
# ---------------------------------------------------------------------------
class MESHTOPOAI_PT_main(Panel):
    bl_label = "Mesh Topo AI"
    bl_idname = "MESHTOPOAI_PT_main"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Mesh Topo AI"

    def draw(self, context):
        layout = self.layout
        props = context.scene.mesh_topo_ai_props

        box = layout.box()
        box.label(text="Backend", icon="URL")
        box.prop(props, "server_url")

        box = layout.box()
        box.label(text="Export", icon="EXPORT")
        box.prop(props, "export_format")

        box = layout.box()
        box.label(text="Import Options", icon="IMPORT")
        box.prop(props, "auto_hide_source")
        box.prop(props, "merge_distance")

        layout.separator()
        layout.operator("meshtopoai.send_inference", icon="PLAY")
        layout.separator()
        layout.operator("meshtopoai.test_mock", icon="MESH_CYLINDER")

# ---------------------------------------------------------------------------
#  Registration
# ---------------------------------------------------------------------------
classes = [
    MeshTopoAIProps,
    MESHTOPOAI_OT_send_inference,
    MESHTOPOAI_OT_test_mock,
    MESHTOPOAI_PT_main,
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mesh_topo_ai_props = bpy.props.PointerProperty(type=MeshTopoAIProps)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.mesh_topo_ai_props

if __name__ == "__main__":
    register()
