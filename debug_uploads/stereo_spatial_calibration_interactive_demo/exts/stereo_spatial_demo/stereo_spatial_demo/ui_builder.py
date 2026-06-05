import omni.ui as ui


def build_ui(controller):
    with ui.VStack(spacing=8):
        ui.Label("Scene Controls")
        ui.Button("Create / Reset Demo Scene")
        ui.Button("Create Stereo Rig")
        ui.Button("Repair Stereo Rig")
        ui.Button("Randomize Target Object Pose")
        ui.Label("Target Object: cube / sphere / cylinder")
        ui.Spacer(height=8)
        ui.Label("Camera Controls")
        ui.Label("baseline_m: 0.10")
        ui.Label("width: 640")
        ui.Label("height: 480")
        ui.Button("Validate Stereo Rig")
        ui.Spacer(height=8)
        ui.Label("Inference Controls")
        ui.Button("Capture Stereo RGB")
        ui.Button("Run FoundationStereo")
        ui.Button(
            "Run Full Localization",
            clicked_fn=lambda: controller.run_full_localization(
                target_object="cube",
                baseline_m=0.10,
                resolution=(640, 480),
                depth_source="foundation_stereo",
                mask_source="native_instance",
            ),
        )
        ui.Button("Generate Visual Report")
        ui.Button("Open Report Folder")
        ui.Spacer(height=8)
        ui.Label("Results Panel")
        ui.Label("Status is written to outputs/interactive_demo/latest/report.json")
