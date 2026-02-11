import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time


class ColorTuner:
    """A class for tuning color thresholds with an advanced GUI.

    This class provides an interactive GUI for tuning color thresholds in different
    color spaces (RGB, HSV, LAB) for different targets (pecan, kernel, crack, background).
    """

    def __init__(self, pecan_video, max_frames=100, frame_step=None):
        """Initialize the ColorTuner with a PecanVideo instance.

        Parameters:
            pecan_video (PecanVideo): The PecanVideo instance to tune thresholds for.
            max_frames (int): Maximum number of frames to load for tuning. If video has more frames,
                             they will be sampled. Default is 100.
            frame_step (int): Step size for frame sampling. If None, automatically calculated
                             based on video length and max_frames.
        """
        self.pecan_video = pecan_video
        self.thresholds = pecan_video.thresholds

        # Handle large videos with frame sampling
        total_frames = len(pecan_video.frames)
        if total_frames > max_frames:
            if frame_step is None:
                frame_step = max(1, total_frames // max_frames)

            # Sample frames for large videos
            self.frames = pecan_video.frames[::frame_step]
            self.original_frame_step = frame_step
            self.total_original_frames = total_frames
            print(f"Large video detected ({total_frames} frames). Sampling every {frame_step} frames ({len(self.frames)} frames for tuning).")
        else:
            self.frames = pecan_video.frames
            self.original_frame_step = 1
            self.total_original_frames = total_frames

        # Variables for resize event handling
        self.resize_timer = None
        self.resize_delay = 100  # milliseconds

        # Default values for each color space
        self.default_thresholds = {
            "rgb": {
                "pecan": {"lower": np.array([0, 0, 0], dtype=np.uint8), "upper": np.array([255, 255, 255], dtype=np.uint8)},
                "kernel": {"lower": np.array([0, 0, 0], dtype=np.uint8), "upper": np.array([255, 255, 255], dtype=np.uint8)},
                "crack": {"lower": np.array([0, 0, 0], dtype=np.uint8), "upper": np.array([50, 50, 50], dtype=np.uint8)},
                "background": {"lower": np.array([0, 0, 0], dtype=np.uint8), "upper": np.array([255, 255, 255], dtype=np.uint8)},
                "damaged_kernel": {"lower": np.array([0, 0, 0], dtype=np.uint8), "upper": np.array([255, 255, 255], dtype=np.uint8)},
            },
            "hsv": {
                "pecan": {"lower": np.array([0, 84, 80], dtype=np.uint8), "upper": np.array([82, 255, 255], dtype=np.uint8)},
                "kernel": {"lower": np.array([0, 0, 0], dtype=np.uint8), "upper": np.array([179, 255, 255], dtype=np.uint8)},
                "crack": {"lower": np.array([0, 0, 0], dtype=np.uint8), "upper": np.array([179, 255, 50], dtype=np.uint8)},
                "background": {"lower": np.array([0, 0, 60], dtype=np.uint8), "upper": np.array([179, 91, 255], dtype=np.uint8)},
                "damaged_kernel": {"lower": np.array([18, 63, 240], dtype=np.uint8), "upper": np.array([31, 138, 255], dtype=np.uint8)},
            },
            "lab": {
                "pecan": {"lower": np.array([0, 0, 0], dtype=np.uint8), "upper": np.array([255, 255, 255], dtype=np.uint8)},
                "kernel": {"lower": np.array([0, 0, 0], dtype=np.uint8), "upper": np.array([255, 255, 255], dtype=np.uint8)},
                "crack": {"lower": np.array([0, 0, 0], dtype=np.uint8), "upper": np.array([100, 128, 128], dtype=np.uint8)},
                "background": {"lower": np.array([0, 0, 0], dtype=np.uint8), "upper": np.array([255, 255, 255], dtype=np.uint8)},
                "damaged_kernel": {"lower": np.array([0, 0, 0], dtype=np.uint8), "upper": np.array([255, 255, 255], dtype=np.uint8)},
            },
        }

        # If RGB is not in the thresholds dictionary, add it
        if "rgb" not in self.thresholds:
            self.thresholds["rgb"] = {
                "pecan": {"lower": np.array([0, 0, 0], dtype=np.uint8), "upper": np.array([255, 255, 255], dtype=np.uint8)},
                "kernel": {"lower": np.array([0, 0, 0], dtype=np.uint8), "upper": np.array([255, 255, 255], dtype=np.uint8)},
                "crack": {"lower": np.array([0, 0, 0], dtype=np.uint8), "upper": np.array([50, 50, 50], dtype=np.uint8)},
                "background": {"lower": np.array([0, 0, 0], dtype=np.uint8), "upper": np.array([255, 255, 255], dtype=np.uint8)},
                "damaged_kernel": {"lower": np.array([0, 0, 0], dtype=np.uint8), "upper": np.array([255, 255, 255], dtype=np.uint8)},
            }

        # If kernel or damaged_kernel is not in the thresholds dictionary, add it
        for color_space in self.thresholds:
            if "kernel" not in self.thresholds[color_space]:
                self.thresholds[color_space]["kernel"] = {"lower": self.default_thresholds[color_space]["kernel"]["lower"].copy(), "upper": self.default_thresholds[color_space]["kernel"]["upper"].copy()}
            if "damaged_kernel" not in self.thresholds[color_space]:
                self.thresholds[color_space]["damaged_kernel"] = {"lower": self.default_thresholds[color_space]["damaged_kernel"]["lower"].copy(), "upper": self.default_thresholds[color_space]["damaged_kernel"]["upper"].copy()}

        # Color space parameters
        self.color_space_params = {
            "rgb": {"channels": ["R", "G", "B"], "max_values": [255, 255, 255], "conversion_code": None, "colors": [(255, 0, 0), (0, 255, 0), (0, 0, 255)]},  # Red, Green, Blue
            "hsv": {"channels": ["H", "S", "V"], "max_values": [179, 255, 255], "conversion_code": cv2.COLOR_BGR2HSV, "colors": [(255, 0, 255), (0, 255, 255), (255, 255, 0)]},  # Magenta, Cyan, Yellow
            "lab": {"channels": ["L", "A", "B"], "max_values": [255, 255, 255], "conversion_code": cv2.COLOR_BGR2LAB, "colors": [(255, 255, 255), (0, 255, 0), (0, 0, 255)]},  # White, Green, Blue
        }

        # Mask colors for different targets
        self.mask_colors = {"pecan": (0, 0, 255), "kernel": (0, 255, 0), "damaged_kernel": (0, 255, 255), "crack": (255, 0, 0), "background": (255, 255, 0)}  # Blue  # Green  # Yellow  # Red  # Cyan

        # Current state
        self.current_target = "pecan"
        self.current_color_space = "hsv"
        self.current_frame_index = 0
        self.running = True

        # Create the main window
        self.root = tk.Tk()
        self.root.title("Color Tuner")
        self.root.geometry("800x600")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.setup_ui()

    def create_color_sliders(self, parent_frame):
        """Create sliders for color thresholds."""
        # Color threshold sliders
        sliders_frame = ttk.LabelFrame(parent_frame, text="Color Thresholds")
        sliders_frame.pack(fill=tk.X, pady=10)

        # Store a reference to the sliders frame for later access
        self.sliders_frame = sliders_frame

        # Clear existing sliders dictionaries
        self.sliders = {}
        self.slider_values = {}

        for i, channel in enumerate(self.color_space_params[self.current_color_space]["channels"]):
            channel_frame = ttk.Frame(sliders_frame)
            channel_frame.pack(fill=tk.X, padx=5, pady=5)

            # Channel label
            ttk.Label(channel_frame, text=f"{channel}:").pack(side=tk.LEFT, padx=5)

            # Min value label and entry
            min_frame = ttk.Frame(channel_frame)
            min_frame.pack(side=tk.LEFT, padx=5)
            ttk.Label(min_frame, text="Min:").pack(side=tk.LEFT)

            self.slider_values[f"{channel}_min"] = tk.StringVar(value="0")
            min_entry = ttk.Entry(min_frame, textvariable=self.slider_values[f"{channel}_min"], width=5)
            min_entry.pack(side=tk.LEFT, padx=2)
            min_entry.bind("<Return>", lambda _: self.on_entry_change(channel))

            # Min slider
            max_val = self.color_space_params[self.current_color_space]["max_values"][i]
            min_slider = ttk.Scale(channel_frame, from_=0, to=max_val, orient=tk.HORIZONTAL, command=lambda val, ch=channel, type_="min": self.on_slider_change(ch, val, type_))
            min_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            self.sliders[f"{channel}_min"] = min_slider

            # Max slider
            max_slider = ttk.Scale(channel_frame, from_=0, to=max_val, orient=tk.HORIZONTAL, command=lambda val, ch=channel, type_="max": self.on_slider_change(ch, val, type_))
            max_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
            self.sliders[f"{channel}_max"] = max_slider

            # Max value label and entry
            max_frame = ttk.Frame(channel_frame)
            max_frame.pack(side=tk.LEFT, padx=5)
            ttk.Label(max_frame, text="Max:").pack(side=tk.LEFT)

            self.slider_values[f"{channel}_max"] = tk.StringVar(value=str(max_val))
            max_entry = ttk.Entry(max_frame, textvariable=self.slider_values[f"{channel}_max"], width=5)
            max_entry.pack(side=tk.LEFT, padx=2)
            max_entry.bind("<Return>", lambda _: self.on_entry_change(channel))

            # Set initial slider values
            lower = self.thresholds[self.current_color_space][self.current_target]["lower"][i]
            upper = self.thresholds[self.current_color_space][self.current_target]["upper"][i]
            self.slider_values[f"{channel}_min"].set(str(lower))
            self.slider_values[f"{channel}_max"].set(str(upper))
            min_slider.set(lower)
            max_slider.set(upper)

    def setup_ui(self):
        """Set up the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Image display area - Create canvases first to avoid AttributeError
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Original image
        original_frame = ttk.LabelFrame(image_frame, text="Original")
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.original_canvas = tk.Canvas(original_frame)
        self.original_canvas.pack(fill=tk.BOTH, expand=True)

        # Mask image
        mask_frame = ttk.LabelFrame(image_frame, text="Mask")
        mask_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.mask_canvas = tk.Canvas(mask_frame)
        self.mask_canvas.pack(fill=tk.BOTH, expand=True)

        # Composite image
        composite_frame = ttk.LabelFrame(image_frame, text="Composite")
        composite_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.composite_canvas = tk.Canvas(composite_frame)
        self.composite_canvas.pack(fill=tk.BOTH, expand=True)

        # Top control panel
        control_panel = ttk.Frame(main_frame)
        control_panel.pack(fill=tk.X, pady=10)

        # Target selection
        target_frame = ttk.LabelFrame(control_panel, text="Target")
        target_frame.pack(side=tk.LEFT, padx=10, pady=5)

        self.target_var = tk.StringVar(value=self.current_target)
        for target in ["pecan", "kernel", "damaged_kernel", "crack", "background"]:
            ttk.Radiobutton(target_frame, text=target.capitalize().replace("_", " "), variable=self.target_var, value=target, command=self.on_target_change).pack(anchor=tk.W, padx=5, pady=2)

        # Color space selection
        color_space_frame = ttk.LabelFrame(control_panel, text="Color Space")
        color_space_frame.pack(side=tk.LEFT, padx=10, pady=5)

        self.color_space_var = tk.StringVar(value=self.current_color_space)
        for color_space in ["rgb", "hsv", "lab"]:
            ttk.Radiobutton(color_space_frame, text=color_space.upper(), variable=self.color_space_var, value=color_space, command=self.on_color_space_change).pack(anchor=tk.W, padx=5, pady=2)

        # Buttons
        button_frame = ttk.Frame(control_panel)
        button_frame.pack(side=tk.LEFT, padx=10, pady=5)

        ttk.Button(button_frame, text="Reset to Min/Max", command=self.reset_to_min_max).pack(pady=2)
        ttk.Button(button_frame, text="Reset to Default", command=self.reset_to_default).pack(pady=2)
        ttk.Button(button_frame, text="Save & Close", command=self.save_and_close).pack(pady=2)

        # Frame slider
        frame_slider_frame = ttk.LabelFrame(control_panel, text="Frame")
        frame_slider_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=5)

        # Frame info label
        frame_info_text = f"Frame {self.current_frame_index + 1}/{len(self.frames)}"
        if self.original_frame_step > 1:
            original_idx = self.get_original_frame_index(self.current_frame_index)
            frame_info_text += f" (Original: {original_idx + 1}/{self.total_original_frames})"

        self.frame_info_label = ttk.Label(frame_slider_frame, text=frame_info_text)
        self.frame_info_label.pack(pady=2)

        self.frame_slider = ttk.Scale(frame_slider_frame, from_=0, to=len(self.frames) - 1, orient=tk.HORIZONTAL, command=self.on_frame_change)
        self.frame_slider.pack(fill=tk.X, padx=5, pady=5)
        self.frame_slider.set(self.current_frame_index)

        # Now create the sliders after the canvases are created
        self.create_color_sliders(main_frame)

        # Bind resize event to update display when window is resized
        self.root.bind("<Configure>", self.on_window_resize)

        # Initialize sliders with current values
        self.update_sliders()

        # Force an initial update of the display after a short delay to ensure canvas sizes are available
        self.root.after(100, self.update_display)

    def on_target_change(self):
        """Handle target selection change."""
        self.current_target = self.target_var.get()
        self.update_sliders()

    def on_color_space_change(self):
        """Handle color space selection change."""
        self.current_color_space = self.color_space_var.get()

        # Find the main frame
        main_frame = None
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.Frame):
                main_frame = widget
                break

        if not main_frame:
            print("Error: Could not find main frame")
            return

        # Remove the existing sliders frame if it exists
        if hasattr(self, "sliders_frame") and self.sliders_frame:
            self.sliders_frame.destroy()

        # Create new sliders for the current color space
        self.create_color_sliders(main_frame)
        self.update_sliders()

        # Force an update of the display
        self.update_display()

    def on_frame_change(self, event):
        """Handle frame slider change."""
        # Get the current value from the slider and ensure it's an integer
        try:
            new_index = int(float(event))
            # Ensure the index is within valid range
            if 0 <= new_index < len(self.frames):
                self.current_frame_index = new_index

                # Update frame info label
                frame_info_text = f"Frame {self.current_frame_index + 1}/{len(self.frames)}"
                if self.original_frame_step > 1:
                    original_idx = self.get_original_frame_index(self.current_frame_index)
                    frame_info_text += f" (Original: {original_idx + 1}/{self.total_original_frames})"
                self.frame_info_label.config(text=frame_info_text)

                # Force an immediate update of the display
                self.update_display()
        except (ValueError, TypeError) as e:
            print(f"Error updating frame index: {e}")

    def on_entry_change(self, channel):
        """Handle entry field value change.

        Parameters:
            channel (str): The color channel being adjusted (e.g., "H", "S", "V")
        """
        index = self.color_space_params[self.current_color_space]["channels"].index(channel)

        try:
            # Get values from entry fields
            min_value = int(self.slider_values[f"{channel}_min"].get())
            max_value = int(self.slider_values[f"{channel}_max"].get())

            # Ensure values are within valid range
            max_val = self.color_space_params[self.current_color_space]["max_values"][index]
            min_value = max(0, min(min_value, max_val))
            max_value = max(0, min(max_value, max_val))

            # Ensure min <= max
            if min_value > max_value:
                min_value = max_value

            # Update thresholds
            self.thresholds[self.current_color_space][self.current_target]["lower"][index] = min_value
            self.thresholds[self.current_color_space][self.current_target]["upper"][index] = max_value

            # Update slider positions
            if f"{channel}_min" in self.sliders:
                self.sliders[f"{channel}_min"].set(min_value)
            if f"{channel}_max" in self.sliders:
                self.sliders[f"{channel}_max"].set(max_value)

            # Force an immediate update of the display
            self.update_display()

        except ValueError:
            # Reset to current values if invalid input
            self.slider_values[f"{channel}_min"].set(str(self.thresholds[self.current_color_space][self.current_target]["lower"][index]))
            self.slider_values[f"{channel}_max"].set(str(self.thresholds[self.current_color_space][self.current_target]["upper"][index]))

    def on_slider_change(self, channel, value, slider_type="min"):
        """Handle slider value change.

        Parameters:
            channel (str): The color channel being adjusted (e.g., "H", "S", "V")
            value (float): The new slider value
            slider_type (str): The type of slider being adjusted ("min" or "max")
        """
        index = self.color_space_params[self.current_color_space]["channels"].index(channel)
        value = int(float(value))

        # Update the threshold values
        if slider_type == "min":
            # Update lower limit
            self.thresholds[self.current_color_space][self.current_target]["lower"][index] = value
            # Ensure upper limit is not less than lower limit
            if self.thresholds[self.current_color_space][self.current_target]["upper"][index] < value:
                self.thresholds[self.current_color_space][self.current_target]["upper"][index] = value
                # Update max slider position
                if f"{channel}_max" in self.sliders:
                    self.sliders[f"{channel}_max"].set(value)
        else:  # slider_type == "max"
            # Update upper limit
            self.thresholds[self.current_color_space][self.current_target]["upper"][index] = value
            # Ensure lower limit is not greater than upper limit
            if self.thresholds[self.current_color_space][self.current_target]["lower"][index] > value:
                self.thresholds[self.current_color_space][self.current_target]["lower"][index] = value
                # Update min slider position
                if f"{channel}_min" in self.sliders:
                    self.sliders[f"{channel}_min"].set(value)

        # Update the slider value labels
        self.slider_values[f"{channel}_min"].set(str(self.thresholds[self.current_color_space][self.current_target]["lower"][index]))
        self.slider_values[f"{channel}_max"].set(str(self.thresholds[self.current_color_space][self.current_target]["upper"][index]))

        # Force an immediate update of the display
        self.update_display()

    def update_sliders(self):
        """Update sliders with current threshold values."""
        for i, channel in enumerate(self.color_space_params[self.current_color_space]["channels"]):
            # Check if the sliders exist for this channel
            if f"{channel}_min" in self.sliders and f"{channel}_max" in self.sliders:
                lower = self.thresholds[self.current_color_space][self.current_target]["lower"][i]
                upper = self.thresholds[self.current_color_space][self.current_target]["upper"][i]

                # Update the slider value labels
                self.slider_values[f"{channel}_min"].set(str(lower))
                self.slider_values[f"{channel}_max"].set(str(upper))

                # Update the slider positions
                self.sliders[f"{channel}_min"].set(lower)
                self.sliders[f"{channel}_max"].set(upper)

    def reset_to_min_max(self):
        """Reset sliders to min/max values."""
        for i in range(len(self.color_space_params[self.current_color_space]["channels"])):
            max_val = self.color_space_params[self.current_color_space]["max_values"][i]
            self.thresholds[self.current_color_space][self.current_target]["lower"][i] = 0
            self.thresholds[self.current_color_space][self.current_target]["upper"][i] = max_val

        self.update_sliders()

    def reset_to_default(self):
        """Reset sliders to default values."""
        self.thresholds[self.current_color_space][self.current_target]["lower"] = self.default_thresholds[self.current_color_space][self.current_target]["lower"].copy()
        self.thresholds[self.current_color_space][self.current_target]["upper"] = self.default_thresholds[self.current_color_space][self.current_target]["upper"].copy()

        self.update_sliders()

    def get_original_frame_index(self, tuning_frame_index):
        """Convert tuning frame index to original video frame index.

        Parameters:
            tuning_frame_index (int): Frame index in the sampled frames used for tuning

        Returns:
            int: Original frame index in the full video
        """
        return tuning_frame_index * self.original_frame_step

    def save_and_close(self):
        """Save the current thresholds and close the window."""
        # Update the PecanVideo thresholds
        self.pecan_video.thresholds = self.thresholds

        # Update the legacy variables for backward compatibility
        self.pecan_video.pecan_lower_limit = self.thresholds["hsv"]["pecan"]["lower"].copy()
        self.pecan_video.pecan_upper_limit = self.thresholds["hsv"]["pecan"]["upper"].copy()
        self.pecan_video.background_lower_limit = self.thresholds["hsv"]["background"]["lower"].copy()
        self.pecan_video.background_upper_limit = self.thresholds["hsv"]["background"]["upper"].copy()
        self.pecan_video.crack_lower_limit = self.thresholds["hsv"]["crack"]["lower"].copy()
        self.pecan_video.crack_upper_limit = self.thresholds["hsv"]["crack"]["upper"].copy()
        self.pecan_video.kernel_lower_limit = self.thresholds["hsv"]["kernel"]["lower"].copy()
        self.pecan_video.kernel_upper_limit = self.thresholds["hsv"]["kernel"]["upper"].copy()
        self.pecan_video.damaged_kernel_lower_limit = self.thresholds["hsv"]["damaged_kernel"]["lower"].copy()
        self.pecan_video.damaged_kernel_upper_limit = self.thresholds["hsv"]["damaged_kernel"]["upper"].copy()
        self.pecan_video.lab_pecan_lower_limit = self.thresholds["lab"]["pecan"]["lower"].copy()
        self.pecan_video.lab_pecan_upper_limit = self.thresholds["lab"]["pecan"]["upper"].copy()
        self.pecan_video.lab_background_lower_limit = self.thresholds["lab"]["background"]["lower"].copy()
        self.pecan_video.lab_background_upper_limit = self.thresholds["lab"]["background"]["upper"].copy()
        self.pecan_video.lab_crack_lower_limit = self.thresholds["lab"]["crack"]["lower"].copy()
        self.pecan_video.lab_crack_upper_limit = self.thresholds["lab"]["crack"]["upper"].copy()
        self.pecan_video.lab_kernel_lower_limit = self.thresholds["lab"]["kernel"]["lower"].copy()
        self.pecan_video.lab_kernel_upper_limit = self.thresholds["lab"]["kernel"]["upper"].copy()
        self.pecan_video.lab_damaged_kernel_lower_limit = self.thresholds["lab"]["damaged_kernel"]["lower"].copy()
        self.pecan_video.lab_damaged_kernel_upper_limit = self.thresholds["lab"]["damaged_kernel"]["upper"].copy()

        # Print summary if frame sampling was used
        if self.original_frame_step > 1:
            print(f"Thresholds saved! Tuned on {len(self.frames)} sampled frames from {self.total_original_frames} total frames.")
            print(f"Frame sampling: every {self.original_frame_step} frames")

        # Close the window
        self.on_close()

    def on_close(self):
        """Handle window close event."""
        self.running = False

        # Simply destroy the root window
        # We'll let Python's garbage collector handle the cleanup
        self.root.destroy()

    @staticmethod
    def create_masks(pecan_video, target="pecan", color_space="hsv", keep_largest_contour=False, apply_morphology=False, close_kernel_size=7, open_kernel_size=5, dilate_kernel_size=3, dilate_iterations=1, erode_iterations=1):
        """Create masks for a specific target and color space from a PecanVideo instance.

        This method creates binary masks for the specified target using the thresholds
        defined in the PecanVideo instance. It returns a BaseVideo containing the masks.

        Parameters:
            pecan_video (PecanVideo): The PecanVideo instance to create masks from.
            target (str): The target to create masks for. Options are "pecan", "kernel",
                         "damaged_kernel", "crack", or "background". Default is "pecan".
            color_space (str): The color space to use for thresholding. Options are "hsv"
                              or "lab". Default is "hsv".
            keep_largest_contour (bool): Whether to keep only the largest contour in the mask.
                                        Default is False, which keeps all contours.
            apply_morphology (bool): Whether to apply morphological operations to clean up the mask.
                                    Default is False.
            close_kernel_size (int): Size of the kernel for morphological closing. Default is 7.
            open_kernel_size (int): Size of the kernel for morphological opening. Default is 5.
            dilate_kernel_size (int): Size of the kernel for dilation and erosion. Default is 3.
            dilate_iterations (int): Number of iterations for dilation. Default is 1.
            erode_iterations (int): Number of iterations for erosion. Default is 1.

        Returns:
            BaseVideo: A BaseVideo instance containing the binary masks.
        """
        from pecan_py.BaseVideo import BaseVideo
        import cv2
        import numpy as np
        from tqdm import tqdm

        # Validate inputs
        if target not in ["pecan", "kernel", "damaged_kernel", "crack", "background"]:
            raise ValueError("target must be one of 'pecan', 'kernel', 'damaged_kernel', 'crack', or 'background'")

        if color_space not in ["hsv", "lab"]:
            raise ValueError("color_space must be one of 'hsv' or 'lab'")

        # Map "damage" to "damaged_kernel" for convenience
        if target == "damage":
            target = "damaged_kernel"

        # Get thresholds for the target
        lower = pecan_video.thresholds[color_space][target]["lower"]
        upper = pecan_video.thresholds[color_space][target]["upper"]

        # Convert frames to the specified color space
        if color_space == "hsv":
            conversion_code = cv2.COLOR_BGR2HSV
        else:  # lab
            conversion_code = cv2.COLOR_BGR2LAB

        # Create masks
        masks = []
        for frame in tqdm(pecan_video.frames, desc=f"Creating {target} masks using {color_space.upper()}"):
            # Convert frame to the specified color space
            converted_frame = cv2.cvtColor(frame, conversion_code)

            # Create mask using the thresholds
            mask = cv2.inRange(converted_frame, lower, upper)

            # Apply morphological operations to clean up the mask if requested
            if apply_morphology:
                kernel_close = np.ones((close_kernel_size, close_kernel_size), np.uint8)
                kernel_open = np.ones((open_kernel_size, open_kernel_size), np.uint8)
                kernel_dilate = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)

                # Close operation helps connect nearby contours (fills small gaps)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

                # Open operation removes small noise (erodes then dilates)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

                # Dilate to expand the mask
                mask = cv2.dilate(mask, kernel_dilate, iterations=dilate_iterations)

                # Erode to shrink the mask back (helps maintain original size while connecting)
                mask = cv2.erode(mask, kernel_dilate, iterations=erode_iterations)

            # Keep only the largest contour if requested
            if keep_largest_contour:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    largest_contour_mask = np.zeros_like(mask)
                    cv2.drawContours(largest_contour_mask, [largest_contour], -1, (255,), thickness=cv2.FILLED)
                    mask = largest_contour_mask

            # Convert to 3-channel for compatibility with BaseVideo
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            masks.append(mask_3ch)

        # Create a BaseVideo instance with the masks
        mask_video = BaseVideo(np.array(masks))

        return mask_video

    def on_window_resize(self, event):
        """Handle window resize event.

        This method is called when the window is resized to update the display.
        Uses a timer to debounce multiple resize events.

        Parameters:
            event: The resize event
        """
        # Only respond to the root window's resize events
        if event.widget == self.root:
            # Cancel the previous timer if it exists
            if self.resize_timer is not None:
                self.root.after_cancel(self.resize_timer)

            # Set a new timer to update the display after a delay
            self.resize_timer = self.root.after(self.resize_delay, self.update_display)

    def resize_image_maintain_aspect(self, image, canvas_width, canvas_height):
        """Resize an image to fit the canvas while maintaining aspect ratio.

        Parameters:
            image: The image to resize (numpy array)
            canvas_width: The width of the canvas
            canvas_height: The height of the canvas

        Returns:
            The resized image as a PIL Image object
        """
        # Get the original image dimensions
        h, w = image.shape[:2]

        # Calculate the aspect ratio
        aspect_ratio = w / h

        # Calculate the new dimensions to fit the canvas while maintaining aspect ratio
        if canvas_width / canvas_height > aspect_ratio:
            # Canvas is wider than the image aspect ratio
            new_height = canvas_height
            new_width = int(new_height * aspect_ratio)
        else:
            # Canvas is taller than the image aspect ratio
            new_width = canvas_width
            new_height = int(new_width / aspect_ratio)

        # Resize the image
        resized_image = cv2.resize(image, (new_width, new_height))

        # Convert to PIL Image
        pil_image = Image.fromarray(resized_image)

        return pil_image, new_width, new_height

    def update_display(self):
        """Update the display with the current frame and masks."""
        if not self.running or self.current_frame_index >= len(self.frames):
            return

        # Check if canvases are created and ready
        if not hasattr(self, "original_canvas") or not hasattr(self, "mask_canvas") or not hasattr(self, "composite_canvas"):
            return

        # Get the current frame
        frame = self.frames[self.current_frame_index].copy()

        # Convert to the current color space
        if self.current_color_space == "rgb":
            converted_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            conversion_code = self.color_space_params[self.current_color_space]["conversion_code"]
            converted_frame = cv2.cvtColor(frame, conversion_code)

        # Create masks for all targets
        masks = {}
        for target in ["pecan", "kernel", "damaged_kernel", "crack", "background"]:
            lower = self.thresholds[self.current_color_space][target]["lower"]
            upper = self.thresholds[self.current_color_space][target]["upper"]
            masks[target] = cv2.inRange(converted_frame, lower, upper)

        # Create a composite mask with different colors for each target
        composite = np.zeros_like(frame)
        for target, mask in masks.items():
            color = self.mask_colors[target]
            colored_mask = np.zeros_like(frame)
            colored_mask[mask > 0] = color
            # Add the colored mask to the composite with alpha blending
            alpha = 0.5
            composite = cv2.addWeighted(composite, 1.0, colored_mask, alpha, 0)

        # Convert images to PIL format for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask_rgb = cv2.cvtColor(masks[self.current_target], cv2.COLOR_GRAY2RGB)
        composite_rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)

        # Get canvas dimensions
        canvas_width = self.original_canvas.winfo_width()
        canvas_height = self.original_canvas.winfo_height()

        # Ensure minimum canvas size for meaningful display
        if canvas_width < 10 or canvas_height < 10:
            # Canvas is too small, try again after a short delay
            self.root.after(100, self.update_display)
            return

        # Canvas is large enough to display images
        # Clear the canvases
        self.original_canvas.delete("all")
        self.mask_canvas.delete("all")
        self.composite_canvas.delete("all")

        # Resize images while maintaining aspect ratio
        frame_pil, frame_width, frame_height = self.resize_image_maintain_aspect(frame_rgb, canvas_width, canvas_height)
        mask_pil, mask_width, mask_height = self.resize_image_maintain_aspect(mask_rgb, canvas_width, canvas_height)
        composite_pil, comp_width, comp_height = self.resize_image_maintain_aspect(composite_rgb, canvas_width, canvas_height)

        # Convert to PhotoImage
        self.frame_photo = ImageTk.PhotoImage(frame_pil)
        self.mask_photo = ImageTk.PhotoImage(mask_pil)
        self.composite_photo = ImageTk.PhotoImage(composite_pil)

        # Calculate center positions for each canvas
        frame_x = (canvas_width - frame_width) // 2
        frame_y = (canvas_height - frame_height) // 2
        mask_x = (canvas_width - mask_width) // 2
        mask_y = (canvas_height - mask_height) // 2
        comp_x = (canvas_width - comp_width) // 2
        comp_y = (canvas_height - comp_height) // 2

        # Update canvases with centered images
        self.original_canvas.create_image(frame_x, frame_y, anchor=tk.NW, image=self.frame_photo)
        self.mask_canvas.create_image(mask_x, mask_y, anchor=tk.NW, image=self.mask_photo)
        self.composite_canvas.create_image(comp_x, comp_y, anchor=tk.NW, image=self.composite_photo)

    def run(self):
        """Run the color tuner."""
        self.root.mainloop()

    @staticmethod
    def filter_small_spots(video, area_threshold):
        """Filter out small bright spots from a BaseVideo based on a threshold value.

        This method processes each frame in the input video and removes bright spots
        (contours) that have an area smaller than the specified threshold.

        Parameters:
            video (BaseVideo): The BaseVideo instance containing binary masks.
            area_threshold (int): The minimum area (in pixels) for a contour to be kept.
                                 Contours with area less than this threshold will be removed.

        Returns:
            BaseVideo: A new BaseVideo instance with filtered masks.
        """
        from pecan_py.BaseVideo import BaseVideo
        import cv2
        import numpy as np
        from tqdm import tqdm

        # Create a list to store the filtered masks
        filtered_masks = []

        # Process each frame in the video
        for frame in tqdm(video.frames, desc="Filtering small spots"):
            # Convert to grayscale if it's a 3-channel image
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()

            # Ensure binary mask (threshold if not already binary)
            _, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

            # Find contours in the binary mask
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create an empty mask to draw the filtered contours
            filtered_mask = np.zeros_like(binary_mask)

            # Draw only contours with area greater than the threshold
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= area_threshold:
                    cv2.drawContours(filtered_mask, [contour], -1, (255,), thickness=cv2.FILLED)

            # Convert back to 3-channel for compatibility with BaseVideo
            filtered_mask_3ch = cv2.cvtColor(filtered_mask, cv2.COLOR_GRAY2BGR)
            filtered_masks.append(filtered_mask_3ch)

        # Create a new BaseVideo with the filtered masks
        filtered_video = BaseVideo(np.array(filtered_masks))

        return filtered_video

    @staticmethod
    def filter_by_overlap(mask_video1, mask_video2):
        """Filter the first mask video to keep only the spot that overlaps with any spot in the second mask video.

        This method:
        1. Flattens both mask videos by combining all frames with bitwise OR
        2. Finds the spot in the first flattened mask that overlaps with any spot in the second flattened mask
        3. Creates a new mask video where only that specific spot is kept in all frames of the first mask

        Parameters:
            mask_video1 (BaseVideo): The first mask video (the one to be filtered)
            mask_video2 (BaseVideo): The second mask video (used to determine which spot to keep)

        Returns:
            BaseVideo: A new BaseVideo instance with only the overlapping spot kept in all frames
        """
        from pecan_py.BaseVideo import BaseVideo
        import cv2
        import numpy as np
        from tqdm import tqdm

        # Step 1: Flatten both mask videos by combining all frames with bitwise OR
        flattened_mask1 = np.zeros_like(mask_video1.frames[0])
        flattened_mask2 = np.zeros_like(mask_video2.frames[0])

        for frame in mask_video1.frames:
            flattened_mask1 = cv2.bitwise_or(flattened_mask1, frame)

        for frame in mask_video2.frames:
            flattened_mask2 = cv2.bitwise_or(flattened_mask2, frame)

        # Step 2: Convert flattened masks to grayscale if they're not already
        if len(flattened_mask1.shape) == 3:
            flattened_mask1_gray = cv2.cvtColor(flattened_mask1, cv2.COLOR_BGR2GRAY)
        else:
            flattened_mask1_gray = flattened_mask1.copy()

        if len(flattened_mask2.shape) == 3:
            flattened_mask2_gray = cv2.cvtColor(flattened_mask2, cv2.COLOR_BGR2GRAY)
        else:
            flattened_mask2_gray = flattened_mask2.copy()

        # Ensure binary masks
        _, flattened_mask1_binary = cv2.threshold(flattened_mask1_gray, 1, 255, cv2.THRESH_BINARY)
        _, flattened_mask2_binary = cv2.threshold(flattened_mask2_gray, 1, 255, cv2.THRESH_BINARY)

        # Step 3: Find the overlap between the two flattened masks
        overlap_mask = cv2.bitwise_and(flattened_mask1_binary, flattened_mask2_binary)

        # Step 4: Find contours in the first flattened mask
        contours, _ = cv2.findContours(flattened_mask1_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Step 5: Find the contour that overlaps with the second mask
        overlapping_contour = None
        max_overlap_area = 0

        for contour in contours:
            # Create a mask for this contour
            contour_mask = np.zeros_like(flattened_mask1_binary)
            cv2.drawContours(contour_mask, [contour], -1, (255,), thickness=cv2.FILLED)

            # Find the overlap between this contour and the overlap mask
            contour_overlap = cv2.bitwise_and(contour_mask, overlap_mask)
            overlap_area = cv2.countNonZero(contour_overlap)

            # If this contour has overlap and it's the largest overlap so far, save it
            if overlap_area > max_overlap_area:
                max_overlap_area = overlap_area
                overlapping_contour = contour

        # Step 6: Create a new mask video with only the overlapping contour
        filtered_masks = []

        # If no overlapping contour was found, return empty masks
        if overlapping_contour is None:
            for _ in range(len(mask_video1.frames)):
                empty_mask = np.zeros_like(mask_video1.frames[0])
                filtered_masks.append(empty_mask)
            return BaseVideo(np.array(filtered_masks))

        # Create a mask with only the overlapping contour
        overlapping_contour_mask = np.zeros_like(flattened_mask1_binary)
        cv2.drawContours(overlapping_contour_mask, [overlapping_contour], -1, (255,), thickness=cv2.FILLED)

        # Process each frame in the first video
        for frame in tqdm(mask_video1.frames, desc="Creating filtered masks"):
            # Convert to grayscale if it's a 3-channel image
            if len(frame.shape) == 3:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame_gray = frame.copy()

            # Ensure binary mask
            _, frame_binary = cv2.threshold(frame_gray, 1, 255, cv2.THRESH_BINARY)

            # Find the intersection of this frame with the overlapping contour mask
            filtered_frame = cv2.bitwise_and(frame_binary, overlapping_contour_mask)

            # Convert back to 3-channel for compatibility with BaseVideo
            filtered_frame_3ch = cv2.cvtColor(filtered_frame, cv2.COLOR_GRAY2BGR)
            filtered_masks.append(filtered_frame_3ch)

        # Create a new BaseVideo with the filtered masks
        filtered_video = BaseVideo(np.array(filtered_masks))

        return filtered_video

    @staticmethod
    def filter_kernel_by_crack_proximity(
        kernel_mask_video,
        crack_mask_video,
        vicinity_pixels=30,
        min_crack_area=50,
    ):
        """Remove kernel mask regions that have no large crack in their vicinity.

        True kernel (visible where shell is cracked) has crack masks nearby;
        false positives often have no crack in the vicinity. Keeps only kernel
        components that have at least min_crack_area crack pixels within
        vicinity_pixels of the component.

        Parameters:
            kernel_mask_video (BaseVideo or PecanVideo): Video of binary kernel masks (each frame 0/255).
            crack_mask_video (BaseVideo or PecanVideo): Video of binary crack masks (same length and frame size).
            vicinity_pixels (int): Radius (in pixels) to dilate each kernel component to define "vicinity".
                Default is 30.
            min_crack_area (int): Minimum number of crack pixels in vicinity to keep a kernel component.
                Default is 50.

        Returns:
            BaseVideo: Filtered kernel mask video (same length and frame size as input).
        """
        from pecan_py.BaseVideo import BaseVideo
        from tqdm import tqdm

        if len(kernel_mask_video.frames) != len(crack_mask_video.frames):
            raise ValueError(
                "kernel_mask_video and crack_mask_video must have the same number of frames"
            )

        def to_binary(frame):
            if frame.ndim == 3:
                g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                g = frame
            _, b = cv2.threshold(g, 1, 255, cv2.THRESH_BINARY)
            return b

        kernel_struct = np.ones(
            (vicinity_pixels * 2 + 1, vicinity_pixels * 2 + 1), dtype=np.uint8
        )
        filtered_frames = []

        for k_frame, c_frame in tqdm(
            zip(kernel_mask_video.frames, crack_mask_video.frames),
            total=len(kernel_mask_video.frames),
            desc="Filtering kernel by crack proximity",
        ):
            k_bin = to_binary(k_frame)
            c_bin = to_binary(c_frame)

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(k_bin)
            out = np.zeros_like(k_bin)

            for i in range(1, num_labels):
                comp_mask = (labels == i).astype(np.uint8) * 255
                dilated = cv2.dilate(comp_mask, kernel_struct)
                crack_in_vicinity = cv2.bitwise_and(c_bin, dilated)
                crack_area = np.count_nonzero(crack_in_vicinity)
                if crack_area >= min_crack_area:
                    out = cv2.bitwise_or(out, comp_mask)

            filtered_frames.append(cv2.cvtColor(out, cv2.COLOR_GRAY2BGR))

        return BaseVideo(np.array(filtered_frames))

    @staticmethod
    def create_ellipse_mask(video, ellipses, margin=0):
        """Create a mask video from ellipses with optional expansion or shrinking.

        This method creates binary masks from ellipses detected in a video. The masks
        can be expanded (positive margin) or shrunk (negative margin) by the specified
        margin value as a percentage of the ellipse size.

        Parameters:
            video (BaseVideo or PecanVideo): The video to match dimensions with
            ellipses (list): List of ellipse parameters (center, axes, angle) as returned
                            by the draw_ellipse_and_axis method
            margin (float): Margin as a percentage to expand (positive) or shrink (negative) the mask.
                         Default is 0 (no change). For example, margin=10 will expand the ellipse
                         by 10% of its original size, while margin=-10 will shrink it by 10%.

        Returns:
            BaseVideo: A BaseVideo instance containing the binary masks
        """
        from pecan_py.BaseVideo import BaseVideo
        import cv2
        import numpy as np
        from tqdm import tqdm

        # Create a list to store the masks
        masks = []

        # Get frame dimensions from the video
        frame_height, frame_width = video.frames[0].shape[:2]

        # Process each frame and corresponding ellipse
        for ellipse in tqdm(ellipses, desc="Creating ellipse masks"):
            # Create a blank mask
            mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

            # Skip if ellipse is None
            if ellipse is None:
                # Convert to 3-channel for compatibility with BaseVideo
                mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                masks.append(mask_3ch)
                continue

            # Draw the ellipse on the mask
            center, axes, angle = ellipse

            # The axes returned by cv2.fitEllipse() are full axes lengths (diameters),
            # but cv2.ellipse() expects half axes lengths (radii).
            # Convert from full axes to half axes by dividing by 2
            half_axes = (axes[0] / 2.0, axes[1] / 2.0)

            # Adjust axes based on margin (as a percentage of the original axes)
            if margin != 0:
                # Calculate the adjustment factor
                factor = 1.0 + (margin / 100.0)  # Convert margin to a percentage

                # Apply the factor to both axes
                adjusted_axes = (max(1, half_axes[0] * factor), max(1, half_axes[1] * factor))
            else:
                adjusted_axes = half_axes

            # Draw the ellipse filled
            cv2.ellipse(mask, (int(center[0]), int(center[1])), (int(adjusted_axes[0]), int(adjusted_axes[1])), angle, 0, 360, 255, thickness=cv2.FILLED)

            # Convert to 3-channel for compatibility with BaseVideo
            mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            masks.append(mask_3ch)

        # Create a BaseVideo instance with the masks
        mask_video = BaseVideo(np.array(masks))

        return mask_video

    @staticmethod
    def detect_seams(mask_video, min_length=50, min_aspect_ratio=3.0, max_width=20, horizontal_angle_threshold=45):
        """Detect and analyze seams (long, thin cracks) in a mask video.

        This method analyzes each frame in the input mask video to find long, thin contours
        that are likely to be seams or cracks. It determines their orientation (horizontal or vertical),
        length, area, and identifies the frame with the largest seam.

        Parameters:
            mask_video (BaseVideo): The BaseVideo instance containing binary masks, typically
                                   created with create_masks(target="crack").
            min_length (int): Minimum length of a contour to be considered a seam. Default is 50 pixels.
            min_aspect_ratio (float): Minimum aspect ratio (length/width) for a contour to be
                                     considered a seam. Default is 3.0.
            max_width (int): Maximum width of a contour to be considered a seam. Default is 20 pixels.
            horizontal_angle_threshold (int): Angle threshold in degrees for determining if a seam
                                            is horizontal. If the angle between the seam's orientation
                                            and the horizontal axis is less than this threshold, the seam
                                            is considered horizontal. Otherwise, it's considered vertical.
                                            Default is 45 degrees.

        Returns:
            dict: A dictionary containing information about the detected seams:
                - 'best_frame_index': Index of the frame with the largest seam
                - 'best_frame_seam_length': Length of the largest seam
                - 'best_frame_seam_area': Area of the largest seam
                - 'best_frame_seam_orientation': Orientation of the largest seam ('horizontal' or 'vertical')
                - 'all_seams': List of dictionaries, one for each frame, containing:
                    - 'frame_index': Frame index
                    - 'seams': List of dictionaries for each seam in the frame, containing:
                        - 'length': Length of the seam
                        - 'width': Width of the seam
                        - 'area': Area of the seam
                        - 'aspect_ratio': Aspect ratio (length/width) of the seam
                        - 'orientation': Orientation of the seam ('horizontal' or 'vertical')
                        - 'angle': Angle of the seam in degrees
        """
        import cv2
        import numpy as np
        from tqdm import tqdm
        import math

        # Initialize variables to track the best frame and seam
        best_frame_index = -1
        best_frame_seam_length = 0
        best_frame_seam_area = 0
        best_frame_seam_orientation = None
        all_seams = []

        # Process each frame in the video
        for frame_idx, frame in enumerate(tqdm(mask_video.frames, desc="Detecting seams")):
            # Convert to grayscale if it's a 3-channel image
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()

            # Ensure binary mask
            _, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

            # Find contours in the binary mask
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Initialize list to store seams for this frame
            frame_seams = []

            # Analyze each contour to find seams
            for contour in contours:
                # Skip contours with too few points
                if len(contour) < 5:
                    continue

                # Calculate area
                area = cv2.contourArea(contour)

                # Fit a rotated rectangle to the contour
                rect = cv2.minAreaRect(contour)
                center, (width, height), angle = rect

                # The length is the maximum of width and height
                length = max(width, height)
                # The width is the minimum of width and height
                width = min(width, height)

                # Calculate aspect ratio
                aspect_ratio = length / width if width > 0 else 0

                # Check if this contour meets the criteria for a seam
                if length >= min_length and aspect_ratio >= min_aspect_ratio and width <= max_width:

                    # Determine orientation based on angle
                    # OpenCV's minAreaRect returns an angle in the range [-90, 0)
                    # We need to adjust it to determine if the seam is horizontal or vertical

                    # If width > height, the angle represents the angle between the horizontal axis
                    # and the first side (width). Otherwise, it's the angle between the horizontal
                    # axis and the second side (height).
                    if width > height:
                        # The longer side is already aligned with the angle
                        adjusted_angle = abs(angle)
                    else:
                        # The longer side is perpendicular to the angle
                        adjusted_angle = abs(angle + 90)

                    # Determine if the seam is horizontal or vertical
                    if adjusted_angle < horizontal_angle_threshold or adjusted_angle > (90 - horizontal_angle_threshold):
                        orientation = "horizontal"
                    else:
                        orientation = "vertical"

                    # Add this seam to the list for this frame
                    frame_seams.append({"length": length, "width": width, "area": area, "aspect_ratio": aspect_ratio, "orientation": orientation, "angle": adjusted_angle})

                    # Check if this is the largest seam so far
                    if length > best_frame_seam_length:
                        best_frame_index = frame_idx
                        best_frame_seam_length = length
                        best_frame_seam_area = area
                        best_frame_seam_orientation = orientation

            # Add the seams for this frame to the overall list
            all_seams.append({"frame_index": frame_idx, "seams": frame_seams})

        # Return the results
        return {"best_frame_index": best_frame_index, "best_frame_seam_length": best_frame_seam_length, "best_frame_seam_area": best_frame_seam_area, "best_frame_seam_orientation": best_frame_seam_orientation, "all_seams": all_seams}

    @staticmethod
    def connect_and_filter_contours(mask_video, min_area=100, max_area=None, distance_threshold=10, keep_largest_only=False, fill_holes=True):
        """Connect nearby contours and filter out noise in a mask video.

        This method processes each frame in the input mask video to:
        1. Connect contours that are within a specified distance of each other
        2. Fill holes in contours if requested
        3. Filter out contours based on area constraints
        4. Optionally keep only the largest contour

        Parameters:
            mask_video (BaseVideo): The BaseVideo instance containing binary masks.
            min_area (int): Minimum contour area to keep. Smaller contours are removed as noise.
                           Default is 100 pixels.
            max_area (int, optional): Maximum contour area to keep. Larger contours are removed.
                                     Default is None (no upper limit).
            distance_threshold (int): Maximum distance between contours to connect them.
                                     Default is 10 pixels.
            keep_largest_only (bool): Whether to keep only the largest contour after processing.
                                     Default is False.
            fill_holes (bool): Whether to fill holes in contours. Default is True.

        Returns:
            BaseVideo: A new BaseVideo instance with processed masks.
        """
        from pecan_py.BaseVideo import BaseVideo
        import cv2
        import numpy as np
        from tqdm import tqdm

        # Create a list to store the processed masks
        processed_masks = []

        # Process each frame in the video
        for frame in tqdm(mask_video.frames, desc="Processing contours"):
            # Convert to grayscale if it's a 3-channel image
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()

            # Ensure binary mask
            _, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

            # Find contours in the binary mask
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create an empty mask for the processed contours
            processed_mask = np.zeros_like(binary_mask)

            # If no contours found, skip further processing
            if not contours:
                processed_mask_3ch = cv2.cvtColor(processed_mask, cv2.COLOR_GRAY2BGR)
                processed_masks.append(processed_mask_3ch)
                continue

            # Filter contours by area
            filtered_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= min_area and (max_area is None or area <= max_area):
                    filtered_contours.append(contour)

            # If no contours remain after filtering, skip further processing
            if not filtered_contours:
                processed_mask_3ch = cv2.cvtColor(processed_mask, cv2.COLOR_GRAY2BGR)
                processed_masks.append(processed_mask_3ch)
                continue

            # Connect nearby contours if distance_threshold > 0
            if distance_threshold > 0 and len(filtered_contours) > 1:
                # Create a mask with all filtered contours
                contour_mask = np.zeros_like(binary_mask)
                cv2.drawContours(contour_mask, filtered_contours, -1, 255, thickness=cv2.FILLED)

                # Create a kernel for dilation based on the distance threshold
                connect_kernel = np.ones((distance_threshold, distance_threshold), np.uint8)

                # Dilate to connect nearby contours
                dilated_mask = cv2.dilate(contour_mask, connect_kernel, iterations=1)

                # Find contours in the dilated mask (these should be connected)
                connected_contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Use the connected contours for further processing
                filtered_contours = connected_contours

            # Keep only the largest contour if requested
            if keep_largest_only and filtered_contours:
                largest_contour = max(filtered_contours, key=cv2.contourArea)
                filtered_contours = [largest_contour]

            # Draw the filtered contours on the processed mask
            cv2.drawContours(processed_mask, filtered_contours, -1, 255, thickness=cv2.FILLED)

            # Fill holes if requested
            if fill_holes:
                # Invert the mask
                inverted_mask = cv2.bitwise_not(processed_mask)

                # Find contours in the inverted mask (these are the holes)
                hole_contours, _ = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Fill all holes except those that touch the image border
                height, width = processed_mask.shape
                for hole in hole_contours:
                    # Check if the hole touches the border
                    touches_border = False
                    for point in hole:
                        x, y = point[0]
                        if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                            touches_border = True
                            break

                    # If the hole doesn't touch the border, fill it
                    if not touches_border:
                        cv2.drawContours(processed_mask, [hole], -1, 255, thickness=cv2.FILLED)

            # Convert back to 3-channel for compatibility with BaseVideo
            processed_mask_3ch = cv2.cvtColor(processed_mask, cv2.COLOR_GRAY2BGR)
            processed_masks.append(processed_mask_3ch)

        # Create a new BaseVideo with the processed masks
        processed_video = BaseVideo(np.array(processed_masks))

        return processed_video
