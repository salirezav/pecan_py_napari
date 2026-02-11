import cv2
import numpy as np
import os
from tqdm import tqdm
from pecan_py.BaseVideo import BaseVideo
from scipy import signal
from scipy.spatial.distance import cosine
from pecan_py.ColorTuner import ColorTuner


class PecanVideo(BaseVideo):
    def __init__(self, video_source, max_frames=None, frame_step=1):
        # Call the parent class constructor with max_frames and frame_step parameters
        super().__init__(video_source, max_frames=max_frames, frame_step=frame_step)

        # Dictionary to store threshold values for different color spaces and targets
        self.thresholds = {
            "hsv": {
                "pecan": {"lower": np.array([0, 96, 95], dtype=np.uint8), "upper": np.array([21, 255, 255], dtype=np.uint8)},
                "background": {"lower": np.array([0, 0, 60], dtype=np.uint8), "upper": np.array([179, 91, 255], dtype=np.uint8)},
                "crack": {"lower": np.array([5, 135, 75], dtype=np.uint8), "upper": np.array([57, 255, 167], dtype=np.uint8)},
                "kernel": {"lower": np.array([0, 0, 0], dtype=np.uint8), "upper": np.array([179, 255, 255], dtype=np.uint8)},
                "damaged_kernel": {"lower": np.array([22, 77, 119], dtype=np.uint8), "upper": np.array([38, 139, 255], dtype=np.uint8)},
            },
            "lab": {
                "pecan": {"lower": np.array([100, 128, 154], dtype=np.uint8), "upper": np.array([215, 147, 211], dtype=np.uint8)},
                "background": {"lower": np.array([0, 0, 0], dtype=np.uint8), "upper": np.array([255, 255, 255], dtype=np.uint8)},
                "crack": {"lower": np.array([0, 128, 0], dtype=np.uint8), "upper": np.array([151, 255, 223], dtype=np.uint8)},
                "kernel": {"lower": np.array([0, 125, 180], dtype=np.uint8), "upper": np.array([255, 146, 255], dtype=np.uint8)},
                "damaged_kernel": {"lower": np.array([201, 109, 157], dtype=np.uint8), "upper": np.array([255, 129, 187], dtype=np.uint8)},
            },
        }
        
        # Selected LAB thresholds:
# Crack lower limit: [  0 137   0]
# Crack upper limit: [109 255 223]

        # Dictionary to store preferred color space for each target
        self.preferred_color_spaces = {"pecan": "hsv", "background": "hsv", "crack": "hsv", "kernel": "lab", "damaged_kernel": "hsv"}  # Use LAB for kernel by default

        # For backward compatibility
        self.pecan_lower_limit = self.thresholds["hsv"]["pecan"]["lower"]
        self.pecan_upper_limit = self.thresholds["hsv"]["pecan"]["upper"]
        self.background_lower_limit = self.thresholds["hsv"]["background"]["lower"]
        self.background_upper_limit = self.thresholds["hsv"]["background"]["upper"]
        self.crack_lower_limit = self.thresholds["hsv"]["crack"]["lower"]
        self.crack_upper_limit = self.thresholds["hsv"]["crack"]["upper"]
        self.kernel_lower_limit = self.thresholds["hsv"]["kernel"]["lower"]
        self.kernel_upper_limit = self.thresholds["hsv"]["kernel"]["upper"]
        self.damaged_kernel_lower_limit = self.thresholds["hsv"]["damaged_kernel"]["lower"]
        self.damaged_kernel_upper_limit = self.thresholds["hsv"]["damaged_kernel"]["upper"]
        self.lab_pecan_lower_limit = self.thresholds["lab"]["pecan"]["lower"]
        self.lab_pecan_upper_limit = self.thresholds["lab"]["pecan"]["upper"]
        self.lab_background_lower_limit = self.thresholds["lab"]["background"]["lower"]
        self.lab_background_upper_limit = self.thresholds["lab"]["background"]["upper"]
        self.lab_crack_lower_limit = self.thresholds["lab"]["crack"]["lower"]
        self.lab_crack_upper_limit = self.thresholds["lab"]["crack"]["upper"]
        self.lab_kernel_lower_limit = self.thresholds["lab"]["kernel"]["lower"]
        self.lab_kernel_upper_limit = self.thresholds["lab"]["kernel"]["upper"]
        self.lab_damaged_kernel_lower_limit = self.thresholds["lab"]["damaged_kernel"]["lower"]
        self.lab_damaged_kernel_upper_limit = self.thresholds["lab"]["damaged_kernel"]["upper"]

        # Initialize ellipses attribute to store ellipse parameters for each frame
        # Each ellipse is a tuple of (center, axes, angle)
        self.ellipses = None

    # get_video_name is inherited from Video class

    def tune_color(self, target="pecan", color_space="hsv", use_defaults=True):
        """Interactive tool to find optimal color thresholds for the video.

        This method displays the video frames and allows the user to adjust color thresholds
        using trackbars. The resulting mask is shown in real-time.

        Parameters:
            target (str): Which thresholds to tune: "pecan", "kernel", "damaged_kernel", "damage", "background", "crack", or "both".
                Default is "pecan".
                - "pecan": Tune thresholds for the pecan itself
                - "kernel": Tune thresholds for the pecan kernel
                - "damaged_kernel" or "damage": Tune thresholds for damaged kernel
                - "background": Tune thresholds for the background
                - "crack": Tune thresholds for cracks on the pecan surface
                - "both": Tune both pecan and background thresholds
            color_space (str): Color space to use for thresholding. Options are "hsv" or "lab".
                Default is "hsv".
            use_defaults (bool): If True, initialize trackbars with current threshold values. Default is True.

        Controls:
            Trackbars: Adjust color channel min/max values
            Q: Quit and save the selected thresholds

        Returns:
            PecanVideo: Returns self for method chaining
        """
        # Map "damage" to "damaged_kernel" for convenience
        if target == "damage":
            target = "damaged_kernel"

        if target not in ["pecan", "kernel", "damaged_kernel", "background", "crack", "both"]:
            raise ValueError("target must be one of 'pecan', 'kernel', 'damaged_kernel', 'background', 'crack', or 'both'")

        if color_space not in ["hsv", "lab"]:
            raise ValueError("color_space must be one of 'hsv' or 'lab'")

        # Set up color space specific parameters
        if color_space == "hsv":
            # Convert frames to HSV and RGB for display
            converted_frames = [cv2.cvtColor(cv2.resize(frame, (frame.shape[1], frame.shape[0])), cv2.COLOR_BGR2HSV) for frame in self.frames]
            frames_resized = [cv2.cvtColor(cv2.resize(frame, (frame.shape[1], frame.shape[0])), cv2.COLOR_BGR2RGB) for frame in self.frames]

            # Channel names and max values for HSV
            channel_names = ["H", "S", "V"]
            max_values = [179, 255, 255]
            channel_descriptions = ["Hue", "Saturation", "Value"]
        else:  # LAB
            # Convert frames to LAB and RGB for display
            converted_frames = [cv2.cvtColor(cv2.resize(frame, (frame.shape[1], frame.shape[0])), cv2.COLOR_BGR2LAB) for frame in self.frames]
            frames_resized = [cv2.cvtColor(cv2.resize(frame, (frame.shape[1], frame.shape[0])), cv2.COLOR_BGR2RGB) for frame in self.frames]

            # Channel names and max values for LAB
            channel_names = ["L", "A", "B"]
            max_values = [255, 255, 255]
            channel_descriptions = ["Lightness", "Green-Red", "Blue-Yellow"]

        # Create windows
        cv2.namedWindow("Mask", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

        # Helper function to create dual sliders (min/max) for color values
        def create_dual_slider(window_name, slider_name, max_value, on_change):
            cv2.createTrackbar(f"{slider_name}Min", window_name, 0, max_value, on_change)
            cv2.createTrackbar(f"{slider_name}Max", window_name, max_value, max_value, on_change)

        # Create trackbars for color adjustment
        for name, max_val, _ in zip(channel_names, max_values, channel_descriptions):
            create_dual_slider("Mask", name, max_val, lambda _: None)  # desc is used for documentation but not accessed

        cv2.createTrackbar("Frame", "Mask", 0, len(self.frames) - 1, lambda _: None)  # Frame seek bar

        # Set initial values for trackbars based on target and color space
        if use_defaults:
            # Get current threshold values from the dictionary
            lower_limit = self.thresholds[color_space][target]["lower"]
            upper_limit = self.thresholds[color_space][target]["upper"]

            # Set trackbar positions
            for i, name in enumerate(channel_names):
                cv2.setTrackbarPos(f"{name}Min", "Mask", int(lower_limit[i]))
                cv2.setTrackbarPos(f"{name}Max", "Mask", int(upper_limit[i]))
        else:
            # Use default max values
            for name, max_val in zip(channel_names, max_values):
                cv2.setTrackbarPos(f"{name}Max", "Mask", max_val)

        cv2.setTrackbarPos("Frame", "Mask", 0)

        while True:
            # Get current positions of all trackbars
            min_values = [cv2.getTrackbarPos(f"{name}Min", "Mask") for name in channel_names]
            max_values = [cv2.getTrackbarPos(f"{name}Max", "Mask") for name in channel_names]
            frame_to_display = cv2.getTrackbarPos("Frame", "Mask")

            # Set the color limits
            lower_limit = np.array(min_values)
            upper_limit = np.array(max_values)

            # Apply the mask
            mask = cv2.inRange(converted_frames[frame_to_display], lower_limit, upper_limit)

            # Create a red overlay for the mask with 80% opacity
            red_mask = np.zeros_like(frames_resized[frame_to_display])
            red_mask[mask > 0] = [0, 10, 100]  # Red color in RGB

            # Overlay the red mask on the original frame with 80% opacity
            alpha = 1  # 80% opacity
            result = cv2.addWeighted(frames_resized[frame_to_display], 1.0, red_mask, alpha, 0)

            # Convert result back to BGR for proper display with cv2.imshow
            result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

            # Display the resulting frame and mask
            cv2.imshow("Mask", mask)
            cv2.imshow("Result", result_bgr)

            # Resize the windows to fit the screen
            cv2.resizeWindow("Mask", 600, 400)
            cv2.resizeWindow("Result", 600, 400)

            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()

        # Update the thresholds dictionary with the new values
        if target == "both":
            # Update both pecan and background thresholds
            self.thresholds[color_space]["pecan"]["lower"] = lower_limit.copy()
            self.thresholds[color_space]["pecan"]["upper"] = upper_limit.copy()
            self.thresholds[color_space]["background"]["lower"] = lower_limit.copy()
            self.thresholds[color_space]["background"]["upper"] = upper_limit.copy()
        else:
            # Update the specific target threshold
            self.thresholds[color_space][target]["lower"] = lower_limit.copy()
            self.thresholds[color_space][target]["upper"] = upper_limit.copy()

        # Update the legacy variables for backward compatibility
        if color_space == "hsv":
            if target == "pecan" or target == "both":
                self.pecan_lower_limit = lower_limit.copy()
                self.pecan_upper_limit = upper_limit.copy()
            if target == "kernel":
                self.kernel_lower_limit = lower_limit.copy()
                self.kernel_upper_limit = upper_limit.copy()
            if target == "damaged_kernel":
                self.damaged_kernel_lower_limit = lower_limit.copy()
                self.damaged_kernel_upper_limit = upper_limit.copy()
            if target == "background" or target == "both":
                self.background_lower_limit = lower_limit.copy()
                self.background_upper_limit = upper_limit.copy()
            if target == "crack":
                self.crack_lower_limit = lower_limit.copy()
                self.crack_upper_limit = upper_limit.copy()
        else:  # LAB
            if target == "pecan" or target == "both":
                self.lab_pecan_lower_limit = lower_limit.copy()
                self.lab_pecan_upper_limit = upper_limit.copy()
            if target == "kernel":
                self.lab_kernel_lower_limit = lower_limit.copy()
                self.lab_kernel_upper_limit = upper_limit.copy()
            if target == "damaged_kernel":
                self.lab_damaged_kernel_lower_limit = lower_limit.copy()
                self.lab_damaged_kernel_upper_limit = upper_limit.copy()
            if target == "background" or target == "both":
                self.lab_background_lower_limit = lower_limit.copy()
                self.lab_background_upper_limit = upper_limit.copy()
            if target == "crack":
                self.lab_crack_lower_limit = lower_limit.copy()
                self.lab_crack_upper_limit = upper_limit.copy()

        # Print the updated threshold values
        print(f"\nSelected {color_space.upper()} thresholds:")
        if target == "pecan" or target == "both":
            print(f"Pecan lower limit: {self.thresholds[color_space]['pecan']['lower']}")
            print(f"Pecan upper limit: {self.thresholds[color_space]['pecan']['upper']}")
        if target == "kernel":
            print(f"Kernel lower limit: {self.thresholds[color_space]['kernel']['lower']}")
            print(f"Kernel upper limit: {self.thresholds[color_space]['kernel']['upper']}")
        if target == "damaged_kernel":
            print(f"Damaged Kernel lower limit: {self.thresholds[color_space]['damaged_kernel']['lower']}")
            print(f"Damaged Kernel upper limit: {self.thresholds[color_space]['damaged_kernel']['upper']}")
        if target == "background" or target == "both":
            print(f"Background lower limit: {self.thresholds[color_space]['background']['lower']}")
            print(f"Background upper limit: {self.thresholds[color_space]['background']['upper']}")
        if target == "crack":
            print(f"Crack lower limit: {self.thresholds[color_space]['crack']['lower']}")
            print(f"Crack upper limit: {self.thresholds[color_space]['crack']['upper']}")

        # Return self for method chaining
        return self

    # For backward compatibility
    def tune_hsv(self, target="pecan", use_defaults=True):
        """Backward compatibility wrapper for tune_color with HSV color space.

        Parameters:
            target (str): Which thresholds to tune: "pecan", "kernel", "background", "crack", or "both".
                Default is "pecan".
            use_defaults (bool): If True, initialize trackbars with current threshold values. Default is True.
        """
        return self.tune_color(target=target, color_space="hsv", use_defaults=use_defaults)

    def tune_lab(self, target="pecan", use_defaults=True):
        """Backward compatibility wrapper for tune_color with LAB color space.

        Parameters:
            target (str): Which thresholds to tune: "pecan", "kernel", "background", "crack", or "both".
                Default is "pecan".
            use_defaults (bool): If True, initialize trackbars with current threshold values. Default is True.
        """
        return self.tune_color(target=target, color_space="lab", use_defaults=use_defaults)

    def tune_color_advanced(self, resize_factor=None):
        """Launch the advanced color tuner GUI.

        This method launches an advanced GUI for tuning color thresholds in different
        color spaces (RGB, HSV, LAB) for different targets (pecan, kernel, crack, background).

        Parameters:
            resize_factor (float, optional): Factor to resize frames for faster processing (0.1-1.0).
                                            If None, automatically resizes to fit within 400x300 pixels.

        Returns:
            PecanVideo: Returns self for method chaining
        """
        # Get original dimensions
        if len(self.frames) == 0:
            raise ValueError("No frames available in the video")

        orig_height, orig_width = self.frames[0].shape[:2]

        # Determine resize factor if not provided
        if resize_factor is None:
            # Calculate resize factors to fit within 400x300
            width_factor = 400 / orig_width if orig_width > 400 else 1.0
            height_factor = 300 / orig_height if orig_height > 300 else 1.0

            # Use the smaller factor to ensure both dimensions fit
            resize_factor = min(width_factor, height_factor)

            # Always resize to ensure dimensions are less than 400x300
            if resize_factor >= 1.0 and (orig_width > 400 or orig_height > 300):
                resize_factor = min(400 / orig_width, 300 / orig_height)

        # Resize frames for faster processing
        print(f"Resizing frames by factor {resize_factor:.2f} for color tuning")
        print(f"Original size: {orig_width}x{orig_height}, New size: {int(orig_width * resize_factor)}x{int(orig_height * resize_factor)}")

        resized_frames = []
        for frame in self.frames:
            new_size = (int(frame.shape[1] * resize_factor), int(frame.shape[0] * resize_factor))
            resized_frames.append(cv2.resize(frame, new_size))

        # Create a temporary PecanVideo with resized frames
        temp_video = PecanVideo(np.array(resized_frames))

        # Copy thresholds to the temporary video
        temp_video.thresholds = self.thresholds.copy()

        # Create and run the color tuner on the resized video
        tuner = ColorTuner(temp_video)

        # Run the color tuner and ignore tkinter cleanup errors
        import sys

        # Save the original excepthook
        original_excepthook = sys.excepthook

        # Define a custom excepthook to ignore tkinter errors during cleanup
        def custom_excepthook(exc_type, exc_value, exc_traceback):
            if "main thread is not in main loop" in str(exc_value):
                # Ignore this specific error
                pass
            else:
                # For all other exceptions, use the original excepthook
                original_excepthook(exc_type, exc_value, exc_traceback)

        # Set our custom excepthook
        sys.excepthook = custom_excepthook

        try:
            # Run the color tuner
            tuner.run()
        finally:
            # Restore the original excepthook
            sys.excepthook = original_excepthook

            # Copy the tuned thresholds back to the original video
            self.thresholds = temp_video.thresholds.copy()

            # Update legacy threshold variables
            self._update_legacy_thresholds()

        # Return self for method chaining
        return self

    def _update_legacy_thresholds(self):
        """Update legacy threshold variables for backward compatibility."""
        self.pecan_lower_limit = self.thresholds["hsv"]["pecan"]["lower"].copy()
        self.pecan_upper_limit = self.thresholds["hsv"]["pecan"]["upper"].copy()
        self.background_lower_limit = self.thresholds["hsv"]["background"]["lower"].copy()
        self.background_upper_limit = self.thresholds["hsv"]["background"]["upper"].copy()
        self.crack_lower_limit = self.thresholds["hsv"]["crack"]["lower"].copy()
        self.crack_upper_limit = self.thresholds["hsv"]["crack"]["upper"].copy()
        self.kernel_lower_limit = self.thresholds["hsv"]["kernel"]["lower"].copy()
        self.kernel_upper_limit = self.thresholds["hsv"]["kernel"]["upper"].copy()
        self.damaged_kernel_lower_limit = self.thresholds["hsv"]["damaged_kernel"]["lower"].copy()
        self.damaged_kernel_upper_limit = self.thresholds["hsv"]["damaged_kernel"]["upper"].copy()
        self.lab_pecan_lower_limit = self.thresholds["lab"]["pecan"]["lower"].copy()
        self.lab_pecan_upper_limit = self.thresholds["lab"]["pecan"]["upper"].copy()
        self.lab_background_lower_limit = self.thresholds["lab"]["background"]["lower"].copy()
        self.lab_background_upper_limit = self.thresholds["lab"]["background"]["upper"].copy()
        self.lab_crack_lower_limit = self.thresholds["lab"]["crack"]["lower"].copy()
        self.lab_crack_upper_limit = self.thresholds["lab"]["crack"]["upper"].copy()
        self.lab_kernel_lower_limit = self.thresholds["lab"]["kernel"]["lower"].copy()
        self.lab_kernel_upper_limit = self.thresholds["lab"]["kernel"]["upper"].copy()
        self.lab_damaged_kernel_lower_limit = self.thresholds["lab"]["damaged_kernel"]["lower"].copy()
        self.lab_damaged_kernel_upper_limit = self.thresholds["lab"]["damaged_kernel"]["upper"].copy()

    def _compute_optical_flow(self):
        """Compute optical flow between consecutive frames.

        This method computes the dense optical flow between consecutive frames
        using the Farneback algorithm.

        Returns:
            list: A list of optical flow vectors for each pair of consecutive frames
        """
        if len(self.frames) < 2:
            raise ValueError("At least 2 frames are required to compute optical flow")

        # Convert frames to grayscale
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in self.frames]

        # Initialize list to store flow vectors
        flow_vectors = []

        # Compute optical flow between consecutive frames
        for i in tqdm(range(len(gray_frames) - 1), desc="Computing optical flow"):
            # Calculate optical flow using Farneback algorithm
            flow = cv2.calcOpticalFlowFarneback(gray_frames[i], gray_frames[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)  # Pyramid scale  # Levels  # Window size  # Iterations  # Poly neighborhood  # Poly sigma  # Flags
            flow_vectors.append(flow)

        return flow_vectors

    def motion_detection(self, threshold=3.0, kernel_size=(5, 5)):
        """Detect moving objects in the video using optical flow.

        This method computes optical flow between consecutive frames and creates
        masks for moving objects based on the magnitude of the flow vectors.

        Parameters:
            threshold (float): Threshold for detecting movement. Default is 3.0.
            kernel_size (tuple): Size of the kernel for morphological operations. Default is (5, 5).

        Returns:
            PecanVideo: A new PecanVideo instance containing the motion masks
        """
        # Check if we have enough frames
        if len(self.frames) < 2:
            raise ValueError("At least 2 frames are required for motion detection")

        # Compute optical flow
        flow_vectors = self._compute_optical_flow()

        # Create motion masks based on the magnitude of flow vectors
        motion_masks = []
        kernel = np.ones(kernel_size, np.uint8)

        for flow in tqdm(flow_vectors, desc="Creating motion masks"):
            # Calculate the magnitude of the flow vectors
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Create a binary mask where magnitude > threshold
            motion_mask = (magnitude > threshold).astype(np.uint8) * 255

            # Apply morphological operations to clean up the mask
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)

            # Keep only the largest contour (assuming it's the pecan)
            contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                largest_contour_mask = np.zeros_like(motion_mask)
                cv2.drawContours(largest_contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
                motion_mask = largest_contour_mask

            motion_masks.append(motion_mask)

        # Add a duplicate of the last mask to match the number of frames
        if len(motion_masks) < len(self.frames):
            motion_masks.append(motion_masks[-1].copy())

        # Convert to 3-channel for compatibility with PecanVideo
        motion_masks_3ch = [cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) for mask in motion_masks]

        # Create a new PecanVideo instance with the motion masks
        motion_video = PecanVideo(np.array(motion_masks_3ch))

        # Copy HSV threshold values and ellipses to the new instance
        motion_video.pecan_lower_limit = self.pecan_lower_limit.copy()
        motion_video.pecan_upper_limit = self.pecan_upper_limit.copy()
        motion_video.background_lower_limit = self.background_lower_limit.copy()
        motion_video.background_upper_limit = self.background_upper_limit.copy()
        motion_video.ellipses = self.ellipses

        print(f"Motion detection completed. {len(motion_masks)} masks created.")

        return motion_video

    def crop_to_pecan_motion(self, motion_masks=None, threshold=3.0):
        """Crop the video to focus on the pecan using motion detection.

        This method uses motion detection to identify the pecan and crop the video
        to focus on it.

        Parameters:
            motion_masks (PecanVideo, optional): Pre-computed motion masks. If None,
                the method will compute them using the motion_detection method.
            threshold (float): Threshold for detecting movement if masks are not provided.
                Default is 3.0.

        Returns:
            tuple: A tuple containing:
                - PecanVideo: A new PecanVideo instance containing the cropped frames
                - PecanVideo: A new PecanVideo instance containing the cropped masks
        """
        # If motion masks are not provided, compute them
        if motion_masks is None:
            motion_masks = self.motion_detection(threshold=threshold)
        elif not isinstance(motion_masks, PecanVideo):
            raise ValueError("motion_masks must be a PecanVideo instance")

        # Convert motion masks to grayscale if they are in BGR format
        masks = []
        for mask in tqdm(motion_masks.frames, desc="Processing motion masks"):
            if len(mask.shape) == 3 and mask.shape[2] == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            masks.append(mask)

        # Function to get bounding box
        def get_bounding_box(mask):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                return 0, 0, mask.shape[1], mask.shape[0]
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            return x, y, w, h

        # Find the overall bounding box to apply to all frames
        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = 0, 0
        for mask in masks:
            x, y, w, h = get_bounding_box(mask)
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)

        # Ensure min_x and min_y are not infinity (in case all masks are empty)
        if min_x == float("inf"):
            min_x = 0
        if min_y == float("inf"):
            min_y = 0

        # Crop all frames using the overall bounding box
        cropped_frames = []
        for frame in tqdm(self.frames, desc="Cropping frames"):
            cropped_frames.append(frame[min_y:max_y, min_x:max_x])

        # Also crop the masks using the same bounding box
        cropped_masks = []
        for mask in tqdm(masks, desc="Cropping masks"):
            cropped_masks.append(mask[min_y:max_y, min_x:max_x])

        # Create a new PecanVideo instance with the cropped frames
        cropped_video = PecanVideo(np.array(cropped_frames))

        # Create a new PecanVideo instance with the cropped masks
        cropped_mask_vid = PecanVideo(np.array([cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) for mask in cropped_masks]))

        # Copy HSV threshold values and ellipses to the new instances
        cropped_video.pecan_lower_limit = self.pecan_lower_limit.copy()
        cropped_video.pecan_upper_limit = self.pecan_upper_limit.copy()
        cropped_video.background_lower_limit = self.background_lower_limit.copy()
        cropped_video.background_upper_limit = self.background_upper_limit.copy()
        cropped_video.ellipses = self.ellipses

        cropped_mask_vid.pecan_lower_limit = self.pecan_lower_limit.copy()
        cropped_mask_vid.pecan_upper_limit = self.pecan_upper_limit.copy()
        cropped_mask_vid.background_lower_limit = self.background_lower_limit.copy()
        cropped_mask_vid.background_upper_limit = self.background_upper_limit.copy()
        cropped_mask_vid.ellipses = self.ellipses

        # Print information about the cropping
        print(f"Cropped video from {self.frames.shape[1:3]} to {cropped_video.frames.shape[1:3]} using motion detection")

        return cropped_video, cropped_mask_vid

    def hybrid_crop_to_pecan(self, hsv_weight=0.5, motion_weight=0.5, motion_threshold=3.0):
        """Crop the video to focus on the pecan using a hybrid of HSV and motion detection.

        This method combines HSV-based segmentation and motion detection to create
        a more robust mask for the pecan, then crops the video to focus on it.

        Parameters:
            hsv_weight (float): Weight for the HSV-based mask. Default is 0.5.
            motion_weight (float): Weight for the motion-based mask. Default is 0.5.
            motion_threshold (float): Threshold for detecting movement. Default is 3.0.

        Returns:
            tuple: A tuple containing:
                - PecanVideo: A new PecanVideo instance containing the cropped frames
                - PecanVideo: A new PecanVideo instance containing the cropped masks
        """
        # Normalize weights
        total_weight = hsv_weight + motion_weight
        hsv_weight = hsv_weight / total_weight
        motion_weight = motion_weight / total_weight

        # Get motion-based masks first (at original size)
        print("Getting motion-based masks...")
        motion_masks_video = self.motion_detection(threshold=motion_threshold)
        motion_masks = []
        for mask in tqdm(motion_masks_video.frames, desc="Processing motion masks"):
            if len(mask.shape) == 3 and mask.shape[2] == 3:
                motion_masks.append(cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))
            else:
                motion_masks.append(mask)

        # Create HSV masks at the original size (without cropping)
        print("Creating HSV-based masks...")
        hsv_masks = []
        # Convert frames to HSV
        hsv_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) for frame in self.frames]
        # Define the kernel for morphological operations
        kernel_close = np.ones((7, 7), np.uint8)
        kernel_open = np.ones((5, 5), np.uint8)
        kernel_dilate = np.ones((3, 3), np.uint8)

        def keep_largest_contour(mask):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                largest_contour_mask = np.zeros_like(mask)
                cv2.drawContours(largest_contour_mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
                return largest_contour_mask
            else:
                return mask

        for _, hsv_frame in tqdm(zip(self.frames, hsv_frames), desc="Creating HSV masks", total=len(self.frames)):
            # Create mask for the pecan using the dictionary-based thresholds
            pecan_mask = cv2.inRange(hsv_frame, self.thresholds["hsv"]["pecan"]["lower"], self.thresholds["hsv"]["pecan"]["upper"])
            # Apply morphological operations
            pecan_mask = cv2.morphologyEx(pecan_mask, cv2.MORPH_CLOSE, kernel_close)
            pecan_mask = cv2.morphologyEx(pecan_mask, cv2.MORPH_OPEN, kernel_open)
            pecan_mask = cv2.dilate(pecan_mask, kernel_dilate, iterations=1)
            pecan_mask = cv2.erode(pecan_mask, kernel_dilate, iterations=1)
            # Keep only the largest contour in the pecan mask
            pecan_mask = keep_largest_contour(pecan_mask)
            # Create mask for the background
            background_mask = cv2.inRange(hsv_frame, self.thresholds["hsv"]["background"]["lower"], self.thresholds["hsv"]["background"]["upper"])
            # Invert the background mask
            inverted_background_mask = cv2.bitwise_not(background_mask)
            # Combine the pecan mask and the inverted background mask
            combined_mask = cv2.bitwise_and(pecan_mask, inverted_background_mask)
            hsv_masks.append(combined_mask)

        # Combine the masks with weighted average
        combined_masks = []
        for hsv_mask, motion_mask in tqdm(zip(hsv_masks, motion_masks), desc="Combining HSV and motion masks", total=len(hsv_masks)):
            # Verify shapes match
            if hsv_mask.shape != motion_mask.shape:
                print(f"Warning: Mask shapes don't match. HSV mask: {hsv_mask.shape}, Motion mask: {motion_mask.shape}")
                # Resize the HSV mask to match the motion mask if needed
                if hsv_mask.shape[0] != motion_mask.shape[0] or hsv_mask.shape[1] != motion_mask.shape[1]:
                    hsv_mask = cv2.resize(hsv_mask, (motion_mask.shape[1], motion_mask.shape[0]))

            # Convert to float for weighted average
            hsv_mask_float = hsv_mask.astype(float) / 255.0
            motion_mask_float = motion_mask.astype(float) / 255.0

            # Weighted average
            combined_mask_float = hsv_weight * hsv_mask_float + motion_weight * motion_mask_float

            # Convert back to uint8
            combined_mask = (combined_mask_float * 255).astype(np.uint8)

            # Threshold to get binary mask
            _, combined_mask_binary = cv2.threshold(combined_mask, 127, 255, cv2.THRESH_BINARY)

            # Apply morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            combined_mask_binary = cv2.morphologyEx(combined_mask_binary, cv2.MORPH_CLOSE, kernel)
            combined_mask_binary = cv2.morphologyEx(combined_mask_binary, cv2.MORPH_OPEN, kernel)

            # Keep only the largest contour (assuming it's the pecan)
            contours, _ = cv2.findContours(combined_mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                largest_contour_mask = np.zeros_like(combined_mask_binary)
                cv2.drawContours(largest_contour_mask, [largest_contour], -1, (255), thickness=cv2.FILLED)
                combined_mask_binary = largest_contour_mask

            combined_masks.append(combined_mask_binary)

        # Function to get bounding box
        def get_bounding_box(mask):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                return 0, 0, mask.shape[1], mask.shape[0]
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            return x, y, w, h

        # Find the overall bounding box to apply to all frames
        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = 0, 0
        for mask in combined_masks:
            x, y, w, h = get_bounding_box(mask)
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)

        # Ensure min_x and min_y are not infinity (in case all masks are empty)
        if min_x == float("inf"):
            min_x = 0
        if min_y == float("inf"):
            min_y = 0

        # Crop all frames using the overall bounding box
        cropped_frames = []
        for frame in tqdm(self.frames, desc="Cropping frames"):
            cropped_frames.append(frame[min_y:max_y, min_x:max_x])

        # Also crop the masks using the same bounding box
        cropped_masks = []
        for mask in tqdm(combined_masks, desc="Cropping masks"):
            cropped_masks.append(mask[min_y:max_y, min_x:max_x])

        # Create a new PecanVideo instance with the cropped frames
        cropped_video = PecanVideo(np.array(cropped_frames))

        # Create a new PecanVideo instance with the cropped masks
        cropped_mask_vid = PecanVideo(np.array([cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) for mask in cropped_masks]))

        # Copy HSV threshold values and ellipses to the new instances
        cropped_video.pecan_lower_limit = self.pecan_lower_limit.copy()
        cropped_video.pecan_upper_limit = self.pecan_upper_limit.copy()
        cropped_video.background_lower_limit = self.background_lower_limit.copy()
        cropped_video.background_upper_limit = self.background_upper_limit.copy()
        cropped_video.ellipses = self.ellipses

        cropped_mask_vid.pecan_lower_limit = self.pecan_lower_limit.copy()
        cropped_mask_vid.pecan_upper_limit = self.pecan_upper_limit.copy()
        cropped_mask_vid.background_lower_limit = self.background_lower_limit.copy()
        cropped_mask_vid.background_upper_limit = self.background_upper_limit.copy()
        cropped_mask_vid.ellipses = self.ellipses

        # Print information about the cropping
        print(f"Cropped video from {self.frames.shape[1:3]} to {cropped_video.frames.shape[1:3]} using hybrid detection")
        print(f"Used HSV weight: {hsv_weight:.2f}, Motion weight: {motion_weight:.2f}")

        return cropped_video, cropped_mask_vid

    def crop_to_pecan(self):
        """Create masks using HSV thresholds and crop both frames and masks to focus on the pecan.

        This method creates masks for the pecan using the HSV thresholds and crops
        both the frames and masks to focus on the pecan.

        Returns:
            tuple: A tuple containing:
                - PecanVideo: A new PecanVideo instance containing the cropped frames
                - PecanVideo: A new PecanVideo instance containing the cropped masks
        """
        # Convert frames to HSV
        hsv_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) for frame in self.frames]
        # Define the kernel for morphological operations
        kernel_close = np.ones((7, 7), np.uint8)
        kernel_open = np.ones((5, 5), np.uint8)
        kernel_dilate = np.ones((3, 3), np.uint8)

        def keep_largest_contour(mask):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                largest_contour_mask = np.zeros_like(mask)
                cv2.drawContours(largest_contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
                return largest_contour_mask
            else:
                return mask

        masks = []
        for _, hsv_frame in zip(self.frames, hsv_frames):
            # Create mask for the pecan using the dictionary-based thresholds
            pecan_mask = cv2.inRange(hsv_frame, self.thresholds["hsv"]["pecan"]["lower"], self.thresholds["hsv"]["pecan"]["upper"])
            # Apply morphological operations
            pecan_mask = cv2.morphologyEx(pecan_mask, cv2.MORPH_CLOSE, kernel_close)
            pecan_mask = cv2.morphologyEx(pecan_mask, cv2.MORPH_OPEN, kernel_open)
            pecan_mask = cv2.dilate(pecan_mask, kernel_dilate, iterations=1)
            pecan_mask = cv2.erode(pecan_mask, kernel_dilate, iterations=1)
            # Keep only the largest contour in the pecan mask
            pecan_mask = keep_largest_contour(pecan_mask)
            # Create mask for the background
            background_mask = cv2.inRange(hsv_frame, self.thresholds["hsv"]["background"]["lower"], self.thresholds["hsv"]["background"]["upper"])
            # Invert the background mask
            inverted_background_mask = cv2.bitwise_not(background_mask)
            # Combine the pecan mask and the inverted background mask
            combined_mask = cv2.bitwise_and(pecan_mask, inverted_background_mask)
            masks.append(combined_mask)

        # We'll create a PecanVideo instance with the cropped masks later

        # Function to get bounding box
        def get_bounding_box(mask):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                return 0, 0, mask.shape[1], mask.shape[0]
            cnt = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            return x, y, w, h

        # Find the overall bounding box to apply to all frames
        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = 0, 0
        for mask in masks:
            x, y, w, h = get_bounding_box(mask)
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)

        # Ensure min_x and min_y are not infinity (in case all masks are empty)
        if min_x == float("inf"):
            min_x = 0
        if min_y == float("inf"):
            min_y = 0

        # Crop all frames using the overall bounding box
        cropped_frames = [frame[min_y:max_y, min_x:max_x] for frame in self.frames]

        # Also crop the masks using the same bounding box
        cropped_masks = [mask[min_y:max_y, min_x:max_x] for mask in masks]

        # Create a new PecanVideo instance with the cropped frames
        cropped_video = PecanVideo(np.array(cropped_frames))

        # Create a new PecanVideo instance with the cropped masks
        cropped_mask_vid = PecanVideo(np.array(cropped_masks))

        # Copy HSV threshold values and ellipses to the new instances
        cropped_video.pecan_lower_limit = self.pecan_lower_limit.copy()
        cropped_video.pecan_upper_limit = self.pecan_upper_limit.copy()
        cropped_video.background_lower_limit = self.background_lower_limit.copy()
        cropped_video.background_upper_limit = self.background_upper_limit.copy()
        cropped_video.ellipses = self.ellipses  # Copy the ellipses to the new instance

        cropped_mask_vid.pecan_lower_limit = self.pecan_lower_limit.copy()
        cropped_mask_vid.pecan_upper_limit = self.pecan_upper_limit.copy()
        cropped_mask_vid.background_lower_limit = self.background_lower_limit.copy()
        cropped_mask_vid.background_upper_limit = self.background_upper_limit.copy()
        cropped_mask_vid.ellipses = self.ellipses  # Copy the ellipses to the new instance

        # Print information about the cropping
        print(f"Cropped video from {self.frames.shape[1:3]} to {cropped_video.frames.shape[1:3]}")

        return cropped_video, cropped_mask_vid

    def edge_detection(self):
        """Detect edges in the video frames.

        This method detects edges in the video frames using the Canny edge detection algorithm.

        Returns:
            PecanVideo: A new PecanVideo instance containing the edge frames
        """
        # Check if frames exist
        if self.frames is None or len(self.frames) == 0:
            raise ValueError("No frames available for edge detection")

        # Initialize list to store edge frames
        edge_frames = []

        # Process each frame
        for i in range(len(self.frames)):
            # Convert the frame to grayscale
            gray = cv2.cvtColor(self.frames[i], cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)

            # Detect edges using Canny edge detection
            edges = cv2.Canny(blurred, 50, 150)

            # Convert back to BGR for compatibility with Video class
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            # Add to the list of edge frames
            edge_frames.append(edges_bgr)

        # Create a new PecanVideo instance with the edge frames
        edge_video = PecanVideo(np.array(edge_frames))

        # Copy HSV threshold values and ellipses to the new instance
        edge_video.pecan_lower_limit = self.pecan_lower_limit.copy()
        edge_video.pecan_upper_limit = self.pecan_upper_limit.copy()
        edge_video.background_lower_limit = self.background_lower_limit.copy()
        edge_video.background_upper_limit = self.background_upper_limit.copy()
        edge_video.ellipses = self.ellipses  # Copy the ellipses to the new instance

        print(f"Edge detection completed. {len(edge_frames)} frames processed.")

        return edge_video

    def shrink(self, kernel_size=(25, 25), iterations=1):
        """Shrink the mask using erosion.

        This method shrinks the mask using erosion.

        Parameters:
            kernel_size (tuple): Size of the kernel for erosion. Default is (25, 25).
            iterations (int): Number of times erosion is applied. Default is 1.

        Returns:
            PecanVideo: A new PecanVideo instance containing the shrunken frames
        """
        # Check if frames exist
        if self.frames is None or len(self.frames) == 0:
            raise ValueError("No frames available for shrinking")

        # Initialize list to store shrunken frames
        shrunken_frames = []

        # Create kernel for erosion
        kernel = np.ones(kernel_size, np.uint8)

        # Process each frame
        for i in range(len(self.frames)):
            # Convert the frame to grayscale
            gray = cv2.cvtColor(self.frames[i], cv2.COLOR_BGR2GRAY)

            # Create a binary mask (threshold the grayscale image)
            _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

            # Apply erosion to shrink the mask
            shrunken_mask = cv2.erode(mask, kernel, iterations=iterations)

            # Convert back to BGR for compatibility with Video class
            shrunken_mask_bgr = cv2.cvtColor(shrunken_mask, cv2.COLOR_GRAY2BGR)

            # Add to the list of shrunken frames
            shrunken_frames.append(shrunken_mask_bgr)

        # Create a new PecanVideo instance with the shrunken frames
        shrunken_video = PecanVideo(np.array(shrunken_frames))

        # Copy HSV threshold values and ellipses to the new instance
        shrunken_video.pecan_lower_limit = self.pecan_lower_limit.copy()
        shrunken_video.pecan_upper_limit = self.pecan_upper_limit.copy()
        shrunken_video.background_lower_limit = self.background_lower_limit.copy()
        shrunken_video.background_upper_limit = self.background_upper_limit.copy()
        shrunken_video.ellipses = self.ellipses  # Copy the ellipses to the new instance

        print(f"Shrinking completed with kernel size {kernel_size} and {iterations} iterations. {len(shrunken_frames)} frames processed.")

        return shrunken_video

    def draw_ellipse_and_axis(self, masks=None):
        """Draw ellipses and their major axes on the video frames.

        This method fits ellipses to the contours in the masks and draws them along with their
        major axes on blank frames of the same size as the video frames.

        Parameters:
            masks (ndarray or PecanVideo, optional): Masks to use for fitting ellipses. If None,
                the method will create masks from the current video frames using the HSV thresholds.

        Returns:
            tuple: A tuple containing:
                - PecanVideo: A new PecanVideo instance containing the frames with drawn ellipses and axes
                - list: A list of ellipse parameters (center, axes, angle) for each frame
        """
        # If masks are not provided, create them using the HSV thresholds
        if masks is None:
            # Create masks using the current video frames and HSV thresholds
            _, masks_video = self.crop_to_pecan()
            masks = masks_video.frames
        elif isinstance(masks, PecanVideo):
            # Extract frames from PecanVideo object
            masks = masks.frames

        # Ensure masks are in the correct format (grayscale)
        if len(masks.shape) == 4 and masks.shape[3] == 3:
            # Convert BGR masks to grayscale
            masks = np.array([cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) for mask in masks])

        # Function to get ellipse parameters from a mask
        def get_ellipse_and_angle(mask):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                return None
            cnt = max(contours, key=cv2.contourArea)
            if len(cnt) < 5:  # Need at least 5 points to fit an ellipse
                return None
            ellipse = cv2.fitEllipse(cnt)
            return ellipse

        ellipses = []
        # Create a blank video array of the same size as the current frames
        blank_video_frames = np.zeros_like(self.frames)

        for mask, frame in zip(masks, blank_video_frames):
            ellipse = get_ellipse_and_angle(mask)
            if ellipse is not None:
                ellipses.append(ellipse)
                center, axes, angle = ellipse
                # Draw the ellipse in green
                cv2.ellipse(frame, ellipse, (0, 255, 0), 2)

                # Determine the angle of the long axis
                # If the first axis is longer than the second, use the given angle
                # Otherwise, add 90 degrees to get the angle of the long axis
                angle = angle if axes[0] > axes[1] else angle + 90

                # Draw the long axis in red
                # Convert angle to radians and calculate the vector for the long axis
                long_axis_vector = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))]) * max(axes)
                point1 = (int(center[0] - long_axis_vector[0]), int(center[1] - long_axis_vector[1]))
                point2 = (int(center[0] + long_axis_vector[0]), int(center[1] + long_axis_vector[1]))
                cv2.line(frame, point1, point2, (0, 0, 255), 2)
            else:
                # If no ellipse could be fitted, add None to the list
                ellipses.append(None)

        # Store the ellipses in the current instance
        self.ellipses = ellipses

        # Create a new PecanVideo instance with the annotated frames
        annotated_video = PecanVideo(np.array(blank_video_frames))

        # Copy HSV threshold values and ellipses to the new instance
        annotated_video.pecan_lower_limit = self.pecan_lower_limit.copy()
        annotated_video.pecan_upper_limit = self.pecan_upper_limit.copy()
        annotated_video.background_lower_limit = self.background_lower_limit.copy()
        annotated_video.background_upper_limit = self.background_upper_limit.copy()
        annotated_video.ellipses = self.ellipses  # Copy the ellipses to the new instance

        print(f"Ellipse and axis drawing completed. {len(ellipses)} frames processed.")

        return annotated_video, ellipses

    def rotate_to_align(self, masks=None, annotations=None):
        """Rotate the video frames to align the pecan's long axis with the X-axis.

        This method rotates the video frames, masks, and annotations to align the pecan's
        long axis with the X-axis. It uses the ellipses drawn by the draw_ellipse_and_axis
        method to determine the angle of rotation.

        Parameters:
            masks (PecanVideo, optional): Masks to use for rotation. If None, the method
                will create masks from the current video frames using the HSV thresholds.
            annotations (PecanVideo, optional): Annotations to use for rotation. If None,
                the method will create annotations using the draw_ellipse_and_axis method.

        Returns:
            tuple: A tuple containing:
                - PecanVideo: A new PecanVideo instance containing the rotated frames
                - PecanVideo: A new PecanVideo instance containing the rotated masks
                - PecanVideo: A new PecanVideo instance containing the rotated annotations
        """
        # If masks are not provided, create them using the HSV thresholds
        if masks is None:
            _, masks_video = self.crop_to_pecan()
            masks = masks_video
        elif not isinstance(masks, PecanVideo):
            raise ValueError("masks must be a PecanVideo instance")

        # If annotations are not provided, create them using the draw_ellipse_and_axis method
        if annotations is None:
            annotations_video, _ = self.draw_ellipse_and_axis(masks)
            annotations = annotations_video
        elif not isinstance(annotations, PecanVideo):
            raise ValueError("annotations must be a PecanVideo instance")

        # Helper function to extract ellipse information from an annotated frame
        def extract_ellipse_info(annotated_frame):
            gray_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(gray_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                return None
            cnt = max(contours, key=cv2.contourArea)
            if len(cnt) < 5:  # Need at least 5 points to fit an ellipse
                return None
            ellipse = cv2.fitEllipse(cnt)
            return ellipse

        # Helper function to rotate a frame
        def rotate_frame(frame, center, angle):
            height, width = frame.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_frame = cv2.warpAffine(frame, rotation_matrix, (width, height))
            return rotated_frame

        rotated_frames = []
        rotated_masks = []
        rotated_annotations = []

        # Process each frame
        for mask_frame, frame, annotated_frame in zip(masks.frames, self.frames, annotations.frames):
            ellipse = extract_ellipse_info(annotated_frame)
            if ellipse is not None:
                center, axes, angle = ellipse
                # Determine the angle of the long axis
                # If the first axis is longer than the second, use the given angle
                # Otherwise, subtract 90 degrees to get the angle of the long axis
                angle = angle if axes[0] > axes[1] else angle - 90

                # Rotate the frame, mask, and annotation to align the long axis with the X-axis
                rotated_frame = rotate_frame(frame, center, angle)
                rotated_mask = rotate_frame(mask_frame, center, angle)
                rotated_annotation = rotate_frame(annotated_frame, center, angle)

                rotated_frames.append(rotated_frame)
                rotated_masks.append(rotated_mask)
                rotated_annotations.append(rotated_annotation)
            else:
                # If no ellipse could be fitted, keep the original frames
                rotated_frames.append(frame)
                rotated_masks.append(mask_frame)
                rotated_annotations.append(annotated_frame)

        # Create new PecanVideo instances with the rotated frames, masks, and annotations
        rotated_video = PecanVideo(np.array(rotated_frames))
        rotated_masks_video = PecanVideo(np.array(rotated_masks))
        rotated_annotations_video = PecanVideo(np.array(rotated_annotations))

        # Copy HSV threshold values and ellipses to the new instances
        rotated_video.pecan_lower_limit = self.pecan_lower_limit.copy()
        rotated_video.pecan_upper_limit = self.pecan_upper_limit.copy()
        rotated_video.background_lower_limit = self.background_lower_limit.copy()
        rotated_video.background_upper_limit = self.background_upper_limit.copy()
        rotated_video.ellipses = self.ellipses  # Copy the ellipses to the new instance

        rotated_masks_video.pecan_lower_limit = self.pecan_lower_limit.copy()
        rotated_masks_video.pecan_upper_limit = self.pecan_upper_limit.copy()
        rotated_masks_video.background_lower_limit = self.background_lower_limit.copy()
        rotated_masks_video.background_upper_limit = self.background_upper_limit.copy()
        rotated_masks_video.ellipses = self.ellipses  # Copy the ellipses to the new instance

        rotated_annotations_video.pecan_lower_limit = self.pecan_lower_limit.copy()
        rotated_annotations_video.pecan_upper_limit = self.pecan_upper_limit.copy()
        rotated_annotations_video.background_lower_limit = self.background_lower_limit.copy()
        rotated_annotations_video.background_upper_limit = self.background_upper_limit.copy()
        rotated_annotations_video.ellipses = self.ellipses  # Copy the ellipses to the new instance

        print(f"Rotation completed. {len(rotated_frames)} frames processed.")

        return rotated_video, rotated_masks_video, rotated_annotations_video

    def center_pecan(self, masks=None, annotations=None):
        """Center the pecan in each frame of the video.

        This method centers the pecan in each frame of the video by translating the frames
        so that the center of the ellipse fitted to the pecan is at the center of the frame.

        Parameters:
            masks (PecanVideo, optional): Masks to use for centering. If None, the method
                will create masks from the current video frames using the HSV thresholds.
            annotations (PecanVideo, optional): Annotations to use for centering. If None,
                the method will create annotations using the draw_ellipse_and_axis method.

        Returns:
            tuple: A tuple containing:
                - PecanVideo: A new PecanVideo instance containing the centered frames
                - PecanVideo: A new PecanVideo instance containing the centered masks
                - PecanVideo: A new PecanVideo instance containing the centered annotations
        """
        # If masks are not provided, create them using the HSV thresholds
        if masks is None:
            _, masks_video = self.crop_to_pecan()
            masks = masks_video
        elif not isinstance(masks, PecanVideo):
            raise ValueError("masks must be a PecanVideo instance")

        # If annotations are not provided, create them using the draw_ellipse_and_axis method
        if annotations is None:
            annotations_video, _ = self.draw_ellipse_and_axis(masks)
            annotations = annotations_video
        elif not isinstance(annotations, PecanVideo):
            raise ValueError("annotations must be a PecanVideo instance")

        # Helper function to extract ellipse information from an annotated frame
        def extract_ellipse_info(annotated_frame):
            gray_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(gray_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                return None
            cnt = max(contours, key=cv2.contourArea)
            if len(cnt) < 5:  # Need at least 5 points to fit an ellipse
                return None
            ellipse = cv2.fitEllipse(cnt)
            return ellipse

        # Helper function to translate a frame
        def translate_frame(frame, translation_matrix):
            height, width = frame.shape[:2]
            translated_frame = cv2.warpAffine(frame, translation_matrix, (width, height))
            return translated_frame

        centered_frames = []
        centered_masks = []
        centered_annotations = []

        # Get the center of the frame
        frame_height, frame_width = self.frames[0].shape[:2]
        center_x, center_y = frame_width // 2, frame_height // 2

        # Process each frame
        for mask_frame, frame, annotated_frame in zip(masks.frames, self.frames, annotations.frames):
            ellipse = extract_ellipse_info(annotated_frame)
            if ellipse is not None:
                (ellipse_center_x, ellipse_center_y), _, _ = ellipse

                # Calculate the translation needed to center the ellipse
                translation_x = center_x - ellipse_center_x
                translation_y = center_y - ellipse_center_y
                translation_matrix = np.float32([[1, 0, translation_x], [0, 1, translation_y]])

                # Translate the frame, mask, and annotation
                centered_frame = translate_frame(frame, translation_matrix)
                centered_mask = translate_frame(mask_frame, translation_matrix)
                centered_annotation = translate_frame(annotated_frame, translation_matrix)

                centered_frames.append(centered_frame)
                centered_masks.append(centered_mask)
                centered_annotations.append(centered_annotation)
            else:
                # If no ellipse could be fitted, keep the original frames
                centered_frames.append(frame)
                centered_masks.append(mask_frame)
                centered_annotations.append(annotated_frame)

        # Create new PecanVideo instances with the centered frames, masks, and annotations
        centered_video = PecanVideo(np.array(centered_frames))
        centered_masks_video = PecanVideo(np.array(centered_masks))
        centered_annotations_video = PecanVideo(np.array(centered_annotations))

        # Copy HSV threshold values and ellipses to the new instances
        centered_video.pecan_lower_limit = self.pecan_lower_limit.copy()
        centered_video.pecan_upper_limit = self.pecan_upper_limit.copy()
        centered_video.background_lower_limit = self.background_lower_limit.copy()
        centered_video.background_upper_limit = self.background_upper_limit.copy()
        centered_video.ellipses = self.ellipses  # Copy the ellipses to the new instance

        centered_masks_video.pecan_lower_limit = self.pecan_lower_limit.copy()
        centered_masks_video.pecan_upper_limit = self.pecan_upper_limit.copy()
        centered_masks_video.background_lower_limit = self.background_lower_limit.copy()
        centered_masks_video.background_upper_limit = self.background_upper_limit.copy()
        centered_masks_video.ellipses = self.ellipses  # Copy the ellipses to the new instance

        centered_annotations_video.pecan_lower_limit = self.pecan_lower_limit.copy()
        centered_annotations_video.pecan_upper_limit = self.pecan_upper_limit.copy()
        centered_annotations_video.background_lower_limit = self.background_lower_limit.copy()
        centered_annotations_video.background_upper_limit = self.background_upper_limit.copy()
        centered_annotations_video.ellipses = self.ellipses  # Copy the ellipses to the new instance

        print(f"Centering completed. {len(centered_frames)} frames processed.")

        return centered_video, centered_masks_video, centered_annotations_video

    def apply_mask(self, masks):
        """Apply masks to the video frames.

        This method applies masks to the video frames, keeping only the pixels where the mask is non-zero.

        Parameters:
            masks (BaseVideo): Masks to apply to the video frames. Can be a BaseVideo or PecanVideo instance.

        Returns:
            PecanVideo: A new PecanVideo instance containing the masked frames.
        """
        if not isinstance(masks, BaseVideo):
            raise ValueError("masks must be a BaseVideo instance")

        if len(self.frames) != len(masks.frames):
            raise ValueError("The video and masks must have the same number of frames.")

        masked_frames = []

        # Process each frame
        for frame, mask_frame in zip(self.frames, masks.frames):
            # Ensure mask frame is single channel
            if len(mask_frame.shape) == 3 and mask_frame.shape[2] == 3:
                mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)

            # Ensure mask is binary (0 or 255) and uint8 type
            if mask_frame.dtype != np.uint8:
                mask_frame = mask_frame.astype(np.uint8)

            # If mask values are not binary, threshold it
            if np.max(mask_frame) != 255 or np.min(mask_frame[mask_frame > 0]) != 255:
                _, mask_frame = cv2.threshold(mask_frame, 127, 255, cv2.THRESH_BINARY)

            # Resize mask if needed to match frame dimensions
            if mask_frame.shape[:2] != frame.shape[:2]:
                mask_frame = cv2.resize(mask_frame, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Apply the mask to the frame
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask_frame)
            masked_frames.append(masked_frame)

        # Create a new PecanVideo instance with the masked frames
        masked_video = PecanVideo(np.array(masked_frames))

        # Copy HSV threshold values and ellipses to the new instance
        masked_video.pecan_lower_limit = self.pecan_lower_limit.copy() if hasattr(self, "pecan_lower_limit") else None
        masked_video.pecan_upper_limit = self.pecan_upper_limit.copy() if hasattr(self, "pecan_upper_limit") else None
        masked_video.background_lower_limit = self.background_lower_limit.copy() if hasattr(self, "background_lower_limit") else None
        masked_video.background_upper_limit = self.background_upper_limit.copy() if hasattr(self, "background_upper_limit") else None

        # Copy thresholds dictionary if it exists
        if hasattr(self, "thresholds"):
            masked_video.thresholds = self.thresholds.copy() if self.thresholds else None

        # Copy ellipses if they exist
        if hasattr(self, "ellipses"):
            masked_video.ellipses = self.ellipses  # Copy the ellipses to the new instance

        print(f"Masking completed. {len(masked_frames)} frames processed.")

        return masked_video

    def rotate_video(self, angle=90):
        """Rotate the video frames by a specified angle.

        This method rotates all frames in the video by a specified angle.
        It supports 90, -90, and 180 degrees rotations using OpenCV's optimized functions.

        Parameters:
            angle (int): The angle to rotate the frames. Supported values are 90, -90, and 180 degrees.
                Default is 90 degrees (clockwise).

        Returns:
            PecanVideo: A new PecanVideo instance containing the rotated frames.
        """
        # Define the rotation type based on the angle
        if angle == 90:
            rotation_code = cv2.ROTATE_90_CLOCKWISE
        elif angle == -90:
            rotation_code = cv2.ROTATE_90_COUNTERCLOCKWISE
        elif angle == 180:
            rotation_code = cv2.ROTATE_180
        else:
            raise ValueError("Angle must be 90, -90, or 180 degrees.")

        # Rotate each frame in the video
        rotated_frames = [cv2.rotate(frame, rotation_code) for frame in self.frames]

        # Create a new PecanVideo instance with the rotated frames
        rotated_video = PecanVideo(np.array(rotated_frames))

        # Copy HSV threshold values and ellipses to the new instance
        rotated_video.pecan_lower_limit = self.pecan_lower_limit.copy()
        rotated_video.pecan_upper_limit = self.pecan_upper_limit.copy()
        rotated_video.background_lower_limit = self.background_lower_limit.copy()
        rotated_video.background_upper_limit = self.background_upper_limit.copy()
        rotated_video.ellipses = self.ellipses  # Copy the ellipses to the new instance

        print(f"Video rotation completed. {len(rotated_frames)} frames rotated by {angle} degrees.")

        return rotated_video

    def create_wide_image(self, strip_width=5, adaptive_width=True, smoothing_window=5, edge_enhance=False, crack_threshold=None, wobble_correction=True, wobble_strength=2.0, height_compensation=True, trim_to_one_rotation=False, similarity_threshold=0.20, min_frames=70, window_size=15):
        """Create a wide image from a rotating pecan video using cylindrical projection.

        This method extracts vertical strips from each frame of the video and stitches them
        together to create an unwrapped view of the pecan surface. It can adapt the strip width
        based on the rotation speed of the pecan and correct for wobbling around the short axis.

        Parameters:
            strip_width (int): Width of the strip to extract from each frame. Default is 5 pixels.
            adaptive_width (bool): Whether to adapt the strip width based on rotation speed. Default is True.
            smoothing_window (int): Window size for smoothing the rotation speed. Default is 5 frames.
            edge_enhance (bool): Whether to enhance edges in the resulting image. Default is False.
            crack_threshold (tuple, optional): Tuple of (lower, upper) thresholds for crack detection.
                If provided, the method will also return a binary mask of detected cracks.
            wobble_correction (bool): Whether to correct for wobbling around the short axis. Default is True.
            wobble_strength (float): Multiplier for the wobble correction. Higher values apply stronger correction. Default is 2.0.
            height_compensation (bool): Whether to compensate for height changes due to wobbling. Default is True.
            trim_to_one_rotation (bool): Whether to trim the video to contain only one full rotation. Default is False.
            similarity_threshold (float): Threshold for considering frames as similar when detecting rotation. Default is 0.20.
            min_frames (int): Minimum number of frames to consider as a full rotation. Default is 70.
            window_size (int): Size of the window for smoothing similarity scores. Default is 15.

        Returns:
            tuple: A tuple containing:
                - numpy.ndarray: The wide image created from the video
                - numpy.ndarray (optional): Binary mask of detected cracks (if crack_threshold is provided)
        """
        # Check if the video has frames
        if len(self.frames) == 0:
            raise ValueError("Video has no frames")

        # Trim the video to one rotation if requested
        if trim_to_one_rotation:
            # Create a copy of the current video
            video_copy = PecanVideo(self.frames.copy())
            video_copy.pecan_lower_limit = self.pecan_lower_limit
            video_copy.pecan_upper_limit = self.pecan_upper_limit

            # Trim the video to one rotation with the specified parameters
            trimmed_video = video_copy.trim_to_one_rotation(similarity_threshold=similarity_threshold, min_frames=min_frames, window_size=window_size)

            # Create the wide image from the trimmed video
            return trimmed_video.create_wide_image(strip_width=strip_width, adaptive_width=adaptive_width, smoothing_window=smoothing_window, edge_enhance=edge_enhance, crack_threshold=crack_threshold, wobble_correction=wobble_correction, wobble_strength=wobble_strength, height_compensation=height_compensation, trim_to_one_rotation=False)  # Prevent infinite recursion

        # Get the dimensions of the frames
        n_frames, height, width, channels = self.frames.shape

        # Calculate the center column of each frame
        center_col = width // 2

        # Initialize the list to store strips
        strips = []

        # If adaptive_width is True, estimate the rotation speed
        if adaptive_width:
            # Calculate frame-to-frame differences to estimate rotation speed
            diffs = []
            for i in range(1, n_frames):
                # Calculate the absolute difference between consecutive frames
                diff = cv2.absdiff(self.frames[i], self.frames[i - 1])
                # Sum the differences to get a single value representing the amount of change
                diff_sum = np.sum(diff)
                diffs.append(diff_sum)

            # Normalize the differences
            diffs = np.array(diffs)
            if np.max(diffs) > 0:  # Avoid division by zero
                diffs = diffs / np.max(diffs)

            # Smooth the differences using a moving average
            if smoothing_window > 1:
                diffs = np.convolve(diffs, np.ones(smoothing_window) / smoothing_window, mode="valid")
                # Pad the beginning to maintain the same length
                diffs = np.pad(diffs, (smoothing_window // 2, (smoothing_window - 1) // 2), "edge")

            # Calculate adaptive strip widths based on rotation speed
            # Slower rotation (smaller diff) = wider strips
            # Faster rotation (larger diff) = narrower strips
            min_width = max(1, strip_width // 2)
            max_width = strip_width * 2

            # Invert the diffs so that smaller diffs (slower rotation) result in wider strips
            inv_diffs = 1.0 - diffs

            # Scale to the range [min_width, max_width]
            strip_widths = min_width + inv_diffs * (max_width - min_width)
            strip_widths = np.round(strip_widths).astype(int)

            # Ensure the first frame has a valid strip width
            strip_widths = np.insert(strip_widths, 0, strip_width)
        else:
            # Use constant strip width for all frames
            strip_widths = np.full(n_frames, strip_width, dtype=int)

        # Detect pecan centers and wobble for each frame if wobble correction is enabled
        if wobble_correction:
            # Convert frames to grayscale for contour detection
            gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in self.frames]

            # Find contours in each frame to detect the pecan
            centers = []
            heights = []
            contours_list = []
            for gray_frame in gray_frames:
                # Apply threshold to separate pecan from background
                _, thresh = cv2.threshold(gray_frame, 10, 255, cv2.THRESH_BINARY)

                # Find contours
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # If contours are found, get the largest one (the pecan)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    contours_list.append(largest_contour)

                    # Calculate the center of the contour
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        centers.append((cx, cy))

                        # Calculate the height of the contour (for height compensation)
                        _, _, _, h = cv2.boundingRect(largest_contour)
                        heights.append(h)
                    else:
                        # If moments calculation fails, use the center of the frame
                        centers.append((width // 2, height // 2))
                        heights.append(height)
                else:
                    # If no contours are found, use the center of the frame
                    centers.append((width // 2, height // 2))
                    heights.append(height)
                    contours_list.append(None)

            # Calculate the average center y-coordinate
            avg_cy = np.mean([cy for _, cy in centers])

            # Calculate the vertical offset for each frame and apply wobble strength
            vertical_offsets = [int((avg_cy - cy) * wobble_strength) for _, cy in centers]

            # Smooth the vertical offsets to avoid jitter
            if smoothing_window > 1:
                vertical_offsets = np.array(vertical_offsets)
                vertical_offsets = np.convolve(vertical_offsets, np.ones(smoothing_window) / smoothing_window, mode="valid")
                vertical_offsets = np.pad(vertical_offsets, (smoothing_window // 2, (smoothing_window - 1) // 2), "edge")
                vertical_offsets = vertical_offsets.astype(int)

            # Calculate height scaling factors for height compensation
            if height_compensation and heights:
                max_height = max(heights)
                height_scales = [max_height / h if h > 0 else 1.0 for h in heights]

                # Smooth the height scales to avoid jitter
                if smoothing_window > 1:
                    height_scales = np.array(height_scales)
                    height_scales = np.convolve(height_scales, np.ones(smoothing_window) / smoothing_window, mode="valid")
                    height_scales = np.pad(height_scales, (smoothing_window // 2, (smoothing_window - 1) // 2), "edge")
            else:
                height_scales = [1.0] * n_frames
        else:
            # No wobble correction
            vertical_offsets = np.zeros(n_frames, dtype=int)
            height_scales = [1.0] * n_frames

        # Extract strips from each frame with wobble correction and height compensation
        total_width = 0
        for i, frame in enumerate(self.frames):
            # Get the strip width for this frame
            current_strip_width = strip_widths[i]

            # Calculate the start and end columns for the strip
            start_col = max(0, center_col - current_strip_width // 2)
            end_col = min(width, center_col + (current_strip_width + 1) // 2)

            # Extract the strip from the frame
            strip = frame[:, start_col:end_col, :].copy()

            # Apply vertical shift to correct for wobbling if needed
            if wobble_correction:
                # Create an affine transformation matrix for both wobble correction and height scaling
                if height_compensation and height_scales[i] != 1.0:
                    # Scale height and apply vertical shift
                    scale_factor = height_scales[i]
                    # Calculate new height after scaling
                    new_height = int(strip.shape[0] * scale_factor)
                    # Calculate vertical offset to keep the strip centered
                    v_offset = vertical_offsets[i] + (strip.shape[0] - new_height) // 2
                    # Resize the strip to compensate for height changes
                    strip = cv2.resize(strip, (strip.shape[1], new_height), interpolation=cv2.INTER_LINEAR)
                    # Apply vertical shift if needed
                    if v_offset != 0:
                        M = np.float32([[1, 0, 0], [0, 1, v_offset]])
                        strip = cv2.warpAffine(strip, M, (strip.shape[1], strip.shape[0]))
                elif vertical_offsets[i] != 0:
                    # Just apply vertical shift without scaling
                    M = np.float32([[1, 0, 0], [0, 1, vertical_offsets[i]]])
                    strip = cv2.warpAffine(strip, M, (strip.shape[1], strip.shape[0]))

            strips.append(strip)

            # Update the total width of the wide image
            total_width += strip.shape[1]

        # Find the maximum height among all strips
        max_strip_height = max([strip.shape[0] for strip in strips])

        # Create the wide image by stitching the strips together
        wide_image = np.zeros((max_strip_height, total_width, channels), dtype=np.uint8)

        # Place each strip in the wide image
        current_col = 0
        for strip in strips:
            strip_width = strip.shape[1]
            strip_height = strip.shape[0]

            # If the strip height is less than the max height, center it vertically
            if strip_height < max_strip_height:
                y_offset = (max_strip_height - strip_height) // 2
                wide_image[y_offset : y_offset + strip_height, current_col : current_col + strip_width, :] = strip
            else:
                wide_image[:, current_col : current_col + strip_width, :] = strip

            current_col += strip_width

        # Create a mask of the pecan area (non-black pixels)
        wide_image_gray = cv2.cvtColor(wide_image, cv2.COLOR_BGR2GRAY)
        _, pecan_mask = cv2.threshold(wide_image_gray, 5, 255, cv2.THRESH_BINARY)

        # Erode the mask slightly to exclude the boundaries
        kernel = np.ones((5, 5), np.uint8)
        pecan_mask_eroded = cv2.erode(pecan_mask, kernel, iterations=2)

        # Enhance edges if requested
        if edge_enhance:
            # Apply edge detection only to the pecan area
            edges = cv2.Canny(wide_image_gray, 50, 150)

            # Apply the eroded mask to the edges to exclude boundaries
            edges = cv2.bitwise_and(edges, edges, mask=pecan_mask_eroded)

            # Dilate the edges to make them more visible
            kernel = np.ones((3, 3), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)

            # Convert back to BGR for overlay
            edges_bgr = cv2.cvtColor(dilated_edges, cv2.COLOR_GRAY2BGR)

            # Overlay the edges on the original image
            wide_image = cv2.addWeighted(wide_image, 0.7, edges_bgr, 0.3, 0)

        # Detect cracks if threshold is provided
        if crack_threshold is not None:
            # Extract only the pecan area from the grayscale image
            pecan_only_gray = cv2.bitwise_and(wide_image_gray, wide_image_gray, mask=pecan_mask_eroded)

            # Apply adaptive thresholding to detect cracks within the pecan area
            # This helps to account for varying brightness across the pecan surface
            lower, _ = crack_threshold

            # Create a binary mask for potential cracks
            # We're looking for darker areas (cracks) within the pecan
            _, crack_mask = cv2.threshold(pecan_only_gray, lower, 255, cv2.THRESH_BINARY_INV)

            # Remove any detection outside the pecan area
            crack_mask = cv2.bitwise_and(crack_mask, pecan_mask_eroded)

            # Apply morphological operations to clean up the mask
            kernel = np.ones((3, 3), np.uint8)
            crack_mask = cv2.morphologyEx(crack_mask, cv2.MORPH_OPEN, kernel)
            crack_mask = cv2.morphologyEx(crack_mask, cv2.MORPH_CLOSE, kernel)

            # Remove small noise
            min_crack_size = 20  # Minimum size of crack to keep (in pixels)
            crack_mask = cv2.morphologyEx(crack_mask, cv2.MORPH_OPEN, kernel)
            # Remove small connected components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(crack_mask, connectivity=8)
            for i in range(1, num_labels):  # Start from 1 to skip background
                if stats[i, cv2.CC_STAT_AREA] < min_crack_size:
                    crack_mask[labels == i] = 0

            return wide_image, crack_mask

        return wide_image

    def detect_full_rotation(self, similarity_threshold=0.85, min_frames=10, max_frames=None, window_size=5):
        """Detect when the pecan has completed a full rotation.

        This method analyzes the similarity between frames to detect when the pecan has completed
        a full rotation. It returns the frame indices that represent one complete rotation.

        Parameters:
            similarity_threshold (float): Threshold for considering frames as similar. Default is 0.85.
            min_frames (int): Minimum number of frames to consider as a full rotation. Default is 10.
            max_frames (int, optional): Maximum number of frames to consider. If None, use all frames.
            window_size (int): Size of the window for smoothing similarity scores. Default is 5.

        Returns:
            tuple: A tuple containing:
                - start_frame (int): Index of the first frame in the rotation
                - end_frame (int): Index of the last frame in the rotation
                - similarity_scores (list): List of similarity scores between the first frame and each subsequent frame
        """
        if len(self.frames) < min_frames:
            raise ValueError("Not enough frames to detect a full rotation")

        # Limit the number of frames to analyze if max_frames is specified
        n_frames = min(len(self.frames), max_frames) if max_frames else len(self.frames)

        # Convert frames to grayscale for feature extraction
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in self.frames[:n_frames]]

        # Extract features from the first frame
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(gray_frames[0], None)

        if des1 is None or len(des1) == 0:
            raise ValueError("Could not detect features in the first frame")

        # Calculate similarity scores between the first frame and each subsequent frame
        similarity_scores = [1.0]  # First frame is identical to itself
        for i in range(1, n_frames):
            kp2, des2 = orb.detectAndCompute(gray_frames[i], None)

            if des2 is None or len(des2) == 0:
                similarity_scores.append(0.0)
                continue

            # Use a feature matcher to find matches between the two sets of descriptors
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)

            # Calculate a similarity score based on the number and quality of matches
            if len(matches) > 0:
                # Sort matches by distance (lower is better)
                matches = sorted(matches, key=lambda x: x.distance)
                # Use the top 75% of matches to calculate the score
                good_matches = matches[: int(len(matches) * 0.75)]
                similarity = len(good_matches) / max(len(kp1), len(kp2))
                similarity_scores.append(similarity)
            else:
                similarity_scores.append(0.0)

        # Smooth the similarity scores to reduce noise
        if window_size > 1:
            smoothed_scores = np.convolve(similarity_scores, np.ones(window_size) / window_size, mode="valid")
            # Pad the beginning to maintain the same length
            smoothed_scores = np.pad(smoothed_scores, (window_size // 2, (window_size - 1) // 2), "edge")
        else:
            smoothed_scores = similarity_scores

        # Find the first peak in similarity after the minimum number of frames
        start_frame = 0
        for i in range(min_frames, n_frames):
            if smoothed_scores[i] >= similarity_threshold:
                # Check if this is a local maximum
                if i + 1 < n_frames and smoothed_scores[i] >= smoothed_scores[i + 1]:
                    end_frame = i
                    return start_frame, end_frame, similarity_scores

        # If no full rotation is detected, return the entire range
        return 0, n_frames - 1, similarity_scores

    def trim_to_one_rotation(self, similarity_threshold=0.20, min_frames=70, window_size=15):
        """Trim the video to contain only one full rotation of the pecan.

        This method detects when the pecan has completed a full rotation and trims the video
        to contain only that rotation.

        Parameters:
            similarity_threshold (float): Threshold for considering frames as similar. Default is 0.20.
            min_frames (int): Minimum number of frames to consider as a full rotation. Default is 70.
            window_size (int): Size of the window for smoothing similarity scores. Default is 15.

        Returns:
            PecanVideo: A new PecanVideo object containing only one full rotation
        """
        # Use the provided parameters for rotation detection
        start_frame, end_frame, _ = self.detect_full_rotation(similarity_threshold=similarity_threshold, min_frames=min_frames, window_size=window_size)

        # Create a new PecanVideo with the trimmed frames
        trimmed_video = PecanVideo(self.frames[start_frame : end_frame + 1])

        # Copy over the HSV thresholds and other properties
        trimmed_video.pecan_lower_limit = self.pecan_lower_limit
        trimmed_video.pecan_upper_limit = self.pecan_upper_limit

        print(f"Trimmed video from {len(self.frames)} frames to {len(trimmed_video.frames)} frames")
        print(f"Rotation detected from frame {start_frame} to frame {end_frame}")

        return trimmed_video

    def detect_cracks(self, masked_pecan=None, color_space="hsv"):
        """Detect cracks on the pecan surface using color thresholds.

        This method uses the crack color thresholds to detect cracks on the pecan surface.
        It works best on a masked pecan image where only the pecan is visible.

        Parameters:
            masked_pecan (PecanVideo, optional): A PecanVideo instance containing only the pecan
                with background removed. If None, the method will use the current video frames.
            color_space (str): Color space to use for thresholding. Options are "hsv" or "lab".
                Default is "hsv".

        Returns:
            PecanVideo: A new PecanVideo instance containing the crack masks
        """
        if color_space not in ["hsv", "lab"]:
            raise ValueError("color_space must be one of 'hsv' or 'lab'")

        # Use provided masked pecan or current frames
        frames_to_process = masked_pecan.frames if masked_pecan is not None else self.frames

        # Convert frames to the specified color space
        if color_space == "hsv":
            converted_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) for frame in frames_to_process]
        else:  # LAB
            converted_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2LAB) for frame in frames_to_process]

        # Create crack masks using the crack thresholds from the dictionary
        crack_masks = []
        for converted_frame in tqdm(converted_frames, desc=f"Detecting cracks using {color_space.upper()}"):
            # Create mask for cracks
            crack_mask = cv2.inRange(converted_frame, self.thresholds[color_space]["crack"]["lower"], self.thresholds[color_space]["crack"]["upper"])

            # Apply morphological operations to clean up the mask
            kernel = np.ones((3, 3), np.uint8)
            crack_mask = cv2.morphologyEx(crack_mask, cv2.MORPH_OPEN, kernel)
            crack_mask = cv2.morphologyEx(crack_mask, cv2.MORPH_CLOSE, kernel)

            crack_masks.append(crack_mask)

        # Convert to 3-channel for compatibility with PecanVideo
        crack_masks_3ch = [cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) for mask in crack_masks]

        # Create a new PecanVideo instance with the crack masks
        crack_video = PecanVideo(np.array(crack_masks_3ch))

        # Copy thresholds dictionary and ellipses to the new instance
        crack_video.thresholds = self.thresholds.copy()

        # For backward compatibility
        crack_video.pecan_lower_limit = self.pecan_lower_limit.copy()
        crack_video.pecan_upper_limit = self.pecan_upper_limit.copy()
        crack_video.background_lower_limit = self.background_lower_limit.copy()
        crack_video.background_upper_limit = self.background_upper_limit.copy()
        crack_video.crack_lower_limit = self.crack_lower_limit.copy()
        crack_video.crack_upper_limit = self.crack_upper_limit.copy()
        crack_video.lab_pecan_lower_limit = self.lab_pecan_lower_limit.copy()
        crack_video.lab_pecan_upper_limit = self.lab_pecan_upper_limit.copy()
        crack_video.lab_background_lower_limit = self.lab_background_lower_limit.copy()
        crack_video.lab_background_upper_limit = self.lab_background_upper_limit.copy()
        crack_video.lab_crack_lower_limit = self.lab_crack_lower_limit.copy()
        crack_video.lab_crack_upper_limit = self.lab_crack_upper_limit.copy()
        crack_video.ellipses = self.ellipses

        print(f"Crack detection completed using {color_space.upper()} color space. {len(crack_masks)} masks created.")

        return crack_video

    def detect_pecan_with_ellipse_detection(self, min_area=500, max_area=None, aspect_ratio_range=(1.2, 3.0), ellipse_fit_threshold=0.85, edge_threshold1=50, edge_threshold2=150):
        """Detect pecans in video frames using ellipse detection.

        This method uses edge detection and contour analysis to find elliptical shapes
        that are likely to be pecans. It's more robust than simple color thresholding
        as it looks specifically for oval-shaped objects.

        Parameters:
            min_area (int): Minimum contour area to consider. Default is 500 pixels.
            max_area (int, optional): Maximum contour area to consider. If None, no upper limit.
            aspect_ratio_range (tuple): Range of acceptable aspect ratios (width/height) for pecans.
                Default is (1.2, 3.0).
            ellipse_fit_threshold (float): Threshold for how well a contour must fit an ellipse.
                Range is 0-1, where 1 is perfect fit. Default is 0.85.
            edge_threshold1 (int): First threshold for Canny edge detection. Default is 50.
            edge_threshold2 (int): Second threshold for Canny edge detection. Default is 150.

        Returns:
            tuple: A tuple containing:
                - PecanVideo: A new PecanVideo instance containing the original frames
                - PecanVideo: A new PecanVideo instance containing the pecan masks
        """
        # Check if frames exist
        if len(self.frames) == 0:
            raise ValueError("No frames available for pecan detection")

        # Initialize list to store pecan masks
        pecan_masks = []

        # Helper function to evaluate how well a contour fits an ellipse
        def evaluate_ellipse_fit(contour, ellipse):
            """Calculate how well a contour fits an ellipse."""
            # Create a blank mask
            mask = np.zeros((self.frames[0].shape[0], self.frames[0].shape[1]), dtype=np.uint8)

            # Draw the contour
            cv2.drawContours(mask, [contour], 0, 255, -1)
            contour_area = cv2.countNonZero(mask)

            # Draw the ellipse
            mask_ellipse = np.zeros_like(mask)
            cv2.ellipse(mask_ellipse, ellipse, 255, -1)
            ellipse_area = cv2.countNonZero(mask_ellipse)

            # Calculate intersection
            intersection = cv2.bitwise_and(mask, mask_ellipse)
            intersection_area = cv2.countNonZero(intersection)

            # Calculate union
            union_area = contour_area + ellipse_area - intersection_area

            # Calculate IoU (Intersection over Union)
            if union_area > 0:
                iou = intersection_area / union_area
                return iou
            return 0

        # Process each frame
        for frame_idx, frame in enumerate(tqdm(self.frames, desc="Detecting pecans with ellipse detection")):
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Edge detection
            edges = cv2.Canny(blurred, edge_threshold1, edge_threshold2)

            # Dilate edges to connect broken contours
            kernel = np.ones((3, 3), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)

            # Find contours
            contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours by area and shape
            pecan_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)

                # Skip if area is too small or too large
                if area < min_area:
                    continue
                if max_area is not None and area > max_area:
                    continue

                # Check if contour is elliptical
                if len(contour) >= 5:  # Need at least 5 points to fit an ellipse
                    try:
                        ellipse = cv2.fitEllipse(contour)
                        _, axes, _ = ellipse

                        # Calculate aspect ratio
                        aspect_ratio = max(axes) / min(axes)

                        # Check if aspect ratio is within acceptable range
                        if aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
                            # Calculate how well the contour fits an ellipse
                            ellipse_fit = evaluate_ellipse_fit(contour, ellipse)

                            if ellipse_fit >= ellipse_fit_threshold:
                                pecan_contours.append(contour)
                    except:
                        # Skip contours that can't be fitted to an ellipse
                        continue

            # Create mask from the detected pecan contours
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, pecan_contours, -1, 255, -1)

            # If no pecan contours were found, try to use the largest contour
            if len(pecan_contours) == 0 and len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) >= min_area:
                    cv2.drawContours(mask, [largest_contour], -1, 255, -1)

            # Apply morphological operations to clean up the mask
            kernel_close = np.ones((7, 7), np.uint8)
            kernel_open = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

            # Add to the list of pecan masks
            pecan_masks.append(mask)

        # Convert to 3-channel for compatibility with PecanVideo
        pecan_masks_3ch = [cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) for mask in pecan_masks]

        # Create a new PecanVideo instance with the pecan masks
        mask_video = PecanVideo(np.array(pecan_masks_3ch))

        # Copy thresholds and ellipses to the new instance
        mask_video.thresholds = self.thresholds.copy()
        mask_video.ellipses = self.ellipses

        print(f"Ellipse-based pecan detection completed. {len(pecan_masks)} masks created.")

        return self, mask_video

    def analyze_cracks(self, crack_mask):
        """Analyze the properties of cracks in a binary mask.

        This method analyzes the properties of cracks in a binary mask, including
        length, width, and area.

        Parameters:
            crack_mask (numpy.ndarray): Binary mask of detected cracks

        Returns:
            dict: A dictionary containing crack properties:
                - 'area': Total area of cracks in pixels
                - 'lengths': List of crack lengths in pixels
                - 'widths': List of crack widths in pixels
                - 'contours': List of contours representing cracks
        """
        # Find contours in the crack mask
        contours, _ = cv2.findContours(crack_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize lists to store crack properties
        areas = []
        lengths = []
        widths = []

        # Analyze each contour
        for contour in contours:
            # Calculate area
            area = cv2.contourArea(contour)
            areas.append(area)

            # Fit a rotated rectangle to the contour
            if len(contour) >= 5:  # Need at least 5 points to fit an ellipse
                rect = cv2.minAreaRect(contour)
                _, (rect_width, rect_height), _ = rect

                # The length is the maximum of width and height
                length = max(rect_width, rect_height)
                # The width is the minimum of width and height
                width = min(rect_width, rect_height)

                lengths.append(length)
                widths.append(width)

        # Calculate total area
        total_area = sum(areas)

        # Return the crack properties
        return {"area": total_area, "lengths": lengths, "widths": widths, "contours": contours}
