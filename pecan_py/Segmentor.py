import cv2
import numpy as np
from pecan_py.PecanVideo import PecanVideo


class Segmentor:
    """Class for creating instance segmentation masks from PecanVideo objects.

    This class provides methods to generate segmentation masks for different targets
    (pecan, kernel, damaged_kernel, crack, background) based on the color thresholds
    defined in a PecanVideo instance.
    """

    def __init__(self):
        """Initialize the Segmentor."""
        # Define unique colors for each target for visualization
        self.target_colors = {"pecan": (0, 0, 255), "kernel": (0, 255, 0), "damaged_kernel": (0, 255, 255), "crack": (255, 0, 0), "background": (255, 255, 0)}  # Red in BGR  # Green in BGR  # Yellow in BGR  # Blue in BGR  # Cyan in BGR

        # Define unique label values for each target for segmentation mask
        self.target_labels = {"pecan": 1, "kernel": 2, "damaged_kernel": 3, "crack": 4, "background": 5}

    def create_segmentation_masks(self, pecan_video, targets=None, color_space="hsv", respect_preferred_color_spaces=True):
        """Create segmentation masks for the specified targets.

        When masks overlap, pixels are assigned based on the following priority order (highest to lowest):
        1. damaged_kernel
        2. kernel
        3. crack
        4. pecan
        5. background

        Parameters:
            pecan_video (PecanVideo): The PecanVideo instance to process.
            targets (list, optional): List of target names to include in the segmentation.
                                     Options: "pecan", "kernel", "damaged_kernel", "crack", "background".
                                     If None, all targets except "background" are included. Default is None.
            color_space (str, optional): Default color space to use for thresholding.
                                        Options: "hsv" or "lab". Default is "hsv".
            respect_preferred_color_spaces (bool, optional): Whether to use the preferred color space for each target
                                                           as specified in pecan_video.preferred_color_spaces.
                                                           If True, the color_space parameter is only used for targets
                                                           that don't have a preferred color space. Default is True.

        Returns:
            tuple: A tuple containing:
                - PecanVideo: A new PecanVideo instance with segmentation masks (each pixel value represents a class)
                - PecanVideo: A new PecanVideo instance with colored visualization masks (for display)
        """
        if not isinstance(pecan_video, PecanVideo):
            raise ValueError("pecan_video must be a PecanVideo instance")

        if color_space not in ["hsv", "lab"]:
            raise ValueError("color_space must be one of 'hsv' or 'lab'")

        # If targets is None, include all targets except background by default
        if targets is None:
            targets = ["pecan", "kernel", "damaged_kernel", "crack"]

        # Validate targets
        for target in targets:
            if target not in ["pecan", "kernel", "damaged_kernel", "crack", "background"]:
                raise ValueError(f"Invalid target: {target}. Must be one of 'pecan', 'kernel', 'damaged_kernel', 'crack', 'background'")

        # Convert frames to both HSV and LAB color spaces
        # We need both because we might use different color spaces for different targets
        hsv_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) for frame in pecan_video.frames]
        lab_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2LAB) for frame in pecan_video.frames]

        # Dictionary to store converted frames for each color space
        converted_frames_dict = {"hsv": hsv_frames, "lab": lab_frames}

        # Create empty segmentation masks and visualization masks
        segmentation_masks = []
        visualization_masks = []

        # Get frame dimensions from the first frame
        frame_height, frame_width = pecan_video.frames[0].shape[:2]

        # Process each frame index
        for frame_idx in range(len(pecan_video.frames)):
            # Create empty segmentation mask (single channel)
            segmentation_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

            # Create empty visualization mask (3 channels for BGR color)
            visualization_mask = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

            # Define priority order (highest to lowest)
            priority_order = ["damaged_kernel", "kernel", "crack", "pecan", "background"]

            # Filter and sort targets based on priority
            prioritized_targets = [target for target in priority_order if target in targets]

            # Process each target in reverse priority order (lowest to highest)
            # This way, higher priority targets will overwrite lower priority ones
            for target in reversed(prioritized_targets):
                # Determine which color space to use for this target
                target_color_space = color_space  # Default

                # If respecting preferred color spaces and the target has a preferred color space
                if respect_preferred_color_spaces and target in pecan_video.preferred_color_spaces:
                    target_color_space = pecan_video.preferred_color_spaces[target]

                # Get the appropriate converted frame for this target's color space
                converted_frame = converted_frames_dict[target_color_space][frame_idx]

                # Get thresholds for the target
                lower = pecan_video.thresholds[target_color_space][target]["lower"]
                upper = pecan_video.thresholds[target_color_space][target]["upper"]

                # Create binary mask for the target
                target_mask = cv2.inRange(converted_frame, lower, upper)

                # Apply morphological operations to clean up the mask
                kernel = np.ones((1, 1), np.uint8)
                target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_CLOSE, kernel)
                target_mask = cv2.morphologyEx(target_mask, cv2.MORPH_OPEN, kernel)

                # Add the target to the segmentation mask with its unique label
                segmentation_mask[target_mask > 0] = self.target_labels[target]

                # Add the target to the visualization mask with its unique color
                visualization_mask[target_mask > 0] = self.target_colors[target]

            # Add the masks to the lists
            segmentation_masks.append(segmentation_mask)
            visualization_masks.append(visualization_mask)

        # Convert segmentation masks to 3-channel for compatibility with PecanVideo
        segmentation_masks_3ch = [cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) for mask in segmentation_masks]

        # Create new PecanVideo instances
        segmentation_video = PecanVideo(np.array(segmentation_masks_3ch))
        visualization_video = PecanVideo(np.array(visualization_masks))

        # Copy thresholds to the new instances
        segmentation_video.thresholds = pecan_video.thresholds.copy()
        visualization_video.thresholds = pecan_video.thresholds.copy()

        return segmentation_video, visualization_video

    def overlay_segmentation_on_original(self, pecan_video, segmentation_video, opacity=0.5):
        """Overlay segmentation visualization on the original video.

        Parameters:
            pecan_video (PecanVideo): The original PecanVideo instance.
            segmentation_video (PecanVideo): The segmentation visualization video.
            opacity (float, optional): Opacity of the overlay (0.0 to 1.0). Default is 0.5.

        Returns:
            PecanVideo: A new PecanVideo instance with the overlay.
        """
        if not isinstance(pecan_video, PecanVideo) or not isinstance(segmentation_video, PecanVideo):
            raise ValueError("Both pecan_video and segmentation_video must be PecanVideo instances")

        if len(pecan_video.frames) != len(segmentation_video.frames):
            raise ValueError("pecan_video and segmentation_video must have the same number of frames")

        if opacity < 0.0 or opacity > 1.0:
            raise ValueError("opacity must be between 0.0 and 1.0")

        # Create empty list for overlay frames
        overlay_frames = []

        # Process each frame
        for original, segmentation in zip(pecan_video.frames, segmentation_video.frames):
            # Blend the original frame and the segmentation mask
            overlay = cv2.addWeighted(original, 1.0, segmentation, opacity, 0)
            overlay_frames.append(overlay)

        # Create a new PecanVideo instance with the overlay frames
        overlay_video = PecanVideo(np.array(overlay_frames))

        # Copy thresholds to the new instance
        overlay_video.thresholds = pecan_video.thresholds.copy()

        return overlay_video
