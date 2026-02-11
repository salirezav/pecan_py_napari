import cv2
import numpy as np
from tqdm import tqdm
from pecan_py.BaseVideo import BaseVideo
from pecan_py.PecanVideo import PecanVideo


class VideoManipulator:
    """Class for performing manipulations on PecanVideo or BaseVideo instances.

    This class provides methods for manipulating videos, such as cropping, rotating,
    trimming, etc. It works with both PecanVideo and BaseVideo instances.
    """

    @staticmethod
    def crop_to_mask(video, mask_video):
        """Crop a video to focus on the area defined by a mask.

        This method takes a video and a mask video, calculates the bounding box
        of the mask, and crops the original video to that bounding box.

        Parameters:
            video (PecanVideo or BaseVideo): The video to crop
            mask_video (BaseVideo): A video containing masks that define the area to crop to

        Returns:
            PecanVideo or BaseVideo: A new video instance containing the cropped frames
        """
        # Validate inputs
        if not isinstance(video, (PecanVideo, BaseVideo)):
            raise ValueError("video must be a PecanVideo or BaseVideo instance")
        if not isinstance(mask_video, (PecanVideo, BaseVideo)):
            raise ValueError("mask_video must be a PecanVideo or BaseVideo instance")

        # Convert mask frames to grayscale if they are in BGR format
        masks = []
        for mask in tqdm(mask_video.frames, desc="Processing masks"):
            if len(mask.shape) == 3 and mask.shape[2] == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            masks.append(mask)

        # Function to get bounding box from a mask
        def get_bounding_box(mask):
            # Ensure binary mask
            _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            
            # Find contours in the binary mask
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) == 0:
                return 0, 0, mask.shape[1], mask.shape[0]
            
            # Find the largest contour by area
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
        for frame in tqdm(video.frames, desc="Cropping frames"):
            cropped_frames.append(frame[min_y:max_y, min_x:max_x])

        # Create a new video instance with the cropped frames
        if isinstance(video, PecanVideo):
            cropped_video = PecanVideo(np.array(cropped_frames))
            
            # Copy properties from the original PecanVideo
            if hasattr(video, 'thresholds'):
                cropped_video.thresholds = video.thresholds.copy()
            if hasattr(video, 'ellipses'):
                cropped_video.ellipses = video.ellipses
                
            # For backward compatibility with older code
            if hasattr(video, 'pecan_lower_limit'):
                cropped_video.pecan_lower_limit = video.pecan_lower_limit.copy() if video.pecan_lower_limit is not None else None
            if hasattr(video, 'pecan_upper_limit'):
                cropped_video.pecan_upper_limit = video.pecan_upper_limit.copy() if video.pecan_upper_limit is not None else None
            if hasattr(video, 'background_lower_limit'):
                cropped_video.background_lower_limit = video.background_lower_limit.copy() if video.background_lower_limit is not None else None
            if hasattr(video, 'background_upper_limit'):
                cropped_video.background_upper_limit = video.background_upper_limit.copy() if video.background_upper_limit is not None else None
        else:
            cropped_video = BaseVideo(np.array(cropped_frames))

        # Print information about the cropping
        print(f"Cropped video from {video.frames.shape[1:3]} to {cropped_video.frames.shape[1:3]} using mask")

        return cropped_video
