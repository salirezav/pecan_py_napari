import cv2
import numpy as np
import os
from tqdm import tqdm
import pathlib


class BaseVideo:
    """Base class for video handling.

    This class provides basic functionality for loading, displaying, and manipulating videos.
    It can be initialized with either a path to a video file or a numpy array of frames.
    Supports slicing operations (e.g., video[:5], video[::2]) to create new videos with subsets of frames.
    """

    def __init__(self, video_source, max_frames=None, frame_step=1):
        """Initialize a BaseVideo object.

        Parameters:
            video_source: Either a path to a video file (str) or a numpy array of frames (ndarray)
            max_frames: Maximum number of frames to load from the video. If None, all frames are loaded.
                        This is useful for quickly loading a subset of frames for testing or preview.
            frame_step: Step size for frame loading. If frame_step=1 (default), every frame is loaded.
                        If frame_step=5, every 5th frame is loaded (frames 0, 5, 10, ...).
        """
        if isinstance(video_source, str):
            # It's a path to a video file
            self.video_path = video_source
            self.frames = self.load_video_to_ndarray(video_source, max_frames=max_frames, frame_step=frame_step)
            self.name = os.path.basename(video_source)
        elif isinstance(video_source, np.ndarray):
            # It's already a numpy array of frames
            self.video_path = None

            # Apply frame_step first to get frames at regular intervals
            if frame_step > 1:
                frames_subset = video_source[::frame_step]
            else:
                frames_subset = video_source

            # Then apply max_frames if specified
            if max_frames is not None and len(frames_subset) > max_frames:
                self.frames = frames_subset[:max_frames]
            else:
                self.frames = frames_subset

            self.name = "unnamed_video"
        else:
            raise ValueError("video_source must be either a path (str) or a numpy array of frames (ndarray)")

    def get_video_name(self):
        """Get the name of the video.

        Returns:
            str: The name of the video
        """
        return self.name

    def __len__(self):
        """Return the number of frames in the video.

        This method allows using the len() function on BaseVideo instances.

        Returns:
            int: The number of frames in the video
        """
        return len(self.frames)

    def __repr__(self):
        """Return a string representation of the BaseVideo object.

        Returns:
            str: A string representation of the BaseVideo object
        """
        if len(self.frames) > 0:
            frame_shape = self.frames[0].shape
            return f"{self.__class__.__name__}(name='{self.name}', frames={len(self.frames)}, shape={frame_shape})"
        else:
            return f"{self.__class__.__name__}(name='{self.name}', frames=0)"

    @staticmethod
    def load_video_to_ndarray(video_path, max_frames=None, frame_step=1):
        """Load a video file into a numpy array.

        Parameters:
            video_path (str): Path to the video file
            max_frames (int, optional): Maximum number of frames to load. If None, all frames are loaded.
            frame_step (int, optional): Step size for frame loading. If frame_step=1 (default), every frame is loaded.
                                       If frame_step=5, every 5th frame is loaded (frames 0, 5, 10, ...).

        Returns:
            ndarray: A numpy array of frames with shape (n_frames, height, width, channels)
        """
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        # List to store individual frames
        frames = []
        frames_read = 0

        # Get total frame count from video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate how many frames we'll actually load
        if max_frames is None:
            frames_to_load = (total_frames + frame_step - 1) // frame_step  # Ceiling division
        else:
            frames_to_load = min(max_frames, (total_frames + frame_step - 1) // frame_step)

        # Add a progress bar if we're loading all frames or a large number
        use_tqdm = frames_to_load > 100

        # Create iterator with or without progress bar
        if use_tqdm:
            desc = f"Loading video frames (step={frame_step})"
            if max_frames is not None:
                desc = f"Loading up to {max_frames} video frames (step={frame_step})"
            iterator = tqdm(range(total_frames), desc=desc, total=frames_to_load)
        else:
            # Simple range without progress bar
            iterator = range(total_frames)

        for frame_idx in iterator:
            # Skip frames according to frame_step
            if frame_idx % frame_step != 0:
                # Skip this frame but still need to read it to advance the video
                ret = cap.grab()
                if not ret:
                    break
                continue

            # Read a frame from the video - use cap.read() for consistency
            ret, frame = cap.read()
            if not ret:
                break

            # Append the frame to the list
            frames.append(frame)
            frames_read += 1

            # Update progress bar to show actual progress
            if use_tqdm and frames_read % 10 == 0:
                iterator.n = frames_read
                iterator.refresh()

            # Stop if we've reached the maximum number of frames
            if max_frames is not None and frames_read >= max_frames:
                break

        # Release the video capture object
        cap.release()

        if use_tqdm:
            iterator.close()

        if len(frames) == 0:
            print(f"Warning: No frames were loaded from {video_path}")
            return np.array([])

        # Return frames as a numpy array
        return np.array(frames)

    def play(self, overlay_videos=None, alphas=None):
        """Display the video with playback controls.

        Parameters:
            overlay_videos (optional): Videos to overlay on the main video. Can be:
                - A single Video object
                - A list of Video objects
                - A single numpy array of frames
                - A list of numpy arrays of frames
            alphas (list, optional): List of alpha values for each video (including the main video).
                If not provided, defaults to 1.0 for the main video and 0.5 for all overlay videos.

        Controls:
            Space: Play/Pause
            A: Previous frame
            D: Next frame
            Q: Quit

        Returns:
            Video: Returns self for method chaining
        """

        def ensure_3_channels(video_array):
            if len(video_array.shape) == 3:
                return np.repeat(video_array[:, :, :, np.newaxis], 3, axis=3)
            return video_array

        def get_frames(video):
            """Extract frames from various video formats."""
            if isinstance(video, BaseVideo):
                return video.frames
            elif isinstance(video, np.ndarray):
                return video
            else:
                raise ValueError(f"Unsupported video type: {type(video)}")

        # Process overlay_videos to handle different input types
        if overlay_videos is not None:
            # Convert single video to list
            if not isinstance(overlay_videos, list):
                overlay_videos = [overlay_videos]

            # Extract frames from each video
            overlay_frames = []
            for video in overlay_videos:
                overlay_frames.append(get_frames(video))

            # Set default alphas if not provided
            if alphas is None:
                alphas = [1.0] + [0.5] * len(overlay_frames)
            elif len(alphas) != len(overlay_frames) + 1:  # +1 for the main video
                raise ValueError(f"Expected {len(overlay_frames) + 1} alpha values (including main video), got {len(alphas)}")

            # Convert all video arrays to 4D with 3 color channels if needed
            overlay_frames = [ensure_3_channels(frames) for frames in overlay_frames]

            # Check if all videos have the same shape
            for i in range(len(overlay_frames)):
                if overlay_frames[i].shape[1:3] != self.frames.shape[1:3]:
                    raise ValueError(f"Overlay video {i} has shape {overlay_frames[i].shape[1:3]}, but main video has shape {self.frames.shape[1:3]}")

        # Create a resizable window
        cv2.namedWindow("Video", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

        # Get the dimensions of the first frame
        if len(self.frames) > 0:
            height, width = self.frames[0].shape[:2]

            # Calculate the scaling factor to fit within 1024x768
            scale_width = min(1.0, 1024.0 / width)
            scale_height = min(1.0, 768.0 / height)
            scale = min(scale_width, scale_height)

            # Set the window size
            new_width = int(width * scale)
            new_height = int(height * scale)
            cv2.resizeWindow("Video", new_width, new_height)

        # Create trackbar for frame seeking
        cv2.createTrackbar("Frame", "Video", 0, len(self.frames) - 1, lambda _: None)
        # Variables to control play/pause and current frame index
        playing = False
        frame_idx = 0
        while True:
            if playing:
                frame_idx = (frame_idx + 1) % len(self.frames)
                cv2.setTrackbarPos("Frame", "Video", frame_idx)
            # Get the current frame position from the trackbar
            frame_idx = cv2.getTrackbarPos("Frame", "Video")
            frame = self.frames[frame_idx].astype(np.float32)
            if overlay_videos is not None:
                combined_frame = frame * alphas[0]
                for i in range(len(overlay_frames)):
                    # Handle case where overlay video has fewer frames than main video
                    if frame_idx < len(overlay_frames[i]):
                        overlay_frame = overlay_frames[i][frame_idx].astype(np.float32)
                        combined_frame += overlay_frame * alphas[i + 1]  # +1 because alphas[0] is for main video
                frame = np.clip(combined_frame, 0, 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
            # Display the current frame
            cv2.imshow("Video", frame)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                playing = not playing
            elif key == ord("d"):  # 'D' key
                frame_idx = min(frame_idx + 1, len(self.frames) - 1)
                cv2.setTrackbarPos("Frame", "Video", frame_idx)
            elif key == ord("a"):  # 'A' key
                frame_idx = max(frame_idx - 1, 0)
                cv2.setTrackbarPos("Frame", "Video", frame_idx)
        cv2.destroyAllWindows()

        # Return self for method chaining
        return self

    def __getitem__(self, key):
        """Enable slicing operations on the video frames.

        This method allows using Python's slice notation to create new video instances
        with subsets of frames. For example:
            - video[:5]     # First 5 frames
            - video[5:10]   # Frames 5-9
            - video[::2]    # Every other frame
            - video[-5:]    # Last 5 frames
            - video[::-1]   # Reversed video

        The returned object will be of the same class as the original video,
        preserving all attributes and methods.

        Parameters:
            key: An integer index or slice object

        Returns:
            Same class as self: A new video instance containing the selected frames

        Raises:
            IndexError: If the index is out of range
        """
        if isinstance(key, slice):
            # Handle slice notation (e.g., video[:5], video[::2])
            sliced_frames = self.frames[key]
            # Create a new instance of the same class as self
            # Pass frame_step=1 and max_frames=None to avoid re-applying these filters
            new_video = self.__class__(sliced_frames, frame_step=1, max_frames=None)

            # Copy attributes from self to new_video
            # Skip frames, video_path, and name which are set by the constructor
            for attr_name, attr_value in self.__dict__.items():
                if attr_name not in ["frames", "video_path", "name"]:
                    try:
                        # Try to copy the attribute (may be a reference or a deep copy)
                        if hasattr(attr_value, "copy") and callable(getattr(attr_value, "copy")):
                            # If the attribute has a copy method, use it
                            setattr(new_video, attr_name, attr_value.copy())
                        else:
                            # Otherwise, just set the reference
                            setattr(new_video, attr_name, attr_value)
                    except (AttributeError, TypeError):
                        # If copying fails, just set the reference
                        setattr(new_video, attr_name, attr_value)

            # Set the name to indicate it's a slice of the original
            new_video.name = f"{self.name}[{key.start if key.start else ''}:{key.stop if key.stop else ''}:{key.step if key.step else ''}]"

            return new_video

        elif isinstance(key, int):
            # Handle single frame access (e.g., video[0])
            # Convert to proper array shape for constructor
            frame = self.frames[key]

            # Create a new instance of the same class as self
            # Pass frame_step=1 and max_frames=None to avoid re-applying these filters
            new_video = self.__class__(np.array([frame]), frame_step=1, max_frames=None)

            # Copy attributes from self to new_video
            # Skip frames, video_path, and name which are set by the constructor
            for attr_name, attr_value in self.__dict__.items():
                if attr_name not in ["frames", "video_path", "name"]:
                    try:
                        # Try to copy the attribute (may be a reference or a deep copy)
                        if hasattr(attr_value, "copy") and callable(getattr(attr_value, "copy")):
                            # If the attribute has a copy method, use it
                            setattr(new_video, attr_name, attr_value.copy())
                        else:
                            # Otherwise, just set the reference
                            setattr(new_video, attr_name, attr_value)
                    except (AttributeError, TypeError):
                        # If copying fails, just set the reference
                        setattr(new_video, attr_name, attr_value)

            # Set the name to indicate it's a single frame from the original
            new_video.name = f"{self.name}[{key}]"

            return new_video
        else:
            raise TypeError(f"Invalid index type: {type(key)}. Use an integer or slice.")

    def save(self, output_path=None, fps=30, codec="mp4v"):
        """Save the video to a file.

        Parameters:
            output_path (str, optional): Path to save the video. If None and the video was loaded from a file,
                                        it will use the original path with '-cropped' appended before the extension.
                                        If None and the video was not loaded from a file, it will raise an error.
            fps (int, optional): Frames per second for the output video. Default is 30.
            codec (str, optional): FourCC codec code. Default is 'mp4v' for MP4 format.
                                  Other common options: 'XVID' for AVI, 'H264' for MP4 with H.264 codec.

        Returns:
            str: The path where the video was saved

        Raises:
            ValueError: If output_path is None and the video was not loaded from a file
        """
        if len(self.frames) == 0:
            raise ValueError("Cannot save an empty video")

        # Determine output path if not provided
        if output_path is None:
            if self.video_path is None:
                raise ValueError("output_path must be provided for videos not loaded from a file")

            # Get the original path and add '-cropped' before the extension
            path_obj = pathlib.Path(self.video_path)
            output_path = str(path_obj.with_name(f"{path_obj.stem}-cropped{path_obj.suffix}"))

        # Get frame dimensions
        height, width = self.frames[0].shape[:2]

        # Create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Write frames to the video
        for frame in tqdm(self.frames, desc=f"Saving video to {output_path}"):
            # Ensure the frame is in the correct format (uint8)
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)

            # Make sure the frame has the right shape and channels
            if len(frame.shape) == 2:  # If grayscale, convert to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:  # If RGBA, convert to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            # Write the frame
            out.write(frame)

        # Release the VideoWriter
        out.release()

        print(f"Video saved to {output_path}")
        return output_path
