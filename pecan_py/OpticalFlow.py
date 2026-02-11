import cv2
import numpy as np
from tqdm import tqdm
from pecan_py.BaseVideo import BaseVideo
from pecan_py.PecanVideo import PecanVideo


class OpticalFlow:
    """Class for calculating and visualizing optical flow in videos.
    
    This class provides methods to calculate optical flow between consecutive frames
    in a video and visualize the results.
    """
    
    def __init__(self):
        """Initialize the OpticalFlow object."""
        pass
    
    def calc_opt_flow(self, pecan_video, pyr_scale=0.5, levels=3, winsize=15, 
                      iterations=3, poly_n=5, poly_sigma=1.2, flags=0, 
                      visualization_scale=4, visualization_color=True):
        """Calculate optical flow for a PecanVideo instance.
        
        This method computes the dense optical flow between consecutive frames
        using the Farneback algorithm and creates a visualization of the flow.
        
        Parameters:
            pecan_video (PecanVideo): The PecanVideo instance to process
            pyr_scale (float): Pyramid scale for the Farneback algorithm (default: 0.5)
            levels (int): Number of pyramid levels (default: 3)
            winsize (int): Window size (default: 15)
            iterations (int): Number of iterations (default: 3)
            poly_n (int): Size of pixel neighborhood (default: 5)
            poly_sigma (float): Standard deviation for Gaussian (default: 1.2)
            flags (int): Flags for the Farneback algorithm (default: 0)
            visualization_scale (int): Scale factor for flow visualization (default: 4)
            visualization_color (bool): Whether to use color for visualization (default: True)
            
        Returns:
            tuple: A tuple containing:
                - PecanVideo: A new PecanVideo instance containing the flow visualization
                - list: A list of optical flow vectors for each pair of consecutive frames
        """
        if not isinstance(pecan_video, PecanVideo) and not isinstance(pecan_video, BaseVideo):
            raise TypeError("pecan_video must be a PecanVideo or BaseVideo instance")
        
        if len(pecan_video.frames) < 2:
            raise ValueError("At least 2 frames are required to compute optical flow")
        
        # Convert frames to grayscale
        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in pecan_video.frames]
        
        # Initialize list to store flow vectors and visualizations
        flow_vectors = []
        flow_visualizations = []
        
        # Compute optical flow between consecutive frames
        for i in tqdm(range(len(gray_frames) - 1), desc="Computing optical flow"):
            # Calculate optical flow using Farneback algorithm
            flow = cv2.calcOpticalFlowFarneback(
                gray_frames[i], 
                gray_frames[i + 1], 
                None, 
                pyr_scale, 
                levels, 
                winsize, 
                iterations, 
                poly_n, 
                poly_sigma, 
                flags
            )
            
            # Store the flow vectors
            flow_vectors.append(flow)
            
            # Create visualization of the flow
            flow_vis = self._create_flow_visualization(
                gray_frames[i], 
                flow, 
                scale=visualization_scale, 
                use_color=visualization_color
            )
            
            # Store the visualization
            flow_visualizations.append(flow_vis)
        
        # Add a duplicate of the last visualization to match the number of frames
        # This ensures the visualization video has the same number of frames as the original
        flow_visualizations.append(flow_visualizations[-1])
        
        # Create a new PecanVideo instance with the flow visualizations
        flow_visualization_video = PecanVideo(np.array(flow_visualizations))
        
        print(f"Optical flow calculation completed. {len(flow_vectors)} flow vectors created.")
        
        return flow_visualization_video, flow_vectors
    
    def _create_flow_visualization(self, frame, flow, scale=4, use_color=True):
        """Create a visualization of optical flow.
        
        Parameters:
            frame (ndarray): The original frame
            flow (ndarray): The optical flow vectors
            scale (int): Scale factor for visualization (default: 4)
            use_color (bool): Whether to use color for visualization (default: True)
            
        Returns:
            ndarray: The visualization image
        """
        # Get the flow magnitude and angle
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        if use_color:
            # Create an HSV image with the flow direction mapped to hue and magnitude to value
            hsv = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
            hsv[..., 0] = angle * 180 / np.pi / 2  # Map angle to hue (0-180)
            hsv[..., 1] = 255  # Full saturation
            hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Map magnitude to value
            
            # Convert HSV to BGR for visualization
            flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            # Create a grayscale visualization based on magnitude
            flow_vis = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            flow_vis = flow_vis.astype(np.uint8)
            flow_vis = cv2.cvtColor(flow_vis, cv2.COLOR_GRAY2BGR)
        
        # Optionally draw flow vectors on the image
        if scale > 0:
            # Create a grid of points to draw flow vectors
            h, w = frame.shape
            step = 16  # Draw a vector every 16 pixels
            y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
            fx, fy = flow[y, x].T
            
            # Create a mask for valid flow vectors
            lines = np.vstack([x, y, x + fx * scale, y + fy * scale]).T.reshape(-1, 2, 2)
            lines = lines.astype(int)
            
            # Draw the flow vectors
            cv2.polylines(flow_vis, lines, False, (0, 255, 0), 1)
            
            # Draw circles at the start points
            for (x1, y1), (x2, y2) in lines:
                cv2.circle(flow_vis, (x1, y1), 1, (0, 255, 0), -1)
        
        return flow_vis
