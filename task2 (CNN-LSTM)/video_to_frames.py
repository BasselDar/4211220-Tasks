import os
import argparse
import cv2
from tqdm import tqdm

def extract_frames(video_path, output_dir, fps=1, max_frames=20):
    """
    Extract frames from a video file.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        fps: Frames per second to extract (default: 1 frame per second)
        max_frames: Maximum number of frames to extract (default: 20)
    
    Returns:
        List of paths to the extracted frames
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
    
    # Get video properties
    video_fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / video_fps
    
    print(f"Video FPS: {video_fps}")
    print(f"Frame Count: {frame_count}")
    print(f"Duration: {duration:.2f} seconds")
    
    # Calculate frame interval
    interval = int(video_fps / fps)
    
    # Extract frames
    frame_paths = []
    count = 0
    
    # Use tqdm for progress bar
    with tqdm(total=min(max_frames, int(frame_count / interval))) as pbar:
        while video.isOpened():
            ret, frame = video.read()
            
            if not ret:
                break
            
            if count % interval == 0:
                # Save the frame
                frame_path = os.path.join(output_dir, f"frame_{count:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                pbar.update(1)
            
            count += 1
            
            # Check if we've extracted enough frames
            if len(frame_paths) >= max_frames:
                break
    
    # Release the video object
    video.release()
    
    print(f"Extracted {len(frame_paths)} frames from {video_path}")
    return frame_paths

def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video")
    parser.add_argument("--video", type=str, required=True, help="Path to the video file")
    parser.add_argument("--output", type=str, default="frames", help="Directory to save extracted frames")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to extract")
    parser.add_argument("--max_frames", type=int, default=20, help="Maximum number of frames to extract")
    
    args = parser.parse_args()
    
    extract_frames(args.video, args.output, args.fps, args.max_frames)

if __name__ == "__main__":
    main() 