
import cv2
import os
import sys

def extract_stills(video_path):
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return

    # Create 'stills' directory in the same folder as the video
    output_dir = os.path.join(os.path.dirname(os.path.abspath(video_path)), "stills")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"Total Frames: {total_frames}")
    
    frame_count = 0
    saved_count = 0
    last_valid_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        last_valid_frame = frame
            
        if frame_count % 24 == 0:
            # Construct output filename
            frame_filename = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{frame_count}.png")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
            sys.stdout.write(f"\rProcessed frame {frame_count}/{total_frames} (Saved {saved_count} stills)")
            sys.stdout.flush()
            
        frame_count += 1

    # Save the last frame if it wasn't already saved
    if last_valid_frame is not None and (frame_count - 1) % 24 != 0:
        last_frame_idx = frame_count - 1
        frame_filename = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{last_frame_idx}.png")
        cv2.imwrite(frame_filename, last_valid_frame)
        saved_count += 1
        print(f"\nSaved last frame: {last_frame_idx}")

    cap.release()
    print(f"\nFinished! Extracted {saved_count} stills to '{output_dir}'.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_name = sys.argv[1]
    else:
        video_name = input("Enter video filename (with extension): ").strip()
    
    extract_stills(video_name)
