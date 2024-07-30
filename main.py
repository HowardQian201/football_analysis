from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner

def main():
    # read video
    video_frames = read_video("input_videos/08fd33_4.mp4")

    # init tracker
    tracker = Tracker("models/best_yolov5xu.pt")
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True, stub_path="stubs/track_stubs.pk1")

    # interpolate ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # assign player teams
    team_assigner = TeamAssigner()
    team_assigner.do_assignment(video_frames, tracks)

    # assign ball possession
    player_ball_assigner = PlayerBallAssigner()
    ball_possession = player_ball_assigner.do_assignment(tracks)
    ball_possession = np.array(ball_possession)

    # draw output
    ## draw object tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, ball_possession)

    # save video
    save_video(output_video_frames, "output_videos/output_video.avi")

    

if __name__ == "__main__":
    main()