import os
from utils import read_video, save_video
from trackers import Tracker
from team_assigner import TeamAssigner

def generate_video(zoom_on_ball=False, output_filename=None):

    # Default output path if not provided
    if output_filename is None:
        output_filename = 'output_videos/output_zoom.mp4' if zoom_on_ball else 'output_videos/output_normal.mp4'

    # Read video frames
    video_frames = read_video('input_videos/08fd33_4.mp4')

    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    # Run tracking (from stub if available)
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path='stubs/track_stubs.pkl'
    )

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Assign Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num],
                track['bbox'],
                player_id
            )
            track['team'] = team
            track['team_color'] = team_assigner.team_colors[team]

    # Path to the football emoji
    emoji_path = os.path.join(os.path.dirname(__file__), 'trackers', 'football.png')

    # Annotate frames
    output_video_frames = tracker.draw_annotations(
        video_frames,
        tracks,
        emoji_path,
        zoom_on_ball=zoom_on_ball
    )

    # Save the result
    save_video(output_video_frames, output_filename)

  
    return output_filename


# Optional CLI test run
def main():
    zoom_on_ball = False  # Set to True to test zoom manually
    generate_video(zoom_on_ball=zoom_on_ball)


if __name__ == '__main__':
    main()
