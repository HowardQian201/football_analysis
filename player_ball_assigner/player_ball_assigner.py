import sys
sys.path.append("../")
from utils import get_center_of_bbox, measure_distance

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 50
        self.team_ball_control = []

    def do_assignment(self, tracks):
        for frame_num, player_track in enumerate(tracks["players"]):
            ball_bbox = tracks["ball"][frame_num][1]["bbox"]
            assigned_player = self.assign_ball_to_player(player_track, ball_bbox)

            if assigned_player != -1:
                tracks["players"][frame_num][assigned_player]["has_ball"] = True
                self.team_ball_control.append(tracks["players"][frame_num][assigned_player]["team"])
            elif self.team_ball_control:
                self.team_ball_control.append(self.team_ball_control[-1])

        # make gk ball assignment to last played team
        self.team_ball_control = self.switch_gk_assignment()

        return self.team_ball_control

    def switch_gk_assignment(self):
        # gk assignment is 0, so make previous seen team be the new assignment
        team_ball_control_copy = self.team_ball_control.copy()
        for i in range(len(self.team_ball_control)-1):
            curr = team_ball_control_copy[i]
            next = team_ball_control_copy[i+1]
            if int(next) == 0:
                team_ball_control_copy[i+1] = curr

            if int(curr) == 0:
                print(curr)

        return team_ball_control_copy

    def assign_ball_to_player(self, players, ball_bbox):

        ball_position = get_center_of_bbox(ball_bbox)
        curr_min_distance = 99999
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player["bbox"]
            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance:
                if distance < curr_min_distance:
                    curr_min_distance = distance
                    assigned_player = player_id

        return assigned_player

