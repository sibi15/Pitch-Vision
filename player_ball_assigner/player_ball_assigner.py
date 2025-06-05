import sys
sys.path.append("../")
from utils import get_center_of_bbox, measure_distance

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70  # in pixels

    def assign_ball_to_player(self, players, ball_bbox):
        ball_position = get_center_of_bbox(ball_bbox)

        minimum_distance = 99999    # for getting closest player to ball
        assigned_player = -1    # -1 means no player assigned (default)

        # Iterate through all players to find the closest one to the ball
        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)  # (x1, y2)
            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position) # (x2, y2)
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance:
                minimum_distance = distance
                assigned_player = player_id

        return assigned_player