from sklearn.cluster import KMeans
import math
import colorspacious # type: ignore
from collections import defaultdict

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.kmeans = None
        self.player_team_dict = {} # player id : team number --- not using because assignments can be wrong if player IDs switched during tracking

    def do_assignment(self, video_frames, tracks):
        self.assign_team_color(video_frames[0], tracks["players"][0])

        players_marked_gk_count = defaultdict(int)
        players_marked_gk_frames = defaultdict(list)

        for frame_num, player_track in enumerate(tracks["players"]):
            for player_id, track in player_track.items():
                team, player_color = self.get_player_team(video_frames[frame_num], track["bbox"], player_id)
                tracks["players"][frame_num][player_id]["team"] = team
                tracks["players"][frame_num][player_id]["team_color"] = self.get_team_colors()[team]
            
                # check for goalkeeper based on color      
                outside_thresh1, deltaE1 = self.outside_gk_color_distance_threshold(player_color, self.get_team_colors()[1], 25)
                outside_thresh2, deltaE2 = self.outside_gk_color_distance_threshold(player_color, self.get_team_colors()[2], 25)
                
                if outside_thresh1 and outside_thresh2:
                    players_marked_gk_count[player_id] += 1
                    players_marked_gk_frames[player_id].append(frame_num)

        # assign gk black color and team
        for player_id, count in players_marked_gk_count.items():
            if count > 10:
                for frame_num in players_marked_gk_frames[player_id]:
                    if tracks["players"][frame_num].get(player_id) is not None:
                        tracks["players"][frame_num][player_id]["team"] = 0
                        tracks["players"][frame_num][player_id]["team_color"] = (0, 0, 0)
            
    def outside_gk_color_distance_threshold(self, color1, color2, threshold):
        # Normalize RGB values to the range 0-1
        rgb1_normalized = [x / 255.0 for x in color1]
        rgb2_normalized = [x / 255.0 for x in color2]
        
        # Convert RGB to LAB using the sRGB1-to-Lab conversion
        lab1 = colorspacious.cspace_convert(rgb1_normalized, "sRGB1", "CIELab")
        lab2 = colorspacious.cspace_convert(rgb2_normalized, "sRGB1", "CIELab")
        
        delta_E = self.calc_deltaE(lab1, lab2)
        
        # print(delta_E)

        if delta_E > threshold:
            return True, delta_E
        else:
            return False, delta_E
    
    def calc_deltaE(self, lab1, lab2):
        L1, a1, b1 = lab1[0], lab1[1], lab1[2]
        L2, a2, b2 = lab2[0], lab2[1], lab2[2]

        # Step 1: Calculate C' and h'
        C1 = math.sqrt(a1**2 + b1**2)
        C2 = math.sqrt(a2**2 + b2**2)
        C_mean = (C1 + C2) / 2
        
        G = 0.5 * (1 - math.sqrt((C_mean**7) / (C_mean**7 + 25**7)))
        
        a1_prime = (1 + G) * a1
        a2_prime = (1 + G) * a2
        
        C1_prime = math.sqrt(a1_prime**2 + b1**2)
        C2_prime = math.sqrt(a2_prime**2 + b2**2)
        
        h1_prime = math.degrees(math.atan2(b1, a1_prime)) % 360
        h2_prime = math.degrees(math.atan2(b2, a2_prime)) % 360
        
        # Step 2: Calculate deltaL', deltaC', and deltah'
        delta_L_prime = L2 - L1
        delta_C_prime = C2_prime - C1_prime
        
        h_prime_diff = h2_prime - h1_prime
        if C1_prime * C2_prime == 0:
            delta_h_prime = 0
        elif abs(h_prime_diff) <= 180:
            delta_h_prime = h_prime_diff
        elif h_prime_diff > 180:
            delta_h_prime = h_prime_diff - 360
        else:
            delta_h_prime = h_prime_diff + 360
        
        delta_H_prime = 2 * math.sqrt(C1_prime * C2_prime) * math.sin(math.radians(delta_h_prime / 2))
        
        # Step 3: Calculate L', C', and H'
        L_mean_prime = (L1 + L2) / 2
        C_mean_prime = (C1_prime + C2_prime) / 2
        
        if C1_prime * C2_prime == 0:
            H_mean_prime = h1_prime + h2_prime
        elif abs(h1_prime - h2_prime) <= 180:
            H_mean_prime = (h1_prime + h2_prime) / 2
        elif abs(h1_prime - h2_prime) > 180 and (h1_prime + h2_prime) < 360:
            H_mean_prime = (h1_prime + h2_prime + 360) / 2
        else:
            H_mean_prime = (h1_prime + h2_prime - 360) / 2
        
        T = 1 - 0.17 * math.cos(math.radians(H_mean_prime - 30)) + \
            0.24 * math.cos(math.radians(2 * H_mean_prime)) + \
            0.32 * math.cos(math.radians(3 * H_mean_prime + 6)) - \
            0.20 * math.cos(math.radians(4 * H_mean_prime - 63))
        
        delta_theta = 30 * math.exp(-((H_mean_prime - 275) / 25)**2)
        R_C = 2 * math.sqrt((C_mean_prime**7) / (C_mean_prime**7 + 25**7))
        S_L = 1 + ((0.015 * (L_mean_prime - 50)**2) / math.sqrt(20 + (L_mean_prime - 50)**2))
        S_C = 1 + 0.045 * C_mean_prime
        S_H = 1 + 0.015 * C_mean_prime * T
        R_T = -math.sin(math.radians(2 * delta_theta)) * R_C
        
        # Step 4: Calculate delta E
        delta_E = math.sqrt(
            (delta_L_prime / S_L)**2 +
            (delta_C_prime / S_C)**2 +
            (delta_H_prime / S_H)**2 +
            R_T * (delta_C_prime / S_C) * (delta_H_prime / S_H)
        )

        return delta_E

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)

            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0] # team 1
        self.team_colors[2] = kmeans.cluster_centers_[1] # team 2

    def get_team_colors(self):
        return self.team_colors
    
    def get_player_team(self, frame, player_bbox, player_id):

        # if player_id in self.player_team_dict:        # commented out because assignments can be wrong if player IDs switched during tracking
        #     return self.player_team_dict[player_id]   # now constantly checking the team assignment each frame 
        
        player_color = self.get_player_color(frame, player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id += 1

        self.player_team_dict[player_id] = team_id

        return team_id, player_color

    def get_player_color(self, frame, bbox):
        # get cropped player image
        cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # take top half of image
        top_half_image = cropped_image[0:int(cropped_image.shape[0]/2),:]

        # cluster image into 2 color clusters to separate background and player jersey
        # reshape image into 2d array
        image_2d = top_half_image.reshape(-1,3)

        # k means clustering
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        # get cluster labels
        labels = kmeans.labels_

        # reshape labels into image
        clustered_image = labels.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # get player cluster
        corner_clusters = [clustered_image[0,0], clustered_image[0,-1], clustered_image[-1,0], clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters), key = corner_clusters.count)
        player_cluster = 1-non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color


    