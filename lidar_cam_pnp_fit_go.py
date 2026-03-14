import cv2
import numpy as np
import argparse
import os
import re
import glob
import json
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.optimize import linear_sum_assignment, least_squares
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN


# --- Configuration Loader ---
def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        data = json.load(f)

    camera_matrix = np.array(data["intrinsics"], dtype=np.float32)
    dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float32)
    t_guess = np.array(data["tf"]["translation"], dtype=np.float64)
    q_guess = np.array(data["tf"]["quaternion"], dtype=np.float64)
    rot = R.from_quat(q_guess)
    rot_inv = rot.inv()
    t_guess_inv = -rot_inv.apply(t_guess)
    rvec_guess_inv = rot_inv.as_rotvec()
    return camera_matrix, dist_coeffs, t_guess_inv, rvec_guess_inv


class PointLoader:
    @staticmethod
    def load_picked_points_from_dir(dir_path):
        data_map = {}
        coord_pattern = re.compile(r"\((.*?);(.*?);(.*?)\)")
        files = glob.glob(os.path.join(dir_path, "manual_corners_*.txt"))
        
        valid_modes = ["pca", "fixed_bottom", "fixed_top"]

        for filepath in files:
            id_match = re.search(r"manual_corners_(\d+)\.txt", os.path.basename(filepath))
            if not id_match:
                continue

            file_id = id_match.group(1)
            pts = []
            override_mode = None

            with open(filepath, "r") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                if not lines:
                    continue
                
                # Check the very last line for an override mode
                last_line = lines[-1].lower()
                for mode in valid_modes:
                    if mode in last_line:
                        override_mode = mode
                        break

                for line in lines:
                    if "[" in line and "Picked" in line:
                        match = coord_pattern.search(line)
                        if match:
                            pts.append(list(map(float, match.groups())))
            
            if len(pts) >= 4:
                # Store points and the override (which may be None)
                data_map[file_id] = (np.array(pts[:4], dtype=np.float32), override_mode)

        return data_map

    @staticmethod
    def load_lidar_bin(bin_path):
        if not os.path.exists(bin_path):
            return None
        return np.fromfile(bin_path, dtype=np.float32).reshape((-1, 4))[:, :3]


class LidarProcessor:
    def __init__(self, picked_points, dims, mode="pca", margin=0.05):
        self.picked_points = picked_points
        self.margin = margin
        self.mode = mode
        self.phys_w = ((dims["cols"] + 1) * dims["square_size"] + 2 * dims["border_w"]) / 1000.0
        self.phys_h = ((dims["rows"] + 1) * dims["square_size"] + 2 * dims["border_h"]) / 1000.0

    def process(self, cloud):
        roi_cloud = self._crop_to_roi(cloud)
        if len(roi_cloud) < 50:
            return None, None, None, None
        best_cluster, rejected = self._cluster_points(roi_cloud)
        if best_cluster is None or len(best_cluster) < 50:
            return None, None, None, None
        plane_cloud, normal, center = self._fit_plane_robust(best_cluster)
        if plane_cloud is None:
            return None, None, None, None
        
        if self.mode == "fixed_bottom":
            corners = self._find_corners_fixed_anchor(plane_cloud, normal, "bottom")
        elif self.mode == "fixed_top":
            corners = self._find_corners_fixed_anchor(plane_cloud, normal, "top")
        else:
            corners = self._find_corners_pca(plane_cloud)

        return self._match_order(corners), best_cluster, (normal, center), rejected

    def _crop_to_roi(self, cloud):
        min_b = np.min(self.picked_points, 0) - self.margin
        max_b = np.max(self.picked_points, 0) + self.margin
        return cloud[np.all((cloud >= min_b) & (cloud <= max_b), axis=1)]

    def _cluster_points(self, points):
        db = DBSCAN(eps=0.2, min_samples=20).fit(points)
        labels = db.labels_
        unique_labels = [l for l in set(labels) if l != -1]
        if not unique_labels:
            return None, points
        m_center = np.mean(self.picked_points, 0)
        best_label = min(
            unique_labels,
            key=lambda l: np.linalg.norm(np.mean(points[labels == l], 0) - m_center),
        )
        return points[labels == best_label], points[labels != best_label]

    def _fit_plane_robust(self, points):
        pca = PCA(n_components=3).fit(points)
        normal, center = pca.components_[2], pca.mean_
        if np.dot(normal, center / np.linalg.norm(center)) > 0:
            normal = -normal
        inliers = np.abs(np.dot(points - center, normal)) < 0.03
        return points[inliers], normal, center

    def _find_corners_pca(self, points):
        pca = PCA(n_components=3).fit(points)
        pts_t = pca.transform(points)
        u_m, v_m = np.min(pts_t[:, :2], 0), np.max(pts_t[:, :2], 0)
        c_pca = np.array([[u_m[0], v_m[0], 0], [v_m[0], v_m[0], 0], [v_m[0], u_m[1], 0], [u_m[0], u_m[1], 0]])
        return pca.inverse_transform(c_pca)

    def _find_corners_fixed_anchor(self, points, normal, anchor):
        z_axis = np.array([0, 0, 1])
        v_vec = np.cross(normal, z_axis)
        c, s = np.dot(normal, z_axis), np.linalg.norm(v_vec)
        if s == 0:
            R_align = np.eye(3)
        else:
            kmat = np.array([[0, -v_vec[2], v_vec[1]], [v_vec[2], 0, -v_vec[0]], [-v_vec[1], v_vec[0], 0]])
            R_align = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))

        center = np.mean(points, 0)
        aligned_2d = np.dot(points - center, R_align.T)[:, :2]
        best_angle, min_area = 0, float("inf")

        for angle in np.linspace(0, 180, 60):
            rad = np.radians(angle)
            rot = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
            rotated = np.dot(aligned_2d, rot.T)
            area = (rotated[:, 0].max() - rotated[:, 0].min()) * (rotated[:, 1].max() - rotated[:, 1].min())
            if area < min_area:
                min_area, best_angle = area, rad

        rot = np.array([[np.cos(best_angle), -np.sin(best_angle)], [np.sin(best_angle), np.cos(best_angle)]])
        final_2d = np.dot(aligned_2d, rot.T)
        u_min, u_max = final_2d[:, 0].min(), final_2d[:, 0].max()
        target_h = self.phys_h if abs((u_max - u_min) - self.phys_w) < abs((u_max - u_min) - self.phys_h) else self.phys_w
        
        v_base = final_2d[:, 1].min() if anchor == "bottom" else final_2d[:, 1].max()
        v_other = v_base + target_h if anchor == "bottom" else v_base - target_h
        v_min, v_max = min(v_base, v_other), max(v_base, v_other)

        corners_2d = np.array([[u_min, v_min], [u_max, v_min], [u_max, v_max], [u_min, v_max]])
        corners_3d = np.hstack((np.dot(corners_2d, rot), np.zeros((4, 1))))
        return np.dot(corners_3d, R_align) + center

    def _match_order(self, corners):
        cost = np.array([[np.linalg.norm(p - c) for c in corners] for p in self.picked_points])
        _, col_ind = linear_sum_assignment(cost)
        return corners[col_ind]


class ImageProcessor:
    def __init__(self, rows, cols, sq, bw, bh):
        self.pattern = (cols, rows)
        self.rows, self.cols, self.sq, self.bw, self.bh = rows, cols, sq, bw, bh
        self.obj_pts = np.zeros((cols * rows, 2), np.float32)
        self.obj_pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    def detect_grid_corners(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.pattern, None)
        if not ret:
            return None
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        H, _ = cv2.findHomography(self.obj_pts, corners.reshape(-1, 2))
        px, py = 1.0 + (self.bw / self.sq), 1.0 + (self.bh / self.sq)
        lc = np.array([[-px, -py], [(self.cols - 1) + px, -py], [(self.cols - 1) + px, (self.rows - 1) + py], [-px, (self.rows - 1) + py]], dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(lc, H).reshape(4, 2)


def visualize_3d_sample(cluster, rejected, corners, plane_data):
    normal, center = plane_data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    pts = cluster[np.random.choice(len(cluster), min(3000, len(cluster)), replace=False)]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="b", s=1, alpha=0.3)
    c_plot = np.vstack((corners, corners[0]))
    ax.plot(c_plot[:, 0], c_plot[:, 1], c_plot[:, 2], c="lime", lw=2)
    n = normal / np.linalg.norm(normal)
    ax.quiver(center[0], center[1], center[2], n[0], n[1], n[2], length=0.3, color="orange")
    limit = (np.array([cluster[:, 0].max() - cluster[:, 0].min(), cluster[:, 1].max() - cluster[:, 1].min(), cluster[:, 2].max() - cluster[:, 2].min()]).max() / 2.0)
    m = np.mean(cluster, 0)
    ax.set_xlim(m[0] - limit, m[0] + limit)
    ax.set_ylim(m[1] - limit, m[1] + limit)
    ax.set_zlim(m[2] - limit, m[2] + limit)
    plt.show()


def reprojection_residuals(params, all_3d, all_2d, K, D):
    rvec, tvec = params[:3], params[3:]
    res = []
    for p3, p2 in zip(all_3d, all_2d):
        proj, _ = cv2.projectPoints(p3, rvec, tvec, K, D)
        res.append((proj.reshape(-1, 2) - p2).ravel())
    return np.concatenate(res)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("coords_dir")
    parser.add_argument("image_dir")
    parser.add_argument("--lidar_dir", required=True)
    parser.add_argument("--config", default="right.json")
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--fit_mode", choices=["pca", "fixed_bottom", "fixed_top"], default="pca")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--viz_3d", action="store_true")
    args = parser.parse_args()

    K, D, t_init, r_init = load_config(args.config)
    manual_data = PointLoader.load_picked_points_from_dir(args.coords_dir)
    img_paths = sorted(glob.glob(os.path.join(args.image_dir, "*.png")))
    dims = {"cols": args.cols, "rows": args.rows, "square_size": 120, "border_w": 70, "border_h": 38}
    img_proc = ImageProcessor(args.rows, args.cols, dims["square_size"], dims["border_w"], dims["border_h"])

    all_3d, all_2d = [], []

    for path in img_paths:
        f_id = os.path.splitext(os.path.basename(path))[0]
        if f_id not in manual_data:
            continue

        picked_pts, override = manual_data[f_id]
        
        # Decide which mode to use
        current_mode = override if override else args.fit_mode
        if override:
            print(f"Sample {f_id}: [!] Overriding fit_mode to '{override}' (from file)")
        else:
            print(f"Sample {f_id}: Using default fit_mode '{current_mode}'")

        cloud = PointLoader.load_lidar_bin(os.path.join(args.lidar_dir, f"{f_id}.bin"))
        img = cv2.imread(path)
        if cloud is None or img is None:
            continue

        lp = LidarProcessor(picked_pts, dims, mode=current_mode)
        pts3d, cluster, plane, rejected = lp.process(cloud)
        if pts3d is None:
            continue

        pts2d = img_proc.detect_grid_corners(img)
        if pts2d is not None:
            all_3d.append(pts3d.astype(np.float64))
            all_2d.append(pts2d.astype(np.float64))
            print(f"Collected Sample: {f_id}")
            if args.viz_3d:
                visualize_3d_sample(cluster, rejected, pts3d, plane)

    if len(all_3d) < 1:
        print("No valid samples collected.")
        return

    print(f"\nOptimizing across {len(all_3d)} frames...")
    res = least_squares(reprojection_residuals, np.hstack((r_init, t_init)), args=(all_3d, all_2d, K, D), method="lm")

    final_rvec, final_tvec = res.x[:3], res.x[3:]
    rot = R.from_rotvec(final_rvec)
    final_quat = rot.as_quat()
    rot_inv = rot.inv()
    final_tvec_inv = -rot_inv.apply(final_tvec)
    final_quat_inv = rot_inv.as_quat()

    print("\n" + "=" * 40 + "\nOPTIMIZED EXTRINSICS (LiDAR -> Camera)\n" + "=" * 40)
    print(f'"translation": {final_tvec.tolist()},\n"quaternion": {final_quat.tolist()}')
    print("\n" + "=" * 40 + "\nINVERSE EXTRINSICS (Camera -> LiDAR)\n" + "=" * 40)
    print(f'"translation": {final_tvec_inv.tolist()},\n"quaternion": {final_quat_inv.tolist()}')
    print("\n" + "-" * 40 + f"\nFinal RMSE: {np.sqrt(np.mean(res.fun**2)):.4f} px\n" + "=" * 40)


if __name__ == "__main__":
    main()
