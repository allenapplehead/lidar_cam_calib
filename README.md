
nstructions: LiDAR-Camera Extrinsics Calibration

Follow these steps to compute the 6-DoF extrinsics calibration by solving the Perspective-n-Point (PnP) problem between 3D LiDAR point clouds and 2D camera images.

### Prerequisites
* Avoid using MATLAB's `cameraLiDARCalibrator`, as it may produce imprecise extrinsics if the calibration board is partially cropped at the edges of the LiDAR FOV.

### Calibration Steps

**1. Select Static Frames**
* Extract 8–15 synchronized image-LiDAR pairs where the calibration board is completely static. 
* *Tip:* Verify stability by ensuring the 2D cross-junctions of the checkerboard do not shift past a set threshold across neighboring frames.

**2. Isolate the 3D Target Cluster**
* Open the LiDAR point cloud (e.g., in CloudCompare) and manually select at least 4 inlier points on the calibration board.
* Run a DBSCAN clustering algorithm on these points to isolate the full calibration board cluster.

**3. Fit a Plane and Extract 3D Corners**
* Apply Principal Component Analysis (PCA) to the isolated cluster. The two largest principal components will define the plane, and the third will define the surface normal.
* Apply RANSAC to filter out any structural outliers (e.g., the mounting stand).
* Extract the 4 physical corners of the calibration board from this fitted 3D plane. 
* *Tip:* If the board is partially cropped in the LiDAR scan, anchor the fitted plane to the visible top or bottom edge to ensure it matches the physical dimensions.

**4. Detect 2D Image Corners**
* Use OpenCV's built-in checkerboard detection methods to automatically locate the 4 corresponding corners of the calibration board in the 2D images.

**5. Solve PnP for Extrinsics**
* Pass the matched 3D plane corners and 2D image corners into a Perspective-n-Point (PnP) solver.
* Use the Levenberg-Marquardt nonlinear optimization method across all selected image-LiDAR pairs to minimize reprojection error and output the final 6-DoF extrinsics.

---

### Expected Reprojection Error (RMSE)
For reference, a successful calibration should yield RMSE values similar to the following:

| Sensor Pair | Target RMSE (px) |
| :--- | :--- |
| Left Camera / LiDAR | ~5.15 |
| Center Camera / LiDAR | ~3.85 |
| Right Camera / LiDAR | ~3.06 |
