# Socket Detection Tool - Best Practices

To ensure consistency and improve model performance:
- Only the central region directly beneath the camera was used for training, as it offered the most undistorted and perpendicular view.
- Manual labelling helped overcome the visual ambiguity between empty and filled sockets.
- ArUco markers were used for camera calibration and to provide a fixed reference point during evaluation.
- Consistent lighting conditions, as well as a fixed camera position and angle, are recommended for both data collection and deployment.
