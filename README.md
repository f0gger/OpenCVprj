# OpenCVprj

Project goals

The pipeline calibrates the camera, corrects lens distortion, thresholds likely road-marking pixels, transforms the road into a bird’s-eye view, fits lane boundaries with second-degree polynomials, estimates lane curvature and vehicle offset, and draws the result back onto the road. It then runs frame-by-frame on videos. The supplied files match a single-forward-camera lane-finding assignment.

Camera calibration

The 20 images in camera_cal/ were checked for a 9×6 grid of chessboard inner corners. Known board coordinates were paired with detected image corners, refined using cv2.cornerSubPix, and passed to cv2.calibrateCamera to estimate the camera matrix and distortion coefficients. In this run, 18 of 20 images were accepted; calibration1.jpg and calibration5.jpg did not yield the full grid and were skipped. The calibration reprojection RMS was approximately 0.865 pixels. This is an aggregate calibration error and does not guarantee perfect correction at every pixel.



Pipeline (single images)

1. Distortion correction

The saved camera matrix is scaled if the test image resolution differs from the 1280×720 calibration resolution. cv2.getOptimalNewCameraMatrix and cv2.remap correct lens distortion; remapping maps are cached by image size for faster video processing.



2. Color and gradient thresholding

threshold_markings() in lane_finding.py converts the image to HLS and HSV. Bright, relatively low-saturation pixels are candidates for white paint, and hue/saturation/value ranges select yellow paint. An x-direction Sobel gradient on the HLS lightness channel adds candidate edges. A trapezoid-shaped region of interest suppresses much of the sky and roadside, and morphological opening removes isolated noise. The result is a binary mask of likely road-marking pixels.



3. Perspective transform

perspective_matrices() defines four source points around the road and maps them to a rectangle. cv2.getPerspectiveTransform creates the forward and inverse transforms, and cv2.warpPerspective creates a bird’s-eye road view. The selected points are proportional to image width and height, so the same approximate transform supports the supplied 1280×720 and 960×540 media. These points are a starting estimate, however, not a calibrated camera-to-road model.



The binary mask is transformed using the same matrix before lane pixels are located:



4. Lane pixels and polynomial fitting

fit_lanes() sums the lower half of the warped binary mask into a column histogram. The strongest left and right peaks seed a nine-window sliding-window search. Pixels close to each window center are collected and each line is fitted with a quadratic polynomial, x(y) = Ay² + By + C, using numpy.polyfit. Plausibility checks reject lane pairs with unlikely width or position.

The solid/dashed labels are based on pixel occupancy along each fitted curve. Shadows, faded paint, partial occlusion, or unusual dash spacing can cause misclassification. Road-edge lines are not separately identified as their own class in this version.

5. Curvature and vehicle position

measure() converts the fitted curves to approximate metre coordinates and calculates their radius of curvature near the bottom of the bird’s-eye image. Vehicle offset is estimated from the difference between image center and the fitted lane center. The conversion assumes a 3.7 m lane width and approximately 30 m of longitudinal span in the warped image. Since these dimensions and the perspective mapping are not surveyed for the actual camera, curvature and offset are estimates rather than precise physical measurements.

6. Overlay

overlay() fills the space between the fitted boundaries with a translucent green polygon, draws the left/right markings in different colors based on their estimated style, inverse-warps the overlay onto the undistorted camera image, and adds curvature/offset text.



Pipeline (video)

The same processing pipeline runs on every decoded video frame. A short rolling history of polynomial fits reduces flicker, and recent fits can be reused briefly while a marking is occluded. The generated MP4s are in output/videos/, and frame counts/estimate rates are summarized in output/processing_summary.md. In this run, outputs were generated for all four included videos; challenge03.mp4 was the hardest sequence and had an estimate/overlay on about 89.6% of frames. That number also includes short-term tracking reuse, aside from new detections.

To process the two videos omitted from the upload, place them under test_videos/ and run python lane_finding.py all --skip-calibration.

Discussion and limitations

Second challenge video has a noticeable overlay distortion at the end, while the third challenge video appears to completely miss. The detector is likely to struggle with deep shadows, glare, faded or occluded paint, sharp bends, road seams that resemble line edges, and roads whose perspective differs from the chosen trapezoid. The histogram/sliding-window approach assumes two coherent boundaries are visible in expected left/right regions. Smoothing reduces flicker but can lag during abrupt changes. Possible improvements include tuning the perspective source points, adaptive thresholds, robust outlier rejection, temporal confidence scoring, an explicit road-edge detector, and quantitative evaluation against hand-labelled images. This is a coursework demonstration and must not be used for vehicle control or safety-critical decisions.
