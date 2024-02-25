# ObjectDetection_using-_python
Creating bounding boxes on the images provided by image sensors and calculating its IOU value and predicting the object with the given coordinates from YOLO data sets 
This task aims to evaluate the detection of the state-of-the-art 2D complex YOLO(You Only Look Once)
object detector on the training split of the Kitti data set. In this case, effectively detecting certain classes of
road elements, cars, cycles, and pedestrians is essential to ensure the safety of the vehicle, its occupants, and
its surroundings. Thus, this effectiveness shall be evaluated using the two most significant metrics for object
detection: Precision and Recall.
The KITTI dataset uses 2D points to represent object-bounding boxes visually. The dataset annotations
provide information on the x and y coordinates, length (l), width (w), and orientation angle (re, im). These
parameters generate shapely polygons to accurately depict the object boundaries in the Bird’s Eye View
(BEV) image. This helps to gain a comprehensive spatial understanding of the objects in the scene and
provides object localization.
We flipped the BEV image according to the offered KITTI image to ensure consistent visualization. Additionally,
we have enlarged the plot where the bounding box was located to enhance the visualization. The
prediction with the highest IOU with the ground truth is chosen when selecting the bounding box for multiple
object predictions.
We use an IoU metric to compare the predicted box with the ground truth. If the value resulting from
this comparison is more significant than a predefined threshold (which is set at 0.5), the object is identified
as ”detected” or ”True Positive.” If the value is below the threshold, the object is considered ”missed” or
”False Positive.” When the model fails to detect an object in the ground truth, it is ”False Negative.”
