import numpy as np
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import math
import cmath
from shapely.ops import unary_union

root = "C:\\KITTI_Selection"
dir_bev = f"{root}\\bev\\"
dir_predictions = f"{root}\\predictions\\"
dir_labels = f"{root}\\labels\\"
dir_img = f"{root}\\images\\"

idxs = ['006067', '006059', '006310', '006227', '006121', '006315', '006211',
        '006042', '006130', '006374', '006037', '006097', '006048', '006206',
        '006253', '006098', '006291', '006312', '006329', '006054']

# Critical images: 4,8,12,16,17
# need to check for precision and recall: 8,6,4,2,0
i = 0

# Read and show the image
img = cv2.imread(f"{dir_img}/{idxs[i]}.png")
plt.figure(figsize=(20, 10))
plt.imshow(img[:, :, [2, 1, 0]])
plt.show()

# Read and show the BEV image
bev = cv2.imread(f"{dir_bev}/{idxs[i]}.png")
plt.figure(figsize=(9, 9))
plt.imshow(bev[:, :, [2, 1, 0]])

# Flip the x-axis and y-axis
x_limits = plt.xlim()
y_limits = plt.ylim()
plt.xlim(x_limits[1], x_limits[0])
plt.ylim(y_limits[1], y_limits[0])
plt.show()

# Load labels and predictions
try:
    predictions = np.loadtxt(f"{dir_predictions}/{idxs[i]}.csv", delimiter=",")
    if len(predictions) == 0:
        raise IOError("Empty predictions file.")
except (ValueError, IOError) as e:
    print(f"Error loading predictions file {dir_predictions}/{idxs[i]}.csv: {e}")

if len(predictions) == 0:
    x, y, w, l, im, re, object_conf, class_score, class_pred = 0, 0, 0, 0, 0, 0, 0, 0, 0
else:
     x, y, w, l, im, re, object_conf, class_score, class_pred = predictions[0]

labels = np.loadtxt(f"{dir_labels}/{idxs[i]}.csv", delimiter=",", ndmin=2)

if len(predictions) > 0:
    # Plot labels on the BEV image
    plt.figure(figsize=(9, 9))
    plt.imshow(bev[:, :, [2, 1, 0]])
    x_limits = plt.xlim()
    y_limits = plt.ylim()
    plt.xlim(x_limits[1], x_limits[0])
    plt.ylim(y_limits[1], y_limits[0])
    plt.scatter(labels[:, 1], labels[:, 2], c=[[0, 0, 1, 1]])
    plt.show()

    # Plot predictions on the BEV image
    plt.figure(figsize=(9, 9))
    plt.imshow(bev[:, :, [2, 1, 0]])
    x_limits = plt.xlim()
    y_limits = plt.ylim()
    plt.xlim(x_limits[1], x_limits[0])
    plt.ylim(y_limits[1], y_limits[0])
    plt.scatter(predictions[:, 0], predictions[:, 1], c=[[0, 1, 0, 1]])
    plt.show()

    # Plot labels and predictions on the same BEV image
    plt.figure(figsize=(9, 9))
    plt.imshow(bev[:, :, [2, 1, 0]])
    x_limits = plt.xlim()
    y_limits = plt.ylim()
    plt.xlim(x_limits[1], x_limits[0])
    plt.ylim(y_limits[1], y_limits[0])
    plt.scatter(labels[:, 1], labels[:, 2], c=[[0, 0, 1, 1]])
    plt.scatter(predictions[:, 0], predictions[:, 1], c=[[0, 1, 0, 1]])
    plt.show()
else:
    print("No data for plotting.")

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Convert bounding box coordinates to Shapely polygons
def create_bounding_box(x, y, w, l, im, re):
    complex_number = complex(re, im)
    angle_rad = cmath.phase(complex_number)
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)

    half_width = w / 2
    half_length = l / 2

    x1 = x - half_width * cos_angle - half_length * sin_angle
    y1 = y - half_width * sin_angle + half_length * cos_angle

    x2 = x + half_width * cos_angle - half_length * sin_angle
    y2 = y + half_width * sin_angle + half_length * cos_angle

    x3 = x + half_width * cos_angle + half_length * sin_angle
    y3 = y + half_width * sin_angle - half_length * cos_angle

    x4 = x - half_width * cos_angle + half_length * sin_angle
    y4 = y - half_width * sin_angle - half_length * cos_angle

    return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

# Create polygons from labels
labels_polygons = [
    Polygon(create_bounding_box(label[1], label[2], label[3], label[4], label[5], label[6]))
    for label in labels
]

# Create polygons from predictions
predictions_polygons = [
    Polygon(create_bounding_box(pred[0], pred[1], pred[2], pred[3], pred[4], pred[5]))
    for pred in predictions
]

# Plot Shapely polygons on the BEV image
plt.figure(figsize=(9, 9))
plt.imshow(bev[:, :, [2, 1, 0]])

for label_polygon, pred_polygon in zip(labels_polygons, predictions_polygons):
    x, y = label_polygon.exterior.xy
    plt.plot(x, y, color='green')
    x, y = pred_polygon.exterior.xy
    plt.plot(x, y, color='red')

#-------------------------------------------------------------------------------
#iou calculation 
#-------------------------------------------------------------------------------
    
def calculate_iou(box1, box2):
    poly1 = Polygon(box1)
    poly2 = Polygon(box2)
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    iou = intersection / union if union > 0 else 0
    return iou

ious_per_label = []

#for i in range(len(labels_polygons)):
for label_polygon in labels_polygons:
    ious_for_i = [] 
    #for j in range(len(predictions_polygons)):
    for pred_polygon in predictions_polygons:
        #label_polygon = labels_polygons[i]
        #pred_polygon = predictions_polygons[j]

        label_coords = label_polygon.exterior.coords
        pred_coords = pred_polygon.exterior.coords

        # Calculate IoU
        iou = calculate_iou(label_coords, pred_coords)
        #print(f'IoU: {iou:.2f}',i,j)
        ious_for_i.append(iou)

    # Check if there are any non-zero IoU values for this i
    if any(value != 0 for value in ious_for_i):
        ious_per_label.append(ious_for_i)
    else:
        ious_per_label.append([0])

# Print the results
non_zero_ious_per_label = []
max_iou_per_label = []
for i, ious_for_i in enumerate(ious_per_label):
    non_zero_ious_for_i = [iou for iou in ious_for_i if iou != 0]
    
    if non_zero_ious_for_i:
        non_zero_ious_per_label.append(non_zero_ious_for_i)
    else:
        non_zero_ious_per_label.append([0])

for i, non_zero_ious_for_i in enumerate(non_zero_ious_per_label):
    max_iou_for_i = max(non_zero_ious_for_i, default=0)
    max_iou_per_label.append(max_iou_for_i)

# Print the results on plot
for i, max_iou_for_i  in enumerate(max_iou_per_label):
    #print(f"For label {i}: {max_iou_for_i }")

    # Assuming you want to print the non-zero IoU values for each label on the plot
    label_x, label_y = np.mean(labels_polygons[i].exterior.coords, axis=0)
    plt.text(label_x - 20, label_y + 42.0, f'IoU: {max_iou_for_i:.2f}', color='white', fontsize=10, ha='left', va='top')

#-------------------------------------------------------------------------------
#precision and recall calculation
#-------------------------------------------------------------------------------

# Set the IoU threshold
iou_threshold = 0.5

# Initialize true positives, false positives, and false negatives
true_positives = 0
false_positives = 0
false_negatives = 0

# Iterate over predictions
for pred_polygon in predictions_polygons:
    iou_with_ground_truths = [calculate_iou(pred_polygon.exterior.coords, label_polygon.exterior.coords) for label_polygon in labels_polygons]
    max_iou = max(iou_with_ground_truths, default=0)  # Set default value to 0 if there are no matches

    if max_iou >= iou_threshold:
        true_positives += 1
    else:
        false_positives += 1

# Calculate false negatives (missed ground truths)
false_negatives = len(labels_polygons) - true_positives

# Calculate precision and recall
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

#converting into percentage
precision = precision*100
recall = recall*100

# Print the precision and recall on the plot
plt.text(10, 10, f'Precision: {precision:.2f}%', color='white', fontsize=10, ha='right', va='bottom')
plt.text(10, 30, f'Recall: {recall:.2f}%', color='white', fontsize=10, ha='right', va='bottom')

#plotting all together on the bev image()
plt.xlim(x_limits[1], x_limits[0])
plt.ylim(y_limits[1], y_limits[0])

if len(predictions) > 0:
    # Plot your existing content
    plt.show()  
else:
    # Plot alternative content when predictions are empty
    plt.figure(figsize=(9, 9))
    plt.imshow(bev[:, :, [2, 1, 0]])

    # Create polygons from labels
    labels_polygons = [
        Polygon(create_bounding_box(label[1], label[2], label[3], label[4], label[5], label[6]))
        for label in labels
    ]

    # Plot Shapely polygons on the BEV image
    for label_polygon in labels_polygons:
        x, y = label_polygon.exterior.xy
        plt.plot(x, y, color='green')

    # Set plot limits
    x_limits = plt.xlim()
    y_limits = plt.ylim()
    plt.xlim(x_limits[1], x_limits[0])
    plt.ylim(y_limits[1], y_limits[0])

    # Scatter plot for labels
    plt.scatter(labels[:, 1], labels[:, 2], c=[[0, 0, 1, 1]])

    # Additional text for indicating no predictions
    plt.text(10, 10, f'Prediction does not exist', color='white', fontsize=10, ha='right', va='bottom')

    # Show the plot
    plt.show()