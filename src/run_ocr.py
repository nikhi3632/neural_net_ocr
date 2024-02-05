import os
import numpy as np
import pickle
import string
import skimage.io
import skimage.morphology
import skimage.transform
from neural_net import forward, sigmoid, softmax
from optical_character_recognition import find_letters
import warnings
warnings.filterwarnings('ignore')

def find_rows(b_boxes):
    # Calculate centers of bounding boxes
    centers_with_bboxes = []
    for bbox in b_boxes:
        minr, minc, maxr, maxc = bbox
        center = ((minr + maxr) // 2, (minc + maxc) // 2)
        centers_with_bboxes.append((center, bbox))
    # Sort based on the x-coordinate of their centers
    points = sorted(centers_with_bboxes, key=lambda c: c[0])
    # Group into rows based on vertical positions and average heights
    rows = []
    for point in points:
        find_matching_row = False
        center = point[0]
        for row in rows:
            # Calculate average height and center position for the current row
            average_height = sum([p[1][2] - p[1][0] for p in row]) / float(len(row))
            average_center = sum([p[0][0] for p in row]) / float(len(row))
            # Check if the current point belongs to the current row
            if abs(center[0] - average_center) < average_height:
                row.append(point)
                find_matching_row = True
                break  # Exit the inner loop when a suitable row is found
        # If no matching row is found, create a new row
        if not find_matching_row:
            rows.append([point])
    # Sort each row based on the y-coordinate
    for i in range(len(rows)):
        rows[i] = sorted(rows[i], key=lambda r: r[0][1])
    return rows

def generate_dataset(rows_, bw_):
    '''
    bw_ (numpy.ndarray): Binary image representing the regions of interest.
    '''
    dataset = []
    for row in rows_:
        data_row = []
        for point in row:
            bbox = point[1]
            minr, minc, maxr, maxc = bbox
            # Crop the image based on bounding box coordinates
            image = bw_[minr : maxr + 1, minc : maxc + 1]
            H, W = image.shape
            if H > W:
                W_left, W_right = (H - W) // 2, (H - W) // 2
                image = np.pad(image, ((H // 20, H // 20), (W_left + H // 20, W_right + H // 20)), "constant", constant_values = (1, 1))
            else:
                H_top, H_bottom = (W - H) // 2, (W - H) // 2
                image = np.pad(image, ((H_top + W // 20, H_bottom + W // 20), (W // 20, W // 20)), "constant", constant_values = (1, 1))
            image = skimage.transform.resize(image, (32, 32))
            image = skimage.morphology.erosion(image, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
            data_row.append(np.transpose(image).flatten())
        dataset.append(np.array(data_row))
    return dataset

def calculate_accuracy(ground_truth, prediction):
    if len(ground_truth) != len(prediction):
        raise ValueError("Input lists must have the same length.")

    correct_count = 0
    total_count = 0

    for gt_list, pred_list in zip(ground_truth, prediction):
        if len(gt_list) != len(pred_list):
            raise ValueError("Each sublist in ground truth and prediction must have the same length.")

        for gt, pred in zip(gt_list, pred_list):
            total_count += 1
            correct_count += (gt == pred)

    accuracy = correct_count / total_count if total_count != 0 else 0
    return accuracy

if __name__ == "__main__":
    ARTIFACTS_DIR = os.getcwd() + "/artifacts"
    IMAGE_DIR = '../data/images'
    PASS_IMAGES = [
        img_filename for img_filename in sorted(os.listdir(IMAGE_DIR))
        if any(ext in img_filename for ext in ['.png', '.jpg', '.jpeg']) and 'fail' not in img_filename
    ]
    ground_truths = [
        [
            'TODOLIST',
            '1MAKEATODOLIST',
            '2CHECKOFFTHEFIRST',
            'THINGONTODOLIST',
            '3REALIZEYOUHAVEALREADY',
            'COMPLETED2THINGS',
            '4REWARDYOURSELFWITH',
            'ANAP'
        ],
        [
            'ABCDEFG',
            'HIJKLMN',
            'OPQRSTU',
            'VWXYZ',
            '1234567890'
        ],
        [
            'HAIKUSAREEASY',
            'BUTSOMETIMESTHEYDONTMAKESENSE',
            'REFRIGERATOR'
        ],
        [
            'DEEPLEARNING',
            'DEEPERLEARNING',
            'DEEPESTLEARNING'
        ]
    ]
    predictions = []
    accuracies = []
    for idx, img_filename in enumerate(PASS_IMAGES):
        img_path = os.path.join(IMAGE_DIR, img_filename)
        im = skimage.img_as_float(skimage.io.imread(os.path.join(img_path)))
        bboxes, bw = find_letters(im)
        # find the rows using..RANSAC, counting, clustering, etc.
        rows = find_rows(bboxes)
        # crop the bounding boxes
        # note.. before flatten, transpose the image (that's how the dataset is!)
        # consider doing a square crop, and even using np.pad() to get the images looking more like the dataset
        dataset = generate_dataset(rows, bw)
        # load the weights
        # run the crops through the neural network and print the outputs
        letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
        params = pickle.load(open(ARTIFACTS_DIR + '/model_weights.pickle','rb'))
        outputs = []
        for row in dataset:
            out = forward(row, params, "layer1", sigmoid)
            probs = forward(out, params, "output", softmax)
            predicted = np.argmax(probs, axis = 1)
            row_pred = ""
            for pred in predicted:
                row_pred += (letters[pred] + " ")
            print(row_pred)
            outputs.append(row_pred.replace(" ", ""))
        predictions.append(outputs)
        accuracy = calculate_accuracy(ground_truths[idx], predictions[idx])
        accuracies.append(accuracy)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print("-" * 60)
    print(f"Average Accuracy: {np.mean(accuracies) * 100:.2f}%")