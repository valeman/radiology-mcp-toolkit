import os
import argparse
import pandas as pd
from typing import List
from collections import defaultdict
from tqdm.auto import tqdm

int_to_label: dict = {0: 'IPH', 1: 'IVH', 2: 'SDH', 3: 'EDH', 4: 'SAH'}
h_types: List[str] = list(int_to_label.values())



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibration_output_dir', type=str, required=True, help='Path to directory containing calibration predictions')
    parser.add_argument('--output_dir', type=str, required=True, default="./", help='Path to directory to save calibration scores')
    args = parser.parse_args()

    # Begin by reading the text files you stored during your model's inference on your calibration dataset.
    calibration_scores: dict = defaultdict(dict)
    """
    calibration_scores = {
        'instance identifier': {
            'IPH': {
                [score, bounding box in xywhn format],
                [score, bounding box in xywhn format],
                ...
            },
            'IVH': ...
        }
    }
    """
    for prediction_file in tqdm(os.listdir(args.calibration_output_dir)):
        with open(os.path.join(args.calibration_output_dir, prediction_file), 'r') as f:
            lines: List[str] = f.readlines()

            classwise_predictions: dict = {h_type: [] for h_type in h_types}
            for line in lines:
                score: float = float(line.split(" ")[-1].split('\n')[0])
                label_int: int = int(float(line.split(" ")[0]))
                bounding_box: str = " ".join(line.split(" ")[1:5])
                classwise_predictions[int_to_label[label_int]].append([score, bounding_box])
        
        calibration_scores[prediction_file.split('.txt')[0]]: dict = classwise_predictions
    print(f"{len(calibration_scores)} sets of raw predictions read and stored in calibration_scores.")

    # Next, perform suppression such that only the highest scoring bounding box for each class is kept.
    retained_predictions: dict = {}
    """
    retained_predictions = {
        'instance identifier': {
            'IPH': {
                'score': score (-1 if no prediction),
                'bounding_box': bounding box in xywhn format ('' if no prediction),
            },
            'IVH': {
                ...
            }
        }
    }
    """
    for instance_identifier, prediction_dict in tqdm(list(calibration_scores.items())):
        
        retained_predictions[instance_identifier]: dict = {}

        for h_type, predictions in prediction_dict.items():

            max_score = -1
            max_score_bounding_box = ''
            
            for bounding_box in predictions:
                if bounding_box[0] > max_score:
                    max_score = bounding_box[0]
                    max_score_bounding_box = bounding_box[1]
            
            retained_predictions[instance_identifier][h_type]: dict = {
                'score': max_score,
                'bounding_box': max_score_bounding_box
            }   
    print(f"Class-wise on-max suppression complete.")

    # Now, condense the retained predictions to only store the hemorrhage type and calibration score.
    presence_calibration_scores: dict = {}
    """
    presence_calibration_scores = {
        'instance identifier': {
            'IPH': score,
            'IVH': score,
            ...
        }
    }
    """
    for instance_identifier, prediction_dict in tqdm(list(retained_predictions.items())):
        
        presence_calibration_scores[instance_identifier]: dict = {}

        for h_type, prediction in prediction_dict.items():
            if prediction['bounding_box'] == '':
                presence_calibration_scores[instance_identifier][h_type] = 0
                continue
            else: 
                presence_calibration_scores[instance_identifier][h_type] = prediction['score']
    print(f"Presence score extraction complete.")

    # Finally, create and save the .csv file containing the calibration scores.
    final_calibration_scores: dict = {'Sample': [], f'{h_types[0]} Presence': [], f'{h_types[1]} Presence': [], f'{h_types[2]} Presence': [], f'{h_types[3]} Presence': [], f'{h_types[4]} Presence': [], f'{h_types[0]} Absence': [], f'{h_types[1]} Absence': [], f'{h_types[2]} Absence': [], f'{h_types[3]} Absence': [], f'{h_types[4]} Absence': []}
    for instance_identifier, prediction_dict in tqdm(list(presence_calibration_scores.items())):
        final_calibration_scores['Sample'].append(instance_identifier)
        for h_type in h_types:
            final_calibration_scores[f'{h_type} Presence'].append(prediction_dict[h_type])
            final_calibration_scores[f'{h_type} Absence'].append(1 - prediction_dict[h_type])
    final_calibration_scores: pd.DataFrame = pd.DataFrame(final_calibration_scores)
    final_calibration_scores.to_csv(os.path.join(args.output_dir, 'calibration_scores.csv'), index=False)
    print(f"Calibration scores saved to {os.path.join(args.output_dir, 'calibration_scores.csv')}.")
