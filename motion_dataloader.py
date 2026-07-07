"""
Unified Motion Dataset Dataloader for MotionFM downstream tasks.

This dataloader supports multiple datasets with different file formats and sensor types.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Union, Literal, Iterable
from dataclasses import dataclass, field
import torch
from torch.utils.data import Dataset, DataLoader
import warnings


# ============================================================================
# Dataset Configuration
# ============================================================================

@dataclass
class DatasetConfig:
    """Configuration for each dataset's file naming and structure."""
    name: str
    # File format type: 'standard', 'embedded_label', 'samosa_imu'
    format_type: str
    # Whether the dataset has separate label files
    has_separate_labels: bool
    # Sensor types available in this dataset
    available_sensors: List[str]
    # Mapping from canonical sensor names to dataset-specific patterns
    sensor_patterns: Dict[str, List[str]]
    # Default sensor if not specified in filename (for accelerometer-only datasets)
    default_sensor: Optional[str] = None
    # Label file postfix pattern (regex)
    label_postfix_pattern: str = r'_labels\.parquet$'
    # Whether label is per-sample (single value) or dense (per-timestep)
    label_type: str = 'dense'  # 'dense' or 'single'
    # Body placement patterns
    placement_patterns: List[str] = field(default_factory=list)
    # Optional default placement constraint (string or list of placement tags).
    # This does not change indexing by itself; callers can choose to apply it.
    default_placement: Optional[Union[str, List[str]]] = None
    # Datasets that are too large for stride-1 overlapped indexing can opt out
    # even when the caller requests an overlap split directory.
    force_nonoverlap: bool = False


# Sensor name mappings - canonical name -> dataset-specific patterns
SENSOR_MAPPINGS = {
    'accelerometer': ['accelerometer', 'Accelerometer', 'acc', 'accel', 'acc1', 'acc2'],
    'gyroscope': ['gyroscope', 'Gyroscope', 'gyro'],
    'magnetometer': ['magnetometer', 'mag'],
}

# ============================================================================
# Label Encoders for Non-Integer Label Datasets
# ============================================================================
# Maps full raw string label vocabularies to contiguous integers [0, num_classes-1]
# for CrossEntropyLoss. Datasets with TOP_K_LABELS enabled may use only a subset
# of these labels by default at runtime.

LABEL_ENCODERS = {
    'wisdm': {
        'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7,
        'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'O': 13, 'P': 14, 'Q': 15,
        'R': 16, 'S': 17,
    },
    'har70plus': {
        # Full raw HAR70Plus label vocabulary.
        # User-approved audited setup keeps raw labels 1/7/6/8/3 as active
        # classes 0..4 and explicitly drops raw labels 4/5 via
        # UNWANTED_LABELS['har70plus'].
        '1': 0, '7': 1, '6': 2, '8': 3, '3': 4,
    },
    'harth': {
        # Kept HARTH activities after audit: drop 14/130/140 via UNWANTED_LABELS.
        '1': 0,   # walking
        '2': 1,   # running
        '3': 2,   # shuffling
        '4': 3,   # stairs up
        '5': 4,   # stairs down
        '6': 5,   # standing
        '7': 6,   # sitting
        '8': 7,   # lying
        '13': 8,  # cycling (sit)
    },
    'PAMAP2_Dataset': {
        # Audited PAMAP2 setup: keep all documented activities and drop raw 0
        # via UNWANTED_LABELS as the null/transient token.
        '1': 0,    # lying
        '2': 1,    # sitting
        '3': 2,    # standing
        '4': 3,    # walking
        '5': 4,    # running
        '6': 5,    # cycling
        '7': 6,    # nordic walking
        '12': 7,   # ascending stairs
        '13': 8,   # descending stairs
        '16': 9,   # vacuum cleaning
        '17': 10,  # ironing
        '24': 11,  # rope jumping
    },
    # 'capture24': {
    #     '7030 sleeping': 0, 'home activity': 1, 'transportation': 2, 'leisure': 3, 'mixed-activity': 4, '': -1,
    # },
    'capture24_willetts': {
        'sleep': 0,
        'sitting': 1,
        'standing': 2,
        'walking': 3,
        'vehicle': 4,
        'bicycling': 5,
        'sports': 6,
        'manual-work': 7,
        'household-chores': 8,
        'mixed-activity': 9,
        '': -1,
    },
    'HHAR': {
        # Activity labels: stand, sit, stairsdown, stairsup, bike, walk
        '0': 0,   # stand
        '1': 1,   # sit
        '2': 2,   # stairsdown
        '3': 3,   # stairsup
        '4': 4,   # bike
        '5': 5,   # walk
    },
    'wear_dataset': {
        'bench-dips': 0, 'burpees': 1, 'jogging': 2, 'jogging (butt-kicks)': 3,
        'jogging (rotating arms)': 4, 'jogging (sidesteps)': 5, 'jogging (skipping)': 6,
        'lunges': 7, 'lunges (complex)': 8, 'push-ups': 9, 'push-ups (complex)': 10,
        'sit-ups': 11, 'sit-ups (complex)': 12, 'stretching (hamstrings)': 13,
        'stretching (lumbar rotation)': 14, 'stretching (lunging)': 15,
        'stretching (shoulders)': 16, 'stretching (triceps)': 17,
    },
    'Recofit': {
        # Audited Recofit setup: drop explicit non-activity/instrumentation labels via
        # UNWANTED_LABELS, then keep the 22 exercise labels with >2000 majority-vote
        # 30 s windows at 20 Hz / 1 s stride. All other exercise labels are intentionally
        # left out of the encoder and map to -1.
        'Walk': 0,
        'Static stretch': 1,
        'Running (treadmill)': 2,
        'Elliptical machine': 3,
        'Static Stretch (at your own pace)': 4,
        'Rowing machine': 5,
        'Dynamic Stretch (at your own pace)': 6,
        'Jump Rope': 7,
        'Plank': 8,
        'Butterfly Sit-up': 9,
        'Lunge (alternating both legs, weight optional)': 10,
        'Wall Squat': 11,
        'Squat (arms in front of body, parallel to ground)': 12,
        'Burpee': 13,
        'Triceps Kickback (knee on bench) (label spans both arms)': 14,
        'Two-arm Dumbbell Curl (both arms, not alternating)': 15,
        'Dumbbell Row (knee on bench) (label spans both arms)': 16,
        'Sit-up (hands positioned behind head)': 17,
        'V-up': 18,
        'Sit-ups': 19,
        'Dumbbell Squat (hands at side)': 20,
        'Russian Twist': 21,
    },
    'SamosaTrainingDataset': {
        'All_Other': 0, 'Bathroom_Brushing_hair': 1, 'Bathroom_Hair_dryer_in_use': 2,
        'Bathroom_Shaver_in_use': 3, 'Bathroom_Toilet_flushing': 4,
        'Bathroom_Toothbrushing': 5, 'Bathroom_Washing_hands': 6,
        'Kitchen_Blender_in_use': 7, 'Kitchen_Chopping': 8, 'Kitchen_Grating': 9,
        'Kitchen_Microwave': 10, 'Kitchen_Pouring_pitcher': 11, 'Kitchen_Twisting_jar': 12,
        'Kitchen_Washing_Utensils': 13, 'Kitchen_Wiping_with_rag': 14, 'Misc_Alarm_clock': 15,
        'Misc_Clapping': 16, 'Misc_Coughing': 17, 'Misc_Drinking': 18, 'Misc_Knocking': 19,
        'Misc_Laughing': 20, 'Misc_Scratching': 21, 'Other_Other': 22,
        'Workshop_Drill in use': 23, 'Workshop_Hammering': 24, 'Workshop_Sanding': 25,
        'Workshop_Screwing': 26, 'Workshop_Vacuum in use': 27,
    },
    'OpportunityUCIDataset': {
        # Audited Opportunity locomotion setup:
        # keep the four documented locomotion classes and map null/missing to -1.
        '1': 0,   # stand
        '2': 1,   # walk
        '4': 2,   # sit
        '5': 3,   # lie
    },
    'MHEALTHDATASET': {
        # Audited MHEALTH setup: keep the 12 documented activities and treat
        # rebuilt `-1` rows as the only null/unwanted token.
        '1': 0,   # standing
        '2': 1,   # sitting
        '3': 2,   # lying down
        '4': 3,   # walking
        '5': 4,   # climbing stairs
        '6': 5,   # waist bends forward
        '7': 6,   # frontal elevation of arms
        '8': 7,   # knees bending
        '9': 8,   # cycling
        '10': 9,  # jogging
        '11': 10,  # running
        '12': 11,  # jump front & back
    },
}

# Create reverse mappings (idx -> label) for each dataset
LABEL_DECODERS = {
    dataset: {v: k for k, v in encoder.items()}
    for dataset, encoder in LABEL_ENCODERS.items()
}

# Corrupted or problematic files to exclude from loading
# Format: dataset_name -> list of filenames (exact match)
# These files will be automatically excluded during sample discovery
# Add files here that have NaN values, corrupted data, or other issues
CORRUPTED_FILES = {
    'wear_dataset': [
        'WEAR_S10_left_arm_acc_20Hz_3985.00s.parquet',  # Raw left-arm stream has a long NaN outage
    ],
    'daphnet_fog': [],
    'FoGTurning': [],
    'OdayFoG': [],
    'capture24': [],
    'capture24_willetts': [],
    'har70plus': [],
    'harth': [],
    'HHAR': [],
    'MHEALTHDATASET': [],
    'OpportunityUCIDataset': [],
    'PAMAP2_Dataset': [],
    'wisdm': [],
    'Recofit': [],
    # Add more datasets and files as needed
}

# Unwanted labels for each dataset - these get mapped to -1 and windows with
# majority -1 are dropped during iteration. Raw label values (as strings).
UNWANTED_LABELS = {
    'HHAR': ['-1'],  # null/transition (no activity)
    'OpportunityUCIDataset': ['-1'],  # audited null/missing token after rebuild
    'MHEALTHDATASET': ['-1'],  # audited null/unlabeled token after rebuild
    'daphnet_fog': [],
    'FoGTurning': [],
    'OdayFoG': [],
    'capture24': [''],
    'capture24_willetts': [''],
    'har70plus': ['4', '5'],  # user-approved rare-class drop list
    'harth': ['14', '130', '140'],  # user-approved drop list for rare cycling-standing/inactive classes
    'wisdm': [],  # keep all
    'wear_dataset': ['nan'],  # unlabeled
    'Recofit': ['Non-Exercise', 'Device on Table', '<Initial Activity>', 'Invalid',
                'Note', 'Tap IMU Device', 'Arm Band Adjustment',
                'Repetitive Stretching'],  # non-activity/instruction-style drop list
    'SamosaTrainingDataset': ['All_Other', 'Other_Other'],  # catch-all categories
    'PAMAP2_Dataset': ['0'],  # null/transient
    'USC-HAD': [],  # keep all
}

TOP_K_LABELS = {
    'daphnet_fog': None,       # binary (freeze of gait detection)
    'FoGTurning': None,        # binary (freeze of gait detection)
    'OdayFoG': None,           # binary (freeze of gait detection)
    'har70plus': None,      # explicit encoder + UNWANTED_LABELS handles the audited 5-class setup
    'harth': None,          # audited as a fixed 9-class dataset after dropping 14/130/140
    'HHAR': None,              # 6 classes
    'MHEALTHDATASET': None,    # 12 classes
    'OpportunityUCIDataset': None,  # audited fixed 4-class locomotion setup
    'PAMAP2_Dataset': None,    # 12 classes
    'Recofit': None,           # audited fixed 22-class exercise setup after dropping non-activity labels
    'SamosaTrainingDataset': 10,  # 28 classes, keep top 10
    'USC-HAD': None,           # 12 classes
    'wear_dataset': None,        # audited 18-class setup; drop only unwanted 'nan'
    'wisdm': None,             # 18 classes
}

PRIORITY_LABELS = {
    'daphnet_fog': [('1', 0.2)],  # Freezing of gait
    'FoGTurning': [('1', 0.2)],  # Freezing of gait
    'OdayFoG': [('1', 0.2)],  # Freezing of gait
}

# Label frequencies for each dataset (sorted by count descending)
# Used for top_k_labels feature
# Note: Integer labels are stored as strings for consistency
LABEL_FREQUENCIES = {
    # String-labeled datasets
    'wisdm': ['H', 'I', 'K', 'Q', 'E', 'R', 'D', 'G', 'S', 'F', 'A', 'J', 'L', 'C', 'M', 'P', 'O', 'B'],
    'har70plus': ['1', '7', '6', '8', '3', '5', '4'],  # raw prevalence order; 4/5 are explicitly unwanted
    # Integer-labeled datasets (stored as strings)
    'daphnet_fog': ['0', '1'],
    'FoGTurning': ['1', '0'],
    'OdayFoG': ['1', '0'],
    'HHAR': ['5', '3', '4', '2', '0', '1', '-1'],  # walk, stairsup, bike, stairsdown, stand, sit, null
    'MHEALTHDATASET': ['-1', '11', '1', '2', '5', '9', '10', '3', '4', '7', '8', '6', '12'],
    'harth': ['7', '1', '6', '8', '13', '2', '3', '4', '5'],
    'PAMAP2_Dataset': ['0', '4', '17', '1', '3', '7', '2', '16', '6', '12', '13', '5', '24'],
    'OpportunityUCIDataset': ['1', '2', '4', '5'],
    'wear_dataset': ['nan', 'jogging', 'jogging (sidesteps)', 'stretching (lunging)', 'lunges (complex)',
                     'sit-ups (complex)', 'sit-ups', 'lunges', 'burpees', 'stretching (triceps)',
                     'stretching (lumbar rotation)', 'push-ups (complex)', 'stretching (hamstrings)',
                     'jogging (skipping)', 'stretching (shoulders)', 'jogging (rotating arms)',
                     'push-ups', 'jogging (butt-kicks)', 'bench-dips'],
    'Recofit': ['Walk', 'Static stretch', 'Running (treadmill)', 'Elliptical machine',
                'Static Stretch (at your own pace)', 'Rowing machine',
                'Dynamic Stretch (at your own pace)', 'Jump Rope', 'Plank',
                'Butterfly Sit-up', 'Lunge (alternating both legs, weight optional)',
                'Wall Squat', 'Squat (arms in front of body, parallel to ground)', 'Burpee',
                'Triceps Kickback (knee on bench) (label spans both arms)',
                'Two-arm Dumbbell Curl (both arms, not alternating)',
                'Dumbbell Row (knee on bench) (label spans both arms)',
                'Sit-up (hands positioned behind head)', 'V-up', 'Sit-ups',
                'Dumbbell Squat (hands at side)', 'Russian Twist'],
    'SamosaTrainingDataset': ['Kitchen_Microwave', 'Other_Other', 'Misc_Alarm_clock', 'All_Other',
                              'Kitchen_Wiping_with_rag', 'Workshop_Vacuum in use', 'Bathroom_Hair_dryer_in_use',
                              'Bathroom_Toothbrushing', 'Bathroom_Shaver_in_use', 'Bathroom_Washing_hands',
                              'Kitchen_Washing_Utensils', 'Bathroom_Brushing_hair', 'Workshop_Screwing',
                              'Kitchen_Chopping', 'Kitchen_Grating', 'Workshop_Sanding', 'Kitchen_Blender_in_use',
                              'Workshop_Hammering', 'Misc_Scratching', 'Misc_Clapping', 'Misc_Laughing',
                              'Kitchen_Pouring_pitcher', 'Misc_Coughing', 'Bathroom_Toilet_flushing',
                              'Misc_Drinking', 'Workshop_Drill in use', 'Kitchen_Twisting_jar', 'Misc_Knocking'],
}

# Dataset configurations
DATASET_CONFIGS = {
    'daphnet_fog': DatasetConfig(
        name='DaphnetFOG',
        format_type='standard',
        has_separate_labels=True,
        available_sensors=['accelerometer'],
        sensor_patterns={'accelerometer': []},  # No explicit sensor in filename
        default_sensor='accelerometer',
        placement_patterns=['ankle', 'thigh', 'trunk'],
        default_placement='ankle',
    ),
    'FoGTurning': DatasetConfig(
        name='FoGTurning',
        format_type='standard',
        has_separate_labels=True,
        available_sensors=['accelerometer', 'gyroscope'],
        sensor_patterns={
            'accelerometer': ['acc'],
            'gyroscope': ['gyro'],
        },
        default_sensor='accelerometer',
        placement_patterns=['shank'],
        default_placement='shank',
    ),
    'OdayFoG': DatasetConfig(
        name='OdayFoG',
        format_type='standard',
        has_separate_labels=True,
        available_sensors=['accelerometer', 'gyroscope'],
        sensor_patterns={
            'accelerometer': ['acc'],
            'gyroscope': ['gyro'],
        },
        default_sensor='accelerometer',
        placement_patterns=['ankle_l', 'ankle_r', 'foot_l', 'foot_r', 'wrist_l', 'wrist_r', 'thigh_l', 'thigh_r', 'chest', 'head', 'lumbar'],
        default_placement='ankle_l',
    ),
    'capture24': DatasetConfig(
        name='Capture24',
        format_type='standard',
        has_separate_labels=True,
        available_sensors=['accelerometer'],
        sensor_patterns={'accelerometer': []},  # No explicit sensor in filename
        default_sensor='accelerometer',
        placement_patterns=['wrist'],
        force_nonoverlap=True,
    ),
    'capture24_willetts': DatasetConfig(
        name='Capture24',
        format_type='standard',
        has_separate_labels=True,
        available_sensors=['accelerometer'],
        sensor_patterns={'accelerometer': []},  # Reuses Capture24 signal filenames.
        default_sensor='accelerometer',
        placement_patterns=['wrist'],
        force_nonoverlap=True,
    ),
    'har70plus': DatasetConfig(
        name='HAR70Plus',
        format_type='standard',
        has_separate_labels=True,
        available_sensors=['accelerometer'],
        sensor_patterns={'accelerometer': []},
        default_sensor='accelerometer',
        placement_patterns=['back', 'thigh'],
        default_placement='thigh',
    ),
    'harth': DatasetConfig(
        name='HARTH',
        format_type='standard',
        has_separate_labels=True,
        available_sensors=['accelerometer'],
        sensor_patterns={'accelerometer': []},
        default_sensor='accelerometer',
        placement_patterns=['back', 'thigh'],
        default_placement='thigh',
    ),
    'HHAR': DatasetConfig(
        name='HHAR',
        format_type='standard',
        has_separate_labels=True,
        available_sensors=['accelerometer', 'gyroscope'],
        sensor_patterns={
            'accelerometer': ['accelerometer'],
            'gyroscope': ['gyroscope'],
        },
        label_postfix_pattern=r'_labels\.parquet$',
        placement_patterns=['phones', 'watch'],
        default_placement='watch',
    ),
    'MHEALTHDATASET': DatasetConfig(
        name='MHEALTH',
        format_type='standard',
        has_separate_labels=True,
        available_sensors=['accelerometer', 'gyroscope', 'magnetometer'],
        sensor_patterns={
            'accelerometer': ['acc'],
            'gyroscope': ['gyro'],
            'magnetometer': ['mag'],
        },
        placement_patterns=['chest', 'lankle', 'rarm'],
        default_placement='rarm',
    ),
    'OpportunityUCIDataset': DatasetConfig(
        name='Opportunity',
        format_type='standard',
        has_separate_labels=True,
        available_sensors=['accelerometer'],
        sensor_patterns={'accelerometer': ['acc']},
        placement_patterns=['BACK', 'HIP', 'LH', 'LUA', 'LWR', 'RH', 'RKN', 'RUA', 'RWR'],
        default_placement=['lwr', 'rwr'],
    ),
    'PAMAP2_Dataset': DatasetConfig(
        name='PAMAP2',
        format_type='standard',
        has_separate_labels=True,
        available_sensors=['accelerometer', 'gyroscope', 'magnetometer'],
        sensor_patterns={
            'accelerometer': ['acc1', 'acc2'],
            'gyroscope': ['gyro'],
            'magnetometer': ['mag'],
        },
        placement_patterns=['ankle', 'chest', 'hand'],
        default_placement='hand',
    ),
    'WearGait-PD': DatasetConfig(
        name='WearGaitPD',
        format_type='standard',
        has_separate_labels=True,
        available_sensors=['accelerometer'],
        sensor_patterns={
            'accelerometer': ['acc'],
        },
        default_sensor='accelerometer',
        placement_patterns=['l-wrist', 'r-wrist'],
        default_placement=['l-wrist', 'r-wrist'],
    ),
    'Recofit': DatasetConfig(
        name='Recofit',
        format_type='embedded_label',
        has_separate_labels=False,
        available_sensors=['accelerometer', 'gyroscope'],
        sensor_patterns={
            'accelerometer': ['Accelerometer'],
            'gyroscope': ['Gyroscope'],
        },
        placement_patterns=['rightarm'],
        default_placement='rightarm',
    ),
    # 'SamosaTrainingDataset': DatasetConfig(
    #     name='SAMoSA',
    #     format_type='samosa_imu',
    #     has_separate_labels=False,  # Label is in filename
    #     available_sensors=['accelerometer', 'gyroscope', 'magnetometer'],
    #     sensor_patterns={
    #         'accelerometer': ['imu'],  # SAMoSA uses 'imu' which contains all sensors
    #         'gyroscope': ['imu'],
    #         'magnetometer': ['imu'],
    #     },
    #     default_sensor='accelerometer',  # Default to accelerometer for matching
    #     label_type='single',  # Activity is from filename
    # ),
    # 'USC-HAD': DatasetConfig(
    #     name='USCHAD',
    #     format_type='embedded_label',
    #     has_separate_labels=False,
    #     available_sensors=['accelerometer', 'gyroscope'],
    #     sensor_patterns={
    #         'accelerometer': ['Accelerometer'],
    #         'gyroscope': ['Gyroscope'],
    #     },
    #     placement_patterns=['waist'],
    # ),
    'wear_dataset': DatasetConfig(
        name='WEAR',
        format_type='standard',
        has_separate_labels=True,
        available_sensors=['accelerometer'],
        sensor_patterns={'accelerometer': []},
        default_sensor='accelerometer',
        placement_patterns=['left_arm', 'right_arm', 'left_leg', 'right_leg'],
        default_placement='right_arm',
    ),
    'wisdm': DatasetConfig(
        name='WISDM',
        format_type='standard',
        has_separate_labels=True,
        available_sensors=['accelerometer', 'gyroscope'],
        sensor_patterns={
            'accelerometer': ['accel'],
            'gyroscope': ['gyro'],
        },
        label_postfix_pattern=r'_labels\.parquet$',
        placement_patterns=['watch'],
        default_placement='watch',
    ),
}


# ============================================================================
# File Parsing Utilities
# ============================================================================

def parse_duration_from_filename(filename: str, dataset_name: str = None) -> Optional[float]:
    """Extract duration in seconds from filename."""
    # Standard pattern: _XXXXs.parquet or _XXXX.XXs.parquet
    match = re.search(r'_(\d+\.?\d*)s\.parquet$', filename)
    if match:
        return float(match.group(1))
    
    # USC-HAD and Recofit: duration is in the filename but without 's' suffix
    # Format: USCHAD_Accelerometer_waist_{duration}_{subject}.parquet
    # Format: Recofit_Accelerometer_rightarm_{duration}_{subject}.parquet
    if dataset_name in ['USC-HAD', 'Recofit']:
        parts = filename.replace('.parquet', '').split('_')
        # Duration is the 4th element (0-indexed: 3)
        if len(parts) >= 5:
            try:
                return float(parts[3])
            except ValueError:
                pass
    
    return None


def parse_subject_id(filename: str, dataset_name: str) -> Optional[str]:
    """Extract subject ID from filename based on dataset format."""
    config = DATASET_CONFIGS.get(dataset_name)
    if not config:
        return None
    
    basename = os.path.basename(filename).replace('.parquet', '')
    
    if dataset_name in ['USC-HAD', 'Recofit']:
        # Format: {dataset}_{sensor}_{placement}_{duration}_{subject_id}
        # e.g., USCHAD_Accelerometer_waist_10_s004
        match = re.search(r'_s(\d+v?\d*)', basename)
        if match:
            return match.group(1)
    elif dataset_name == 'SamosaTrainingDataset':
        # Format: SAMoSA_P007_...
        match = re.search(r'_P(\d+)_', basename)
        if match:
            return f'P{match.group(1)}'
    elif dataset_name == 'OpportunityUCIDataset':
        # Format: Opportunity_S1-ADL1_BACK_acc_20Hz_1703.85s.parquet
        # Session token is Sx-ADLy / Sx-Drill. For subject-grouped splits we want the subject (Sx).
        match = re.search(r'_(S\d+)-', basename)
        if match:
            return match.group(1)
    elif dataset_name == 'MHEALTHDATASET':
        # Format: MHEALTH_mHealth_subject10_chest_acc_20Hz_1966.05s.parquet
        # We want group-by-subject to use the per-person identifier (e.g., subject10).
        match = re.search(r'_(subject\d+)_', basename)
        if match:
            return match.group(1)
    else:
        # Standard format: {dataset}_{subject_id}_...
        parts = basename.split('_')
        if len(parts) >= 2:
            return parts[1]
    
    return None


def get_sensor_type_from_filename(filename: str, dataset_name: str) -> Optional[str]:
    """Extract sensor type from filename."""
    config = DATASET_CONFIGS.get(dataset_name)
    if not config:
        return None
    
    # Special case for SAMoSA - imu files contain all sensors
    if dataset_name == 'SamosaTrainingDataset':
        if '_imu_' in filename.lower():
            return 'accelerometer'  # Return accelerometer as the primary sensor
        return None
    
    # If dataset has default sensor (accelerometer-only), return that
    if config.default_sensor and not config.sensor_patterns.get(config.default_sensor):
        return config.default_sensor
    
    filename_lower = filename.lower()
    
    for canonical_name, patterns in config.sensor_patterns.items():
        for pattern in patterns:
            if pattern.lower() in filename_lower:
                return canonical_name
    
    return config.default_sensor


def get_placement_from_filename(filename: str, dataset_name: str) -> Optional[str]:
    """Extract body placement from filename based on dataset config.
    
    Returns the matched placement pattern or None if no match found.
    """
    config = DATASET_CONFIGS.get(dataset_name)
    if not config or not config.placement_patterns:
        return None
    
    filename_lower = filename.lower()
    
    for placement in config.placement_patterns:
        if placement.lower() in filename_lower:
            return placement.lower()
    
    return None


def get_label_file_for_data_file(data_file: str, dataset_name: str, processed_dir: str) -> Optional[str]:
    """Find the corresponding label file for a data file."""
    config = DATASET_CONFIGS.get(dataset_name)
    if not config or not config.has_separate_labels:
        return None
    
    basename = os.path.basename(data_file)
    subject_id = parse_subject_id(data_file, dataset_name)
    
    if dataset_name == 'wisdm':
        # WISDM: label file has same name + _labels
        label_name = basename.replace('.parquet', '_labels.parquet')
        label_path = os.path.join(processed_dir, label_name)
        if os.path.exists(label_path):
            return label_path
    elif dataset_name == 'WearGait-PD':
        # WearGait-PD: label file has same name + _labels
        label_name = basename.replace('.parquet', '_labels.parquet')
        label_path = os.path.join(processed_dir, label_name)
        if os.path.exists(label_path):
            return label_path
    elif dataset_name == 'HHAR' or dataset_name == "FoGTurning":
        # HHAR: label file has same name + _labels
        label_name = basename.replace('.parquet', '_labels.parquet')
        label_path = os.path.join(processed_dir, label_name)
        if os.path.exists(label_path):
            return label_path
    elif dataset_name == "OdayFoG":
        # OdayFoG: label file is {stem}_labels.parquet (stem excludes placement/sensor)
        # Data file: OdayFoG_subject1_v24_t1_ankle_l_acc_20Hz_83.92s.parquet
        # Label file: OdayFoG_subject1_v24_t1_labels.parquet
        placement = get_placement_from_filename(basename, dataset_name)
        if placement:
            split_token = f'_{placement}'
            if split_token in basename:
                stem = basename.split(split_token)[0]
                label_name = f'{stem}_labels.parquet'
                label_path = os.path.join(processed_dir, label_name)
                if os.path.exists(label_path):
                    return label_path
    elif dataset_name == 'MHEALTHDATASET':
        # MHEALTH: label file is MHEALTH_mHealth_{subject}_labels.parquet
        # Signal file: MHEALTH_mHealth_subject10_chest_acc_20Hz_1966.05s.parquet
        # Extract subject identifier (e.g., subject10)
        match = re.search(r'_(subject\d+)_', basename)
        if match:
            subject = match.group(1)
            label_name = f'MHEALTH_mHealth_{subject}_labels.parquet'
            label_path = os.path.join(processed_dir, label_name)
            if os.path.exists(label_path):
                return label_path
    elif dataset_name == 'OpportunityUCIDataset':
        # Opportunity: label file is Opportunity_{session}_labels.parquet
        # Signal file: Opportunity_S1-ADL1_BACK_acc_20Hz_1703.85s.parquet
        # Extract session identifier (e.g., S1-ADL1 or S3-Drill)
        match = re.search(r'_(S\d+-[A-Za-z]+\d*)_', basename)
        if match:
            session = match.group(1)
            label_name = f'Opportunity_{session}_labels.parquet'
            label_path = os.path.join(processed_dir, label_name)
            if os.path.exists(label_path):
                return label_path
    elif dataset_name == 'har70plus':
        # Strict-boundary HAR70Plus rebuild emits one label parquet per subject segment:
        #   signal: HAR70Plus_501_seg000_thigh_acc_20Hz_123.45s.parquet
        #   label:  HAR70Plus_501_seg000_labels.parquet
        match = re.match(r'^(HAR70Plus_\d+_seg\d+)_', basename)
        if match:
            label_name = f'{match.group(1)}_labels.parquet'
            label_path = os.path.join(processed_dir, label_name)
            if os.path.exists(label_path):
                return label_path
        if subject_id:
            pattern = os.path.join(processed_dir, f'{config.name}_{subject_id}_labels*.parquet')
            matches = glob.glob(pattern)
            if matches:
                return matches[0]
    else:
        # Standard: {dataset_name}_{subject_id}_labels.parquet
        if subject_id:
            # Search for matching label file
            pattern = os.path.join(processed_dir, f'{config.name}_{subject_id}_labels*.parquet')
            matches = glob.glob(pattern)
            if matches:
                return matches[0]
    
    return None

def normalize_label_to_string(label) -> str:
    """Convert any label type to consistent string for lookup.
    
    Handles: int, float, np.int*, np.float*, str
    Always returns string representation that matches LABEL_FREQUENCIES format.
    """
    if pd.isna(label):
        return 'nan'
    
    if isinstance(label, (int, np.integer)):
        return str(int(label))
    elif isinstance(label, (float, np.floating)):
        # Check if it's actually an integer value
        if label == int(label):
            return str(int(label))
        return str(label)
    elif isinstance(label, str):
        # Try to parse and normalize numeric strings
        try:
            val = float(label)
            if val == int(val):
                return str(int(val))
            return label
        except ValueError:
            return label
    else:
        return str(label)

# ============================================================================
# Main Dataset Class
# ============================================================================

class MotionDataset(Dataset):
    """
    Unified Motion Dataset for downstream tasks.
    
    Modes:
        - 'nonoverlap': Non-overlapping windows (stride = window_size). Use for evaluation.
        - 'overlap': Overlapping windows (stride = 1). Use for training with shuffle=True.
          Naturally duration-weighted since longer recordings have more valid positions.
    
    Label Encoding:
        For datasets with string labels (wisdm, har70plus, wear_dataset, Recofit, 
        SamosaTrainingDataset), labels are automatically encoded to integers [0, num_classes-1]
        for compatibility with CrossEntropyLoss. Access the mappings via:
        - self.label_to_idx: dict mapping label string -> integer index
        - self.idx_to_label: dict mapping integer index -> label string
        - self.num_classes: number of classes
    
    Preloading:
        Set preload=True to load all recordings into memory at initialization.
        This significantly speeds up training by avoiding disk I/O on every sample.
        Memory usage scales with dataset size (check with get_memory_usage()).
    
    Args:
        data_root: Root directory containing all datasets
        dataset_name: Name of the dataset (must be in DATASET_CONFIGS)
        sampling_rate: Target sampling rate (must be <= 20 Hz)
        window_size: Window size in number of samples at target sampling rate
        sensor_types: List of sensor types to load ['accelerometer', 'gyroscope', 'magnetometer']
        axial_mode: 'triaxial' for (x,y,z) or 'uniaxial' for magnitude
        placement: Optional filter for body placement (e.g., 'wrist', 'ankle')
        label_column: Name of the label column to use (for multi-label datasets)
        top_k_labels: If set, only keep the top K most frequent labels and map all 
                      others (including 'nan') to an 'other' class. This reduces 
                      num_classes to K+1 (K main classes + 1 'other' class).
        preload: If True, load all recordings into memory at initialization.
                 Dramatically speeds up training. Default False.
        mode: 'nonoverlap' for evaluation (non-overlapping windows), 
              'overlap' for training (stride=1, naturally duration-weighted).
        return_majority_label: If True, return a single majority label per window 
                               instead of dense labels. Useful for classification 
                               tasks with CrossEntropyLoss. Default False.
    """
    
    def __init__(
        self,
        data_root: str,
        dataset_name: str,
        sampling_rate: float = 20.0,
        window_size: int = 200,
        sensor_types: Union[str, List[str]] = 'accelerometer',
        axial_mode: Literal['triaxial', 'uniaxial'] = 'triaxial',
        placement: Optional[Union[str, List[str]]] = None,
        label_column: Optional[str] = None,
        top_k_labels: Optional[int] = None,
        preload: bool = False,
        labels_only: bool = False,
        mode: Literal['overlap', 'nonoverlap'] = 'nonoverlap',
        overlap_stride_samples: int = 1,
        return_majority_label: bool = True,
        max_unwanted_frac: float = 0.5,
        max_files_per_dataset: Optional[int] = None,  # NEW
        max_windows: Optional[int] = None,  # NEW
        max_windows_per_class: Optional[int] = None,  # NEW
        subsample_ratio: Optional[float] = None,  # NEW
        lazy_cache: bool = True,
    ):
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.axial_mode = axial_mode
        # Placement filtering is used for dataset window indexing.
        # If placement is not provided, default to the dataset's configured default placement
        # (so frozen split JSONs generated for default placement remain consistent).
        self.placement = placement
        self.label_column = label_column
        self.preload = preload
        # When True (default), each worker will memoize a recording the first time
        # it reads it. Set False to fall back to per-window disk reads (saves RAM
        # at the cost of large I/O amplification on overlapping-window modes).
        self.lazy_cache = bool(lazy_cache)
        self.labels_only = bool(labels_only)
        self.mode = mode
        try:
            overlap_stride_samples = int(overlap_stride_samples)
        except Exception as e:
            raise ValueError(
                f"overlap_stride_samples must be an int, got {overlap_stride_samples!r}: {e}"
            )
        if overlap_stride_samples < 1:
            raise ValueError(f"overlap_stride_samples must be >= 1, got {overlap_stride_samples}")
        self.overlap_stride_samples = overlap_stride_samples
        self.return_majority_label = return_majority_label
        self.max_unwanted_frac = float(max_unwanted_frac)
        self.max_files_per_dataset = max_files_per_dataset
        self.max_windows = max_windows
        self.max_windows_per_class = max_windows_per_class
        self.subsample_ratio = subsample_ratio
        
        # Store priority labels for this dataset
        self.priority_labels = PRIORITY_LABELS.get(dataset_name, [])
        # Allow callers (e.g. eval/baselines) to override top_k_labels.
        self.top_k_labels = TOP_K_LABELS.get(dataset_name, None) if top_k_labels is None else int(top_k_labels)

        if not (0.0 <= self.max_unwanted_frac <= 1.0):
            raise ValueError(f"max_unwanted_frac must be in [0,1], got {self.max_unwanted_frac}")

        # Validate sampling rate
        if sampling_rate > 20.0:
            raise ValueError(f"Sampling rate must be <= 20 Hz, got {sampling_rate}")
        
        # Normalize sensor_types to list
        if isinstance(sensor_types, str):
            self.sensor_types = [sensor_types]
        else:
            self.sensor_types = list(sensor_types)
        
        # Get dataset config
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")
        self.config = DATASET_CONFIGS[dataset_name]
        if self.config.force_nonoverlap and self.mode != "nonoverlap":
            print(f"  {dataset_name}: forcing mode=nonoverlap (overlap indexing disabled for this dataset)")
            self.mode = "nonoverlap"

        # Apply default placement if caller didn't specify one.
        # Supports either a single placement string or a list of placements.
        if self.placement is None and self.config.default_placement is not None:
            self.placement = self.config.default_placement

        # Normalize placement filter into a list of lowercase tags for fast membership checks.
        self._placement_tags: Optional[List[str]]
        if self.placement is None:
            self._placement_tags = None
        elif isinstance(self.placement, str):
            tag = self.placement.strip().lower()
            self._placement_tags = [tag] if tag else None
        else:
            tags: List[str] = []
            for p in list(self.placement):
                s = str(p).strip().lower()
                if s:
                    tags.append(s)
            self._placement_tags = sorted(set(tags)) if tags else None
        
        # Validate sensor types
        for st in self.sensor_types:
            if st not in self.config.available_sensors:
                raise ValueError(
                    f"Sensor type '{st}' not available in {dataset_name}. "
                    f"Available: {self.config.available_sensors}"
                )
        
        # Set up paths
        # Allow data_root to be a comma-separated list of candidate roots (mirrors inertia1 patient dataset behavior).
        data_root_str = str(data_root)
        candidate_roots = [r.strip() for r in data_root_str.split(",") if r.strip()]
        if not candidate_roots:
            candidate_roots = [data_root_str]

        chosen_root: Optional[str] = None
        processed_dir: Optional[str] = None
        tried: List[str] = []
        for root in candidate_roots:
            pdir = os.path.join(root, dataset_name, "processed")
            tried.append(pdir)
            if os.path.exists(pdir):
                chosen_root = root
                processed_dir = pdir
                break

        if processed_dir is None:
            tried_msg = "\n".join(f"  - {p}" for p in tried)
            raise ValueError(
                "Processed directory not found. Tried:\n" + tried_msg
            )

        self.data_root = chosen_root if chosen_root is not None else data_root_str
        self.processed_dir = processed_dir
        
        # Discover samples (but don't build index yet)
        self.samples = self._discover_samples()
        
        # Compute number of channels
        self.num_channels = self._compute_num_channels()
        
        # Set up label encoding BEFORE building sample index
        # (needed for filtering unwanted labels)
        if self.top_k_labels is not None and dataset_name in LABEL_FREQUENCIES:
            # Use top K labels + 'other' class (works for both string and integer labels)
            self._setup_top_k_encoding(self.top_k_labels)
        elif dataset_name in LABEL_ENCODERS:
            # String-labeled dataset: use full label encoding
            self.label_to_idx = LABEL_ENCODERS[dataset_name].copy()
            self.idx_to_label = LABEL_DECODERS[dataset_name].copy()
            self.num_classes = max(v for v in self.label_to_idx.values() if v >= 0) + 1
        else:
            # Integer-labeled dataset without top_k: no encoding needed
            self.label_to_idx = None
            self.idx_to_label = None
            self.num_classes = None  # Will need to be inferred from data
        
        # Initialize caches BEFORE building index.
        # (index building may use caches for label filtering)
        self._sample_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._label_cache: Dict[int, np.ndarray] = {}
        
        # Calculate number of windows per sample and build index
        # This filters out windows with majority unwanted labels
        self._build_sample_index()
        
        # Preload all samples into memory if requested
        if preload:
            self._preload_all_samples()

    def _get_sample_labels(self, sample_idx: int) -> Optional[np.ndarray]:
        """Load and encode labels for a sample without loading sensor data.

        This is primarily used to speed up frozen split generation for large
        datasets where reading full sensor parquets is expensive.
        """
        if sample_idx in self._label_cache:
            return self._label_cache[sample_idx]

        sample = self.samples[sample_idx]

        # If labels are in a separate file, load just the label parquet.
        if sample.get('label_file') is not None:
            # Fast path: only read the single column we need. For high-rate
            # label files (capture24's annotation parquets are ~2M rows each)
            # skipping the timestamp column is a significant IO+parse win.
            label_path = sample['label_file']
            preferred_cols = []
            if self.label_column:
                preferred_cols.append(self.label_column)
            preferred_cols += ['label', 'activity', 'activityID', 'annotation']

            labels = None
            label_df = None
            for col in preferred_cols:
                try:
                    label_df = pd.read_parquet(label_path, columns=[col])
                    labels = label_df[col].values
                    break
                except (KeyError, ValueError, OSError):
                    continue

            if labels is None:
                # Fallback: read the whole file and pick the first non-time column
                label_df = pd.read_parquet(label_path)
                for col in label_df.columns:
                    if col not in ['timestamp', 'timestamp_s', 'timestamp_sec', 'time_s', 'subject_id']:
                        labels = label_df[col].values
                        break

            if labels is None:
                return None

            if len(labels) == 1 and (isinstance(labels[0], (np.ndarray, list)) or (hasattr(labels[0], '__len__') and not isinstance(labels[0], str))):
                labels = np.asarray(labels[0])
            
            # Handle case where labels are stored as a list/array in a single row (e.g. OdayFoG)
            if len(labels) == 1 and isinstance(labels[0], (list, np.ndarray)):
                labels = np.array(labels[0])

            labels = self._resample_labels(labels)
            labels = self._encode_labels(labels)
            self._label_cache[sample_idx] = labels
            return labels

        # Otherwise fall back to full loading (labels embedded in sensor file).
        _signal, labels = self._get_sample_data(sample_idx)
        self._label_cache[sample_idx] = labels
        return labels
    
    def _setup_top_k_encoding(self, k: int):
        """Set up label encoding using only top K most frequent labels.
        
        FIXED: Properly handles label type normalization.
        """
        freq_list = LABEL_FREQUENCIES[self.dataset_name]
        
        # Get unwanted labels (already strings)
        unwanted = set(UNWANTED_LABELS.get(self.dataset_name, []))
        
        # Filter unwanted from frequency list
        freq_list_filtered = [l for l in freq_list if l not in unwanted]
        
        # Get top K labels
        top_k = freq_list_filtered[:k]
        
        # Debug output
        print(f"\n=== Top-K Label Setup for {self.dataset_name} ===")
        print(f"Top {k} labels selected: {top_k}")
        print(f"Unwanted labels: {unwanted}")
        
        # Create encoding: top K labels get indices 0 to K-1
        self.label_to_idx = {}
        self.idx_to_label = {}
        
        for i, label in enumerate(top_k):
            self.label_to_idx[label] = i
            self.idx_to_label[i] = label
        
        # Store for fast lookup
        self._top_k_set = set(top_k)
        self._unwanted_set = unwanted
        
        self.num_classes = k
        
        print(f"Number of classes: {self.num_classes}")
        print(f"Label encoding: {self.label_to_idx}")
    
    def _preload_all_samples(self):
        """Preload all recordings into memory for faster access during training.
        
        This loads all sample data once and stores it in self._sample_cache.
        Subsequent calls to _get_sample_data will return cached data directly.
        """
        import sys
        
        total_samples = len(self.samples)
        total_bytes = 0
        
        print(f"Preloading {total_samples} recordings into memory...")
        
        for i, sample in enumerate(self.samples):
            signal, labels = self._load_sample_data(sample)
            self._sample_cache[i] = (signal, labels)
            
            # Track memory usage
            total_bytes += signal.nbytes + labels.nbytes
            
            # Progress indicator every 10% or every 50 samples
            if (i + 1) % max(1, total_samples // 10) == 0 or (i + 1) == total_samples:
                mb_used = total_bytes / (1024 * 1024)
                print(f"  Loaded {i + 1}/{total_samples} samples ({mb_used:.1f} MB)")
        
        mb_total = total_bytes / (1024 * 1024)
        print(f"Preload complete: {total_samples} samples, {mb_total:.1f} MB in memory")
    
    def _get_sample_data(self, sample_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get sample data, using cache if available.

        Caching policy:
        - If preload=True, the cache is populated up-front in `_preload_all_samples`.
        - Otherwise we *lazily* cache each recording the first time it's read from
          disk. In overlap modes (e.g. stride=1), a single recording supplies many
          windows; without lazy caching we re-read the same parquet file for every
          window, which dominates training time. Lazy caching lives per-worker
          process (DataLoader fork copies the cache dict), so memory growth is
          bounded by what each worker actually touches.

        Args:
            sample_idx: Index into self.samples list

        Returns:
            signal: numpy array of shape (num_timesteps, num_channels)
            labels: numpy array of shape (num_timesteps,)
        """
        cached = self._sample_cache.get(sample_idx)
        if cached is not None:
            return cached

        sample = self.samples[sample_idx]
        signal, labels = self._load_sample_data(sample)

        # Memoize the loaded recording so subsequent windows from the same recording
        # don't re-read the parquet file. Disabled only when explicitly requested
        # (e.g. extremely large datasets where per-worker caching would OOM).
        if getattr(self, "lazy_cache", True):
            self._sample_cache[sample_idx] = (signal, labels)

        return signal, labels

    def clear_cache(self):
        """Clear the sample cache to free memory."""
        self._sample_cache.clear()

    def get_sample_placement(self, sample_idx: int) -> Optional[str]:
        """Return the body placement/device tag for a discovered sample, if available.

        Placement is inferred during discovery from the filename (see
        get_placement_from_filename) and stored on each per-sensor file record.
        """
        if sample_idx < 0 or sample_idx >= len(self.samples):
            raise IndexError(f"sample_idx {sample_idx} out of range [0, {len(self.samples)})")

        sample = self.samples[sample_idx]
        files = sample.get('files')
        if not isinstance(files, dict):
            return None

        # Placement should be consistent across matched sensors; take the first non-empty.
        for st in self.sensor_types:
            fi = files.get(st)
            if isinstance(fi, dict):
                placement = fi.get('placement')
                if placement:
                    return str(placement).lower()
        return None

    def get_window_placement(self, window_idx: int) -> Optional[str]:
        """Return placement tag for a window index into this dataset."""
        if window_idx < 0 or window_idx >= len(self):
            raise IndexError(f"window_idx {window_idx} out of range [0, {len(self)})")
        if self.mode == 'overlap':
            sample_idx, _start = self.valid_random_starts[window_idx]
        else:
            sample_idx, _window_offset = self.valid_windows[window_idx]
        return self.get_sample_placement(int(sample_idx))
    
    def get_memory_usage(self) -> float:
        """Get current cache memory usage in MB."""
        total_bytes = 0
        for signal, labels in self._sample_cache.values():
            total_bytes += signal.nbytes + labels.nbytes
        return total_bytes / (1024 * 1024)
    
    def _discover_samples(self) -> List[Dict]:
        """Discover all valid samples in the dataset."""
        samples = []
        
        # Deterministic ordering is important because frozen splits store window indices
        # into this dataset instance. If discovery order changes across runs, the same
        # saved indices will point to different windows.
        all_files = sorted([f for f in os.listdir(self.processed_dir) if f.endswith('.parquet')])
        
        if self.config.format_type == 'samosa_imu':
            # SAMoSA: filter for IMU files only
            data_files = [f for f in all_files if '_imu_' in f]
        else:
            # Filter out label files
            data_files = [f for f in all_files if 'labels' not in f.lower()]

        data_files = sorted(data_files)
        
        # Get list of corrupted files to exclude
        corrupted_files = set(CORRUPTED_FILES.get(self.dataset_name, []))
        excluded_count = 0
        
        # Filter out corrupted files
        original_count = len(data_files)
        data_files = [f for f in data_files if f not in corrupted_files]
        excluded_count = original_count - len(data_files)
        
        if excluded_count > 0:
            print(f"  Excluding {excluded_count} known corrupted file(s) from {self.dataset_name}")
        
        # Group files by subject
        subject_files = {}
        for f in data_files:
            subject_id = parse_subject_id(f, self.dataset_name)
            sensor_type = get_sensor_type_from_filename(f, self.dataset_name)

            session_id = None
            if self.dataset_name == 'OpportunityUCIDataset':
                # Example: Opportunity_S1-ADL1_BACK_acc_20Hz_1703.85s.parquet -> S1-ADL1
                m = re.search(r'_(S\d+-[A-Za-z]+\d*)_', f)
                if m:
                    session_id = m.group(1)
            
            if subject_id is None:
                continue
            
            # Filter by placement if specified
            if self._placement_tags is not None:
                f_low = f.lower()
                if not any(tag in f_low for tag in self._placement_tags):
                    continue
            
            # For SAMoSA, the IMU file contains all sensors, so we don't filter by sensor type
            if self.config.format_type == 'samosa_imu':
                sensor_type = 'accelerometer'  # Use as primary key
            elif sensor_type not in self.sensor_types:
                continue
            
            # Get duration
            duration = parse_duration_from_filename(f, self.dataset_name)
            if duration is None:
                # Try to infer from file if embedded label format
                if self.config.format_type == 'embedded_label':
                    try:
                        filepath = os.path.join(self.processed_dir, f)
                        df = pd.read_parquet(filepath)
                        if 'time_s' in df.columns:
                            duration = df['time_s'].max() - df['time_s'].min()
                        else:
                            duration = len(df) / 20.0  # Assume 20Hz
                    except:
                        continue
                else:
                    continue
            
            # Calculate window duration at original 20Hz
            window_duration_s = self.window_size / self.sampling_rate
            
            # Check if sample is long enough
            if duration < window_duration_s:
                continue
            
            # Extract placement for multi-sensor matching
            placement_tag = get_placement_from_filename(f, self.dataset_name)
            
            key = (subject_id, sensor_type)
            if key not in subject_files:
                subject_files[key] = []
            subject_files[key].append({
                'filename': f,
                'filepath': os.path.join(self.processed_dir, f),
                'subject_id': subject_id,
                'session_id': session_id,
                'sensor_type': sensor_type,
                'duration': duration,
                'placement': placement_tag,
            })

        # Deterministic ordering within each (subject, sensor) bucket.
        for key in list(subject_files.keys()):
            subject_files[key] = sorted(subject_files[key], key=lambda d: d.get('filename', ''))
        
        # For each subject, group sensors together
        # We need all requested sensor types to be available
        subjects = sorted(set(sid for sid, _ in subject_files.keys()))
        
        for subject_id in subjects:
            # Special case for SAMoSA - all sensors in one file
            if self.config.format_type == 'samosa_imu':
                key = (subject_id, 'accelerometer')  # We stored all IMU files under 'accelerometer'
                if key in subject_files:
                    for file_info in subject_files[key]:
                        # Create a files dict with all requested sensors pointing to the same file
                        files_dict = {st: file_info for st in self.sensor_types}
                        samples.append({
                            'subject_id': subject_id,
                            'files': files_dict,
                            'duration': file_info['duration'],
                            'label_file': None,  # Label from filename
                        })
                continue
            
            # Check if all required sensor types are available
            available_sensors = {}
            for st in self.sensor_types:
                key = (subject_id, st)
                if key in subject_files:
                    available_sensors[st] = subject_files[key]
            
            # Skip if not all sensors available
            if len(available_sensors) != len(self.sensor_types):
                continue
            
            # For standard datasets, we might have multiple placements
            # We need to match files by placement/duration
            if len(self.sensor_types) == 1:
                # Single sensor type - add each file as a sample
                for file_info in sorted(available_sensors[self.sensor_types[0]], key=lambda d: d.get('filename', '')):
                    label_file = get_label_file_for_data_file(
                        file_info['filepath'], self.dataset_name, self.processed_dir
                    )
                    samples.append({
                        'subject_id': subject_id,
                        'session_id': file_info.get('session_id'),
                        'files': {self.sensor_types[0]: file_info},
                        'duration': file_info['duration'],
                        'label_file': label_file,
                    })
            else:
                # Multiple sensor types - need to match by duration AND placement
                # Use flexible matching: allow larger duration difference for some datasets
                duration_tolerance = 0.1  # Default: strict matching
                if self.dataset_name in ['wisdm', 'HHAR']:
                    duration_tolerance = 100.0  # More lenient for datasets with variable sensor durations

                def fogturning_record_key(filename: str) -> Optional[str]:
                    if self.dataset_name != "FoGTurning":
                        return None
                    match = re.match(
                        r"^(FoGTurning_SUB\d+_\d+)_shank_(?:acc|gyro)_20Hz_[0-9.]+s\.parquet$",
                        filename,
                    )
                    return match.group(1) if match else None
                
                primary_files = available_sensors[self.sensor_types[0]]
                for pf in primary_files:
                    matched_files = {self.sensor_types[0]: pf}
                    all_matched = True
                    primary_placement = pf.get('placement')
                    primary_record_key = fogturning_record_key(pf.get("filename", ""))
                    
                    for st in self.sensor_types[1:]:
                        matched = None
                        best_match_diff = float('inf')
                        for sf in available_sensors[st]:
                            # Check placement match first (if placements are available)
                            sf_placement = sf.get('placement')
                            if primary_placement is not None and sf_placement is not None:
                                if primary_placement != sf_placement:
                                    continue  # Skip files from different placements

                            if primary_record_key is not None:
                                if fogturning_record_key(sf.get("filename", "")) != primary_record_key:
                                    continue
                            
                            # Then check duration match
                            diff = abs(sf['duration'] - pf['duration'])
                            if diff < duration_tolerance and diff < best_match_diff:
                                matched = sf
                                best_match_diff = diff
                        if matched:
                            matched_files[st] = matched
                        else:
                            all_matched = False
                            break
                    
                    if all_matched:
                        # Use the minimum duration across all matched sensors
                        min_duration = min(f['duration'] for f in matched_files.values())
                        
                        # Label comes from primary sensor's file
                        label_file = get_label_file_for_data_file(
                            pf['filepath'], self.dataset_name, self.processed_dir
                        )
                        samples.append({
                            'subject_id': subject_id,
                            'session_id': pf.get('session_id'),
                            'files': matched_files,
                            'duration': min_duration,
                            'label_file': label_file,
                        })
        
         # APPLY max_files_per_dataset LIMIT
        if self.max_files_per_dataset is not None and len(samples) > self.max_files_per_dataset:
            print(f"  Limiting to {self.max_files_per_dataset} files (from {len(samples)} discovered)")
            samples = samples[:self.max_files_per_dataset]

        return samples
    
    def _build_sample_index(self):
        """Build index for window sampling and track labels efficiently."""
        self.valid_windows = []
        self.valid_random_starts = []
        self.total_windows = 0
        self.total_windows_before_filter = 0

        # When in overlap mode and returning majority labels, we also track labels
        # aligned to valid_random_starts so get_all_labels() matches len(self).
        random_start_labels_list = (
            [] if (self.mode == 'overlap' and self.return_majority_label) else None
        )
        
        # Track labels if return_majority_label is True
        window_labels_list = [] if self.return_majority_label else None
        
        for sample_idx, sample in enumerate(self.samples):
            resampled_length = int(sample['duration'] * self.sampling_rate)
            num_windows = resampled_length // self.window_size
            sample['num_windows'] = num_windows
            self.total_windows_before_filter += num_windows
            
            # Load labels (avoid loading full sensor files when possible)
            encoded_labels = self._get_sample_labels(sample_idx)
            
            if encoded_labels is not None:
                # Resample to target rate
                if len(encoded_labels) != resampled_length:
                    indices = np.linspace(0, len(encoded_labels) - 1, resampled_length).astype(int)
                    encoded_labels = encoded_labels[indices]
                
                # Pre-compute for efficient checking
                is_unwanted = (encoded_labels == -1).astype(np.int32)
                cumsum = np.concatenate([[0], np.cumsum(is_unwanted)])
                threshold = int(self.window_size * self.max_unwanted_frac)
                
                # Check non-overlapping windows
                for window_offset in range(num_windows):
                    start = window_offset * self.window_size
                    end = start + self.window_size
                    num_unwanted = cumsum[end] - cumsum[start]
                    
                    if num_unwanted <= threshold:
                        self.valid_windows.append((sample_idx, window_offset))
                        
                        # Compute and store window label if needed
                        if window_labels_list is not None:
                            label_window = encoded_labels[start:end]
                            window_label = self._get_window_label(label_window)
                            window_labels_list.append(window_label)

                # Handle random starts for overlap mode (stride=1).
                # IMPORTANT: Only compute these when actually in overlap mode;
                # otherwise this is O(T) per recording and can be extremely slow.
                if self.mode == 'overlap':
                    max_start = resampled_length - self.window_size
                    if max_start > 0:
                        stride = int(getattr(self, "overlap_stride_samples", 1))
                        all_starts = np.arange(0, max_start + 1, stride, dtype=np.int64)
                        all_ends = all_starts + self.window_size
                        all_num_unwanted = cumsum[all_ends] - cumsum[all_starts]
                        valid_mask = all_num_unwanted <= threshold
                        valid_starts = all_starts[valid_mask]

                        for start in valid_starts:
                            self.valid_random_starts.append((sample_idx, int(start)))
                            if random_start_labels_list is not None:
                                label_window = encoded_labels[int(start):int(start) + self.window_size]
                                random_start_labels_list.append(self._get_window_label(label_window))
            else:
                # No labels - keep all windows
                for window_offset in range(num_windows):
                    self.valid_windows.append((sample_idx, window_offset))
                    if window_labels_list is not None:
                        window_labels_list.append(0)  # Dummy label

                if self.mode == 'overlap':
                    max_start = resampled_length - self.window_size
                    if max_start > 0:
                        stride = int(getattr(self, "overlap_stride_samples", 1))
                        for start in range(0, max_start + 1, stride):
                            self.valid_random_starts.append((sample_idx, start))
                            if random_start_labels_list is not None:
                                random_start_labels_list.append(0)
        
        # Convert to numpy array for fast access
        if window_labels_list is not None:
            self.window_labels = np.array(window_labels_list, dtype=np.int64)
            
            # Compute label distribution
            valid_labels = self.window_labels[self.window_labels != -1]
            unique, counts = np.unique(valid_labels, return_counts=True)
            self.label_distribution = dict(zip(unique.tolist(), counts.tolist()))
            
            print(f"\n{'='*60}")
            print(f"Label Distribution in Valid Windows")
            print(f"{'='*60}")
            for lbl in sorted(self.label_distribution.keys()):
                if hasattr(self, 'idx_to_label') and self.idx_to_label:
                    label_name = self.idx_to_label.get(lbl, f"Label_{lbl}")
                    print(f"  {lbl} ({label_name}): {self.label_distribution[lbl]:,} windows")
                else:
                    print(f"  {lbl}: {self.label_distribution[lbl]:,} windows")
            print(f"{'='*60}\n")
        
        # Report filtering stats
        windows_after_filter = len(self.valid_windows)
        if self.total_windows_before_filter > 0:
            drop_rate = 1.0 - windows_after_filter / self.total_windows_before_filter
            print(f"\n=== Window Filtering Stats ===")
            print(f"Windows before filtering: {self.total_windows_before_filter:,}")
            print(f"Windows after filtering: {windows_after_filter:,}")
            print(f"Dropped: {self.total_windows_before_filter - windows_after_filter:,} ({drop_rate*100:.1f}%)")
        
        # STEP 2: Apply max_windows limit (hard cap on total windows)
        # STEP 2a: Optional per-class cap (applies only when majority labels are available)
        if (
            self.max_windows_per_class is not None
            and self.max_windows_per_class > 0
            and hasattr(self, "window_labels")
            and self.window_labels is not None
            and len(self.window_labels) == len(self.valid_windows)
        ):
            k = int(self.max_windows_per_class)
            labels = self.window_labels
            # Ignore unwanted label (-1) for capping; keep all unwanted windows (if any) as-is.
            kept_indices = []
            unwanted_idx = np.flatnonzero(labels == -1)
            if unwanted_idx.size:
                kept_indices.append(unwanted_idx)

            for lbl in np.unique(labels[labels != -1]):
                idx = np.flatnonzero(labels == lbl)
                if idx.size:
                    kept_indices.append(idx[:k])

            if kept_indices:
                keep = np.sort(np.concatenate(kept_indices))
                if keep.size < len(self.valid_windows):
                    print(
                        f"  Applying max_windows_per_class={k}: "
                        f"{len(self.valid_windows)} -> {int(keep.size)} windows"
                    )
                    self.valid_windows = [self.valid_windows[i] for i in keep.tolist()]
                    self.window_labels = self.window_labels[keep]

        if self.max_windows is not None and len(self.valid_windows) > self.max_windows:
            print(f"  Limiting to {self.max_windows} windows (from {len(self.valid_windows)} valid)")
            self.valid_windows = self.valid_windows[:self.max_windows]
            if window_labels_list is not None:
                self.window_labels = self.window_labels[:self.max_windows]
            # Also limit random starts proportionally
            if len(self.valid_random_starts) > 0:
                ratio = self.max_windows / windows_after_filter
                limit_random = int(len(self.valid_random_starts) * ratio)
                self.valid_random_starts = self.valid_random_starts[:limit_random]
        
        # STEP 3: Apply subsample_ratio (random sampling of windows)
        if self.subsample_ratio is not None and self.subsample_ratio < 1.0:
            import random
            n_keep = int(len(self.valid_windows) * self.subsample_ratio)
            print(f"  Subsampling {self.subsample_ratio*100:.1f}% of windows: "
                f"{len(self.valid_windows)} → {n_keep}")
            
            # Shuffle and keep first n_keep windows
            indices = list(range(len(self.valid_windows)))
            random.shuffle(indices)
            self.valid_windows = [self.valid_windows[i] for i in indices[:n_keep]]

            if window_labels_list is not None:
                self.window_labels = self.window_labels[indices[:n_keep]]
            
            # Also subsample random starts
            if len(self.valid_random_starts) > 0:
                n_keep_random = int(len(self.valid_random_starts) * self.subsample_ratio)
                indices_random = list(range(len(self.valid_random_starts)))
                random.shuffle(indices_random)
                self.valid_random_starts = [self.valid_random_starts[i] for i in indices_random[:n_keep_random]]
        
        self.total_windows = len(self.valid_windows)
        self.total_random_positions = len(self.valid_random_starts)

        if random_start_labels_list is not None:
            self.random_start_labels = np.asarray(random_start_labels_list, dtype=np.int64)

        if self.mode == 'overlap':
            print(
                f"  Final dataset size: {self.total_random_positions} windows "
                f"(overlap stride={getattr(self, 'overlap_stride_samples', 1)} samples; "
                f"nonoverlap_windows={self.total_windows})"
            )
        else:
            print(f"  Final dataset size: {self.total_windows} windows")
    
    def _compute_num_channels(self) -> int:
        """Compute number of output channels based on sensor types and axial mode."""
        channels_per_sensor = 1 if self.axial_mode == 'uniaxial' else 3
        return len(self.sensor_types) * channels_per_sensor
    
    def __len__(self) -> int:
        if self.mode == 'overlap':
            return len(self.valid_random_starts)
        else:
            return len(self.valid_windows)
    
    def _find_sample_for_index(self, idx: int) -> Tuple[int, int]:
        """Find which sample and window offset for a given index.
        
        Uses the pre-computed valid_windows list that excludes windows
        with majority unwanted labels.
        """
        if idx < 0 or idx >= len(self.valid_windows):
            raise IndexError(f"Index {idx} out of range [0, {len(self.valid_windows)})")
        return self.valid_windows[idx]
    
    
    def _resample_data(self, data: np.ndarray, original_rate: float = 20.0) -> np.ndarray:
        """Resample data to target sampling rate using linear interpolation."""
        if self.sampling_rate == original_rate:
            return data
        
        original_len = len(data)
        original_time = np.arange(original_len) / original_rate
        
        # New time points
        new_duration = original_time[-1]
        new_len = int(new_duration * self.sampling_rate)
        new_time = np.arange(new_len) / self.sampling_rate
        
        # Interpolate each channel
        new_data = np.zeros((new_len, data.shape[1]))
        for i in range(data.shape[1]):
            new_data[:, i] = np.interp(new_time, original_time, data[:, i])
        
        return new_data
    
    def _resample_labels(self, labels: np.ndarray, original_rate: float = 20.0) -> np.ndarray:
        """Resample labels to target sampling rate using nearest neighbor."""
        if self.sampling_rate == original_rate:
            return labels
        
        original_len = len(labels)
        original_time = np.arange(original_len) / original_rate
        
        new_duration = original_time[-1]
        new_len = int(new_duration * self.sampling_rate)
        new_time = np.arange(new_len) / self.sampling_rate
        
        # Use nearest neighbor for labels
        indices = np.searchsorted(original_time, new_time)
        indices = np.clip(indices, 0, original_len - 1)
        
        return labels[indices]
    
    def _load_sample_data(self, sample: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Load and process data for a sample.
        
        FIXED: Only encode labels ONCE at the very end.
        """
        all_channels = []
        labels = None
        
        # Fixed order for sensor types
        sensor_order = ['accelerometer', 'gyroscope', 'magnetometer']
        
        for sensor_type in sensor_order:
            if sensor_type not in self.sensor_types:
                continue
            
            file_info = sample['files'].get(sensor_type)
            if file_info is None:
                continue
            
            filepath = file_info['filepath']
            
            if self.config.format_type == 'samosa_imu':
                # SAMoSA: load specific columns
                df = pd.read_parquet(filepath)
                if sensor_type == 'accelerometer':
                    data = df[['acc_x', 'acc_y', 'acc_z']].values
                elif sensor_type == 'gyroscope':
                    data = df[['gyro_x', 'gyro_y', 'gyro_z']].values
                elif sensor_type == 'magnetometer':
                    data = df[['mag_x', 'mag_y', 'mag_z']].values
            elif self.config.format_type == 'embedded_label':
                df = pd.read_parquet(filepath)
                data = df[['x', 'y', 'z']].values
                if labels is None:
                    labels = df['label'].values  # RAW labels, not encoded yet
                    # IMPORTANT: for embedded-label datasets (e.g., Recofit), labels live at the
                    # original sampling rate in the sensor parquet. If we downsample the signal
                    # (e.g., to 5 Hz) but do not resample the labels, the later length-alignment
                    # step will truncate labels to the beginning of the recording, effectively
                    # deleting later activities and causing many windows to become -1.
                    labels = self._resample_labels(labels)
            else:
                df = pd.read_parquet(filepath)
                # Most datasets use explicit x/y/z columns, but some (e.g. OdayFoG)
                # store 3-axis accelerometer vectors in a single object column.
                if 'values' in df.columns and len(df.columns) == 1:
                    v = df['values'].to_numpy()
                    if v.size == 0:
                        data = np.zeros((0, 3), dtype=np.float32)
                    else:
                        # Two common layouts:
                        #  (A) one row per timestep, where each row is a length-3 vector
                        #  (B) a single row where the entry is a length-T list/array of 3-vectors (OdayFoG)
                        try:
                            if v.size == 1 and isinstance(v[0], (list, np.ndarray)):
                                inner = np.asarray(v[0], dtype=object)
                                if inner.ndim == 1 and inner.size > 0 and isinstance(inner[0], (list, np.ndarray)):
                                    data = np.stack([np.asarray(x, dtype=np.float32) for x in inner], axis=0)
                                else:
                                    data = np.asarray(inner, dtype=np.float32)
                            else:
                                data = np.stack([np.asarray(row, dtype=np.float32) for row in v], axis=0)
                        except Exception as e:
                            raise TypeError(
                                f"Failed to unpack '{self.dataset_name}' parquet column 'values' into numeric array; "
                                f"n_rows={len(df)}, example type={type(v[0]) if v.size else None}, error={e}"
                            )
                elif all(c in df.columns for c in ['x', 'y', 'z']):
                    data = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
                elif all(c in df.columns for c in ['acc_x', 'acc_y', 'acc_z']):
                    data = df[['acc_x', 'acc_y', 'acc_z']].to_numpy(dtype=np.float32)
                elif all(c in df.columns for c in ['accel_x', 'accel_y', 'accel_z']):
                    data = df[['accel_x', 'accel_y', 'accel_z']].to_numpy(dtype=np.float32)
                else:
                    # Fall back to the first 3 numeric-looking columns, ignoring common metadata.
                    ignore = {
                        'timestamp', 'timestamp_s', 'timestamp_sec', 'time', 'time_s', 'time_sec',
                        'label', 'activity', 'activityID', 'subject_id'
                    }
                    cols = [c for c in df.columns if str(c) not in ignore]
                    if len(cols) < 3:
                        cols = list(df.columns)
                    data = df[cols[:3]].to_numpy()
                    # Coerce to float32 if needed (handles object dtypes from mixed columns).
                    if not np.issubdtype(data.dtype, np.number):
                        data = pd.DataFrame(data).apply(pd.to_numeric, errors='coerce').to_numpy(dtype=np.float32)
                    else:
                        data = data.astype(np.float32, copy=False)
            
            # Resample data
            data = self._resample_data(data)
            
            # Apply uniaxial aggregation if needed
            if self.axial_mode == 'uniaxial':
                magnitude = np.sqrt(np.sum(data ** 2, axis=1, keepdims=True))
                all_channels.append(magnitude)
            else:
                all_channels.append(data)
        
        # Truncate all channels to same length
        min_len = min(ch.shape[0] for ch in all_channels)
        all_channels = [ch[:min_len] for ch in all_channels]
        
        # Stack all channels
        signal = np.concatenate(all_channels, axis=1)

        # Some datasets contain NaN/Inf in the raw sensor stream (notably OdayFoG).
        # Replace with zeros so downstream embedding + metrics don't blow up.
        if signal.size and not np.isfinite(signal).all():
            signal = signal.astype(np.float32, copy=False)
            np.nan_to_num(signal, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Load labels if not embedded
        if labels is None and sample['label_file'] is not None:
            label_df = pd.read_parquet(sample['label_file'])
            
            # Find the label column
            if self.label_column and self.label_column in label_df.columns:
                labels = label_df[self.label_column].values
            elif 'label' in label_df.columns:
                labels = label_df['label'].values
            elif 'activity' in label_df.columns:
                labels = label_df['activity'].values
            elif 'activityID' in label_df.columns:
                labels = label_df['activityID'].values
            else:
                # Use first non-timestamp column
                for col in label_df.columns:
                    if col not in ['timestamp', 'timestamp_s', 'timestamp_sec', 'time_s', 'subject_id']:
                        labels = label_df[col].values
                        break
            
            if labels is not None:
                # DEBUG PRINT
                # print(f"DEBUG: Loaded labels. Shape: {labels.shape}, Sample first: {labels[0] if len(labels)>0 else 'empty'}")
                if len(labels) == 1:
                     if os.environ.get("MOTION_DEBUG_LABELS", "0") == "1":
                         print(f"INFO: Single row label file. Type of first element: {type(labels[0])}")

                # Handle case where labels are stored as a list/array in a single row
                if len(labels) == 1 and isinstance(labels[0], (list, np.ndarray)):
                    if os.environ.get("MOTION_DEBUG_LABELS", "0") == "1":
                        print(f"DEBUG: Unwrapping label array. Original shape: {labels.shape}, Inner type: {type(labels[0])}")
                    labels = np.array(labels[0])
                    if os.environ.get("MOTION_DEBUG_LABELS", "0") == "1":
                        print(f"DEBUG: Unwrapped labels shape: {labels.shape}, dtype: {labels.dtype}")
                    
                labels = self._resample_labels(labels)
        
        # Handle SAMoSA labels from filename
        if labels is None and self.config.format_type == 'samosa_imu':
            filename = list(sample['files'].values())[0]['filename']
            parts = filename.split('_')
            activity_parts = []
            started = False
            for p in parts:
                if p.startswith('P') and p[1:].isdigit():
                    started = True
                    continue
                if started and p.isdigit() and len(p) <= 2:
                    break
                if started:
                    activity_parts.append(p)
            activity = '_'.join(activity_parts)
            labels = np.full(len(signal), activity)
        
        # If still no labels, create dummy labels
        if labels is None:
            labels = np.zeros(len(signal), dtype=np.int64)
        
        # Ensure labels match signal length BEFORE encoding
        if len(labels) != len(signal):
            if len(labels) > len(signal):
                labels = labels[:len(signal)]
            else:
                # Extend with last value
                last_val = labels[-1] if len(labels) > 0 else 0
                labels = np.concatenate([
                    labels,
                    np.full(len(signal) - len(labels), last_val)
                ])
        
        # ===== ENCODE LABELS ONLY ONCE, AT THE END =====
        labels = self._encode_labels(labels)
        
        return signal, labels
    
    def _encode_labels(self, labels: np.ndarray) -> np.ndarray:
        """Encode labels to integers with FIXED type handling.

        Returns:
            Array of integer labels where:
            - Valid labels: 0 to num_classes-1
            - Invalid labels (unwanted/not in top-k): -1

        Implementation note: per-element python loops were a major bottleneck
        for high-rate datasets like capture24 (one label per 50 ms => ~2M rows
        per recording * 151 recordings). Since the number of *distinct* raw
        labels per file is tiny (single digits for capture24, low-double for
        most HAR datasets), we encode each unique value once and broadcast
        through an inverse index. The legacy per-element path is retained as
        a fallback for inputs whose dtype/values are not hashable by numpy.
        """
        n = len(labels)
        if n == 0:
            return np.zeros(0, dtype=np.int64)

        unwanted = set(UNWANTED_LABELS.get(self.dataset_name, []))

        if self.top_k_labels is not None and hasattr(self, '_top_k_set'):
            mode = 'topk'
        elif self.label_to_idx is not None:
            mode = 'encoder'
        else:
            mode = 'passthrough_int'

        def _encode_one(lab):
            s = normalize_label_to_string(lab)
            if mode == 'topk':
                if s in unwanted or s not in self._top_k_set:
                    return -1
                return int(self.label_to_idx[s])
            if mode == 'encoder':
                if s in unwanted:
                    return -1
                idx = self.label_to_idx.get(s, None)
                return -1 if idx is None else int(idx)
            # passthrough_int
            if s in unwanted:
                return -1
            try:
                return int(float(lab))
            except (ValueError, TypeError):
                return -1

        try:
            uniques, inverse = np.unique(labels, return_inverse=True)
            unique_encoded = np.fromiter(
                (_encode_one(u) for u in uniques),
                dtype=np.int64,
                count=len(uniques),
            )
            return unique_encoded[inverse]
        except TypeError:
            pass

        # Fallback: original per-element loop for unhashable / mixed object
        # arrays that np.unique can't sort.
        encoded = np.empty(n, dtype=np.int64)
        for i, label in enumerate(labels):
            encoded[i] = _encode_one(label)
        return encoded
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a window of data.
        
        In 'nonoverlap' mode: Returns non-overlapping windows (stride = window_size).
        In 'overlap' mode: Returns overlapping windows (stride = 1).
        
        Both modes filter out windows with majority unwanted labels.
        """
        if getattr(self, "labels_only", False):
            raise RuntimeError(
                "MotionDataset was created with labels_only=True; signal windows are not available. "
                "Recreate the dataset with labels_only=False for training/evaluation."
            )
        if self.mode == 'overlap':
            # Overlap mode: index into valid_random_starts
            if idx < 0 or idx >= len(self.valid_random_starts):
                raise IndexError(f"Index {idx} out of range [0, {len(self.valid_random_starts)})")
            sample_idx, start = self.valid_random_starts[idx]
        else:
            # Nonoverlap mode: index into valid_windows
            sample_idx, window_offset = self._find_sample_for_index(idx)
            start = window_offset * self.window_size
        
        # Load full sample data (uses cache if preloaded)
        signal, labels = self._get_sample_data(sample_idx)
        
        end = start + self.window_size
        
        # Extract window
        signal_window = signal[start:end]
        label_window = labels[start:end]
        
        # Pad if necessary (shouldn't happen with valid positions, but safety check)
        if len(signal_window) < self.window_size:
            pad_len = self.window_size - len(signal_window)
            signal_window = np.pad(signal_window, ((0, pad_len), (0, 0)), mode='constant')
            label_window = np.pad(label_window, (0, pad_len), mode='constant', constant_values=label_window[-1] if len(label_window) > 0 else 0)
        
        # Convert to tensors
        # Shape: (num_channels, window_size)
        signal_tensor = torch.from_numpy(signal_window.T).float()
        
        # Handle labels - return as tensor if numeric, numpy array if strings
        if label_window.dtype == object or (len(label_window) > 0 and isinstance(label_window[0], str)):
            # String labels: return as numpy array
            if self.return_majority_label:
                majority_label = self._get_window_label(label_window)
                return signal_tensor, majority_label
            return signal_tensor, label_window
        else:
            # Numeric labels
            if self.return_majority_label:
                majority_label = self._get_window_label(label_window)
                return signal_tensor, torch.tensor(majority_label, dtype=torch.long)
            label_tensor = torch.from_numpy(label_window).long()
            return signal_tensor, label_tensor
    
    def _get_window_label(self, label_window: np.ndarray) -> int:
        """Get the label for a window.
        
        Priority logic:
        1. If dataset has priority labels with thresholds, check if threshold is met
           - If yes, return that priority label
           - Threshold is the minimum fraction of window with that label
        2. Otherwise, return majority label, excluding -1 (unwanted)
        3. If all labels are -1, return -1
        
        Args:
            label_window: Array of encoded labels in the window
            
        Returns:
            Single integer label for the window
        """
        window_size = len(label_window)
        
        # Check for priority labels first
        if self.priority_labels:
            for priority_label_str, threshold in self.priority_labels:
                priority_label_int = int(priority_label_str)
                
                # Count occurrences of this priority label
                count = np.sum(label_window == priority_label_int)
                fraction = count / window_size if window_size > 0 else 0.0
                
                # Check if threshold is met
                if fraction >= threshold:
                    return priority_label_int
        
        # No priority label threshold met - use majority voting, excluding -1 (unwanted)
        valid_mask = label_window != -1
        if valid_mask.any():
            valid_labels = label_window[valid_mask]
            unique, counts = np.unique(valid_labels, return_counts=True)
            majority_label = unique[np.argmax(counts)]
            return int(majority_label)
        else:
            # All labels are unwanted
            return -1
        
    def get_all_labels(self) -> np.ndarray:
        """Get all window labels efficiently.
        
        Returns array of labels corresponding to each window.
        Only works when return_majority_label=True.
        """
        if not self.return_majority_label:
            raise ValueError("get_all_labels() only works with return_majority_label=True")

        if self.mode == 'overlap':
            if not hasattr(self, 'random_start_labels') or self.random_start_labels is None:
                raise RuntimeError("Overlap-mode labels not computed during initialization")
            if len(self.random_start_labels) != len(self.valid_random_starts):
                raise RuntimeError(
                    f"Overlap-mode label length mismatch: labels={len(self.random_start_labels)} vs windows={len(self.valid_random_starts)}"
                )
            return self.random_start_labels

        if self.window_labels is None:
            raise RuntimeError("Window labels not computed during initialization")
        
        return self.window_labels

def create_motion_dataloader(
    data_root: str,
    dataset_name: str,
    batch_size: int = 32,
    sampling_rate: float = 20.0,
    window_size: int = 200,
    sensor_types: Union[str, List[str]] = 'accelerometer',
    axial_mode: Literal['triaxial', 'uniaxial'] = 'triaxial',
    placement: Optional[Union[str, List[str]]] = None,
    label_column: Optional[str] = None,
    top_k_labels: Optional[int] = None,
    preload: bool = False,
    mode: Literal['overlap', 'nonoverlap'] = 'nonoverlap',
    return_majority_label: bool = True,
    max_unwanted_frac: float = 0.5,
    max_files_per_dataset: Optional[int] = None,
    max_windows: Optional[int] = None,
    subsample_ratio: Optional[float] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader for motion data.
    
    Args:
        data_root: Root directory containing all datasets
        dataset_name: Name of the dataset
        batch_size: Batch size
        sampling_rate: Target sampling rate (<= 20 Hz)
        window_size: Window size in samples at target rate
        sensor_types: Sensor types to load
        axial_mode: 'triaxial' or 'uniaxial'
        placement: Body placement filter
        label_column: Label column name
        preload: If True, load all recordings into memory at initialization
        mode: 'nonoverlap' for evaluation (non-overlapping windows),
              'overlap' for training (stride=1, naturally duration-weighted)
        return_majority_label: If True, return single majority label per window
        max_files_per_dataset: Limit number of recording files to load
        max_windows: Hard cap on total number of windows
        subsample_ratio: Keep only this fraction of windows (e.g., 0.1 for 10%)
        shuffle: Whether to shuffle (recommended True for mode='overlap')
        num_workers: Number of worker processes
        **kwargs: Additional arguments for DataLoader
    
    Returns:
        DataLoader instance
    """
    dataset = MotionDataset(
        data_root=data_root,
        dataset_name=dataset_name,
        sampling_rate=sampling_rate,
        window_size=window_size,
        sensor_types=sensor_types,
        axial_mode=axial_mode,
        placement=placement,
        label_column=label_column,
        top_k_labels=top_k_labels,
        mode=mode,
        preload=preload,
        return_majority_label=return_majority_label,
        max_unwanted_frac=max_unwanted_frac,
        max_files_per_dataset=max_files_per_dataset,
        max_windows=max_windows,
        subsample_ratio=subsample_ratio,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs
    )



# =========================================================================
# Frozen split JSON utilities (downstream evaluation)
#
# Split files store train/val/test indices into MotionDataset's window list.
# For correctness, the MotionDataset instance must be constructed with the
# same indexing-related knobs used when generating the split.
# =========================================================================

def load_frozen_split_payload(split_path: Union[str, "os.PathLike[str]"]) -> Dict:
    """Load a frozen split JSON payload from disk."""
    import json
    from pathlib import Path

    p = Path(split_path)
    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or "splits" not in payload:
        raise ValueError(f"Invalid split JSON (expected object with 'splits'): {p}")
    if not isinstance(payload.get("splits"), dict):
        raise ValueError(f"Invalid split JSON (expected splits dict): {p}")
    return payload


def resolve_split_path(
    split_path_or_dir: Union[str, "os.PathLike[str]"],
    *,
    dataset_name: str,
    placement: Optional[Union[str, List[str]]] = None,
) -> str:
    """Resolve a {dataset}_splits.json path from either a directory or a file path."""
    from pathlib import Path

    p = Path(split_path_or_dir)
    if p.is_dir():
        # Only resolve to a placement-specific split file if placement is a single string.
        if isinstance(placement, str) and placement:
            placement_norm = str(placement).strip().lower()
            if placement_norm:
                cand = p / f"{dataset_name}_{placement_norm}_only_splits.json"
                if cand.exists():
                    return str(cand)
        return str(p / f"{dataset_name}_splits.json")
    return str(p)


def get_split_indices_for_dataset(
    ds: "MotionDataset",
    split_path_or_dir: Union[str, "os.PathLike[str]"],
    *,
    allow_oob: bool = False,
    placement: Optional[Union[str, List[str]]] = None,
) -> Tuple[Dict[str, List[int]], Dict]:
    """Load train/val/test indices for a MotionDataset from a frozen split JSON.

    Returns:
        splits: dict with keys train/val/test and list[int] indices.
        payload: the full JSON payload (metadata + splits).
    """
    import json
    import warnings
    from pathlib import Path

    split_path = Path(
        resolve_split_path(split_path_or_dir, dataset_name=ds.dataset_name, placement=placement)
    )
    if not split_path.exists():
        raise FileNotFoundError(f"Frozen split file not found: {split_path}")

    payload = load_frozen_split_payload(str(split_path))
    splits_raw = payload.get("splits", {}) or {}

    train_idx = [int(i) for i in (splits_raw.get("train", []) or [])]
    val_idx = [int(i) for i in (splits_raw.get("val", []) or [])]
    test_idx = [int(i) for i in (splits_raw.get("test", []) or [])]

    n = int(len(ds))
    expected_n = payload.get("n_windows", None)

    def _oob_count(idxs: List[int]) -> int:
        return sum(1 for i in idxs if i < 0 or i >= n)

    oob = _oob_count(train_idx) + _oob_count(val_idx) + _oob_count(test_idx)
    mismatch = (expected_n is not None and int(expected_n) != n)
    if oob > 0 or mismatch:
        msg = (
            f"Frozen split mismatch for {ds.dataset_name}: current_len={n}, split_file_n_windows={expected_n}, "
            f"oob_total={oob}. This usually means dataset window indexing changed (window_size/sampling_rate/mode/"
            "top_k_labels/unwanted-label filtering/max_windows/subsample_ratio/max_files_per_dataset/etc)."
        )
        if not allow_oob:
            raise ValueError(msg + " Set allow_oob=True to drop OOB indices, or regenerate splits.")

        def _filter_in_range(idxs: List[int]) -> List[int]:
            return [int(i) for i in idxs if 0 <= int(i) < n]

        train_idx = _filter_in_range(train_idx)
        val_idx = _filter_in_range(val_idx)
        test_idx = _filter_in_range(test_idx)
        warnings.warn(
            msg
            + f" Dropped OOB indices. Sizes now train/val/test={len(train_idx)}/{len(val_idx)}/{len(test_idx)}"
        )

    # Optional placement subsetting:
    # IMPORTANT: do NOT construct ds with placement filtering when using frozen splits, because
    # split indices are defined against the unfiltered dataset's window indexing.
    if placement:
        # Normalize into a sorted list of placement tags.
        if isinstance(placement, str):
            placement_list = [placement]
        else:
            placement_list = list(placement)
        placement_norms = sorted({str(p).strip().lower() for p in placement_list if str(p).strip()})
        if placement_norms:
            # If we resolved to a placement-specific split file (e.g. *_wrist_only_splits.json),
            # trust it and just annotate metadata.
            if len(placement_norms) == 1 and split_path.name.endswith(f"_{placement_norms[0]}_only_splits.json"):
                payload.setdefault("metadata", {})
                payload["metadata"]["placement_filter"] = placement_norms[0]
            else:
                by_place_path = split_path.parent / f"{ds.dataset_name}_splits_by_placement.json"
                if by_place_path.exists():
                    with by_place_path.open("r", encoding="utf-8") as f:
                        by_place = json.load(f)
                    if not isinstance(by_place, dict):
                        raise ValueError(f"Invalid splits_by_placement JSON: {by_place_path}")

                    def _from_map_union(split_name: str) -> List[int]:
                        block = by_place.get(split_name, {}) or {}
                        if not isinstance(block, dict):
                            raise ValueError(f"Invalid splits_by_placement JSON (expected dict at {split_name}): {by_place_path}")
                        avail = {str(k).strip().lower() for k in block.keys()}
                        missing = [p for p in placement_norms if p not in avail]
                        if missing:
                            raise KeyError(
                                f"Placements {missing} not found in {by_place_path} for split '{split_name}'. "
                                f"Available: {sorted(avail)}"
                            )
                        idx_set = set()
                        for ptag in placement_norms:
                            for i in (block.get(ptag, []) or []):
                                idx_set.add(int(i))
                        return sorted(idx_set)

                    train_idx = _from_map_union("train")
                    val_idx = _from_map_union("val")
                    test_idx = _from_map_union("test")
                    payload.setdefault("metadata", {})
                    payload["metadata"]["placement_filter"] = placement_norms
                    payload["metadata"]["placement_source"] = str(by_place_path)
                else:
                    # Fallback: filter the loaded indices by inspecting ds window metadata.
                    # This is slower but avoids needing a *_splits_by_placement.json sidecar.
                    def _keep(idxs: List[int]) -> List[int]:
                        kept: List[int] = []
                        for i in idxs:
                            try:
                                ptag = ds.get_window_placement(int(i))
                            except Exception:
                                ptag = None
                            if ptag is not None and str(ptag).lower() in set(placement_norms):
                                kept.append(int(i))
                        return kept

                    train_idx = _keep(train_idx)
                    val_idx = _keep(val_idx)
                    test_idx = _keep(test_idx)
                    payload.setdefault("metadata", {})
                    payload["metadata"]["placement_filter"] = placement_norms
                    payload["metadata"]["placement_source"] = "ds.get_window_placement"

    return {"train": train_idx, "val": val_idx, "test": test_idx}, payload


# ============================================================================
# Utility Functions
# ============================================================================

def list_available_datasets() -> List[str]:
    """List all available datasets."""
    return list(DATASET_CONFIGS.keys())


def get_dataset_info(dataset_name: str) -> Dict:
    """Get information about a dataset."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    config = DATASET_CONFIGS[dataset_name]
    return {
        'name': config.name,
        'available_sensors': config.available_sensors,
        'has_separate_labels': config.has_separate_labels,
        'placements': config.placement_patterns,
        'default_placement': config.default_placement,
        'format_type': config.format_type,
    }


# =========================================================================
# Frozen splits + label distribution plots
#
# These utilities are used by inertia1/eval/motion_split_analysis.py to create
# per-dataset frozen train/val/test splits (optionally grouped by subject_id)
# and to visualize label distributions.
# =========================================================================


def _stratified_split_indices(labels: np.ndarray, seed: int, train_frac: float, val_frac: float):
    """Simple stratified split for window-level labels."""
    rng = np.random.default_rng(int(seed))
    labels = np.asarray(labels)
    n = int(labels.shape[0])
    idx = np.arange(n)

    unique = np.unique(labels)
    if unique.size == 0:
        return [], [], []

    # If any class is too small, fall back to random.
    min_per_class = int(min(np.sum(labels == c) for c in unique))
    if min_per_class < 3:
        rng.shuffle(idx)
        n_train = max(1, int(train_frac * n))
        n_val = max(0, int(val_frac * n))
        return idx[:n_train].tolist(), idx[n_train:n_train + n_val].tolist(), idx[n_train + n_val:].tolist()

    train_idx, val_idx, test_idx = [], [], []
    for c in unique:
        c_idx = idx[labels == c]
        rng.shuffle(c_idx)
        n_c = int(c_idx.shape[0])
        n_train = max(1, int(round(train_frac * n_c)))
        n_val = max(0, int(round(val_frac * n_c)))
        n_train = min(n_train, n_c - 1)
        n_val = min(n_val, n_c - n_train - 1)
        train_idx.append(c_idx[:n_train])
        val_idx.append(c_idx[n_train:n_train + n_val])
        test_idx.append(c_idx[n_train + n_val:])

    train_idx = np.concatenate(train_idx) if train_idx else np.array([], dtype=int)
    val_idx = np.concatenate(val_idx) if val_idx else np.array([], dtype=int)
    test_idx = np.concatenate(test_idx) if test_idx else np.array([], dtype=int)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def _group_stratified_split_indices(
    labels: np.ndarray,
    groups: List[str],
    *,
    seed: int,
    train_frac: float,
    val_frac: float,
    n_tries: int = 200,
):
    """Heuristic subject-grouped stratified split.

    Keeps all windows from the same group (e.g., subject) in one split while
    trying to match global label distribution across splits.
    """
    base_seed = int(seed)
    rng = np.random.default_rng(base_seed)
    labels = np.asarray(labels)
    groups = np.asarray(groups)
    if labels.shape[0] != groups.shape[0]:
        raise ValueError(f"labels/groups length mismatch: {labels.shape[0]} vs {groups.shape[0]}")

    unique_labels = np.unique(labels)
    label_to_i = {int(l): i for i, l in enumerate(unique_labels.tolist())}
    k = int(unique_labels.size)

    # Build per-group histograms
    group_names = np.unique(groups)
    group_hists: Dict[str, np.ndarray] = {}
    group_sizes: Dict[str, int] = {}
    group_indices: Dict[str, np.ndarray] = {}

    for g in group_names:
        mask = (groups == g)
        idxs = np.where(mask)[0]
        group_indices[str(g)] = idxs
        group_sizes[str(g)] = int(idxs.size)
        hist = np.zeros(k, dtype=np.int64)
        for l in labels[mask]:
            hist[label_to_i[int(l)]] += 1
        group_hists[str(g)] = hist

    total = int(labels.shape[0])
    target_sizes = {
        "train": int(round(train_frac * total)),
        "val": int(round(val_frac * total)),
    }
    target_sizes["test"] = max(0, total - target_sizes["train"] - target_sizes["val"])

    global_hist = np.zeros(k, dtype=np.int64)
    for h in group_hists.values():
        global_hist += h
    global_total = int(global_hist.sum())

    # Target per-split label histograms: match global label proportions at the desired split sizes.
    split_fracs = {
        "train": float(train_frac),
        "val": float(val_frac),
        "test": float(max(0.0, 1.0 - float(train_frac) - float(val_frac))),
    }
    target_hists = {
        sp: (global_hist.astype(np.float64) * split_fracs[sp]) for sp in ("train", "val", "test")
    }

    # Randomized initialization + local search; pick best overall objective.
    base_order = [str(g) for g in group_names.tolist()]

    def _size_err(split_name: str, size: int) -> float:
        tgt = int(target_sizes.get(split_name, 0))
        return float(abs(int(size) - tgt)) / float(max(1, tgt))

    def _hist_err(split_name: str, hist: np.ndarray) -> float:
        # L1 distance to the target histogram (normalized by global_total).
        tgt = target_hists[split_name]
        return float(np.abs(hist.astype(np.float64) - tgt).sum()) / float(max(1, global_total))

    def _final_objective(split_hist: Dict[str, np.ndarray], split_size: Dict[str, int]) -> float:
        obj = 0.0
        size_weight = 3.0
        hist_weight = 2.0
        for sp in ("train", "val", "test"):
            obj += hist_weight * _hist_err(sp, split_hist[sp])
            obj += size_weight * _size_err(sp, split_size[sp])
        return obj

    best_assign = None
    best_obj = float("inf")

    groups_list = [str(g) for g in base_order]
    n_groups = len(groups_list)
    hists = np.stack([group_hists[g] for g in groups_list], axis=0)  # (G, K)
    sizes = np.asarray([group_sizes[g] for g in groups_list], dtype=np.int64)  # (G,)

    split_names = ("train", "val", "test")

    # Minimum number of groups per split (helps avoid tiny val/test splits).
    # Heuristic based on number of available groups.
    min_groups_per_split = 1
    if n_groups >= 6:
        min_groups_per_split = 2
    if n_groups >= 9:
        min_groups_per_split = 3

    def init_assignment(trial_rng: np.random.Generator) -> np.ndarray:
        # Assign groups while roughly meeting size targets.
        order = np.arange(n_groups)
        trial_rng.shuffle(order)
        assign = np.full(n_groups, -1, dtype=np.int64)
        cur_sizes = np.zeros(3, dtype=np.int64)
        targets = np.asarray([target_sizes["train"], target_sizes["val"], target_sizes["test"]], dtype=np.int64)

        # Seed each split with at least `min_groups_per_split` groups if possible.
        cursor = 0
        for sp_i in range(3):
            for _ in range(int(min_groups_per_split)):
                if cursor >= n_groups:
                    break
                gi = int(order[cursor])
                assign[gi] = sp_i
                cur_sizes[sp_i] += int(sizes[gi])
                cursor += 1

        for gi in order[cursor:]:
            gi = int(gi)
            if assign[gi] != -1:
                continue
            # Prefer the split most under target.
            deficits = targets - cur_sizes
            # If all are over target (can happen due to coarse group sizes), pick smallest split.
            if np.all(deficits <= 0):
                choice = int(np.argmin(cur_sizes))
            else:
                choice = int(np.argmax(deficits))
            assign[gi] = choice
            cur_sizes[choice] += int(sizes[gi])
        return assign

    def assignment_stats(assign: np.ndarray):
        split_hist = {sp: np.zeros(k, dtype=np.int64) for sp in split_names}
        split_size = {sp: 0 for sp in split_names}
        for sp_i, sp in enumerate(split_names):
            idxs = np.where(assign == sp_i)[0]
            if idxs.size:
                split_hist[sp] = hists[idxs].sum(axis=0)
                split_size[sp] = int(sizes[idxs].sum())
        return split_hist, split_size

    def objective(assign: np.ndarray) -> float:
        split_hist, split_size = assignment_stats(assign)
        return _final_objective(split_hist, split_size)

    n_tries = int(max(1, n_tries))
    for t in range(n_tries):
        trial_rng = np.random.default_rng(base_seed + 1337 * t)
        assign = init_assignment(trial_rng)

        # Local search: single-group moves that improve objective.
        cur_obj = objective(assign)
        # Limit iterations; increase a bit with number of groups.
        n_steps = int(max(500, 80 * n_groups))

        for _ in range(n_steps):
            gi = int(trial_rng.integers(0, n_groups))
            src = int(assign[gi])
            dst = int(trial_rng.integers(0, 3))
            if dst == src:
                continue

            # Avoid emptying a split completely.
            if int(np.sum(assign == src)) <= int(min_groups_per_split):
                continue

            # Soft constraint: don't let a move blow up size mismatch too much.
            split_hist, split_size = assignment_stats(assign)
            src_name = split_names[src]
            dst_name = split_names[dst]
            src_size_new = int(split_size[src_name] - int(sizes[gi]))
            dst_size_new = int(split_size[dst_name] + int(sizes[gi]))
            if src_size_new <= 0:
                continue
            # If a move overshoots the destination massively, skip.
            if dst_size_new > int(target_sizes[dst_name] * 1.25) and split_size[dst_name] > int(target_sizes[dst_name] * 0.9):
                continue

            trial = assign.copy()
            trial[gi] = dst
            new_obj = objective(trial)
            if new_obj < cur_obj:
                assign = trial
                cur_obj = new_obj

        if cur_obj < best_obj:
            best_obj = cur_obj
            best_assign = assign

    if best_assign is None:
        best_assign = np.zeros(n_groups, dtype=np.int64)

    splits = {"train": [], "val": [], "test": []}
    for sp_i, sp in enumerate(split_names):
        idxs = np.where(best_assign == sp_i)[0]
        splits[sp] = [groups_list[int(i)] for i in idxs.tolist()]

    # Expand back to window indices
    train_idx = np.concatenate([group_indices[g] for g in splits["train"]]) if splits["train"] else np.array([], dtype=int)
    val_idx = np.concatenate([group_indices[g] for g in splits["val"]]) if splits["val"] else np.array([], dtype=int)
    test_idx = np.concatenate([group_indices[g] for g in splits["test"]]) if splits["test"] else np.array([], dtype=int)

    # Shuffle within each split
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def compute_window_labels_and_groups(ds: "MotionDataset", *, group_key: str = "subject_id") -> Tuple[np.ndarray, List[str]]:
    """Compute per-window labels and aligned group ids.

    Returns:
        labels: (N,) int labels
        groups: (N,) group ids aligned to labels
    """
    if not hasattr(ds, "valid_windows"):
        raise ValueError("Dataset is missing valid_windows; did initialization fail?")

    # Use dataset-majority label logic to match downstream behavior.
    if not getattr(ds, "return_majority_label", False):
        raise ValueError("compute_window_labels_and_groups requires return_majority_label=True")

    group_key = str(group_key)
    groups: List[str] = []
    if getattr(ds, "mode", None) == "overlap":
        window_refs = getattr(ds, "valid_random_starts", [])
    else:
        window_refs = getattr(ds, "valid_windows", [])

    for (sample_idx, _window_offset) in window_refs:
        g = ds.samples[int(sample_idx)].get(group_key, "unknown")
        groups.append(str(g))

    # ds.get_all_labels() iterates windows in the same order as the active
    # index space: valid_random_starts for overlap mode, valid_windows otherwise.
    ys = ds.get_all_labels()
    if len(groups) != int(ys.shape[0]):
        # Fallback (should be rare): align to min length.
        n = min(len(groups), int(ys.shape[0]))
        ys = ys[:n]
        groups = groups[:n]

    labels = ys.astype(np.int64, copy=False)
    return np.asarray(labels, dtype=np.int64), groups


def make_or_load_frozen_split(
    ds: "MotionDataset",
    *,
    seed: int = 0,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    group_by_subject: bool = False,
    group_key: str = "subject_id",
    group_split_restarts: int = 200,
    ensure_label_coverage: bool = False,
    strict_label_coverage: bool = False,
    min_groups_per_label: int = 3,
    split_path: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, List[int]]:
    """Create (or load) a frozen split file for a dataset.

    The stored indices are indices into this dataset instance (window indices).
    If you change filtering knobs like max_windows/subsample_ratio/window_size,
    you should generate a new split file.

    Note: with group_by_subject=True, ensure_label_coverage applies a best-effort
    subject-level adjustment to cover feasible labels in train/val/test.
    """
    import json
    from collections import Counter
    from pathlib import Path

    if split_path is not None:
        split_file = Path(split_path)
        if split_file.exists() and not overwrite:
            with split_file.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            return {
                "train": list(payload["splits"]["train"]),
                "val": list(payload["splits"]["val"]),
                "test": list(payload["splits"]["test"]),
            }

    ys, groups = compute_window_labels_and_groups(ds, group_key=str(group_key))
    if ys.size == 0:
        return {"train": [], "val": [], "test": []}

    def _enforce_group_label_coverage(
        labels: np.ndarray,
        group_ids: List[str],
        splits_in: Dict[str, List[int]],
        *,
        min_groups_per_label: int,
        strict: bool,
        seed: int,
    ) -> Dict[str, List[int]]:
        """Best-effort enforcement that each feasible label appears in train/val/test.

        Operates at the group (subject) level to avoid leakage.
        """
        rng = np.random.default_rng(int(seed))

        labels = np.asarray(labels, dtype=np.int64)
        group_arr = np.asarray([str(g) for g in group_ids])
        if labels.shape[0] != group_arr.shape[0]:
            raise ValueError(f"labels/groups length mismatch: {labels.shape[0]} vs {group_arr.shape[0]}")

        unique_labels = np.unique(labels)
        group_names = np.unique(group_arr)

        # Build per-group indices and label-sets
        group_to_indices: Dict[str, np.ndarray] = {}
        group_to_labelset: Dict[str, set] = {}
        label_to_groups: Dict[int, set] = {int(l): set() for l in unique_labels.tolist()}

        for g in group_names:
            g = str(g)
            idxs = np.where(group_arr == g)[0]
            group_to_indices[g] = idxs
            ls = set(int(x) for x in np.unique(labels[idxs]).tolist())
            group_to_labelset[g] = ls
            for l in ls:
                label_to_groups[int(l)].add(g)

        # Only enforce labels that appear in enough distinct groups.
        enforce_labels = [l for l, gs in label_to_groups.items() if len(gs) >= int(min_groups_per_label)]

        # Convert existing window-index splits into group splits.
        split_to_groups: Dict[str, set] = {"train": set(), "val": set(), "test": set()}
        for sp in ("train", "val", "test"):
            idxs = np.asarray(splits_in.get(sp, []) or [], dtype=int)
            if idxs.size == 0:
                continue
            split_to_groups[sp] = set(str(g) for g in np.unique(group_arr[idxs]).tolist())

        # Helper: which labels are currently covered by a split
        def covered_labels(split_name: str) -> set:
            cov = set()
            for g in split_to_groups[split_name]:
                cov |= group_to_labelset.get(g, set())
            return cov

        # Iterate; try to fix missing labels by moving ONE group at a time.
        # Keep it conservative to avoid thrashing.
        for _pass in range(25):
            changed = False

            cov = {sp: covered_labels(sp) for sp in ("train", "val", "test")}
            missing = {
                sp: [l for l in enforce_labels if int(l) not in cov[sp]]
                for sp in ("train", "val", "test")
            }
            if not any(missing[sp] for sp in ("train", "val", "test")):
                break

            # Address the split with the most missing labels first.
            target_split = max(("train", "val", "test"), key=lambda sp: len(missing[sp]))
            if not missing[target_split]:
                break

            l = int(missing[target_split][0])

            # Candidate donor splits are those that have the label.
            donor_splits = [sp for sp in ("train", "val", "test") if sp != target_split and l in cov[sp]]
            rng.shuffle(donor_splits)

            moved = False
            for donor in donor_splits:
                # Find groups in donor that contain label l
                donor_groups = [g for g in split_to_groups[donor] if l in group_to_labelset.get(g, set())]
                rng.shuffle(donor_groups)
                for g in donor_groups:
                    # Do not move if it would make donor lose label l
                    donor_cov_without = cov[donor].copy()
                    # If g is the only provider of some enforced label in donor, avoid moving.
                    # Compute donor coverage without g cheaply by checking if any other group provides each label.
                    ok = True
                    for l2 in enforce_labels:
                        l2 = int(l2)
                        if l2 not in cov[donor]:
                            continue
                        if l2 in group_to_labelset.get(g, set()):
                            # would donor still have l2?
                            still = False
                            for gg in split_to_groups[donor]:
                                if gg == g:
                                    continue
                                if l2 in group_to_labelset.get(gg, set()):
                                    still = True
                                    break
                            if not still:
                                ok = False
                                break
                    if not ok:
                        continue

                    # Move group
                    split_to_groups[donor].remove(g)
                    split_to_groups[target_split].add(g)
                    changed = True
                    moved = True
                    break
                if moved:
                    break

            if not moved:
                if strict:
                    raise RuntimeError(
                        f"Unable to enforce label coverage for dataset={getattr(ds, 'dataset_name', 'unknown')} "
                        f"(missing label {l} in split {target_split})."
                    )
                # Best effort: stop trying.
                break

            if not changed:
                break

        # Expand groups back to window indices.
        out: Dict[str, List[int]] = {}
        for sp in ("train", "val", "test"):
            idxs = []
            for g in split_to_groups[sp]:
                idxs.append(group_to_indices[g])
            if idxs:
                merged = np.concatenate(idxs)
            else:
                merged = np.array([], dtype=int)
            rng.shuffle(merged)
            out[sp] = merged.tolist()
        return out

    if group_by_subject:
        train_idx, val_idx, test_idx = _group_stratified_split_indices(
            ys,
            groups,
            seed=seed,
            train_frac=train_frac,
            val_frac=val_frac,
            n_tries=int(group_split_restarts),
        )
    else:
        train_idx, val_idx, test_idx = _stratified_split_indices(ys, seed=seed, train_frac=train_frac, val_frac=val_frac)

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    if ensure_label_coverage and group_by_subject:
        splits = _enforce_group_label_coverage(
            ys,
            groups,
            splits,
            min_groups_per_label=int(min_groups_per_label),
            strict=bool(strict_label_coverage),
            seed=int(seed),
        )

    # For rare labels (appearing in fewer than min_groups_per_label groups), it is generally
    # impossible to cover train/val/test without leakage. As a pragmatic default, keep these
    # rare-label groups in the train split so downstream training can still see them.
    if group_by_subject and int(min_groups_per_label) > 1:
        rng = np.random.default_rng(int(seed) + 4242)
        labels = np.asarray(ys, dtype=np.int64)
        group_arr = np.asarray([str(g) for g in groups])

        # Build label->groups cardinalities
        label_to_groups: Dict[int, set] = {}
        for l in np.unique(labels).tolist():
            l = int(l)
            label_to_groups[l] = set(str(g) for g in np.unique(group_arr[labels == l]).tolist())
        rare_labels = [l for l, gs in label_to_groups.items() if len(gs) < int(min_groups_per_label)]
        if rare_labels:
            # Precompute group -> window indices
            group_names = np.unique(group_arr)
            group_to_indices: Dict[str, np.ndarray] = {str(g): np.where(group_arr == str(g))[0] for g in group_names.tolist()}

            train_set = set(int(i) for i in (splits.get("train", []) or []))
            val_set = set(int(i) for i in (splits.get("val", []) or []))
            test_set = set(int(i) for i in (splits.get("test", []) or []))

            for l in rare_labels:
                for g in label_to_groups[int(l)]:
                    idxs = set(int(i) for i in group_to_indices.get(str(g), np.array([], dtype=int)).tolist())
                    if not idxs:
                        continue
                    # Move the entire group into train.
                    val_set -= idxs
                    test_set -= idxs
                    train_set |= idxs

            # After forcing rare-label groups into train, re-run best-effort coverage enforcement
            # (for feasible labels) to avoid accidentally removing a label from val/test.
            splits = {"train": sorted(train_set), "val": sorted(val_set), "test": sorted(test_set)}
            if ensure_label_coverage:
                try:
                    splits = _enforce_group_label_coverage(
                        ys,
                        groups,
                        splits,
                        min_groups_per_label=int(min_groups_per_label),
                        strict=False,
                        seed=int(seed) + 99,
                    )
                except Exception:
                    # Keep best-effort; do not fail split creation here.
                    pass

                # Guarantee rare-label groups remain in train.
                train_set = set(int(i) for i in (splits.get("train", []) or []))
                val_set = set(int(i) for i in (splits.get("val", []) or []))
                test_set = set(int(i) for i in (splits.get("test", []) or []))
                for l in rare_labels:
                    for g in label_to_groups[int(l)]:
                        idxs = set(int(i) for i in group_to_indices.get(str(g), np.array([], dtype=int)).tolist())
                        val_set -= idxs
                        test_set -= idxs
                        train_set |= idxs

            train_idx = np.fromiter(train_set, dtype=int)
            val_idx = np.fromiter(val_set, dtype=int)
            test_idx = np.fromiter(test_set, dtype=int)
            rng.shuffle(train_idx)
            rng.shuffle(val_idx)
            rng.shuffle(test_idx)
            splits = {"train": train_idx.tolist(), "val": val_idx.tolist(), "test": test_idx.tolist()}

    if split_path is not None:
        split_file = Path(split_path)
        split_file.parent.mkdir(parents=True, exist_ok=True)

        def _counts(idxs: List[int]) -> Dict[str, int]:
            c = Counter(ys[np.asarray(idxs, dtype=int)].tolist())
            return {str(k): int(v) for k, v in sorted(c.items(), key=lambda kv: int(kv[0]))}

        payload = {
            "dataset": getattr(ds, "dataset_name", "unknown"),
            "dataset_name": getattr(ds, "dataset_name", "unknown"),
            "window_size": int(getattr(ds, "window_size", len(ys))),
            "window_duration_s": float(getattr(ds, "window_size", len(ys)) / getattr(ds, "sampling_rate", 1.0)),
            "sampling_rate": float(getattr(ds, "sampling_rate", 1.0)),
            "seed": int(seed),
            "train_frac": float(train_frac),
            "val_frac": float(val_frac),
            "group_by_subject": bool(group_by_subject),
            "group_key": str(group_key),
            "n_windows": int(ys.size),
            "splits": splits,
            "label_counts": {
                "train": _counts(list(splits.get("train", []) or [])),
                "val": _counts(list(splits.get("val", []) or [])),
                "test": _counts(list(splits.get("test", []) or [])),
            },
            "metadata": {
                "train_frac": float(train_frac),
                "val_frac": float(val_frac),
                "seed": int(seed),
                "strategy": "group_stratified_subject" if group_by_subject else "stratified",
                "placement": getattr(ds, "placement", None),
                "mode": getattr(ds, "mode", "nonoverlap"),
                "stride_samples": int(
                    getattr(ds, "overlap_stride_samples", 1)
                    if getattr(ds, "mode", "nonoverlap") == "overlap"
                    else getattr(ds, "window_size", len(ys))
                ),
                "stride_duration_s": float(
                    (
                        getattr(ds, "overlap_stride_samples", 1)
                        if getattr(ds, "mode", "nonoverlap") == "overlap"
                        else getattr(ds, "window_size", len(ys))
                    ) / getattr(ds, "sampling_rate", 1.0)
                ),
                "source_processed_dir": getattr(ds, "processed_dir", None),
            },
        }
        with split_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return splits


def plot_split_label_distributions(
    labels: np.ndarray,
    splits: Dict[str, List[int]],
    *,
    title: str,
    save_path: str,
    idx_to_label: Optional[Dict[int, str]] = None,
):
    """Plot per-split label distribution (proportions) and save as PNG."""
    from collections import Counter
    from pathlib import Path

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"matplotlib is not available; cannot plot distributions: {e}")

    labels = np.asarray(labels)
    all_classes = sorted({int(x) for x in np.unique(labels).tolist()})
    class_names = [idx_to_label.get(c, str(c)) if idx_to_label else str(c) for c in all_classes]

    split_names = ["train", "val", "test"]
    counts = {sp: np.zeros(len(all_classes), dtype=np.int64) for sp in split_names}
    totals = {sp: 0 for sp in split_names}
    for sp in split_names:
        idxs = np.asarray(splits.get(sp, []), dtype=int)
        totals[sp] = int(idxs.size)
        if idxs.size == 0:
            continue
        c = Counter(labels[idxs].tolist())
        for j, cls in enumerate(all_classes):
            counts[sp][j] = int(c.get(cls, 0))

    props = {sp: (counts[sp].astype(np.float64) / max(1, totals[sp])) for sp in split_names}

    fig = plt.figure(figsize=(max(10, 0.45 * len(all_classes)), 5))
    ax = fig.add_subplot(1, 1, 1)
    x = np.arange(len(all_classes))
    w = 0.27
    ax.bar(x - w, props["train"], width=w, label=f"train (N={totals['train']})")
    ax.bar(x, props["val"], width=w, label=f"val (N={totals['val']})")
    ax.bar(x + w, props["test"], width=w, label=f"test (N={totals['test']})")
    ax.set_title(title)
    ax.set_xlabel("Label")
    ax.set_ylabel("Proportion within split")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()

    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    # Quick smoke test. Point this at your own preprocessed data root
    # (<data_root>/<dataset_name>/processed/*.parquet).
    data_root = os.environ.get('MOTION_DATA_ROOT', './data/downstream')
    
    print("Available datasets:", list_available_datasets())

    # Test with DaphnetFOG
    print("\n=== Testing DaphnetFOG ===")
    try:
        dataset = MotionDataset(
            data_root=data_root,
            dataset_name='daphnet_fog',
            sampling_rate=20.0,
            window_size=200,
            sensor_types='accelerometer',
            axial_mode='triaxial',
        )
        print(f"Dataset length: {len(dataset)}")
        print(f"Num channels: {dataset.num_channels}")
        print(f"Num samples: {len(dataset.samples)}")
        
        if len(dataset) > 0:
            signal, label = dataset[0]
            print(f"Signal shape: {signal.shape}")
            print(f"Label shape: {label.shape}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test with USC-HAD (embedded labels)
    print("\n=== Testing USC-HAD ===")
    try:
        dataset = MotionDataset(
            data_root=data_root,
            dataset_name='USC-HAD',
            sampling_rate=10.0,  # Test resampling
            window_size=100,
            sensor_types='accelerometer',
            axial_mode='uniaxial',
        )
        print(f"Dataset length: {len(dataset)}")
        print(f"Num channels: {dataset.num_channels}")
        
        if len(dataset) > 0:
            signal, label = dataset[0]
            print(f"Signal shape: {signal.shape}")
            print(f"Label shape: {label.shape}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test with MHEALTH (multiple sensors)
    print("\n=== Testing MHEALTH with multiple sensors ===")
    try:
        dataset = MotionDataset(
            data_root=data_root,
            dataset_name='MHEALTHDATASET',
            sampling_rate=20.0,
            window_size=200,
            sensor_types=['accelerometer', 'gyroscope'],
            axial_mode='triaxial',
        )
        print(f"Dataset length: {len(dataset)}")
        print(f"Num channels: {dataset.num_channels}")
        
        if len(dataset) > 0:
            signal, label = dataset[0]
            print(f"Signal shape: {signal.shape}")
            print(f"Label shape: {label.shape}")
    except Exception as e:
        print(f"Error: {e}")