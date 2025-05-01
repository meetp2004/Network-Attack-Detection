import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, classification_report
from sklearn.utils import compute_class_weight, shuffle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# This function reads the CSVs, turns it into a parquet file, and then preprocesses the data, placing them into train and test parquet files
def preprocess ():

    # This function reads all the CSVs and returns a main parquet file
    def process_files_to_parquet():
        path = "TrafficLabelling/" #Directory with all the CSVs (Make sure to have this first)
        data_frames = []

        file_paths = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        
        for file_path in file_paths:
            df_added = pd.read_csv(file_path)
            data_frames.append(df_added)
        if data_frames:
            final_df = pd.concat(data_frames,ignore_index=True)
            final_df.to_parquet('data/traffic_data_1.parquet',index=False)
            
    def load_parquet(file_path):
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:  # Catch specific exceptions if needed
            print(f"Error loading parquet file: {e}")
            return None  # Return None to indicate failure
        return df

    def replace_inf(df):
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=['Flow Bytes/s', ' Flow Packets/s'], inplace=True)
        assert not df.isin([np.inf, -np.inf]).any().any(), "Infinity values still exist!"
        assert not df.isna().any().any(), "NaN values still exist!"
        return df

    def get_data(file_path, target_col_name, feature_col_name):
        df = load_parquet(file_path)
        if df is None:
            raise ValueError("Failed to load parquet file. Please check the file path or content.")
        
        df = replace_inf(df)
        
        Y = df.loc[:, target_col_name]
        X = df.loc[:, feature_col_name]
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
        return X_train, X_test, Y_train, Y_test
    
    process_files_to_parquet()
    features_list = [
            ' Flow Duration', 'Total Length of Fwd Packets', ' Total Length of Bwd Packets', 
            ' Fwd Packet Length Max', ' Fwd Packet Length Min', ' Fwd Packet Length Mean', 
            ' Fwd Packet Length Std', 'Bwd Packet Length Max', ' Bwd Packet Length Min', 
            ' Bwd Packet Length Mean', ' Bwd Packet Length Std', 'Flow Bytes/s', ' Flow Packets/s', 
            ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min', 'Fwd IAT Total', 
            ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max', ' Fwd IAT Min', 'Bwd IAT Total', 
            ' Bwd IAT Mean', ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min', 'Fwd PSH Flags', 
            ' Fwd URG Flags', ' Fwd Header Length', ' Bwd Header Length', 'Fwd Packets/s', 
            ' Bwd Packets/s', ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean', 
            ' Packet Length Std', ' Packet Length Variance', 'FIN Flag Count', ' SYN Flag Count', 
            ' RST Flag Count', ' PSH Flag Count', ' ACK Flag Count', ' URG Flag Count', 
            ' CWE Flag Count', ' ECE Flag Count', ' Down/Up Ratio', ' Average Packet Size', 
            ' Avg Fwd Segment Size', ' Avg Bwd Segment Size', ' Fwd Header Length.1', 
            'Subflow Fwd Packets', ' Subflow Fwd Bytes', ' Subflow Bwd Packets', 
            ' Subflow Bwd Bytes', 'Init_Win_bytes_forward', ' Init_Win_bytes_backward', 
            ' act_data_pkt_fwd', ' min_seg_size_forward', 'Active Mean', ' Active Std', 
            ' Active Max', ' Active Min', 'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min'
    ]

    X_train, X_test, Y_train, Y_test = get_data('data/traffic_data.parquet', ' Label', features_list) 
    
    Y_train = Y_train.to_numpy().reshape(-1, 1)
    Y_test = Y_test.to_numpy().reshape(-1, 1)

    train = pd.concat([X_train, pd.DataFrame(Y_train, columns=[' Label'], index=X_train.index)], axis=1)
    test = pd.concat([X_test, pd.DataFrame(Y_test, columns=[' Label'], index=X_test.index)], axis=1)

    train.to_parquet('data/train_features.parquet', index=False)
    test.to_parquet('data/test_features.parquet', index=False)

def train_SVM():
    # Define attack types for each level
    level_1_labels = {
        'BENIGN': 0,
        'ATTACK': 1
    }


    group_labels = {
        'DoS/DDoS': ['DoS GoldenEye', 'DoS Hulk', 'DoS Slowhttptest', 'DoS slowloris', 'DDoS'],
        'Brute Force': ['FTP-Patator', 'SSH-Patator', 'Heartbleed', 'Bot'],
        'Reconnaissance': ['PortScan', 'Infiltration']
    }

    # Map the level 2 groups to numerical encoding
    group_mapping = {
        'DoS/DDoS': 0,
        'Brute Force': 1,
        'Reconnaissance': 2
    }
    
    # Function to create hierarchical labels
    def create_hierarchical_labels(data):
        """Create hierarchical labels for each level of classification"""
        
        # Level 1: Binary classification (BENIGN vs ATTACK)
        data['Level_1_Label'] = data[' Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
        
        # Level 2: Attack group classification
        def get_group_label(attack_type):
            if attack_type == 'BENIGN':
                return -1
            for group, attacks in group_labels.items():
                if attack_type in attacks:
                    return group_mapping[group]
            return -1
        
        data['Level_2_Label'] = data[' Label'].apply(get_group_label)
        
        
        # Level 3: Specific attack type within group
        attack_type_mapping = {}
        for group, attacks in group_labels.items():
            for i, attack in enumerate(attacks):
                attack_type_mapping[attack] = i
        
        def get_specific_attack_label(row):
            if row[' Label'] == 'BENIGN':
                return -1
            return attack_type_mapping.get(row[' Label'], -1)
        
        data['Level_3_Label'] = data.apply(get_specific_attack_label, axis=1)
        
        return data

    def prepare_data_for_cv():
        print("Loading and preprocessing data...")
        file_path = 'data/train_features.parquet'
        data = pd.read_parquet(file_path)
        data = create_hierarchical_labels(data)

        # Balance the dataset
        balanced_samples = []
        max_samples_per_class = 5000
        for label in np.unique(data[' Label']):
            class_data = data[data[' Label'] == label]
            if label != 'BENIGN':
                if len(class_data) > max_samples_per_class:
                    class_data = class_data.sample(max_samples_per_class, random_state=42)
            else:
                if len(class_data) > max_samples_per_class:
                    class_data = class_data.sample(50000, random_state=42)
            balanced_samples.append(class_data)
        
        balanced_data = pd.concat(balanced_samples)
        
        X = balanced_data.drop(columns=[' Label', 'Level_1_Label', 'Level_2_Label', 'Level_3_Label'])
        y_level_1 = balanced_data['Level_1_Label']
        y_level_2 = balanced_data['Level_2_Label']
        y_level_3 = balanced_data['Level_3_Label']
        
        return X, y_level_1, y_level_2, y_level_3

    def cross_validate_hierarchical(X, y_level_1, y_level_2, y_level_3, n_splits=5):
        level_2_names = {
        0: 'DoS/DDoS',
        1: 'Brute Force',
        2: 'Reconnaissance'
        }
    
        level_3_names = {
            0: {  # DoS/DDoS attacks
                0: 'DoS GoldenEye',
                1: 'DoS Hulk',
                2: 'DoS Slowhttptest',
                3: 'DoS slowloris',
                4: 'DDoS'
            },
            1: {  # Brute Force attacks
                0: 'FTP-Patator',
                1: 'SSH-Patator',
                2: 'Heartbleed',
                3: 'Bot'
            },
            2: {  # Reconnaissance attacks
                0: 'PortScan',
                1: 'Infiltration'
            }
        }
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Store metrics for each fold
        metrics = {
            'level_1': [],
            'level_2': [],
            'level_3': {0: [], 1: [], 2: []}
        }
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_level_1)):
            
            print(f"\nFold {fold + 1}/{n_splits}")
            
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train_l1, y_test_l1 = y_level_1.iloc[train_idx], y_level_1.iloc[test_idx]
            y_train_l2, y_test_l2 = y_level_2.iloc[train_idx], y_level_2.iloc[test_idx]
            y_train_l3, y_test_l3 = y_level_3.iloc[train_idx], y_level_3.iloc[test_idx]
            
            # Scale features
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Level 1: Binary classification
            level_1_model = SVC(kernel='rbf', random_state=42, C=1)
            level_1_model.fit(X_train_scaled, y_train_l1)
            l1_preds = level_1_model.predict(X_test_scaled)
            l1_report = classification_report(y_test_l1, l1_preds, target_names=['BENIGN', 'ATTACK'], output_dict=True)
            metrics['level_1'].append(l1_report)
            
            # Level 2: Attack group classification
            attack_mask_train = y_train_l1 == 1
            attack_mask_test = y_test_l1 == 1
            
            level_2_model = SVC(kernel='linear', random_state=42, C=0.5)
            valid_l2_mask_train = attack_mask_train & (y_train_l2 != -1)
            valid_l2_mask_test = attack_mask_test & (y_test_l2 != -1)
            
            if np.any(valid_l2_mask_train):
                level_2_model.fit(X_train_scaled[valid_l2_mask_train], 
                                y_train_l2[valid_l2_mask_train])
                l2_preds = level_2_model.predict(X_test_scaled[valid_l2_mask_test])
                l2_report = classification_report(y_test_l2[valid_l2_mask_test], l2_preds, target_names=[level_2_names[i] for i in range(3)], output_dict=True)
                metrics['level_2'].append(l2_report)
            level_3_models = {}
            # Level 3: Specific attack classification
            for group_id in range(3):
                print(group_id)
                group_mask_train = (y_train_l2 == group_id) & attack_mask_train
                group_mask_test = (y_test_l2 == group_id) & attack_mask_test
                
                if np.any(group_mask_train):
                    X_group_train = X_train_scaled[group_mask_train & (y_train_l3 != -1)]
                    y_group_train = y_train_l3[group_mask_train & (y_train_l3 != -1)]
                    
                    # Apply SMOTE
                    smote = SMOTE(random_state=42)
                    X_group_train_smote, y_group_train_smote = smote.fit_resample(
                        X_group_train, y_group_train)
                    
                    group_model = SVC(kernel='linear', random_state=42, C=0.01)
                    group_model.fit(X_group_train_smote, y_group_train_smote)
                    level_3_models[group_id] = group_model
                    valid_l3_mask_test = group_mask_test & (y_test_l3 != -1)
                    if np.any(valid_l3_mask_test):
                        valid_labels = sorted(list(set(y_test_l3[valid_l3_mask_test])))
                        l3_preds = group_model.predict(X_test_scaled[valid_l3_mask_test])
                        l3_report = classification_report(y_test_l3[valid_l3_mask_test], 
                                                    l3_preds, target_names=[level_3_names[group_id][i] for i in valid_labels], output_dict=True, labels=valid_labels)
                        metrics['level_3'][group_id].append(l3_report)
        
        return scaler, metrics, level_1_model, level_2_model, level_3_models

    def print_cv_results(metrics):
        
        # Level 1 results
        l1_accuracy = np.mean([m['accuracy'] for m in metrics['level_1']])
        l1_std = np.std([m['accuracy'] for m in metrics['level_1']])
        print(f"\nLevel 1 CV Results:")
        print(f"Average Accuracy: {l1_accuracy:.3f} ± {l1_std:.3f}")
        
        # Level 2 results
        if metrics['level_2']:
            l2_accuracy = np.mean([m['accuracy'] for m in metrics['level_2']])
            l2_std = np.std([m['accuracy'] for m in metrics['level_2']])
            print(f"\nLevel 2 CV Results:")
            print(f"Average Accuracy: {l2_accuracy:.3f} ± {l2_std:.3f}")
        
        # Level 3 results
        print("\nLevel 3 CV Results by Group:")
        for group_id in range(3):
            if metrics['level_3'][group_id]:
                l3_accuracy = np.mean([m['accuracy'] for m in metrics['level_3'][group_id]])
                l3_std = np.std([m['accuracy'] for m in metrics['level_3'][group_id]])
                print(f"Group {group_id} Average Accuracy: {l3_accuracy:.3f} ± {l3_std:.3f}")

    
    X, y_level_1, y_level_2, y_level_3 = prepare_data_for_cv()
    scaler, metrics, level_1_model, level_2_model, level_3_models = cross_validate_hierarchical(X, y_level_1, y_level_2, y_level_3)
    print_cv_results(metrics)

    # Save models to files
    with open('weights/level_1_model.pkl', 'wb') as f:
        pickle.dump(level_1_model, f)
    with open('weights/level_2_model.pkl', 'wb') as f:
        pickle.dump(level_2_model, f)
    with open('weights/level_3_models.pkl', 'wb') as f:
        pickle.dump(level_3_models, f)
    with open('weights/SVM_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("Models saved successfully.")

def train_Random_Forest():
    # Load and preprocess data
    file_path = 'data/train_features.parquet'
    data = pd.read_parquet(file_path)

    # Encode labels
    label_encoder = LabelEncoder()
    data[' Label'] = label_encoder.fit_transform(data[' Label'])

    # Prepare features and labels
    features = data.drop(columns=[' Label'])
    constant_features_idx = [28, 40, 44, 45]
    features = features.drop(features.columns[constant_features_idx], axis=1)
    labels = data[' Label']

    # Define configurations
    configurations = [
        {'n_estimators': 100, 'criterion': 'entropy', 'name': 'rf_entropy_100'},
        {'n_estimators': 150, 'criterion': 'entropy', 'name': 'rf_entropy_150'},
        {'n_estimators': 200, 'criterion': 'entropy', 'name': 'rf_entropy_200'},
        {'n_estimators': 100, 'criterion': 'gini', 'name': 'rf_gini_100'},
        {'n_estimators': 150, 'criterion': 'gini', 'name': 'rf_gini_150'},
        {'n_estimators': 200, 'criterion': 'gini', 'name': 'rf_gini_200'}
    ]

    # Initialize cross-validation
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Store best models and scores for each configuration
    best_models = {}
    best_scalers = {}
    best_f1_scores = {config['name']: -1 for config in configurations}

    for config in configurations:
        print(f"\nTraining model with {config['criterion']} criterion and {config['n_estimators']} trees")
        cv_scores = []

        for fold, (train_index, test_index) in enumerate(skf.split(features, labels)):
            print(f"\nFold {fold + 1}/{n_splits}")
            
            # Split data for current fold
            X_train, X_test = features.iloc[train_index], features.iloc[test_index]
            y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
            
            # Balance classes
            benign_class = 0
            X_train_benign = X_train[y_train == benign_class]
            y_train_benign = y_train[y_train == benign_class]
            X_train_minority = X_train[y_train != benign_class]
            y_train_minority = y_train[y_train != benign_class]
            
            X_train_combined = np.vstack([X_train_benign, X_train_minority])
            y_train_combined = np.hstack([y_train_benign, y_train_minority])
            
            # Downsample to max_samples_per_class
            max_samples_per_class = 150000
            X_train_resampled = []
            y_train_resampled = []
            
            for class_label in np.unique(y_train_combined):
                mask = y_train_combined == class_label
                X_class = X_train_combined[mask]
                y_class = y_train_combined[mask]
                
                if len(X_class) > max_samples_per_class:
                    indices = np.random.choice(len(X_class), max_samples_per_class, replace=False)
                    X_class = X_class[indices]
                    y_class = y_class[indices]
                
                X_train_resampled.append(X_class)
                y_train_resampled.append(y_class)
            
            X_train_resampled = np.vstack(X_train_resampled)
            y_train_resampled = np.hstack(y_train_resampled)
            
            # Apply SMOTE
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_resampled, y_train_resampled)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_balanced)
            X_test_scaled = scaler.transform(X_test)
            
            # Train and evaluate model
            model = RandomForestClassifier(
                n_estimators=config['n_estimators'],
                criterion=config['criterion'],
                random_state=42
            )
            model.fit(X_train_scaled, y_train_balanced)
            
            y_pred = model.predict(X_test_scaled)
            fold_f1 = f1_score(y_test, y_pred, average='macro')
            cv_scores.append(fold_f1)
            
            print(f"Fold {fold + 1} F1 Score: {fold_f1:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
            
            if fold_f1 > best_f1_scores[config['name']]:
                best_f1_scores[config['name']] = fold_f1
                best_models[config['name']] = model
                best_scalers[config['name']] = scaler

        # Print results for current configuration
        print(f"\nResults for {config['name']}:")
        print(f"Mean F1 Score: {np.mean(cv_scores):.4f} Â± {np.std(cv_scores):.4f}")

    # Save all best models and their scalers
    for config in configurations:
        model_name = config['name']
        if model_name in best_models:
            # Save model
            model_filename = f"weights/{model_name}.pkl"
            with open(model_filename, "wb") as f:
                pickle.dump(best_models[model_name], f)
            print(f"Saved model as '{model_filename}'")
            
            # Save scaler
            scaler_filename = f"weights/{model_name}_scaler.pkl"
            with open(scaler_filename, "wb") as f:
                pickle.dump(best_scalers[model_name], f)
            print(f"Saved scaler as '{scaler_filename}'")

def train_cnn ():

    def build_cnn(input_shape, num_classes):
        model = Sequential([
            # First Conv1D Layer
            Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            
            Dropout(0.5),
                        
            # Second Conv1D Layer
            Conv1D(filters=16, kernel_size=3, activation='relu'),
            BatchNormalization(),
            
            Dropout(0.5),
                        
            # Third Conv1D Layer
            Conv1D(filters=16, kernel_size=3, activation='relu'),
            BatchNormalization(),
            
            Dropout(0.5),
                        
            # Fourth Conv1D Layer
            Conv1D(filters=16, kernel_size=3, activation='relu'),
            BatchNormalization(),
                        
            Dropout(0.5),            
                        
            # Fifth Conv1D Layer
            Conv1D(filters=16, kernel_size=3, activation='relu'),
            BatchNormalization(),
                        
            # Dropout Layer
            Dropout(0.5),
            
            # Max Pooling
            MaxPooling1D(pool_size=2),
            
            # Flatten Layer
            Flatten(),
            
            # Dense Layers
            Dense(100, activation='relu'),
            Dense(75, activation='relu'),
            Dense(50, activation='relu'),
            Dense(25, activation='relu'),
            Dense(num_classes, activation='softmax')  # Final output layer
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def map_labels (df, col_name):
        mapping = {}
        val = 0
        for label in df[col_name].unique():
            mapping[label] = val
            val += 1
        df[col_name] = df[col_name].map(mapping)
        return df, mapping
    
    training_data = pd.read_parquet("data/train_features.parquet")
    training_data, mapping = map_labels(training_data, ' Label')
 
    new_map = {}
    for key, value in mapping.items():
        new_map[value] = key
    
    X = training_data.iloc[:, :-1].to_numpy()
    Y = training_data.iloc[:, -1].to_numpy()
        
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.fit_transform(X_val)
    
    desired_samples_per_class = 150_000
    sampling_strategy_undersample = {cls : desired_samples_per_class for cls in Counter(Y_train).keys()}
    
    pipeline = Pipeline([
        ('smote', SMOTE(sampling_strategy='auto', random_state=42)),
        ('undersample', RandomUnderSampler(sampling_strategy=sampling_strategy_undersample, random_state=42))
    ])
    
    print("Original Training set class distribution (unchanged):", Counter(Y_train))
    X_train_resampled, Y_train_resampled = pipeline.fit_resample(X_train, Y_train)
    print("New Training set class distribution (unchanged):", Counter(Y_train_resampled))
        
    # # Reshape for CNN
    X_train_resampled = X_train_resampled.reshape(X_train_resampled.shape[0], X_train_resampled.shape[1], 1)
    X_val_cnn = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    
    # Print class distribution
    print("Resampled Training Data Class Distribution:")
    unique, counts = np.unique(Y_train_resampled, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"Class {u} ('{new_map[u]}'): {c} samples")
    
    # Callbacks for training
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True
    )
    
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=5, 
        min_lr=1e-6
    )
    
    # Build and train the model
    input_shape = (X_train_resampled.shape[1], X_train_resampled.shape[2])
    num_classes = len(np.unique(Y_train))
    
    model = build_cnn(input_shape, num_classes)
    
    history = model.fit(
        X_train_resampled, Y_train_resampled,
        validation_data=(X_val_cnn, Y_val),
        epochs=15,  # Increased Ts for more complex training
        batch_size=32,  # Adjusted batch size
        callbacks=[early_stopping, lr_scheduler]
    )
    
    weights = model.get_weights()
    with open("weights/new_cnn_weights.pkl", 'wb') as f:
        pickle.dump(weights, f)
    print("Model weights saved to the file!")

print("Preprocessing Stage...")
preprocess()
print("CNN Training Stage...")
train_cnn()
print("SVM Training Stage...")
train_SVM()
print("Random Forest Training Stage...")
train_Random_Forest()
print("Training Done...")