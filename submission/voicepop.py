import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import pickle
import librosa
import random
from tqdm import tqdm
import winsound
from imblearn.over_sampling import SMOTE
from gammatone.filters import make_erb_filters, erb_filterbank
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


# read in dataset and populate dictionary with the labels for each
# audio file
def load_labels_from_dataset(path):
    labels = {}
    total_real = 0
    total_fake = 0

    with open(path, "r") as f:
        print("Loading labels")
        for label in tqdm(f, desc="Processing"):
            line_split = label.strip().split()
            file, label = line_split[1], line_split[4]
            if label == "bonafide" and total_real <= 2580:
                labels[file] = 1
                total_real += 1
            elif label == "spoof" and total_fake <= 2580:
                labels[file] = 0
                total_fake += 1

    print(f"Total Real Labels Loaded {total_real}")
    print(f"Total Fake Labels Loaded {total_fake}")
    return labels


# load in FLAC audio files from the dataset, extract pop noises
# and gfcc features in the audio file and then return
# the audio frequencies and labels for each file
def load_voices_from_dataset(path, labels):
    gfcc_features, associated_label = [], []
    total = 0

    # shuffle the labels
    shuffled_labels = list(labels.items())
    random.shuffle(shuffled_labels)

    # start the feature extraction
    print("Extracting GFCC features from audio samples.")
    for fn, label in tqdm(shuffled_labels, desc="Processing"):
        # get the path of the current filename
        voice_path = os.path.join(path, fn + ".flac")

        if os.path.exists(voice_path):
            # extract numerical rep of audio and sound rate
            audio, sr = librosa.load(voice_path, sr=16000)

            # extract the pop noises
            pop_noise = detect_pop_noise(audio, sr)

            # extract the gfcc features
            gfcc = gfcc_extraction(audio, sr, pop_noise)

            # if gfcc features found, append to the lists
            if gfcc is not None:
                gfcc_features.append(gfcc)
                associated_label.append(label)
                print(
                    f"Added GFCC for {fn}, Current count: x: {len(gfcc_features)}, y: {len(associated_label)}"
                )
            else:
                print(
                    f"No GFCC detected for {fn}, Current count: x: {len(gfcc_features)}, y: {len(associated_label)}"
                )
        total += 1

    # if gfcc_features is not empty, flatten the features and return
    if gfcc_features:
        gfcc_stack = np.vstack(gfcc_features)
        labels = np.array(associated_label)
        print(f"Loaded {len(labels)} labels and {gfcc_stack.shape[0]} GFCC samples")
        return gfcc_stack, labels
    else:
        print("No valid GFCC files found")
        return np.array([]), np.array([])


# get the pop noises at any sequence and return at which they mett the threshold
def detect_pop_noise(audio, sample_rate):
    # extract all frquencies within the audio sample and get the magnitude
    short_time_fourier_transform = np.abs(librosa.stft(audio))
    # key index for the stft matrix
    frequency = librosa.fft_frequencies(sr=sample_rate)
    # get the frequency keys for any value below 100 hz
    index = np.where(frequency <= 100)[0]

    # sum the time magnitudes of energy for pop noises under 100hz
    low_freq_energy = np.sum(short_time_fourier_transform[index, :], axis=0)
    # take an average of the pop noises and get a general threshold
    threshold = np.mean(low_freq_energy) + 2 * np.std(low_freq_energy)
    return np.where(low_freq_energy > threshold)[0]


# extracting GFCC features based on pop noises
def gfcc_extraction(audio, sample_rate, frames):
    # if no pop noises exist, simply return None
    if len(frames) == 0:
        return None

    # set up 32 gammatone filters to divide the signal to freq bands
    gfcc_feats = []
    gammatone_filters = make_erb_filters(sample_rate, 64)

    # process the pop noise frames
    for frame in frames:

        # get the window of the start and end of the pop noise
        start = int((frame - 1) * 512)
        end = int((frame + 1) * 512)
        # extract the part where the pop occured
        segment = audio[start:end]

        # if no frame was detected, skip it
        if len(segment) == 0:
            print(f"Empty frame detected at {frame}")
            continue

        try:
            # get the frequency bands for the segment
            filtered_segment = erb_filterbank(segment, gammatone_filters)

            # transform the energies to filter out background noise
            # focus on the pop noise and compress the background noise
            gfcc = np.log(np.abs(filtered_segment) + 1e-8)
            gfcc_mean = np.mean(gfcc, axis=1)

            # compress the features if we have more than 9 features
            if gfcc_mean.shape[0] >= 9:
                # timing difference between spectral features
                delta_gfcc = librosa.feature.delta(gfcc_mean)
                # acceleration of the change between background noise and a pop noise
                delta_order2_gfcc = librosa.feature.delta(gfcc_mean, order=2)

                # combine the features into a single feature vector
                combined_features = np.hstack(
                    (gfcc_mean, delta_gfcc, delta_order2_gfcc)
                )
            else:
                combined_features = gfcc_mean
            # append to the gfcc features
            gfcc_feats.append(combined_features)
        except Exception as e:
            print(f"Error with extracting GFCC: {e}")
            continue
    if gfcc_feats:
        return np.mean(gfcc_feats, axis=0)

    return None


def train_svm_model(x, y):
    # compute the weights of the classes
    class_weights_comp = compute_class_weight(
        class_weight="balanced", classes=np.array([0, 1]), y=y
    )
    class_weight_dict = {0: class_weights_comp[0], 1: class_weights_comp[1]}

    # imputer used for parts of the GFCC extraction that returned None
    imputer = SimpleImputer(strategy="mean")
    x_imputed = imputer.fit_transform(x)

    # scaling all inputs to ensure each feature is close to unit variance and zero mean
    # and no feature overpowers the other
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_imputed)

    # #handle the class imbalance issue (creates syntehtic samples)
    smote = SMOTE(random_state=42)
    x_resample, y_resample = smote.fit_resample(x_scaled, y)

    # hyperparams for training
    hyperparams = {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
    }
    svm = GridSearchCV(
        SVC(probability=True, class_weight=class_weight_dict),
        hyperparams,
        cv=5,
        n_jobs=-1,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x_resample, y_resample, test_size=0.2, random_state=42, stratify=y_resample
    )

    # get the best model
    svm.fit(x_train, y_train)
    best_model = svm.best_estimator_

    y_pred = best_model.predict(x_test)
    y_proba = best_model.predict_proba(x_test)[:, 1]

    # accuracy report and ROC curve score
    print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred, target_names=["Spoof", "Real"]))
    print(f"AUC: {roc_auc_score(y_test, y_proba)}")

    winsound.PlaySound("DING.WAV", winsound.SND_FILENAME)

    # save the model with the scaler
    save_file = input("Save model as (no extension): ")
    with open(save_file + ".pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open(save_file + "_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("Model has been saved")


def main():
    # proto = labels, dataset_path = audio files
    proto_path = (
        "./ASVspoof/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
    )
    dataset_path = "./ASVspoof/ASVspoof2019_LA_train/flac"

    # get the labels
    labels = load_labels_from_dataset(proto_path)

    # extract gfcc features
    gfcc_train, label_train = load_voices_from_dataset(dataset_path, labels)

    # train model
    if gfcc_train.size > 0 and label_train.size > 0:
        print("Starting training")
        train_svm_model(gfcc_train, label_train)
    else:
        print("Training data is not valid.")


main()
