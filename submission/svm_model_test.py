import pickle
import librosa
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from gammatone.filters import make_erb_filters, erb_filterbank
from multiprocessing import Pool
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# extract the labels for the evaluation set
def label_extract(path):
    labels = {}
    with open(path, "r") as f:
        for label in f:
            line_split = label.strip().split()
            file, label = line_split[1], line_split[4]
            if label == "bonafide":
                labels[file] = "Real"
            else:
                labels[file] = "Fake"
    return labels


# extract the GFCC features
def gfcc_extraction(audio, sr):

    # set up 32 gammatone filters to divide the signal to freq bands
    gfcc_feats = []
    gammatone_filters = make_erb_filters(sr, 64)

    # process the frames of the audio
    for i in range(0, len(audio), 512):
        segment = audio[i : i + 512]
        # if no frame was detected, skip it
        if len(segment) == 0:
            print(f"Empty frame detected at index {i}")
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
            print(f"Error extracting GFCC: {e}")
            continue

    if gfcc_feats:
        return np.mean(gfcc_feats, axis=0)
    return None


# classification process based on the saved model and scaler
def process_file(file, threshold, svm, scaler):
    label = None  # label classification from model
    prob = None  # exact probability value

    # load the audio file, extract gfcc features, and get probability
    # of being real or fake
    try:
        audio, sr = librosa.load(file, sr=16000)
        gfcc = gfcc_extraction(audio, sr).reshape(1, -1)
        gfcc_scaled = scaler.transform(gfcc)
        prob_pred = svm.predict_proba(gfcc_scaled)
        prob = np.round(prob_pred[0][1], 2)

        print(prob)

        # assign label based on threshold
        if prob >= threshold:
            label = "Real"
        else:
            label = "Fake"
    except Exception as e:
        print(f"Error with processing file {e}")
    return prob, label


# process entire evaluation dataset and get overall model
# performance
def process_audio_files(path, threshold, dataset_labels):
    total_real = 0
    total_fake = 0
    total_errors = 0

    probabilities = []  # probabilities of all audio files (fake or real)
    base_truth_labels = []  # total real
    predicted_labels = []  # total fake

    # load the entire dataset files and load the models
    files = glob.glob(os.path.join(path, "*"))
    with open("./models_used/even_test_cv_5.pkl", "rb") as f:
        svm = pickle.load(f)
    with open("./models_used/even_test_cv_5_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # process the entire dataset and get the predictions
    with Pool(processes=4) as pool:
        predictions = pool.starmap(
            process_file, [(file, threshold, svm, scaler) for file in files]
        )

    # keep track of all real and fake
    for file, (prob, label) in zip(files, predictions):
        if label == "Fake":
            total_fake += 1
        elif label == "Real":
            total_real += 1
        elif label is None or prob is None:
            total_errors += 1
            continue
        probabilities.append(prob)

        # get the base truth label of the current file
        file_name = os.path.basename(file).split(".")[0]
        dataset_label = dataset_labels.get(file_name, None)

        # keep track of the base truth and the model predicted label
        if dataset_label == "Real" or dataset_label == "Fake":
            base_truth_labels.append(dataset_label)
            predicted_labels.append(label)

    print("Done!")
    print(f"Total Real Identified {total_real}")
    print(f"Total Fake Identified {total_fake}")
    print(f"Total Files Tested {len(probabilities)}")
    print(f"Average Prob {np.mean(probabilities)}")

    # return the probabilities, base truth, and predicted labels
    return probabilities, base_truth_labels, predicted_labels


def plot_confusion_matrix(dataset_labels, predicted_labels):
    cm = confusion_matrix(dataset_labels, predicted_labels, labels=["Real", "Fake"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()


def plot_auc(prob, threshold):
    kde = gaussian_kde(np.array(prob))
    x = np.linspace(0, 1, 100)
    density = kde(x)

    plt.figure(figsize=(10, 6))
    plt.plot(x, density, label="Probability Density Curve")
    plt.axvline(
        threshold, color="red", linestyle="--", label=f"Threshold = {threshold}"
    )
    plt.fill_between(
        x,
        density,
        where=(x >= threshold),
        alpha=0.3,
        color="green",
        label="Accepted (Real)",
    )
    plt.fill_between(
        x,
        density,
        where=(x < threshold),
        alpha=0.3,
        color="blue",
        label="Rejected (Fake)",
    )

    # Add labels and legend
    plt.title("Bell Curve of Acceptance Rate", fontsize=14)
    plt.xlabel("Probability of Being Real", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.4)
    plt.show()


if __name__ == "__main__":

    def main():
        # define threshold for acceptance
        threshold = 0.5

        # load dataset
        dataset_path = "./ASVspoof/ASVspoof2019_LA_eval/flac"
        # dataset_path = "./fake_tests" ASVspoof\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.train.trn.txt
        dataset_label_path = (
            "ASVspoof\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.eval.trl.txt"
        )

        # extract the labels
        dataset_labels = label_extract(dataset_label_path)

        print("Processing audio")
        prob, base, pred = process_audio_files(dataset_path, threshold, dataset_labels)

        # plot the probability dist and the confusion matrix
        plot_auc(prob, threshold)
        plot_confusion_matrix(base, pred)

    main()
