def count_labels(path):
    total_real = 0
    total_fake = 0
    labels = {}
    with open(path, "r") as f:
        for label in f:
            line_split = label.strip().split()
            file, label = line_split[1], line_split[4]
            if label == "bonafide":
                total_real += 1
            else:
                total_fake += 1
    return total_real, total_fake

def main():
    real, fake = count_labels("./ASVspoof/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt")
    print(f"Total Real: {real}")
    print(f"Total Fake: {fake}")

main()