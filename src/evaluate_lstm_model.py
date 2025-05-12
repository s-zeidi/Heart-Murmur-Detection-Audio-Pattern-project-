import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from model_dataloader import MyHeartDataset
from torch.utils.data import DataLoader
import pandas as pd
import os


class LSTMSoundClassifier(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, num_layers=2, num_classes=3):
        super(LSTMSoundClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        x = x.squeeze(1).permute(0, 2, 1)  # [B, 1, 128, T] -> [B, T, 128]
        output, _ = self.lstm(x)
        output = self.dropout(output[:, -1, :])
        return self.fc(output)


def evaluate_lstm_model(model_path, test_csv, model_name="lstm_eval", batch_size=32):
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    test_dataset = MyHeartDataset(test_csv)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = LSTMSoundClassifier()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    report = classification_report(
        all_labels,
        all_preds,
        target_names=["Absent", "Present", "Unknown"],
        output_dict=True
    )

    print("\n📊 Classification Report on Test Set:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=["Absent", "Present", "Unknown"]
    ))

    overall_f1 = f1_score(all_labels, all_preds, average="macro")
    accuracy = report["accuracy"]
    macro_precision = report["macro avg"]["precision"]
    macro_recall = report["macro avg"]["recall"]

    print(f"\n🔢 Overall Macro F1 Score: {overall_f1:.4f}")

    results_row = {
        "Model Name": model_name,
        "Accuracy": round(accuracy, 4),
        "Macro F1": round(overall_f1, 4),
        "Macro Precision": round(macro_precision, 4),
        "Macro Recall": round(macro_recall, 4),
        "Absent F1": round(report["Absent"]["f1-score"], 4),
        "Present F1": round(report["Present"]["f1-score"], 4),
        "Unknown F1": round(report["Unknown"]["f1-score"], 4),
    }

    csv_path = "../results/evaluation_results.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([results_row])], ignore_index=True)
    else:
        df = pd.DataFrame([results_row])

    df.to_csv(csv_path, index=False)
    print(f"\n📁 Evaluation results saved to: {csv_path}")

    return all_preds, all_labels
