from dataset_preparer import load_patient_dataset
from data_converter import convert_to_full_file_spectrograms,convert_to_segment_level
from data_splitter import split_segmented_data_by_subject
from model_dataloader import get_loaders, MyHeartDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from model_dataloader import get_loaders
#from train_and_save_model import train_and_save_model
#from evaluate_model import evaluate_on_test
from evaluate_lstm_model import evaluate_lstm_model
from train_attention_model import train_attention_model
from train_lstm_model import train_lstm_model


# Step 1: Load data
##df = load_patient_dataset()

# Step 2: Convert to full-length spectrograms and then segmented
##spectro_df = convert_to_full_file_spectrograms()
##df_segmented = convert_to_segment_level(spectro_df)

# Step 3: Split data into train,test,dev dataset by considering subject id
#train_df, val_df, test_df = split_segmented_data_by_subject()

# Step 4 : Model running

# Data
#train_csv = "../data/generated_data/train_segments.csv"
#val_csv = "../data/generated_data/val_segments.csv"

#model = train_and_save_model(
#    train_csv="../data/generated_data/train_segments.csv",
#    val_csv="../data/generated_data/val_segments.csv",
#    model_path="../models/resnet18_fine_tuned_epochs4.pth",
#    batch_size=32,
#    epochs=4
#)

#train_lstm_model(
 #   train_csv="../data/generated_data/train_segments.csv",
 #   val_csv="../data/generated_data/val_segments.csv",
#    model_path="../models/lstm_finetuned_epoch10.pth",
#    batch_size=32,
#    epochs=10
#)


#evaluate_on_test(
#    model_path="../models/lstm_finetuned_epoch10.pth",
#    test_csv="../data/generated_data/test_segments.csv",
#    model_name="lstm_finetuned_epoch10",
#)


#train_attention_model(
#    train_csv="../data/generated_data/train_segments.csv",
#    val_csv="../data/generated_data/val_segments.csv",
#    model_path="../models/resnet34_attention_epoch5.pth",
#    batch_size=32,
#    epochs=6
#)
from evaluate_attention_model import evaluate_attention_model

evaluate_attention_model(
    model_path="../models/resnet34_attention_epoch5.pth",
    test_csv="../data/generated_data/test_segments.csv",
    model_name="resnet34_attention_epoch6"
)