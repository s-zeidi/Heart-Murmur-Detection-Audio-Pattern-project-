from dataset_preparer import load_patient_dataset
from data_converter import convert_to_full_file_spectrograms,convert_to_segment_level
from data_splitter import split_segmented_data_by_subject
from data_splitter import split_segmented_data_patient_dependent
import os
from data_splitter import split_segmented_data_like_article
from model_dataloader import get_loaders, MyHeartDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from model_dataloader import get_loaders
#from train_and_save_model import train_and_save_model
from evaluate_model import evaluate_on_test
from evaluate_lstm_model import evaluate_lstm_model
from train_attention_model import train_attention_model
from train_lstm_model import train_lstm_model



# Step 1: Load data
#df = load_patient_dataset()

# Step 2: Convert to full-length spectrograms and then segmented
#spectro_df = convert_to_full_file_spectrograms()
#df_segmented = convert_to_segment_level(spectro_df)

# Step 3: Split data into train,test,dev dataset by considering subject id
#train_df, val_df, test_df = split_segmented_data_by_subject()
#train_df, val_df, test_df =split_segmented_data_like_article()
#train_df, val_df, test_df =split_segmented_data_patient_dependent()
# Step 4 : Model running

# Data
##train_csv = "../data/generated_data/patient-dependent/train_random.csv"
##val_csv = "../data/generated_data/patient-dependent/val_random.csv"
##test_csv = "../data/generated_data/patient-dependent/test_random.csv"

#train_csv = "../data/generated_data/article_like_split/train/segment_level_data(128)train.csv"
#val_csv = "../data/generated_data/article_like_split/val/segment_level_data(128)val.csv"
#test_csv = "../data/generated_data/article_like_split/test/segment_level_data(128)test.csv"
#model = train_and_save_model(
#    train_csv=train_csv,
#    val_csv=val_csv,
#    model_path="../models/resnet18_fine_tuned_epochs5_aticle_based.pth",
#    batch_size=32,
#    epochs=5
#)


#from ResNet_trainer import train_resnet_model
#train_resnet_model(
##    model_type="resnet34",18
#    train_csv=train_csv,
#    val_csv=val_csv,
#    model_path="../models/resnet34_finetuned_epoch100_article_based.pth",
#    batch_size=32,
#    epochs=100
#)

#evaluate_on_test(
#    model_path="../models/resnet34_attention_epoch40_patient-dependent_epoch1.pth",
#    test_csv=test_csv,
#    model_name="resnet34_attention_epoch40_patient-dependent_epoch1",
#    model_type="attention_resnet34", # options: "resnet18", "resnet34", "attention_resnet34"
#    batch_size=32
#)

#base_model_path = "../models/resnet34_attention_epoch100_article_based_epoch{}.pth"
#test_csv = "../data/generated_data/random_split/test_random.csv"
#batch_size = 32
#model_type = "attention_resnet34"

# Evaluate epochs 1 through 40
#for epoch in range(1, 101):
 #   model_path = base_model_path.format(epoch)
 #   model_name = f"resnet34_attention_epoch100_article_based_epoch{epoch}"

#    print(f"\n📊 Evaluating model: {model_name}")
#    evaluate_on_test(
#        model_path=model_path,
#        test_csv=test_csv,
#        model_name=model_name,
#        model_type=model_type,
#        batch_size=batch_size
#    )

#train_lstm_model(
#    train_csv=train_csv,
#    val_csv=val_csv,
#    model_dir="../models/lstm_finetuned_epoch5_patient_dependent",
#    batch_size=32,
#    epochs=5
#)

#base_model_path = "../models/lstm_finetuned_epoch5_patient_dependent/lstm_epoch{}.pth"
#test_csv = "../data/generated_data/random_split/test_random.csv"
#batch_size = 32

# Evaluate epochs 1 through 40
#for epoch in range(1, 6):
#    model_path = base_model_path.format(epoch)
#    model_name = f"lstm_epoch{epoch}"

#    print(f"\n📊 Evaluating model: {model_name}")
#    evaluate_lstm_model(
#        model_path=model_path,
#        test_csv=test_csv,
#        model_name="lstm_epoch5_patient_dependent.pth",
#        batch_size=batch_size
#    )

#evaluate_lstm_model(
#    model_path="../models/lstm_finetuned_epoch20_article_based.pth",
#    test_csv=test_csv,
#    model_name="lstm_epoch20_article_based.pth",
#    batch_size=32
#)
#train_attention_model(
#    train_csv=train_csv,
#    val_csv=val_csv,
#    base_model_path="../models/resnet34_attention_epoch100_article_based.pth",
#    batch_size=32,
#    epochs=100
#)
from evaluate_attention_model import evaluate_attention_model

#evaluate_attention_model(
#    model_path="../models/resnet34_attention_epoch40_article_based.pth",
#    test_csv=test_csv,
#    model_name="resnet34_attention_epoch7_article_based"
#)
import pandas as pd
from segment_multiwindow_converter import convert_to_multi_resolution_segments


#train = "../data/generated_data/multi_resolution_segments_3s/train_segment_data(128).csv"
#val = "../data/generated_data/multi_resolution_segments_3s/val_segment_data(128).csv"
#test = "../data/generated_data/multi_resolution_segments_3s/test_segment_data(128).csv"

train = "../data/generated_data/patient-dependent/multi_resolution_segments_3s/train_segmented_multi.csv"
val = "../data/generated_data/patient-dependent/multi_resolution_segments_3s/val_segmented_multi.csv"
test = "../data/generated_data/patient-dependent/multi_resolution_segments_3s/test_segmented_multi.csv"
# Train
#train_df = pd.read_csv(train)
#convert_to_multi_resolution_segments(train_df, split_name="train")

# Validation
#val_df = pd.read_csv(val_csv)
#convert_to_multi_resolution_segments(val_df, split_name="val")

# Test
#test_df = pd.read_csv(test_csv)
#convert_to_multi_resolution_segments(test_df, split_name="test")

from multi_chaneel_trainer import train_heart_model

#train_heart_model(
#    train_csv=train,
#    val_csv=val,
#    model_path="../models/resnet34_patient-dependent_epoch50_multi.pth",
#    model_type="resnet34",  # or "resnet18"
#    batch_size=32,
#    epochs=50
#)

from evaluate_multi_channel import evaluate_multi_channel_model

#evaluate_multi_channel_model(
#    model_path="../models/resnet34_article_epoch7_multi.pth",
#    test_csv=test,
#    model_name="resnet34_multi_res_epoch7",
#    model_type="resnet34"
#)
# Evaluate epochs 1 through 40
base_model_path = "../models/resnet34_patient-dependent_epoch50_multi_epoch{}.pth"
#test_csv = "../data/generated_data/random_split/test_random.csv"
batch_size = 32
for epoch in range(1, 101):
    model_path = base_model_path.format(epoch)#
    model_name = f"resnet34_patient-dependent_epoch50_multi_epoch{epoch}"

    print(f"\n📊 Evaluating model: {model_name}")
    evaluate_multi_channel_model(
        model_path=model_path,
        test_csv=test,
        model_name=model_name,
        model_type="resnet34"
    )
