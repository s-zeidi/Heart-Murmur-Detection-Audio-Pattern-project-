import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_segmented_data_by_subject(input_csv="../data/generated_data/segment_level_data.csv",
                                     output_dir="../data/generated_data/",
                                     subject_column="subject_id",
                                     label_column="murmur_label",
                                     test_size=0.15,
                                     val_size=0.15,
                                     random_state=42):
    df = pd.read_csv(input_csv)
    subject_df = df[[subject_column, label_column]].drop_duplicates()

    train_subjects, temp_subjects = train_test_split(
        subject_df,
        test_size=val_size + test_size,
        stratify=subject_df[label_column],
        random_state=random_state
    )

    val_ratio = val_size / (val_size + test_size)
    val_subjects, test_subjects = train_test_split(
        temp_subjects,
        test_size=1 - val_ratio,
        stratify=temp_subjects[label_column],
        random_state=random_state
    )

    train_df = df[df[subject_column].isin(train_subjects[subject_column])]
    val_df = df[df[subject_column].isin(val_subjects[subject_column])]
    test_df = df[df[subject_column].isin(test_subjects[subject_column])]

    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train_segments.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_segments.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_segments.csv"), index=False)

    # Print formatted table
    def summarize(name, data):
        total = len(data)
        summary = data[label_column].value_counts()
        summary_ratio = data[label_column].value_counts(normalize=True)
        return {
            "Split": name,
            "Total": total,
            **{f"{label}": f"{summary[label]} ({summary_ratio[label]:.2%})"
               for label in summary.index}
        }

    print("\n✅ Dataset split complete and saved:")
    summaries = [summarize("Train", train_df),
                 summarize("Validation", val_df),
                 summarize("Test", test_df)]

    # Convert to table
    summary_df = pd.DataFrame(summaries)
    print("\n📊 Final Segment-Level Split Summary:")
    print(summary_df.to_string(index=False))

    return train_df, val_df, test_df
