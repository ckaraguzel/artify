import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class SaveResults:
    def __init__(self, base_path, model_name):
        self.folder_path = os.path.join(base_path, model_name)
        os.makedirs(self.folder_path, exist_ok=True)

    def save_results_to_csv(self, file_name, headers, data):
        file_path = os.path.join(self.folder_path, file_name)
        file_exists = os.path.isfile(file_path)

        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow(data)

    def save_classification_report_to_csv(self, file_name, report_dict):
        file_path = os.path.join(self.folder_path, file_name)
        report_df = pd.DataFrame(report_dict).transpose()
        report_df = report_df.map(lambda x: round(x, 2) if isinstance(x, (int, float)) else x)

        if not os.path.isfile(file_path):
            report_df.to_csv(file_path, mode='w', index=True, index_label="Class")
        else:
            report_df.to_csv(file_path, mode='a', index=True, index_label="Class", header=False)

    def save_confusion_matrix_as_png(self, file_name, conf_matrix, class_names):
        file_path = os.path.join(self.folder_path, file_name)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={"shrink": 0.75},
            linewidths=0.5,
            linecolor="gray"
        )
        plt.xlabel("Predicted Labels", fontsize=12)
        plt.ylabel("True Labels", fontsize=12)
        plt.title("Confusion Matrix", fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.savefig(file_path, bbox_inches="tight", dpi=300)
        plt.close()

    def plot_and_save_graphs(self, file_name, train_losses, val_losses, train_accuracies, val_accuracies):
        epochs = range(1, len(train_losses) + 1)
        plt.figure(figsize=(10, 5))

        # Loss Plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label="Train Loss", color="blue", marker="o")
        plt.plot(epochs, val_losses, label="Validation Loss", color="red", marker="x")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()

        # Accuracy Plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, label="Train Accuracy", color="blue", marker="o")
        plt.plot(epochs, val_accuracies, label="Validation Accuracy", color="red", marker="x")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.title("Training vs Validation Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.folder_path, file_name))
        plt.close()

    def save_csv_as_png(self, csv_file_name, output_png_name):
        csv_file_path = os.path.join(self.folder_path, csv_file_name)
        output_png_path = os.path.join(self.folder_path, output_png_name)

        # Load the CSV file into a DataFrame
        try:
            df = pd.read_csv(csv_file_path)
        except FileNotFoundError:
            print(f"File not found: {csv_file_path}")
            return

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(12, len(df) * 0.5))  # Adjust size based on rows
        ax.axis('tight')
        ax.axis('off')

        # Create a table from the DataFrame
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center',
        )

        # Adjust table properties
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(df.columns))))

        # Save the table as a PNG image
        plt.savefig(output_png_path, bbox_inches='tight', dpi=300)
        plt.close(fig)