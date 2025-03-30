import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime


# Tạo thư mục để lưu các hình ảnh đồ thị
def create_plots_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = f"plots_{timestamp}"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    return plots_dir


def plot_accuracy_loss(history, model_name, data_type="raw"):
    """
    Vẽ đồ thị accuracy và loss của một mô hình

    Parameters:
        history (dict): Dictionary chứa lịch sử huấn luyện
        model_name (str): Tên của mô hình
        data_type (str): Loại dữ liệu (raw hoặc processed)
    """
    plots_dir = create_plots_dir()

    # Tạo hình vẽ
    plt.figure(figsize=(12, 5))

    # Vẽ đồ thị accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train')
    if 'val_accuracy' in history:
        plt.plot(history['val_accuracy'], label='Validation')
    plt.title(f'{model_name} - Accuracy ({data_type} data)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Vẽ đồ thị loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation')
    plt.title(f'{model_name} - Loss ({data_type} data)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{model_name}_{data_type}_accuracy_loss.png"))
    plt.close()


def compare_models_accuracy(results, data_type="raw"):
    """
    Vẽ biểu đồ so sánh độ chính xác của các mô hình

    Parameters:
        results (dict): Dictionary chứa kết quả của các mô hình
        data_type (str): Loại dữ liệu (raw hoặc processed)
    """
    plots_dir = create_plots_dir()

    # Lấy tên các mô hình và độ chính xác tương ứng
    model_names = list(results.keys())
    accuracies = [results[model]["accuracy"] for model in model_names]

    # Tạo hình vẽ
    plt.figure(figsize=(10, 6))

    # Vẽ biểu đồ cột
    bars = plt.bar(model_names, accuracies, color=['blue', 'green', 'red', 'orange'])

    # Thêm giá trị độ chính xác lên mỗi cột
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')

    plt.title(f'So sánh độ chính xác giữa các mô hình ({data_type} data)')
    plt.xlabel('Mô hình')
    plt.ylabel('Độ chính xác')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"comparison_{data_type}_accuracy.png"))
    plt.close()


def plot_training_time(results, data_type="raw"):
    """
    Vẽ biểu đồ so sánh thời gian huấn luyện của các mô hình

    Parameters:
        results (dict): Dictionary chứa kết quả của các mô hình
        data_type (str): Loại dữ liệu (raw hoặc processed)
    """
    plots_dir = create_plots_dir()

    # Lấy tên các mô hình và thời gian huấn luyện tương ứng
    model_names = list(results.keys())
    training_times = [results[model]["time"] for model in model_names]

    # Tạo hình vẽ
    plt.figure(figsize=(10, 6))

    # Vẽ biểu đồ cột
    bars = plt.bar(model_names, training_times, color=['blue', 'green', 'red', 'orange'])

    # Thêm giá trị thời gian lên mỗi cột
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 1,
                 f'{height:.2f}s', ha='center', va='bottom')

    plt.title(f'So sánh thời gian huấn luyện giữa các mô hình ({data_type} data)')
    plt.xlabel('Mô hình')
    plt.ylabel('Thời gian (giây)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"comparison_{data_type}_training_time.png"))
    plt.close()


def plot_confusion_matrix(cm, classes, model_name, data_type="raw"):
    """
    Vẽ ma trận nhầm lẫn

    Parameters:
        cm (array): Ma trận nhầm lẫn
        classes (list): Danh sách các lớp
        model_name (str): Tên mô hình
        data_type (str): Loại dữ liệu (raw hoặc processed)
    """
    plots_dir = create_plots_dir()

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Ma trận nhầm lẫn - {model_name} ({data_type} data)')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Nhãn thực tế')
    plt.xlabel('Nhãn dự đoán')

    plt.savefig(os.path.join(plots_dir, f"{model_name}_{data_type}_confusion_matrix.png"))
    plt.close()