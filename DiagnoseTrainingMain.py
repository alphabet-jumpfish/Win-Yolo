from service.DiagnoseTrainingService import DiagnoseTraining
from pathlib import Path

if __name__ == '__main__':
    diagnose = DiagnoseTraining()
    train_dir_path = Path("training/warZ/dataset/images/train")
    diagnose.check_training_images(path=train_dir_path)
    label_dir_path = Path("training/warZ/dataset/labels/train")
    diagnose.check_labels(path=label_dir_path)
