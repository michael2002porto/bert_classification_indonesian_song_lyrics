import argparse
import sys
import os

from utils.preprocessor_class import PreprocessorClass
from models.multi_class_model import MultiClassModel

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

import lightning as L

def collect_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_nodes", type=int, default=1)

    parser.add_argument("--max_length", type=int,  default=100)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--max_epochs", type=int, default=10)

    parser.add_argument("--preprocessed_dir", type=str, default="data/preprocessed/original")
    parser.add_argument("--train_data_dir", type=str, default="data/training.res")
    parser.add_argument("--test_data_dir", type=str, default="data/testing.res")

    parser.add_argument("--pretrained_checkpoint", type=str, default="checkpoints/split_synthesized.ckpt")
    parser.add_argument("--model_name", type=str, default="original_split_synthesized")

    return parser.parse_args()

if __name__ == '__main__':
    args = collect_parser()

    #"bert_classification/data/preprocessed"

    # print(args.batch_size)

    dm = PreprocessorClass(
        preprocessed_dir = args.preprocessed_dir,
        batch_size = args.batch_size,
        max_length = args.max_length
    )

    # Pre-training
    # model = MultiClassModel(
    #     n_out = 4,
    #     dropout = 0.3,  # dropout tentuin sendiri
    #     lr = 1e-5   # 1e-3 = 0.0001
    # )

    # Fine-tuning
    model = MultiClassModel.load_from_checkpoint(
        args.pretrained_checkpoint,
        n_out = 4,
        dropout = 0.3,  # dropout tentuin sendiri
        lr = 1e-5
    )

    model_name = args.model_name if args.model_name else os.path.basename(os.path.normpath(args.preprocessed_dir))

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=4, verbose=False, mode="min")
    logger = CSVLogger("logs", name=model_name)
    trainer = L.Trainer(
        accelerator = args.accelerator,
        max_epochs = args.max_epochs,
        default_root_dir = f'logs/indobert/{model_name}',
        callbacks=[early_stop_callback],
        logger=logger
    )

    # Ini bagian training 
    trainer.fit(model, datamodule = dm)

    # Testing model
    trainer.test(datamodule = dm, ckpt_path = 'best')
    # sama saja dengan trainer.test(datamodule = dm, ckpt_path = 'best')
    # trainer.test(model=model, datamodule=dm, ckpt_path='best')
    
    # Prediction
    # hasil = trainer.predict(model = model, datamodule = dm)