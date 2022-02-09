import argparse
import os
import numpy as np
from tqdm import tqdm
import logging
from utils.common import read_yaml, create_directories
import tensorflow as tf
import io

STAGE = "creating base model"

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    
    )

def main(config_path):
    ## read config files
    config = read_yaml(config_path)

    ## get the data
    (X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train_full = X_train_full / 255.0
    X_test = X_test / 255.0
    X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
    y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

    ## set the seeds
    seed = 2021 ## get it from config
    tf.random.set_seed(seed)
    np.random.seed(seed)

    ## define layers
    LAYERS = [
          tf.keras.layers.Flatten(input_shape=[28,28], name="inputlayer"),
          tf.keras.layers.Dense(300, name="hiddenlayer1"),
          tf.keras.layers.LeakyReLU(), ## alternative way
          tf.keras.layers.Dense(100, name="hiddenlayer2"),
          tf.keras.layers.LeakyReLU(),
          tf.keras.layers.Dense(10,activation="softmax", name="outputlayer")
    ]

    ## define the model and compile it
    model = tf.keras.models.Sequential(LAYERS)


    LOSS = "sparse_categorical_crossentropy"
    OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=1e-3)
    METRICS = ["accuracy"]

    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS) 

    ## log our model summary information in logs
    def _log_model_summary(model):
        with io.StringIO() as stream:
            model.summary(print_fn=lambda x:stream.write(f"{x}\n"))
            summary_str=stream.getvalue()
        return summary_str
    
    ## model.summary()
    logging.info(f"base model summary: \n{_log_model_summary(model)}")

    ## Train the model
    history = model.fit(
        X_train, y_train,
        epochs=5,
        validation_data=(X_valid, y_valid),
        verbose=2)

    ## save the base model - 
    model_dir_path = os.path.join("artifacts","models")
    create_directories([model_dir_path])
    
    model_file_path = os.path.join(model_dir_path, "base_model.h5")
    model.save(model_file_path)

    logging.info(f"base model is saved at {model_file_path}")
    logging.info(f"evaluation metrics {model.evaluate(X_test,y_test)}")

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n************************************************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e