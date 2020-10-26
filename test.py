from pathlib import Path
import os
import unisal
import argparse
# def parse_args():
#     """
#     Parse input arguments
#     """
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--train_id',default='2020-09-23_09:50:34_unisal',type=str)
#     #parser.add_argument('--image_size',default=(96，128),type=tuple)
#     parser.add_argument('--input_dir',default='2020-09-23_09:50:34_unisal',type=str)
#     parser.add_argument('--otput_dir',default='2020-09-23_09:50:34_unisal',type=str)
#     args = parser.parse_args()
#     return args

def load_trainer(train_id=None):
    """Instantiate Trainer class from saved kwargs."""
    if train_id is None:
        train_id = 'pretrained_unisal'
    print(f"Train ID: {train_id}")
    train_dir = Path(os.environ["TRAIN_DIR"])
    train_dir = train_dir / train_id
    return unisal.train.Trainer.init_from_cfg_dir(train_dir)

def score_model(
        train_id=None,
        sources=('SALICON',),
        **kwargs):
    """Compute the scores for a trained model."""

    trainer = load_trainer(train_id)
    for source in sources:
        trainer.score_model(source=source, **kwargs)

def predictions_from_folder(
        folder_path, is_video, source=None, train_id=None, model_domain=None):
    """Generate predictions of files in a folder with a trained model."""
    trainer = load_trainer(train_id)
    trainer.generate_predictions_from_path(
        folder_path, is_video, source=source, model_domain=model_domain)


def test(train_id='2020-10-23_15:07:04_unisal'):
    # args=parse_args()
    # train_id=args.train_id
    for example_folder in (Path(__file__).resolve().parent / "data").glob("*"):
        if not example_folder.is_dir():
            continue

        source = example_folder.name
        is_video = source not in ('SALICON', 'MIT1003')

        print(f"\nGenerating predictions for {'video' if is_video else 'image'} "
              f"folder\n{str(source)}")

        if is_video:
            if not example_folder.is_dir():
                continue
            for video_folder in example_folder.glob('*'):
                predictions_from_folder(
                    video_folder, is_video, train_id=train_id, source=source)

        else:
            predictions_from_folder(
                example_folder, is_video, train_id=train_id, source=source)


if __name__ == "__main__":
    #args=parse_args()
    test()
   
    

