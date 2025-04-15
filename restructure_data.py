import json
import os
import pandas as pd
import numpy as np

with open("./dataset/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)['dataset']

def restruct():
  images_path = config['images_path']
  anno_train_path = config['annotations']['train']
  anno_val_path = config['annotations']['val']
  anno_test_path = config['annotations']['test']

  def load_annotations(anno_path):
      with open(anno_path, 'r', encoding="utf-8") as f:
          annotations = json.load(f)
          label_annos = annotations['annotations']
          image_annos = annotations['images']
      return label_annos, image_annos
  
  train_label_annos, train_image_anno = load_annotations(anno_train_path)
  val_label_annos, val_image_annos = load_annotations(anno_val_path)
  test_label_annos, test_image_annos = load_annotations(anno_test_path)

  def combine_data(annotation, images):
    data = []
    for i, anno in enumerate(annotation):
        image_id, question, answer = anno['image_id'], anno['question'], anno['answers']
        for image in images:
            if image['id'] == image_id:
                data.append({
                    "image_name": image['filename'],
                    "question": question,
                    "answer": answer[0]
                })
    return data
  
  train_data = combine_data(train_label_annos, train_image_anno)
  dev_data = combine_data(val_label_annos, val_image_annos)
  test_data = combine_data(test_label_annos, test_image_annos)

  train_df = pd.DataFrame(train_data)
  dev_df = pd.DataFrame(dev_data)
  test_df = pd.DataFrame(test_data)

  train_df.to_csv('./dataset/csv/train_data.csv', index=False)
  dev_df.to_csv('./dataset/csv/dev_data.csv', index=False)
  test_df.to_csv('./dataset/csv/test_data.csv', index=False)

restruct()
      