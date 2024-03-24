import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import json
import os
import glob
from PIL import Image
from utils import transform

def test():
    path = '/home/cesare/Projects/datasets/LaSOT/'
    dataset = LasotDataset(path)
    for data in dataset:
        print(data)
        


class LasotDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder):
        """
        :param data_folder: folder where data files are stored
        """
        self.data_folder = data_folder

        classes_paths = glob.glob(os.path.join(self.data_folder,'*'))
        self.label_types = [os.path.basename(path) for path in classes_paths if os.path.isdir(path)]
        self.label_types_num = len(self.label_types)

        self.images = []
        self.labels = []
        self.sequences = []
        self.progress = []
        self.bboxs = []
        self.occlusions = []
        self.hidden = []
        
        for label_num, label_obj in enumerate(self.label_types):
            seq_paths = glob.glob(os.path.join(self.data_folder, label_obj, label_obj+'-*'))
            seq_paths = filter(lambda path:os.path.isdir(path),seq_paths)
            seq_paths = sorted(seq_paths, key=lambda path: int(path.split('-')[-1]))
            for seq_path in seq_paths:
                seq = os.path.basename(seq_path)
                seq_num = int(seq.split('-')[-1])
                with open(os.path.join(seq_path,'groundtruth.txt'),'r') as f: labels_bbox = f.readlines()
                with open(os.path.join(seq_path,'full_occlusion.txt'),'r') as f: labels_occ = f.readline()
                with open(os.path.join(seq_path,'out_of_view.txt'),'r') as f: labels_hidden = f.readline()

                labels_bbox = [[int(coord) for coord in bbox.split(',')] for bbox in labels_bbox]
                labels_occ = [occ == '1' for occ in labels_occ.split(',')]
                labels_hidden = [hidden == '1' for hidden in labels_hidden.split(',')]

                images_path = glob.glob(os.path.join(seq_path, 'img', '*.jpg'))
                images_path = sorted(images_path)
                img_cnt = len(images_path)

                labels_seq = [seq_num] * img_cnt
                labels_progress = torch.linspace(0,1,img_cnt)
                labels_class = [label_num] * img_cnt
                
                self.images.extend(images_path)
                self.labels.extend(labels_class)
                self.sequences.extend(labels_seq)
                self.progress.extend(labels_progress)
                self.bboxs.extend(labels_bbox)
                self.occlusions.extend(labels_occ)
                self.hidden.extend(labels_hidden)


        assert len(self.images) == len(self.labels) == len(self.sequences) == len(self.progress) == len(self.bboxs) == len(self.occlusions) == len(self.hidden)

    def __getitem__(self, i):
        # Read image
        image = Image.open(self.images[i], mode='r')
        image = image.convert('RGB')
        w,h = image.width, image.height

        label = torch.IntTensor([self.labels[i]])
        seq_num = torch.IntTensor([self.sequences[i]])
        progress = self.progress[i]
        bbox = torch.FloatTensor(self.bboxs[i]) / torch.FloatTensor([w,h,w,h])
        occlusion = torch.BoolTensor([self.occlusions[i]]) 
        hidden = torch.BoolTensor([self.hidden[i]])
        
        return image, label, seq_num, progress, bbox, occlusion, hidden

    def __len__(self):
        return len(self.images)

if __name__ == '__main__': test()