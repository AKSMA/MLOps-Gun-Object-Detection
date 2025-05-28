import os 
import cv2 
import torch 
import numpy as np
from torch.utils.data import Dataset 
from src.logger import get_logger 
from src.custom_exception import CustomException

logger = get_logger(__name__)

import os 
class GunDataset(Dataset):

    def __init__(self, root:str , device:str="cpu"):
        self.image_path = os.path.join(root, "Images")
        self.labels_path = os.path.join(root, "Labels")
        self.device = device
        self.img_name = sorted(os.listdir(self.image_path))
        self.label_name = sorted(os.listdir(self.labels_path))
        
        logger.info("Data Processing Initialized")

    def __getitem__(self , idx):
        try:
            logger.info(f"Loading Image and Label {idx}")
            
            image_path = os.path.join(self.image_path , str(self.img_name[idx]))
            logger.info(f'Image Path : {image_path} ')
            image = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(image , cv2.COLOR_BGR2RGB).astype(np.float32)

            img_res = img_rgb / 255 
            img_res = torch.as_tensor(img_res).permute(2 , 0 , 1)

            
            
            label_name = self.img_name[idx].rsplit('.' , 1)[0] + ".txt"
            label_path = os.path.join(self.labels_path , str(label_name))

            if not os.path.exists(label_path):
                raise FileNotFoundError(f'Label file not found : {label_path}')
            
            target = {
                'boxes': torch.as_tensor([]) ,
                'area': torch.as_tensor([]) ,
                'image_id': torch.as_tensor([idx]) , 
                'labels': torch.as_tensor([] , dtype=torch.int64)
            }
            
            with open(label_path , "r") as label_file:
                l_count = int(label_file.readline())
                box = []
                for i in range(l_count):
                    box.append(list(map(int, label_file.readline().split())))

            
            # area = []
            # labels = []

            if box:
                area = [(box[i][2] - box[i][0]) * (box[i][3] - box[i][1]) for b in box]
                labels = [1] * len(box)

                target['boxes']= torch.as_tensor(box, dtype=torch.float32) 
                target['area']= torch.as_tensor(area, dtype=torch.float32) 
                target['labels']= torch.as_tensor(labels , dtype=torch.int64)

            img_res = img_res.to(self.device)

            for key in target:
                if isinstance(target[key] , torch.Tensor):
                    target[key] = target[key].to(self.device)

            return img_res , target 
        except Exception as e:
            logger.error(f"Error in loading image and label {idx}: {e}")
            raise CustomException("Failed to load data ", e)

    def __len__(self):
        return len(self.img_name)

if __name__ == "__main__":
    root_path = "D:/ML_Ops/ML_OPS_project_4/artifacts/raw"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dataset = GunDataset(root=root_path , device = device)
    image, target = dataset[0]
    
    print(f"Image shape: {image.shape}")
    print(f"Target: {target.keys()}")
    print("Bounding boxes: ", target['boxes'])    