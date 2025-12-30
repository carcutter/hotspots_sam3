"""
General idea here: 
Given the 4 views, we generated a 360° video, 20sec length. 
Idea here is to set N hotspots on the any of the four views, and track them in the 360° video.
We're not going to use the .mp4 output here, only 180 views extracted from the 360° video.
"""

import torch
import sys
import sam3
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from typing import Tuple
import glob
import os 
import cv2 
from PIL import Image, ImageChops
import numpy as np 
from tqdm import tqdm 
import time 
import yaml 
from typing import List
import json 

GLOBAL_COUNTER = 1 

from sam3.train.data.sam3_image_dataset import InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint

from sam3.train.data.collator import collate_fn_api as collate
from sam3.model.utils.misc import copy_data_to_device


torch.autocast('cuda',dtype=torch.float16).__enter__()
torch.inference_mode().__enter__()
torch.is_inference_mode_enabled()

GENERIC_HOTSPOT_DICT = {'hs_0':{'sam3_prompt':'license plate', # Hotspot name
                                'expected_positions':['front','rear'], # Position observation
                                'name': 'lp'}, # Generic name

                        'hs_1':{'sam3_prompt':'door handles',  # Door handles
                                'expected_positions':['front-left', 'front-right', 'rear-left', 'rear-right'],
                                'name': 'door-handle'},

                        'hs_2':{'sam3_prompt':'wheels', # Wheels
                                'expected_positions':['front-left', 'front-right', 'rear-left', 'rear-right'],
                                'name': 'wheel'},

                        'hs_3':{'sam3_prompt':'headlights', # Headlights
                                'expected_positions':['front-left', 'front-right'],
                                'name': 'headlight'},

                        'hs_4':{'sam3_prompt':'taillights', # Taillights
                                'expected_positions':['rear-left', 'rear-right'],
                                'name': 'taillight'},

                        'hs_5':{'sam3_prompt':'side mirrors', # Side mirrors
                                'expected_positions':['front-left', 'front-right'],
                                'name': 'mirror'}
                        }

nb_hotspots = len(GENERIC_HOTSPOT_DICT)

ANCHORS_IMG_INDEX = [0,44,89,134] # Correspond to the cardinal views index that were used to generate the 360° videos. 
INDEX_TO_ANCHOR_VIEW = {0:'front',1:'right-side',2:'rear',3:'left-side'}


RANGE_LP_DETECTION = list(range(0, 30)) + list(range(70, 110)) + list(range(160, 180))
RANGE_MIRROR_HEADLIGHT_DETECTION = list(range(0, 80)) + list(range(100, 180))
RANGE_TAILLIGHT_DETECTION = list(range(60,120))
RANGE_WHEEL_DOOR_HANDLE_DETECTION = list(range(20,70)) + list(range(110,160))


def create_empty_datapoint():
    return Datapoint(find_queries = [], images = [])

                            
class SAMHotspots:
    def __init__(self, scene_dir: str,batch_process:bool = False, device: str = "cuda"):

        # Load S3 model. 
        self.model = build_sam3_image_model()
        self.processor = Sam3Processor(self.model)

        # Load the scene. 
        self.scene_dir = scene_dir
        self.video = os.path.join(scene_dir,'videos', "360_final.mp4")
        self.images = os.listdir(os.path.join(scene_dir,'images'))
        self.images.sort()
        
       
        print(f'Found {len(self.images)} images')

        if batch_process:
            self.datapoint1 = create_empty_datapoint()
            self.set_transform()
            self.set_postprocessor()
        
        self.json_hotspot_creation()
        

    def json_hotspot_creation(self):
       # Create a dictionary to store the hotspots.
       # Currently and as is, this dictionnary is pretty heavy, since absolutely all the hotspots are stored in it. 
       
        self.hotspots_dict = {}
        
        # Loop over all the hotspots.
        for key, value in GENERIC_HOTSPOT_DICT.items():

            generic_hs_name = value['name']
            hs_expected_positions = value['expected_positions'] 
            
            # For each viewpoint where such a hotspot is observed. 
            for pos in hs_expected_positions:

                unique_id = f"{generic_hs_name}_{pos}"
                
                # Set a new entry with the unique_id as key and keep its generic name and the orientation it has. 
                self.hotspots_dict[unique_id] = {
                    'generic_hs_name': generic_hs_name, # Generic hotspot name
                    'position': pos, # vp_obs = Viewpoint Observation
                    'frame_index': {}
                }
                
             

    def set_transform(self):
        from sam3.train.transforms.basic_for_api import ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI
        self.transform_batch = ComposeAPI([RandomResizeAPI(sizes = 1008,max_size=1008,square=True,consistent_transform=True),
                                    ToTensorAPI(), 
                                    NormalizeAPI(mean = [0.5,0.5,0.5],std = [0.5,0.5,0.5])])

    def set_postprocessor(self):
        from sam3.eval.postprocessors import PostProcessImage
        self.postprocessor = PostProcessImage(max_dets_per_img = 3,
                                            iou_type = "segm",
                                            use_original_sizes_box = True,
                                            use_original_sizes_mask=True,
                                            convert_mask_to_rle = False,
                                            detection_threshold = 0.7,
                                            to_cpu = False)

    @staticmethod   
    def set_image(datapoint,pil_image):
        w,h = pil_image.size
        datapoint.images = [SAMImage(data = pil_image,
                                    objects = [],
                                    size = [h,w])]
    @staticmethod
    def add_text_prompt(datapoint: Datapoint, text_query: str):
        global GLOBAL_COUNTER

        assert len(datapoint.images) ==1, "Set the image first"
        w,h = datapoint.images[0].size
        datapoint.find_queries.append(
            FindQueryLoaded(query_text = text_query,
            image_id = 0,
            object_ids_output = [],
            is_exhaustive = [],
            query_processing_order = 0,
            inference_metadata = InferenceMetadata(coco_image_id = GLOBAL_COUNTER,
                                                    original_image_id = GLOBAL_COUNTER,
                                                    original_category_id = 1,
                                                    original_size= [w,h],
                                                    object_id = 0,
                                                    frame_index = 0,
                                                    )
                                )
        )   
        GLOBAL_COUNTER += 1
        return GLOBAL_COUNTER - 1

    @staticmethod
    def add_visual_prompt(datapoint: Datapoint, boxes:List[List[float]], labels:List[bool], text_prompt="visual"):
        """ Add a visual query to the datapoint.
        The bboxes are expected in XYXY format (top left and bottom right corners)
        For each bbox, we expect a label (true or false). The model tries to find boxes that ressemble the positive ones while avoiding the negative ones
        We can also give a text_prompt as an additional hint. It's not mandatory, leave it to "visual" if you want the model to solely rely on the boxes.

        Note that the model expects the prompt to be consistent. If the text reads "elephant" but the provided boxe points to a dog, the results will be undefined.
        """

        global GLOBAL_COUNTER
        # in this function, we require that the image is already set.
        # that's because we'll get its size to figure out what dimension to resize masks and boxes
        # In practice you're free to set any size you want, just edit the rest of the function
        assert len(datapoint.images) == 1, "please set the image first"
        assert len(boxes) > 0, "please provide at least one box"
        assert len(boxes) == len(labels), f"Expecting one label per box. Found {len(boxes)} boxes but {len(labels)} labels"
        for b in boxes:
            assert len(b) == 4, f"Boxes must have 4 coordinates, found {len(b)}"

        labels = torch.tensor(labels, dtype=torch.bool).view(-1)
        if not labels.any().item() and text_prompt=="visual":
            print("Warning: you provided no positive box, nor any text prompt. The prompt is ambiguous and the results will be undefined")
        w, h = datapoint.images[0].size
        datapoint.find_queries.append(
            FindQueryLoaded(
                query_text=text_prompt,
                image_id=0,
                object_ids_output=[], # unused for inference
                is_exhaustive=True, # unused for inference
                query_processing_order=0,
                input_bbox=torch.tensor(boxes, dtype=torch.float).view(-1,4),
                input_bbox_label=labels,
                inference_metadata=InferenceMetadata(
                    coco_image_id=GLOBAL_COUNTER,
                    original_image_id=GLOBAL_COUNTER,
                    original_category_id=1,
                    original_size=[w, h],
                    object_id=0,
                    frame_index=0,
                )
            )
        )
        GLOBAL_COUNTER += 1
        return GLOBAL_COUNTER - 1
    def save_hotspots(self):
        with open(os.path.join(self.scene_dir,'hotspots.json'), 'w') as f:
            json.dump(self.hotspots_dict, f, indent=4, sort_keys=True)

    def reset(self):
        self.datapoint1 = create_empty_datapoint()
    
    def prepare_batch_inference(self,frame:Image.Image):
        SAMHotspots.set_image(self.datapoint1,frame)
        for i in range(nb_hotspots): 
            setattr(self, f'id{i}', SAMHotspots.add_text_prompt(self.datapoint1, GENERIC_HOTSPOT_DICT[f'hs_{i}']['sam3_prompt']))

        self.datapoint1 = self.transform_batch(self.datapoint1) 

    def make_batch(self,datapoint):
        batch =  collate([datapoint],dict_key='dummy')['dummy']
        batch = copy_data_to_device(batch,device = "cuda")
        return batch

    def _draw_annotation(self, image, detection, label):
        box = detection['box']
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        cv2.putText(image, label, (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return image

    def _assign_hotspots(self, name, detections, side, width, front_or_rear, frame_index):
        assignments = []
        if not detections:
            return assignments

        # Sort by x-center (left to right)
        detections.sort(key=lambda x: x['center'][0])
        
        if name == 'lp':
            if frame_index in RANGE_LP_DETECTION:
                assignments.append((f"{name}_{front_or_rear}", detections[0]))
            
        elif name in ['mirror', 'headlight']:
            u_right = f"{name}_front-right"
            u_left = f"{name}_front-left"
            if frame_index in RANGE_MIRROR_HEADLIGHT_DETECTION:
                if len(detections) == 2:
                    assignments.append((u_right, detections[0])) # Left in image -> Right component
                    assignments.append((u_left, detections[1]))  # Right in image -> Left component
                elif len(detections) == 1:
                    if side == 'right': assignments.append((u_right, detections[0]))
                    elif side == 'left': assignments.append((u_left, detections[0]))
                
        elif name == 'taillight':
            u_right = f"{name}_rear-right"
            u_left = f"{name}_rear-left"
            if frame_index in RANGE_TAILLIGHT_DETECTION:
                if len(detections) == 2:
                    assignments.append((u_right, detections[0]))
                    assignments.append((u_left, detections[1]))
                elif len(detections) == 1:
                    cx = detections[0]['center'][0]
                    if side == 'right' and cx > width/2:
                        assignments.append((u_right, detections[0]))
                    elif side == 'left' and cx < width/2:
                        assignments.append((u_left, detections[0]))

        elif name in ['wheel', 'door-handle']:
            u_front = f"{name}_front-{side}"
            u_rear = f"{name}_rear-{side}"
            if frame_index in RANGE_WHEEL_DOOR_HANDLE_DETECTION:
                if side == 'right':
                    if len(detections) == 2:
                        assignments.append((u_rear, detections[0]))
                        assignments.append((u_front, detections[1]))
                    elif len(detections) == 1:
                        if detections[0]['center'][0] < width/2: assignments.append((u_rear, detections[0]))
                        else: assignments.append((u_front, detections[0]))
                elif side == 'left':
                    if len(detections) == 2:
                        assignments.append((u_front, detections[0]))
                        assignments.append((u_rear, detections[1]))
                    elif len(detections) == 1:
                        if detections[0]['center'][0] < width/2: assignments.append((u_front, detections[0]))
                    else: assignments.append((u_rear, detections[0]))
                    
        return assignments

    def batch_inference(self):
        cap = cv2.VideoCapture(self.video)
       
        # Get video properties for the output writer
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object
        output_filename = os.path.join(self.scene_dir,"sam_hotspots.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4 files
        mask_writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    
        for i in tqdm(range(len(self.images))):
            pil_image = Image.open(os.path.join(self.scene_dir, 'images', self.images[i])).convert('RGB')
            np_image = np.array(pil_image)[:, :, ::-1].copy()

            self.prepare_batch_inference(pil_image)
            batch = self.make_batch(self.datapoint1)   

            # This is our prior knowledge of the viewpoint, based on the index of the image. 
            # Determinie which anchor view we're processing. 
            if i<44: which_side = 'front-right'
            elif i<89: which_side = 'rear-right'
            elif i<134: which_side = 'rear-left'
            else: which_side = 'front-left'

            front_or_rear, side = which_side.split('-')

            with torch.no_grad():
                output = self.model(batch)
                processed_output = self.postprocessor.process_results(output, batch.find_metadatas)

            for k in range(nb_hotspots):
                
                # Retrive which hotspots we're processing. 
                curr_hs_name = GENERIC_HOTSPOT_DICT[f'hs_{k}']['name']
                
                pred_key = getattr(self, f'id{k}')
                
                masks = processed_output[pred_key]['masks']
                boxes = processed_output[pred_key]['boxes']
                scores = processed_output[pred_key]['scores']
                
                if len(masks) == 0:
                    continue
                
                # For all the detections, get the BB, score and bb center. 
                detections = []
                for j in range(len(masks)):
                    box = boxes[j].cpu().numpy()
                    x_center = float((box[0] + box[2]) / 2)
                    y_center = float((box[1] + box[3]) / 2)
                    detections.append({
                        'center': (x_center, y_center),
                        'box': box.tolist(),
                        'score': scores[j].item()
                    })
                # Assign the detections to the hotspots, based on the side we're processing, the frame index and the hotspot name. 
                assignments = self._assign_hotspots(curr_hs_name, detections, side, width, front_or_rear,i)
                
                # Populate the hotspots dictionary with the assignments, made BB drawing for logs. 
                for unique_id, detection in assignments:
                    if unique_id in self.hotspots_dict:
                        self.hotspots_dict[unique_id]['frame_index'][i] = detection
                    np_image = self._draw_annotation(np_image, detection, unique_id)
            
            self.reset()

            mask_writer.write(np_image)
        mask_writer.release()
        self.save_hotspots()

if __name__=='__main__':
    
    sam_hotspots = SAMHotspots(scene_dir="/home/gaetan/data/nextgen360/Hyundai/",batch_process=True)
    sam_hotspots.batch_inference()
    print(sam_hotspots.hotspots_dict)
   


