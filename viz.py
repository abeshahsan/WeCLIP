import numpy as np
import cv2
from pathlib import Path

# # VOC colormap for Pascal VOC dataset classes
# VOC_COLORMAP = np.array([
#     [0, 0, 0],        # 0=background
#     [128, 0, 0],      # 1=aeroplane
#     [0, 128, 0],      # 2=bicycle
#     [128, 128, 0],    # 3=bird
#     [0, 0, 128],      # 4=boat
#     [128, 0, 128],    # 5=bottle
#     [0, 128, 128],    # 6=bus
#     [128, 128, 128],  # 7=car
#     [64, 0, 0],       # 8=cat
#     [192, 0, 0],      # 9=chair
#     [64, 128, 0],     # 10=cow
#     [192, 128, 0],    # 11=diningtable
#     [64, 0, 128],     # 12=dog
#     [192, 0, 128],    # 13=horse
#     [64, 128, 128],   # 14=motorbike
#     [192, 128, 128],  # 15=person
#     [0, 64, 0],       # 16=potted plant
#     [128, 64, 0],     # 17=sheep
#     [0, 192, 0],      # 18=sofa
#     [128, 192, 0],    # 19=train
#     [0, 64, 128],     # 20=tv/monitor
# ])

# VOC class names
voc_classes = [
    "background","aeroplane","bicycle","bird","boat","bottle","bus",
    "car","cat","chair","cow","diningtable","dog","horse","motorbike",
    "person","potted plant","sheep","sofa","train","tv/monitor"
]

# VOC colormap
def voc_colormap():
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)
    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3
    return colormap

VOC_COLORMAP = voc_colormap().astype(np.uint8)

def visualize_cams():
    predictions_folder = Path("predictions")
    viz_folder = Path("color_predictions")
    viz_folder.mkdir(exist_ok=True)

    ind = 0
    
    for img_file in predictions_folder.glob("*.png"):
        ind += 1
        if ind == 2:
            break
        # Load label image
        labels_processed = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)

        print(np.unique(labels_processed))
        
        # # Reverse the preprocessing done in VOC:
        # # Original processing: subtract 1, remove 255/254
        # # So we need to: add 1 back, handle special values
        # labels_processed = labels.copy().astype(np.int32)
        # labels_processed = labels_processed + 1  # Add 1 back
        
        # # Handle edge cases - clip values to valid range (0-20 for VOC)
        # labels_processed = np.clip(labels_processed, 0, 20)

        
        print(f"Processing {img_file.name}: unique labels = {np.unique(labels_processed)}")
        
        # Create colored image
        h, w = labels_processed.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        
        for label_index in np.unique(labels_processed):
            if label_index < len(VOC_COLORMAP):
                colored[labels_processed == label_index] = VOC_COLORMAP[label_index]
            else:
                print(f"Warning: label {label_index} exceeds colormap range.")
        
        # Save
        output_path = viz_folder / f"{img_file.stem}_colored.png"
        cv2.imwrite(str(output_path), colored)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    visualize_cams()