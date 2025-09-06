import numpy as np
import cv2

def analyze_voc_image():
    # Load the image
    img_path = r'C:\Users\abesh\Downloads\archive\VOC2012\SegmentationClassAug\2007_000032.png'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    if img is not None:
        print(f'Image shape: {img.shape}')
        print(f'Image dtype: {img.dtype}')
        
        # Get middle row
        middle_row_idx = img.shape[0] // 2
        middle_row = img[middle_row_idx, :]
        
        print(f'Middle row (row {middle_row_idx}):')
        print(middle_row)
        
        print(f'Unique values in the entire image:')
        unique_vals = np.unique(img)
        print(unique_vals)
        
        print(f'Number of unique values: {len(unique_vals)}')
        
        # Save results to a text file
        with open('voc_image_analysis.txt', 'w') as f:
            f.write(f'Image Analysis for: {img_path}\n')
            f.write(f'Image shape: {img.shape}\n')
            f.write(f'Image dtype: {img.dtype}\n\n')
            f.write(f'Middle row (row {middle_row_idx}):\n')
            f.write(f'{middle_row}\n\n')
            f.write(f'Unique values in the entire image:\n')
            f.write(f'{unique_vals}\n')
            f.write(f'Number of unique values: {len(unique_vals)}\n')
        
        print("Analysis saved to 'voc_image_analysis.txt'")
        
    else:
        print('Failed to load image. Check if the path is correct.')

if __name__ == "__main__":
    analyze_voc_image()
