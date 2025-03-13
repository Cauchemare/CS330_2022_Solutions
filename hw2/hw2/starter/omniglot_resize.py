import cv2
from glob import glob
from tqdm import tqdm 
from pathlib import Path 
sr_dir_path ='/Users/luyaoli/Desktop/学习竞赛/学习课程/CS330/w2/hw1_starter_code/omniglot_resized/'
ds_dir='/Users/luyaoli/Desktop/学习竞赛/学习课程/CS330/w3/2/hw2/starter/omniglot_resized'
for file_path  in   tqdm(glob(f'{sr_dir_path}/**/*.*',recursive=True)):
	image_file_recursive=  file_path.split(sr_dir_path)[1]
	
	N_SPLIT =  image_file_recursive.split('/')
	if len(N_SPLIT)<3:
		print(f'file_path ={file_path} \n image_file_recursive  ={image_file_recursive } \n ds_file_path={ds_file_path}')
		break
	
	ds_file_path = Path(ds_dir).joinpath( image_file_recursive )
	
	ds_file_path.parent.mkdir(parents=True,exist_ok=True)
	
	image =cv2.imread(file_path,cv2.IMREAD_GRAYSCALE)
	image_resize=cv2.resize(image,(28,28)) 
	cv2.imwrite(str(ds_file_path),image_resize)