cd ..
python run.py \
	--multigpu=y \
	--mode=save_pb \
	--freeze_weights_file_path='./save_h5_model/weights_097.h5' \
	--num_classes=54
