# pretreat train set
python misc/pretreatment_grew.py --input_path "/path/to/GREW/train" --output_path "GREW-64-44-pkl/train" --img_h 64 --img_w 44 --subset "train"
# pretreat test-gallery set
python misc/pretreatment_grew.py --input_path "/path/to/GREW/test/gallery" --output_path "GREW-64-44-pkl/test/gallery" --img_h 64 --img_w 44 --subset "test/gallery"
# pretreat test-probe set
python misc/pretreatment_grew_probe.py --input_path "/path/to/GREW/test/probe" --output_path "GREW-64-44-pkl/test/probe" --img_h 64 --img_w 44