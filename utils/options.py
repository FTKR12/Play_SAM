import argparse

def get_args():
    parser = argparse.ArgumentParser(description="SAM test")
    ########## base options ##########
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--device', default='cuda')
    ########## sam options ##########
    parser.add_argument('--mode', default='auto', help='[auto, point, box]')
    parser.add_argument('--model_type', default='vit_h')
    parser.add_argument('--checkpoint', default='download_model/sam_vit_h_4b8939.pth')
    ########## prompt options ##########
    parser.add_argument('--point', default=[[75,75]], type=list)
    parser.add_argument('--point_label', default=[1], type=list)
    parser.add_argument('--box', default=[[0,0,150,150]], type=list)
    ########## dataset options ##########
    parser.add_argument('--data_type', default='sperm', help='[test_image, sperm]')
    parser.add_argument('--image_path', default='image/groceries.jpg')
    parser.add_argument('--sperm_path', default='datasets/771_1stframe_img.pkl')
    parser.add_argument('--sperm_id', default='027')

    args = parser.parse_args()
    return args