from configs.base_config import BaseConfig

def create_tags(tag_path):
    tags = {}
    with open(tag_path, 'r') as f:
        for _i, _line in enumerate(f.readlines()):
            if _line != '':
                tags.update({_line.strip().split('@')[0]: _i})
    return tags

class BCLabelInfoConfig(BaseConfig):
    """
    这里参数：格式
    model = 'model'
    """
    model = 'BCLabelInfo'
    roberta_model_path = 'chinese-roberta-wwm-ext'
    tags_path = 'dataset/tool_data/label.txt'
    tags = create_tags(tags_path)
    input_features = 768
    hidden_features = 768 * 2
    train_batch_size = 8
    test_batch_size = 16
    dev_batch_size = 8
    mask = False
    dw = 0.