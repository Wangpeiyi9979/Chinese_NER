from configs.base_config import BaseConfig


class BertConfig(BaseConfig):
    """
    这里参数：格式
    model = 'model'
    """
    model = 'Bert'
    roberta_model_path = 'chinese-roberta-wwm-ext'
    input_features: int = 768
    use_lstm = False
