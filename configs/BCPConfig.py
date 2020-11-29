from configs.base_config import BaseConfig


class BCPConfig(BaseConfig):
    """
    这里参数：格式
    model = 'model'
    """
    model = 'BCP'
    roberta_model_path = 'chinese-roberta-wwm-ext'
    data_dir = './dataset/bcp_processed_data'
    train_batch_size = 64
    dev_batch_size = 500
    test_batch_size = 500
    lr = 1e-4
    hidden_size=300
    input_feature: int = 768
    tao = 0.1
