config = {
    "patch_size": 4,                # 32/4 = 8 -> 64 патча
    "hidden_size": 192,             # Размер эмбеддингов (было 48)
    "num_hidden_layers": 12,        # Количество трансформер-блоков (было 4)
    "num_attention_heads": 6,       # 192 / 6 = 32 (должно делиться)
    "intermediate_size": 768,        # 4 * hidden_size
    "hidden_dropout_prob": 0.1,      # dropout после каждого блока
    "attention_probs_dropout_prob": 0.1,
    "initializer_range": 0.02,
    "image_size": 32,
    "num_classes": 10,               # для CIFAR-100 измените
    "num_channels": 3,
    "qkv_bias": True,
    "stochastic_depth_rate": 0.1,    # вероятность дропа целого блока
}