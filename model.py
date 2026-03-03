import torch
import torch.nn as nn
import math
import sys

sys.path.insert(1, r"/wd/Optical_matrix_multiplication")
try:
    import source
except ImportError:
    source = None

pixel_size = 3.6e-6
device = "cuda" if torch.cuda.is_available() else "cpu"


def optics_matmul(sim, tensor_1, tensor_2):
    # Шаг 0: Дополняем размерность
    tensor_1 = tensor_1[None,:,:,:]
    tensor_2 = tensor_2[None,:,:,:]

    # Шаг 1: Разделяем на положительные и отрицательные части
    # A_pos содержит все положительные значения из A, остальные 0
    # A_neg содержит модули всех отрицательных значений из A, остальные 0
    A_pos = torch.clamp(tensor_1, min=0)      # A⁺ = max(A, 0)
    A_neg = torch.clamp(-tensor_1, min=0)     # A⁻ = max(-A, 0)
    B_pos = torch.clamp(tensor_2, min=0)      # B⁺ = max(B, 0)
    B_neg = torch.clamp(-tensor_2, min=0)     # B⁻ = max(-B, 0)
    
    # Шаг 2: Находим максимальные значения для нормировки
    max_A_pos = torch.max(A_pos)  # Может быть 0, если нет положительных значений
    max_A_neg = torch.max(A_neg)  # Может быть 0, если нет отрицательных значений
    max_B_pos = torch.max(B_pos)
    max_B_neg = torch.max(B_neg)

    # Заранее создаём шаблон нулевого тензора
    shape = (tensor_1.shape[0], tensor_1.shape[1], tensor_1.shape[2], tensor_2.shape[3])
    
    # Шаг 3: Вычисляем 4 компонента с защитой от деления на 0
    
    # Компонент 1: A⁺ × B⁺
    if max_A_pos > 0 and max_B_pos > 0:
        term1 = sim(A_pos / max_A_pos, B_pos / max_B_pos) * max_A_pos * max_B_pos
    else:
        term1 = torch.zeros(shape, device=tensor_1.device, dtype=tensor_1.dtype)
    
    # Компонент 2: A⁺ × B⁻ (со знаком минус в финальной формуле)
    if max_A_pos > 0 and max_B_neg > 0:
        term2 = sim(A_pos / max_A_pos, B_neg / max_B_neg) * max_A_pos * max_B_neg
    else:
        term2 = torch.zeros(shape, device=tensor_1.device, dtype=tensor_1.dtype)
    
    # Компонент 3: A⁻ × B⁺ (со знаком минус в финальной формуле)
    if max_A_neg > 0 and max_B_pos > 0:
        term3 = sim(A_neg / max_A_neg, B_pos / max_B_pos) * max_A_neg * max_B_pos
    else:
        term3 = torch.zeros(shape, device=tensor_1.device, dtype=tensor_1.dtype)
    
    # Компонент 4: A⁻ × B⁻
    if max_A_neg > 0 and max_B_neg > 0:
        term4 = sim(A_neg / max_A_neg, B_neg / max_B_neg) * max_A_neg * max_B_neg
    else:
        term4 = torch.zeros(shape, device=tensor_1.device, dtype=tensor_1.dtype)
    
    # Шаг 4: Собираем результат по формуле A⁺B⁺ - A⁺B⁻ - A⁻B⁺ + A⁻B⁻
    result = term1 - term2 - term3 + term4
    return result[0,:,:,:]

# Вспомогательная функция для DropPath (Stochastic Depth)
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415

    Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py
    """

    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

class PatchEmbeddings(nn.Module):
    """
    Convert the image into patches and then project them into a vector space.
    """

    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]
        # Calculate the number of patches from the image size and patch size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        # Create a projection layer to convert the image into patches
        # The layer projects each patch into a vector of size hidden_size
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        # (batch_size, num_channels, image_size, image_size) -> (batch_size, num_patches, hidden_size)
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Embeddings(nn.Module):
    """
    Combine the patch embeddings with the class token and position embeddings.
    """
        
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)
        # Create a learnable [CLS] token
        # Similar to BERT, the [CLS] token is added to the beginning of the input sequence
        # and is used to classify the entire sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["hidden_size"]))
        # Create position embeddings for the [CLS] token and the patch embeddings
        # Add 1 to the sequence length for the [CLS] token
        self.position_embeddings = \
            nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, config["hidden_size"]))
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.patch_embeddings(x)
        batch_size, _, _ = x.size()
        # Expand the [CLS] token to the batch size
        # (1, 1, hidden_size) -> (batch_size, 1, hidden_size)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # Concatenate the [CLS] token to the beginning of the input sequence
        # This results in a sequence length of (num_patches + 1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x

class AttentionHead(nn.Module):
    def __init__(self, hidden_size, attention_head_size, dropout, bias=True, use_optical=False, simulator=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_head_size = attention_head_size
        self.use_optical = use_optical
        self.sim = simulator

        self.query = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, attention_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, attention_head_size, bias=bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        if self.use_optical and self.sim is not None:
            # Используем вашу оптическую функцию
            attention_scores = optics_matmul(self.sim, query, key.transpose(-1, -2))
        else:
            # Обычное матричное умножение (torch)
            attention_scores = torch.matmul(query, key.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        if self.use_optical and self.sim is not None:
            attention_output = optics_matmul(self.sim, attention_probs, value)
        else:
            attention_output = torch.matmul(attention_probs, value)

        return attention_output, attention_probs

class MultiHeadAttention(nn.Module):
    def __init__(self, config, simulator=None, layer_idx=0, total_layers=1):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.qkv_bias = config.get("qkv_bias", True)
        self.use_optical = config.get("use_optical", False)  # флаг из конфига
        self.optical_layers = config.get("optical_layers", 0)
        self.noise_std = config.get("noise_std", 0.0)

        # Определяем, использовать ли оптику в этом слое
        if self.use_optical:
            if self.optical_layers == 0:
                # 0 означает использовать оптику во всех слоях
                self.layer_use_optical = True
            else:
                # Распределяем optical_layers равномерно по всем слоям
                # Например, если optical_layers=4 и total_layers=12, то используем оптику в слоях 0,3,6,9
                step = max(1, total_layers // self.optical_layers)
                self.layer_use_optical = (
                    layer_idx % step == 0 and layer_idx < self.optical_layers * step
                )
        else:
            self.layer_use_optical = False

        self.heads = nn.ModuleList(
            [
                AttentionHead(
                    self.hidden_size,
                    self.attention_head_size,
                    config["attention_probs_dropout_prob"],
                    bias=self.qkv_bias,
                    use_optical=self.layer_use_optical,
                    simulator=simulator,
                )
                for _ in range(self.num_attention_heads)
            ]
        )

        self.output_projection = nn.Linear(self.all_head_size, self.hidden_size)
        self.output_dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x, output_attentions=False):
        attention_outputs = [head(x) for head in self.heads]
        attention_output = torch.cat([out for out, _ in attention_outputs], dim=-1)
        attention_output = self.output_projection(attention_output)
        attention_output = self.output_dropout(attention_output)

        if not output_attentions:
            return attention_output, None
        else:
            attention_probs = torch.stack([probs for _, probs in attention_outputs], dim=1)
            return attention_output, attention_probs

class MLP(nn.Module):
    """
    A multi-layer perceptron module.
    """

    def __init__(self, config):
        super().__init__()
        self.dense_1 = nn.Linear(config["hidden_size"], config["intermediate_size"])
        self.activation = NewGELUActivation()
        self.dense_2 = nn.Linear(config["intermediate_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])

    def forward(self, x):
        x = self.dense_1(x)
        x = self.activation(x)
        x = self.dense_2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config, simulator=None, drop_path_rate=0.0, layer_idx=0, total_layers=1):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.attention = MultiHeadAttention(config, simulator, layer_idx, total_layers)
        self.layernorm_1 = nn.LayerNorm(self.hidden_size)
        self.mlp = MLP(config)
        self.layernorm_2 = nn.LayerNorm(self.hidden_size)

        # LayerScale: обучаемые параметры
        self.ls1 = nn.Parameter(torch.ones(self.hidden_size) * 1e-4)
        self.ls2 = nn.Parameter(torch.ones(self.hidden_size) * 1e-4)

        # Stochastic Depth
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x, output_attentions=False):
        # Self-attention с pre-norm
        attn_out, attn_probs = self.attention(self.layernorm_1(x), output_attentions=output_attentions)
        x = x + self.drop_path(self.ls1 * attn_out)   # residual с LayerScale и DropPath

        # MLP с pre-norm
        mlp_out = self.mlp(self.layernorm_2(x))
        x = x + self.drop_path(self.ls2 * mlp_out)

        if not output_attentions:
            return x, None
        else:
            return x, attn_probs

class Encoder(nn.Module):
    def __init__(self, config, simulator=None):
        super().__init__()
        self.blocks = nn.ModuleList()
        # Линейное возрастание вероятности stochastic depth от 0 до config["stochastic_depth_rate"]
        total_blocks = config["num_hidden_layers"]
        dpr = [x.item() for x in torch.linspace(0, config.get("stochastic_depth_rate", 0.0), total_blocks)]
        for i in range(total_blocks):
            block = Block(config, simulator, drop_path_rate=dpr[i], layer_idx=i, total_layers=total_blocks,)
            self.blocks.append(block)

    def forward(self, x, output_attentions=False):
        all_attentions = []
        for block in self.blocks:
            x, attn_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attn_probs)
        if not output_attentions:
            return x, None
        else:
            return x, all_attentions

class ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_optical = config.get("use_optical", False)

        if self.use_optical:
            # Инициализируем симулятор только если нужна оптика
            self.simulator = source.OpticalDataParallel(
                source.OpticalMul(
                    source.Config(
                        right_matrix_count_columns=512,
                        right_matrix_count_rows=512,
                        right_matrix_width=pixel_size * 512,
                        right_matrix_height=pixel_size * 512,
                        min_height_gap=pixel_size,
                        right_matrix_split_x=2,
                        right_matrix_split_y=2,
                        left_matrix_split_x=2,
                        left_matrix_split_y=2,
                        result_matrix_split=2,
                        distance=0.01,
                ))).to(device)
        else:
            self.simulator = None

        self.embedding = Embeddings(config)
        self.encoder = Encoder(config, self.simulator)
        self.classifier = nn.Linear(config["hidden_size"], config["num_classes"])

        self.apply(self._init_weights)

    def forward(self, x, output_attentions=False):
        embedding_output = self.embedding(x)
        encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions)
        logits = self.classifier(encoder_output[:, 0, :])  # [CLS] token
        return logits, all_attentions

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, Embeddings):
            nn.init.trunc_normal_(module.position_embeddings, std=self.config["initializer_range"])
            nn.init.trunc_normal_(module.cls_token, std=self.config["initializer_range"])