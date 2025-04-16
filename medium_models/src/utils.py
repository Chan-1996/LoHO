def convert_masking_prob(model_name, prob):
    if '125m' in model_name:
        embed_dim = 768
        num_layers = 12
    elif '1.3b' in model_name:
        embed_dim = 2048
        num_layers = 24
    elif '13b' in model_name:
        embed_dim = 5120
        num_layers = 40
    elif 'llama-7b' in model_name:
        embed_dim = 4096
        num_layers = 32
    elif 'opt-6.7b' in model_name:
        embed_dim = 4096
        num_layers = 32
    else:
        raise NotImplementedError

    ffn_dim = 4 * embed_dim

    embed = 50272 * embed_dim + 2050 * embed_dim
    final_layer_norm = embed_dim * 2

    attn = embed_dim * embed_dim * 4 + embed_dim * 4
    linear = embed_dim * ffn_dim * 2 + embed_dim + ffn_dim
    layer_norm = embed_dim * 4

    param_count = num_layers * (attn + linear + layer_norm) + embed + final_layer_norm

    num_remaining_param = param_count * (1 - prob)
    param_per_linear = num_remaining_param / num_layers / 2
    true_masking_prob = 1 - param_per_linear / embed_dim / embed_dim
    if true_masking_prob > 1 or true_masking_prob < 0:
        raise ValueError
    return true_masking_prob