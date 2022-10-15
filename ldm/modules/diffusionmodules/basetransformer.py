import torch
from torch import nn
import torch.nn.functional as F

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm
    class LayerNorm(FusedLayerNorm):
        def __init__(self, *args, pb_relax=False, **kwargs):
            super().__init__(*args, **kwargs)
            self.pb_relax = pb_relax

        def forward(self, x):
            if not self.pb_relax:
                return super().forward(x)
            return super().forward(x / (x.abs().max().detach() / 8))
except ModuleNotFoundError:
    print('Please install apex to use fused_layer_norm, fall back to torch.nn.LayerNorm')
    from  torch.nn import LayerNorm
    
def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, '{} is not divisible by {}'.format(
        numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator

def sqrt(x):
    return int(math.sqrt(x) + 1e-4)

def unscaled_init_method(sigma):
    """Init method based on N(0, sigma)."""
    def init_(tensor, **kwargs):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_

def scaled_init_method(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)
    def init_(tensor, **kwargs):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_

@torch.jit.script
def gelu_impl(x):
     """OpenAI's gelu implementation."""
     return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *
                                        (1.0 + 0.044715 * x * x)))

def gelu(x): 
    return gelu_impl(x)


class SelfAttention(nn.Module):
    def __init__(self,):
        super(SelfAttention, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        if hidden_size_per_attention_head is None:
            self.hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)
        else:
            self.hidden_size_per_attention_head = hidden_size_per_attention_head
        self.inner_hidden_size = num_attention_heads * self.hidden_size_per_attention_head
        
        self.query_key_value = 
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)
        
        self.dense = 
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)
    
    def forward(self, hidden_states, **kw_args):
        pass
        
        
class CrossAttention(nn.Module):
    def __init__(self,):
        super().__init__()
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.layer_id = layer_id
        
        self.hidden_size = hidden_size
        if hidden_size_per_attention_head is None:
            self.hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)
        else:
            self.hidden_size_per_attention_head = hidden_size_per_attention_head
        self.inner_hidden_size = num_attention_heads * self.hidden_size_per_attention_head
        
        self.query = 
        self.key_value = 
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)
        
        # Output.
        self.dense =
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)
    
    def forward(self, hidden_states, context, **kw_args):
        pass

class MLP(nn.Module):
    def __init__(self,):
        super(MLP, self).__init__()
        self.layer_id = layer_id
        self.activation_func = activation_func
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        
        # Project to 4h.
        self.hidden_size = hidden_size
        if inner_hidden_size is None:
            inner_hidden_size = 4 * hidden_size
        self.inner_hidden_size = inner_hidden_size
        self.dense_h_to_4h = 
        # Project back to h.
        self.dense_4h_to_h =
        self.dropout = torch.nn.Dropout(output_dropout_prob)
        
    def forward(self, hidden_states, **kw_args):
        pass

class BaseTransformerLayer(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_attention_heads,
                 attention_dropout_prob,
                 output_dropout_prob,
                 layernorm_epsilon,
                 init_method,
                 layer_id,
                 inner_hidden_size=None,
                 hidden_size_per_attention_head=None,
                 output_layer_init_method=None,
                 layernorm_order='pre',
                 layernorm=LayerNorm,
                 use_bias=True,
                 activation_func=gelu,
                 skip_init=False,
    ):
        super(BaseTransformerLayer, self).__init__()
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.layer_id = layer_id
        self.layernorm_order = layernorm_order
        
        # Layernorm on the input data.
        self.input_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)
        
        self.attention = SelfAttention(
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            init_method,
            layer_id,
            hidden_size_per_attention_head=hidden_size_per_attention_head,
            output_layer_init_method=output_layer_init_method,
            bias=use_bias,
            
        )
        
        # Layernorm on the input data.
        self.post_attention_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)
        if self.layernorm_order == 'sandwich':
            self.third_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)
            self.fourth_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)
            
        self.cross_attention = CrossAttention(
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            init_method,
            layer_id,
            hidden_size_per_attention_head=hidden_size_per_attention_head,
            output_layer_init_method=output_layer_init_method,
            bias=use_bias,
        )
        self.post_cross_attention_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)
        
        # MLP
        self.mlp = MLP(
            hidden_size,
            output_dropout_prob,
            init_method,
            inner_hidden_size=inner_hidden_size,
            output_layer_init_method=output_layer_init_method,
            bias=use_bias,
            layer_id=layer_id,
            activation_func=activation_func,
        )
    
    def forward(self, hidden_states, *kw_args):
        pass

class BaseTransformer(nn.Module):
    def __init__(self, ):
        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_num_layers = checkpoint_num_layers
        self.max_sequence_length = max_sequence_length
        self.layernorm_order = layernorm_order
        
        # create all layers
        if init_method is None:
            self.output_layer_init_method = scaled_init_method(init_method_std, num_layers)
            self.init_method = unscaled_init_method(init_method_std)
        else:
            self.output_layer_init_method = init_method
            self.init_method = init_method
            
        def get_layer(layer_id):
            return BaseTransformerLayer(
                hidden_size,
                num_attention_heads,
                attention_dropout_prob,
                output_dropout_prob,
                layernorm_epsilon,
                self.init_method,
                layer_id,
                inner_hidden_size=inner_hidden_size,
                hidden_size_per_attention_head=hidden_size_per_attention_head,
                output_layer_init_method=self.output_layer_init_method,
                layernorm_order=layernorm_order,
                layernorm=layernorm,
                use_bias=use_bias,
                activation_func=activation_func,
            )
        
        self.layers = nn.ModuleList(
            [get_layer(layer_id) for layer_id in range(num_layers)]
        )
        
        # Final layer norm before output.
        self.use_final_layernorm = use_final_layernorm
        if use_final_layernorm:
            self.final_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)
        
        
    def forward(self, ):
        
        
        output_per_layers, output_this_layer = [], []
        for i, layer in enumerate(self.layers):
            args = [hidden_states, attention_mask]
            
            output_this_layer_obj, output_cross_layer_obj = {}, {}
            layer_ret = layer(*args, layer_id=torch.tensor(i), **kw_args, **output_cross_layer,
                output_this_layer=output_this_layer_obj, output_cross_layer=output_cross_layer_obj)
            if isinstance(layer_ret, tuple):
                layer_ret = layer_ret[0]
            hidden_states, output_this_layer, output_cross_layer = layer_ret, output_this_layer_obj, output_cross_layer_obj
            
            if output_hidden_states:
                output_this_layer['hidden_states'] = hidden_states
            output_per_layers.append(output_this_layer)
        
        if self.use_final_layernorm:
            logits = self.final_layernorm(hidden_states)
        else:
            logits = hidden_states
            
        outputs = [logits] + output_per_layers
        return outputs