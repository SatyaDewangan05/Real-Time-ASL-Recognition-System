import re

import keras as K
from .common import *
from .rexnet import KerasReXNetV1
from .efficientnet import KerasEfficientNet
from .transformer import KerasTransformer

import torch
import transformers

import timm
from efficientnet_pytorch import EfficientNet

def replace(s):
    replace_list = [
        ("/", "."), 
        ("gamma", "weight"), 
        ("beta", "bias"), 
        ("embeddings:0", "weight:0"), 
        ("depthwise_kernel:0", "weight:0"), 
        ("kernel:0", "weight:0"), 
        ("moving", "running"), 
        ("variance", "var")
    ]
    for a, b in replace_list:
        s = s.replace(a, b)
    s = re.sub("(keras_depthwise_conv2d|depthwise_conv2d|conv2d|batch_normalization)(_[0-9]+)?\.", "", s)
    return s

def modify(name, original_name, newstt):
    w = newstt[name[6:]]
    if "kernel:0" in original_name and "depthwise" not in original_name:
        w = w.permute(2, 3, 1, 0) if len(w.shape) == 4 else w.T
    elif "depthwise" in original_name:
        w = w.permute(2, 3, 0, 1)[...,0]
    return w


def transfer_efficientnet(torch_model_path, model_name, size = (160, 80)):
    stt = torch.load(torch_model_path, map_location = "cpu")["model"]
    model = EfficientNet.from_name('efficientnet-b0', num_classes=250, in_channels=3)
    model.eval()
    model.load_state_dict(stt)


    input_layer = K.layers.Input((*size, 3), batch_size = 1, name = "inputs")
    model_layer = KerasEfficientNet(model, name = "model")(input_layer)
    output_layer = KerasIdentity(None, name = "outputs")(model_layer)
    keras_model = K.models.Model(input_layer, output_layer, name = model_name)

    newstt = {k[1:].replace("._", "."): v for k, v in model.state_dict().items()}

    original_names = [_.name for _ in keras_model.weights]
    names = [replace(_.name)[:-2] for _ in keras_model.weights]
    shapes = {replace(_.name)[:-2]: _.numpy().shape for _ in keras_model.weights}
    for name, original_name in zip(names, original_names):
        if modify(name, original_name, newstt).shape != shapes[name]:
            print(name, ", ", modify(name, original_name, newstt).shape, "-->", shapes[name])

    keras_model.set_weights([modify(name, original_name, newstt) for name, original_name in zip(names, original_names)])
    return model, keras_model


def transfer_rexnet(torch_model_path, model_name, size = (160, 80)):
    stt = torch.load(torch_model_path, map_location = "cpu")["model"]
    model = timm.create_model("rexnet_100", num_classes = 250)
    model.eval()
    model.load_state_dict(stt)

    input_layer = K.layers.Input((*size, 3), batch_size = 1, name = "inputs")
    model_layer = KerasReXNetV1(model, name = "model")(input_layer)
    output_layer = KerasIdentity(None, name = "outputs")(model_layer)
    keras_model = K.models.Model(input_layer, output_layer, name = model_name)

    newstt = {k: v for k, v in model.state_dict().items()}

    original_names = [_.name for _ in keras_model.weights]
    names = [replace(_.name)[:-2] for _ in keras_model.weights]
    shapes = {replace(_.name)[:-2]: _.numpy().shape for _ in keras_model.weights}
    for name, original_name in zip(names, original_names):
        if modify(name, original_name, newstt).shape != shapes[name]:
            print(name, ", ", modify(name, original_name, newstt).shape, "-->", shapes[name])

    keras_model.set_weights([modify(name, original_name, newstt) for name, original_name in zip(names, original_names)])
    return model, keras_model


def transfer_transformer(torch_model_path, model_name):
    ckpt = torch.load(torch_model_path, map_location = "cpu")
    config = ckpt["hyper_parameters"]["config"]
    stt = {k[6:]: v for k, v in ckpt["state_dict"].items()}
    if any([k.startswith("student.") for k in stt]):
        stt = {k[8:]: v for k, v in stt.items() if k.startswith("student.")}
    num_layers = [re.search("layer.[0-9]", k) for k in stt]
    num_layers = len(set([_[0] for _ in num_layers if _ is not None]))
    if config["model_type"] == "bert":
        config = transformers.BertConfig(**config)
    else:
        config = transformers.DebertaV2Config(**config)
    config.num_hidden_layers = num_layers
    config.num_labels = 250

    input_layer = K.layers.Input((None, config.pre_emb_size), batch_size = 1, name = "inputs")
    model_layer = KerasTransformer(config, softmax = False, name = "model")(input_layer)
    output_layer = KerasIdentity(None, name = "outputs")(model_layer)
    keras_model = K.models.Model(input_layer, output_layer, name = model_name)

    newstt = {k: v for k, v in stt.items()}

    original_names = [_.name for _ in keras_model.weights]
    names = [replace(_.name)[:-2] for _ in keras_model.weights]
    shapes = {replace(_.name)[:-2]: _.numpy().shape for _ in keras_model.weights}
    for name, original_name in zip(names, original_names):
        if modify(name, original_name, newstt).shape != shapes[name]:
            print(name, ", ", modify(name, original_name, newstt).shape, "-->", shapes[name])

    keras_model.set_weights([modify(name, original_name, newstt) for name, original_name in zip(names, original_names)])
    return config, keras_model

if __name__ == "__main__":
    from utils.keras_models.preprocess import KerasPreprocessing, KerasTransformerPreprocess
    size = (160, 80)

    torch_eff, keras_eff = transfer_efficientnet("logs/cnn/160x80_8951/checkpoints/img_v0_fold0_final_loop_best_main_0.8951_0.9507.pth", "eff", size = size)
    torch_rex, keras_rex = transfer_rexnet("logs/cnn/Anthony_White_ep171/checkpoints/timm_fold0_final_loop_best_main_0.8953_0.9499.pth", "rex", size = size)

    bert_config, keras_bert = transfer_transformer("logs/transformer/Tonya_DarkViolet_f0_bert/d1/epoch=81_valid_top1=0.841.ckpt", "bert")
    deberta_config, keras_deberta = transfer_transformer("logs/transformer/Tonya_DarkViolet_f0_deberta-v2/d1/epoch=74_valid_top1=0.839.ckpt", "deberta")

    input_layer = K.layers.Input((543, 3), name = "inputs")
    preprocess_layer = KerasPreprocessing(size = size, name = "preprocess")(input_layer)
    transformer_preprocess_layer = KerasTransformerPreprocess(bert_config, name = "trans_preprocess")(input_layer)

    eff_out = keras_eff(preprocess_layer)
    rex_out = keras_rex(preprocess_layer)
    bert_out = keras_bert(transformer_preprocess_layer)
    deberta_out = keras_deberta(transformer_preprocess_layer)

    ensemble_out = rex_out * 0.6 + bert_out * 0.2 + deberta_out * 0.2
    ensemble_out = ensemble_out[0]

    output_layer = KerasIdentity(None, name = "outputs")(ensemble_out)

    keras_model = K.models.Model(input_layer, output_layer)