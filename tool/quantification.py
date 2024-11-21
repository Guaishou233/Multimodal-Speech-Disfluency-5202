import torch
import torch.nn as nn
import torch.ao.quantization as quant



def quantification_net(model,dataloader):
    # define a floating point model where some layers could be statically quantized
    # create a model instance
    model_fp32 = model.to('cpu')


    # model must be set to eval mode for static quantization logic to work
    model_fp32.eval()

    # attach a global qconfig, which contains information about what kind
    # of observers to attach. Use 'x86' for server inference and 'qnnpack'
    # for mobile inference. Other quantization configurations such as selecting
    # symmetric or asymmetric quantization and MinMax or L2Norm calibration techniques
    # can be specified here.
    # Note: the old 'fbgemm' is still available but 'x86' is the recommended default
    # for server inference.
    # model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
    model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('x86')

    model_fp32.clf_head[0].qconfig = torch.quantization.default_qconfig
    model_fp32.clf_head[1].qconfig = torch.quantization.default_qconfig
    model_fp32.clf_head[2].qconfig = torch.quantization.default_qconfig
    model_fp32.clf_head[3].qconfig = torch.quantization.default_qconfig
    model_fp32.clf_head[4].qconfig = torch.quantization.default_qconfig

    # for name, module in model_fp32_fused.named_modules():
    #     device = next(module.parameters(), torch.tensor([])).device
    #     print(f"Module: {name}, Device before: {device}")
    #     module.to('cpu')
    #     print(f"Module: {name}, Device after: {next(module.parameters(), torch.tensor([])).device}")


    # Prepare the model for static quantization. This inserts observers in
    # the model that will observe activation tensors during calibration.
    model_fp32_prepared = quant.prepare(model_fp32)

    # calibrate the prepared model to determine quantization parameters for activations
    # in a real world setting, the calibration would be done with a representative dataset
    for i, (feat_a, feat_v, y) in enumerate(dataloader):
        feat_a = feat_a.float()
        feat_v = feat_v.float()
        model_fp32_prepared(feat_a,feat_v)

    # Convert the observed model to a quantized model. This does several things:
    # quantizes the weights, computes and stores the scale and bias value to be
    # used with each activation tensor, and replaces key operators with quantized
    # implementations.
    model_int8 = quant.convert(model_fp32_prepared)

    return model_int8

