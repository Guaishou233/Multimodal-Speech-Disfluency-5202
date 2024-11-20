import torch
import torch.nn as nn
import torch.ao.quantization as quant



def quantification_net(model,rep_data_a,req_data_v):
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

    # Fuse the activations to preceding layers, where applicable.
    # This needs to be done manually depending on the model architecture.
    # Common fusions include `conv + relu` and `conv + batchnorm + relu`
    model_fp32_fused = quant.fuse_modules(model_fp32, [
        ['temporal_summarization_a.0','temporal_summarization_a.1'],  # Conv1d + ReLU
        ['layer1_a.0','layer1_bn_a','layer1_a.1'],  # Conv1d + ReLU
        ['layer1_v.0','layer1_bn_v','layer1_v.1'],  # Conv1d + ReLU
    ])

    # for name, module in model_fp32_fused.named_modules():
    #     device = next(module.parameters(), torch.tensor([])).device
    #     print(f"Module: {name}, Device before: {device}")
    #     module.to('cpu')
    #     print(f"Module: {name}, Device after: {next(module.parameters(), torch.tensor([])).device}")


    # Prepare the model for static quantization. This inserts observers in
    # the model that will observe activation tensors during calibration.
    model_fp32_prepared = quant.prepare(model_fp32_fused)

    # calibrate the prepared model to determine quantization parameters for activations
    # in a real world setting, the calibration would be done with a representative dataset
    model_fp32_prepared(rep_data_a,req_data_v)

    # Convert the observed model to a quantized model. This does several things:
    # quantizes the weights, computes and stores the scale and bias value to be
    # used with each activation tensor, and replaces key operators with quantized
    # implementations.
    model_int8 = quant.convert(model_fp32_prepared)

    return model_int8

