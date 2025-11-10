"""
The Analyzer gives you a structural dump you can machine-parse; the Interpreter lets you tell which tensors are constants 
(weights), their dtypes, and whether they carry quantization scales/zero-points. 
Comparing models pinpoints how many parameters were added by EE heads, removed by pruning, 
and how many tensors are quantized vs float. For pruning, if you exported sparsity, 
the flatbuffer SparsityParameters indicate blocked sparsity and let you estimate density; 
"""
import json
from tflite_analyzer import dump_analyzer
from tflite_interpretor import interpreter_details
from parse_param_counts import parse_param_counts_from_analyzer
from utils import count_params_from_interpreter, diff

def summarize(model_path):

    txt = dump_analyzer(model_path)
    total_est, by_dtype_est, per_tensor_est = parse_param_counts_from_analyzer(txt)
    tens, const_flags = interpreter_details(model_path)
    interp_stats = count_params_from_interpreter(tens, const_flags)

    return {
        'model': model_path,
        'analyzer_param_estimate_total': total_est,
        'analyzer_params_by_dtype_estimate': dict(by_dtype_est),
        'interpreter_param_total_const': interp_stats['const_param_total'],
        'params_by_dtype_const': interp_stats['const_params_by_dtype'],
        'num_quantized_tensors': interp_stats['num_quantized_tensors'],
        'num_float_tensors': interp_stats['num_float_tensors'],
        'sample_quantized_tensor_scales': interp_stats['quantized_tensor_scales'],
        'graph_dump_head': "\n".join(dump_analyzer(model_path).splitlines()[:200])
    }

if __name__ == "__main__":
    
    baseline_fp32 = "models/YOLOv8/yolov8-obstacle-detection-18-objects/results/quantized/FLOAT16 and 32/best_yolov8_float32.tflite"
    ee_fp32      = "models/YOLOv8/early-exit based/backbone + neck/results/EE_backbone_neck_manual_pruned_float32.tflite"
    pruned_fp32  = "models/YOLOv8/yolov8-obstacle-detection-18-objects/results/pruned/customized_without_EE_pruned_float32.tflite"
    quant_int8   = "models/YOLOv8/yolov8-obstacle-detection-18-objects/results/quantized/INT8/best_yolov8_int8.tflite"

    reports = { 'baseline': summarize(baseline_fp32) }
    for name, path in [('early_exit', ee_fp32), ('pruned', pruned_fp32), ('quant', quant_int8)]:
        reports[name] = summarize(path)

    added_removed_ee   = diff(reports['baseline'], reports['early_exit'])
    added_removed_prun = diff(reports['baseline'], reports['pruned'])
    added_removed_quant= diff(reports['baseline'], reports['quant'])

    print("# === PARAM SUMMARY ===")
    print(json.dumps({
        'baseline': reports['baseline']|{'graph_dump_head': reports['baseline']['graph_dump_head']},
        'early_exit': reports['early_exit']|{'graph_dump_head': reports['early_exit']['graph_dump_head']},
        'pruned': reports['pruned']|{'graph_dump_head': reports['pruned']['graph_dump_head']},
        'quant': reports['quant']|{'graph_dump_head': reports['quant']['graph_dump_head']},
        'diff_vs_baseline': {
            'early_exit': added_removed_ee,
            'pruned': added_removed_prun,
            'quant': added_removed_quant
        }
    }, indent=2))
