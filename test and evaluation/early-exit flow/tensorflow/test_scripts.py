import json, collections
import numpy as np
import tensorflow as tf
from image_loader import test_images
from fire_detector import which_exit_fired
from estimate_macs import estimate_macs_from_analyzer
from group_detection_outputs import group_detection_outputs


def run_ee_analysis(model_path, image_loader, folder_path):
    """Run early-exit model on images from image_loader, collect usage stats.
    Parameters
    ----------
    model_path : str
        Path to TFLite early-exit model file.
    image_loader : callable
        Function or generator yielding preprocessed input images as numpy arrays.
    Returns
    -------
    dict
        Dictionary with usage counts and average confidence by exit.
    """
    interp = tf.lite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    in_det = interp.get_input_details()
    out_det = interp.get_output_details()

    usage = collections.Counter()
    confs = collections.defaultdict(list)

    for img in image_loader(folder_path):

        interp.set_tensor(in_det[0]['index'], img)
        interp.invoke()

        outputs = group_detection_outputs(out_det, interp, 18)
        if not outputs:
            # Nothing matching detection head shape; skip or log
            continue

        winner, conf = which_exit_fired(outputs, per_exit_threshold=0.25)

        print(f"Chosen exit: {winner} with confidence {conf:.4f}")
        
        usage[winner] += 1
        confs[winner].append(conf)

    return {
        'usage_counts': dict(usage),
        'avg_conf_by_exit': {k: float(np.mean(v)) for k, v in confs.items() if v}
    }


if __name__ == "__main__":

    
    ee_model = "models/YOLOv8/early-exit based/backbone + neck/quantized/EE_Backbone_Neck_float32.tflite"

    ee_stats = run_ee_analysis(
        model_path= ee_model,
        image_loader= test_images,
        folder_path="C:/Users/abt/Documents/Real-time-obstacle-detector/data sets/dataset/dataset/test/images"
    )

    print("# === Early-exit usage ===")
    print(json.dumps(ee_stats, indent=2))

    from tensorflow.lite.experimental import Analyzer
    dump = Analyzer.analyze(model_path=ee_model)
    macs = estimate_macs_from_analyzer(dump)
    print("# === Estimated MACs ===")
    print(json.dumps(macs, indent=2))
