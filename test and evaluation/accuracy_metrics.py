def calculate_accuracy_metrics(model, data_yaml, split, imgsz, batch, device, conf, iou, verbose, plots):
    """
    Compute detection accuracy metrics using the `val()` method of an Ultralytics YOLO model.

    This function runs validation on a given data split (e.g. test or validation), using parameters
    such as image size, confidence threshold, IoU threshold, etc. It returns key metrics including
    mAP at IoU thresholds, precision, and recall, which are useful for assessing model performance.

    Args:
        model: An Ultralytics YOLO model object (possibly with early‐exit layers and quantization/pruning executed) loaded via YOLO(...).
        data_yaml (str): Path to the data configuration YAML file. This file should include paths
            to images/labels for different splits (train/val/test) and class names.
        split (str): Which dataset split to evaluate, typically 'test' or 'val'.
        imgsz (int): Input image size (width = height) used during validation; images are resized to this.
        batch (int): Batch size to use during model.val(...).
        device (str or int or None): Device identifier for inference/validation, e.g. "cpu", "cuda:0", or GPU index.
        conf (float): Confidence threshold – predictions below this confidence are ignored in evaluation.
        iou (float): IoU (Intersection over Union) threshold for determining matches between predicted boxes and ground truths
            and for Non-Maximum Suppression (NMS) in evaluation.
        verbose (bool): If True, show detailed logging/messages during validation.
        plots (bool): If True, generate and/or save visual plots like PR curves, confusion matrices, etc.

    Returns:
        tuple of float: (map50_95, map50, precision, recall) where:
            - map50_95: Mean Average Precision over IoU thresholds from 0.50 to 0.95 (in steps of 0.05).  
            - map50: Average Precision at IoU = 0.50.  
            - precision: Mean precision across all classes. The ratio of true positive detections to all positive predictions.  
            - recall: Mean recall across all classes. The ratio of true positive detections to all ground truth instances.

    Raises:
        Exception: If model.val(...) fails or the expected metrics are not available in metrics.box.  
        ValueError: If conversion of metric values to float fails, or if results_dict fallback is missing required keys.

    """
    
    metrics = model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=conf,   
        iou=iou,
        verbose=verbose,
        plots=plots
    )
    
    # Pull out key metrics (mAP50-95, mAP50, plus precision/recall if available)
    try:
        map_50_95 = float(metrics.box.map)
        map_50    = float(metrics.box.map50)    
        precision  = float(getattr(metrics.box, "mp", float("nan")))
        recall     = float(getattr(metrics.box, "mr", float("nan")))

        return map_50_95, map_50, precision, recall
    
    except Exception:
        # Fallback: try results_dict if present
        rd = getattr(metrics, "results_dict", {})
        map_50_95 = float(rd.get("metrics/mAP50-95(B)", float("nan")))
        map_50    = float(rd.get("metrics/mAP50(B)", float("nan")))
        precision = float(rd.get("metrics/precision(B)", float("nan")))
        recall    = float(rd.get("metrics/recall(B)", float("nan")))

        return map_50_95, map_50, precision, recall