import argparse
import os
import torch
from ultralytics import YOLO

def check_hardware_device(requested_device):
    """Checks for hardware acceleration and returns the optimal device string."""
    if requested_device is not None and requested_device.lower() != "auto":
        return requested_device

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Hardware Check: Found {torch.cuda.device_count()} GPU(s) -> [{gpu_name}]. Using CUDA.")
        return "0" 
    elif torch.backends.mps.is_available():
        print("Hardware Check: Found Apple Silicon. Using MPS.")
        return "mps"
    else:
        print("Hardware Check: No GPU detected. Falling back to CPU.")
        return "cpu"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Universal Object Detection Training Pipeline")

    # Core required arguments
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Model architecture or path to weights")
    parser.add_argument("--data", type=str, default="data.yaml", help="Path to the dataset YAML file")
    
    # Standard Ultralytics Parameters (Default to None so YOLO handles its own defaults unless overridden)
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Image resolution")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience")
    parser.add_argument("--workers", type=int, default=None, help="Dataloader workers (reduce if out of RAM)")
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate")
    parser.add_argument("--optimizer", type=str, default=None, choices=['SGD', 'Adam', 'Adamax', 'AdamW', 'NAdam', 'RAdam', 'RMSProp'], help="Optimizer algorithm")
    
    # FIXED: Replaced type=bool with string choices to prevent string evaluation bugs in argparse
    parser.add_argument("--amp", type=str, default=None, choices=["True", "False"], help="Enable Automatic Mixed Precision (AMP)")
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")

    # Custom Parameters
    parser.add_argument("--fl_gamma", type=float, default=0.0, help="Focal Loss gamma for class imbalance")
    parser.add_argument("--device", type=str, default="auto", help="Hardware override: '0', '0,1', 'cpu', 'mps', or 'auto'")

    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Extract the clean model name for output directories
    model_base_name = os.path.splitext(os.path.basename(args.model))[0]
    target_project = "Result"
    target_name = model_base_name

    print("="*60)
    print("Initializing Universal Training Pipeline")
    print(f"Model:   {args.model}")
    print(f"Data:    {args.data}")
    print(f"Output:  {target_project}/{target_name}")
    print("="*60)

    optimal_device = check_hardware_device(args.device)

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"CRITICAL ERROR: Dataset file '{args.data}' not found.")

    try:
        model = YOLO(args.model)
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{args.model}'. Error: {e}")

    # Base kwargs that are always required or dynamically computed by our script
    training_kwargs = {
        'data': args.data,
        'project': target_project,
        'name': target_name,
        'device': optimal_device,
    }

    # Standard YOLO Arguments Injection
    # this will only override defaults if the user explicitly provided a value, otherwise YOLO's internal defaults will be used.
    standard_args = ['epochs', 'batch', 'imgsz', 'patience', 'workers', 'lr0', 'optimizer']
    
    for arg in standard_args:
        value = getattr(args, arg)
        if value is not None:
            training_kwargs[arg] = value
            print(f"Parameter Override: {arg} = {value}")

    # Explicitly handle the string-to-boolean conversion for AMP safely
    if args.amp is not None:
        training_kwargs['amp'] = (args.amp == "True")
        print(f"Parameter Override: amp = {training_kwargs['amp']}")

    # Handle Special Flags
    if args.resume:
        training_kwargs['resume'] = True
        print("Parameter Override: Resuming training from checkpoint.")

    if args.fl_gamma > 0.0:
        training_kwargs['fl_gamma'] = args.fl_gamma
        print(f"Focal Loss Enabled (gamma={args.fl_gamma}).")

    # Execute Training
    try:
        print("\nStarting training loop...")
        results = model.train(**training_kwargs)
        print(f"\nTraining complete! Check the '{target_project}' directory for results.")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nTraining failed due to an error: {e}")

if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
    
    
    
    
#python train.py --data data.yaml --model yolov8n.pt --epochs 50 --batch 16 --imgsz 640 --fl_gamma 1.5