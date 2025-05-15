from typing import List, Optional
from cvnets import get_model
from options.opts import get_training_arguments
import torch


def main(args: Optional[List[str]] = None, **kwargs):
    opts = get_training_arguments(args=args)
    print(opts)
    
    model = get_model(opts)
    
    torch.save(model, "model/coco-ssd-mobilevitv2-0.75_structure.pt")

    model.info()

if __name__ == "__main__":
    main()