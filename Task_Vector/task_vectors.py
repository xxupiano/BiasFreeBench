import torch
from transformers import AutoModelForCausalLM



class TaskVector():
    def __init__(self, pretrained_checkpoint=None, finetuned_checkpoint=None, vector=None, torch_dtype=torch.bfloat16):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.
        
        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        self.torch_dtype = torch_dtype

        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint is not None and finetuned_checkpoint is not None
            with torch.no_grad():
                pretrained_state_dict= self.load_state_dict(pretrained_checkpoint)
                finetuned_state_dict= self.load_state_dict(finetuned_checkpoint)
                self.vector = {}
                for key in pretrained_state_dict:
                    if key not in finetuned_state_dict:
                        continue
                    if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                        continue
                    try:
                        self.vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
                    except Exception as e:
                        print(f"Error computing vector for key {key}: {e}")
    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f'Warning, key {key} is not present in both task vectors.')
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector, torch_dtype=self.torch_dtype)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = - self.vector[key]
        return TaskVector(vector=new_vector, torch_dtype=self.torch_dtype)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pretrained_model = AutoModelForCausalLM.from_pretrained(
                pretrained_checkpoint,
                device_map="cpu",
                torch_dtype=self.torch_dtype,  # or bfloat16 if your checkpoint uses it
                low_cpu_mem_usage=True,
                trust_remote_code=True,  # if you are using models from Huggingface Hub
            )
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(f'Warning: key {key} is present in the pretrained state dict but not in the task vector')
                    continue
                new_state_dict[key] = pretrained_state_dict[key] + scaling_coef * self.vector[key]
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model

    def load_state_dict(self, model_name_or_path):
        with torch.no_grad():
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                device_map="cpu",
                torch_dtype=self.torch_dtype,  # or bfloat16 if your checkpoint uses it
                low_cpu_mem_usage=True,
                trust_remote_code=True,  # if you are using models from Huggingface Hub
            )
        return model.state_dict()
