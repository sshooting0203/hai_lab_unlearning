from omegaconf import OmegaConf
import torch, transformers, logging, os
from peft_egg.src.peft.tuners.base import Gate
from utils import *
################################3
from peft_egg.src.peft.mapping import get_peft_model
from peft_egg.src.peft.peft_model import PeftModel
from peft_egg.src.peft.utils import get_peft_model_state_dict
from peft_egg.src.peft.tuners.elder import ElderConfig, ElderLinear, ElderGraceLinear

LOG = logging.getLogger(__name__)


def translate_tokens(tokens, from_tok, to_tok):
    tokens = tokens.masked_fill(tokens == -100, from_tok.pad_token_id)
    text = from_tok.batch_decode(tokens, skip_special_tokens=True)
    return to_tok(text, return_tensors="pt")["input_ids"].to(tokens.device)


class LORA(torch.nn.Module):
    def __init__(self, model, config, model_tok,scale=None):
        super(LORA, self).__init__()
        self.config = config

        '''Apply_lora
        '''
        # r_num = config.grace.num_block * config.grace.num_rank_per_block

        self.lora_config = ElderConfig( # @ Elder Config로 바꿔야함
            r = config.lora.lora_r, # @ 수정
            lora_alpha = config.lora.lora_alpha, # @ 수정
            lora_dropout = config.lora.lora_dropout, # @ 수정
            
            num_experts = config.num_experts,
            target_modules= list(config.model.target_modules), #
            task_type = config.lora_task_type, # @ 수정
            fan_in_fan_out= config.model.fan_in_fan_out, #

            
            grace_layer = config.model.grace_layer, #
            grace_config= OmegaConf.to_object(config.grace) #
        )
        self.log_dict = {}

        '''Load
        '''
        # self.original_model = model
        # self.model = model

        print(type(self.lora_config), self.lora_config)
        if not config.check_dir:
            self.model = get_peft_model(model, self.lora_config) # @ 실제로 MoE 모듈이 구성되는 위치 
        else:
            save_path = os.path.join(config.base_dir, "checkpoint", config.check_dir)
            self.load_from_checkpoint(save_path)

        self.lora_list = self.named_lora_modules() # @ experts들이라고 보면 됨 
        self.grace_layer = self.named_grace_layer() # @ routing network라고 보면 됨 
        # self.register_lora_backward_hooks(lora_backward_hook)

        '''Load Tokenizer
        '''
        self.model_tok = model_tok
        # self.classifier_tok = transformers.AutoTokenizer.from_pretrained(config.lora.cls_name) # @ 수정

        '''Parameters to be optimized
        '''
        self.opt_params = self.optim_parameters()
        pass

    def optim_parameters(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad==True and 'lora' not in name:
                if "gate" in name:
                    continue
                param.requires_grad = False
        lora_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        return lora_params

    def load_from_checkpoint(self, save_path):
        print(save_path)

    def save_classifier_weights(self,cls_dir):
        if not os.path.exists(cls_dir):
            os.makedirs(cls_dir,exist_ok=True)
        torch.save(self.classifier.state_dict(),f"{cls_dir}/classifier.pt")
    def save_lora_weights(self,lora_dir):
        self.model.save_pretrained(lora_dir+"/lora_checkpoint")

    def reset_lora(self):
        for key in self.lora_list:
            self.model.get_submodule(key).reset_lora_parameters('default')

    def named_lora_modules(self):
        module_list = [key for key,_ in self.model.named_modules()]
        lora_list = []
        for key in module_list:
            # if isinstance(self.model.get_submodule(key),LoraLayer):
            if isinstance(self.model.get_submodule(key), ElderLinear): # @ 수정
                lora_list.append(key)
        return lora_list

    def named_grace_layer(self) -> str:
        module_list = [key for key, _ in self.model.named_modules()]
        grace_list = []
        for key in module_list:
            # if isinstance(self.model.get_submodule(key), GraceLayer):
            if isinstance(self.model.get_submodule(key), ElderGraceLinear): # @ 수정
                grace_list.append(key)
        assert len(grace_list) == 1, "At Most One Grace Layer"
        return grace_list[0]

    def register_lora_backward_hooks(self,backward_hook_fn):
        for key in self.lora_list:
            self.model.get_submodule(key).register_backward_hook(backward_hook_fn)

    def disable_melo(self):
        self.model.base_model.disable_adapter_layers()
        self.model.base_model.disable_grace_layer()

    def enable_melo(self):
        self.model.base_model.enable_adapter_layers()
        self.model.base_model.enable_grace_layer()

    def edit(self, tokens):
        debug_trainables(self.model)
        optimizer = torch.optim.Adam(self.optim_parameters(), self.config.grace.edit_lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.5)
        # --- pass edit label, training mode, and key_id into GRACE ---
        # setattr(self.model.get_submodule(self.grace_layer), "training", True) 
        setattr(self.model.get_submodule(self.grace_layer), "editing", True) # @ 수정
        setattr(self.model.get_submodule(self.grace_layer), "is_last_iter", False) # @ 수정
        setattr(self.model.get_submodule(self.grace_layer), "edit_label", tokens["labels"])
        for _, m in self.model.named_modules():
            if hasattr(m, "gate") and isinstance(m.gate, Gate):
                m.gate.editing = True

        self.losses = []
        for i in range(self.config.grace.num_iter):
            # --- insert iteration into each layer (only initiate keys on first iteration) ---
            setattr(self.model.get_submodule(self.grace_layer), "batch_iter", i)

            if i == self.config.grace.num_iter - 1: # @ 수정
                setattr(self.model.get_submodule(self.grace_layer), "is_last_iter", True)
            else:
                setattr(self.model.get_submodule(self.grace_layer), "is_last_iter", False)
  
            # --- pass tokens through model (including through the GRACE layer) ---
            outputs = self.model.model(**tokens)
            lm_loss = outputs.loss

            gate_loss_sum = 0.0
            for _, m in self.model.named_modules():
                if hasattr(m, "gate_loss") and m.gate_loss is not None:
                    print("this is gate loss : ", m.gate_loss)
                    gate_loss_sum = gate_loss_sum + m.gate_loss
            loss = lm_loss + self.config.gate_lambda * gate_loss_sum
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            self.losses.append(loss.detach().cpu().numpy())
            LOG.info(f'batch loss in iter {i}: {loss.detach().cpu().numpy()}')

        self.loss = loss # Log final loss
        # setattr(self.model.get_submodule(self.grace_layer), "training", False) # @ 수정
        setattr(self.model.get_submodule(self.grace_layer), "editing", False) 
        for _, m in self.model.named_modules():
            if hasattr(m, "gate") and isinstance(m.gate, Gate):
                m.gate.editing = False
                # m.gate.set_target(target, set_mode="soft") # @ target 붙여야

    def generate(self, *args, **kwargs):
        return self.model.model.generate(*args, **kwargs)

    def get_VecDB_info(self):
        VecDB_logdict = {}
        VecDB_logdict["num_cluster"] = len(getattr(self.model.get_submodule(self.grace_layer), "VecDB"))
        VecDB_logdict["conflict_num"] = getattr(self.model.get_submodule(self.grace_layer), "VecDB").conflict_num
        VecDB_logdict["forget_keys"] = len(getattr(self.model.get_submodule(self.grace_layer), "VecDB").forget_keys)
        return VecDB_logdict

if __name__ == '__main__':
    pass


def debug_trainables(model):
    print("model is  : : ", model)
    lora_cnt = gate_cnt = other_cnt = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            if "lora_" in n:
                lora_cnt += 1
            elif "gate" in n:
                gate_cnt += 1
            else:
                other_cnt += 1
    print(f"[dbg] trainable params: lora={lora_cnt}, gate={gate_cnt}, other={other_cnt}")



















