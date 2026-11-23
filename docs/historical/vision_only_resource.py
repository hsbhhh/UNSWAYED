"""Opt-in vision-only InternVideo2 loading; no text model/weights constructed."""
import gc
import importlib
import json
import sys
import torch
sys.path.insert(0, '/root/autodl-tmp/hsb/get_data')
sys.path.insert(0, '/root/autodl-tmp/hsb/get_data/fine-turn')
from adapters import internvideo2_adapter as base_adapter


class VisionOnlyAdapter(base_adapter.InternVideo2Adapter):
    """Load official weights on CPU before moving the used vision path to GPU.

    q_sem and VGC tokens use encode_vision only. Preserve official vision and
    CLIP-projector loading, skip text construction/loading, and use a real
    video forward probe to determine feature_dim.
    """
    def load(self):
        discovery = self.discover()
        one = base_adapter._single_or_raise
        config_path = one('InternVideo2 config', discovery.config_candidates)
        pretrained = one('InternVideo2 pretrained checkpoint', discovery.weight_candidates['pretrained'])
        tokenizer = one('InternVideo2 tokenizer', discovery.weight_candidates['tokenizer'])
        package_root, prefix = base_adapter.resolve_import_package(discovery.multi_modality_root)
        base_adapter._ensure_path(package_root)
        base_adapter._ensure_path(discovery.multi_modality_root)
        models = importlib.import_module(prefix + '.models')
        sys.modules['models'] = models
        sys.modules['utils'] = importlib.import_module(prefix + '.utils')
        config_module = importlib.import_module(prefix + '.utils.config')
        cfg = config_module.eval_dict_leaf(config_module.Config.from_file(str(config_path)))
        cfg.device = self.device_text
        cfg.num_frames = cfg.num_frames_test = self.num_frames
        cfg.model.vision_encoder.num_frames = self.num_frames
        cfg.model.tokenizer_path = cfg.model.text_encoder.llama_path = str(tokenizer)
        cfg.model.vision_ckpt_path = str(one('InternVideo2 vision checkpoint', discovery.weight_candidates['vision']))
        cfg.model.text_encoder.use_lora = False
        cfg.model.extra_ckpt_path = None
        cfg.deepspeed.enable = False
        cfg.use_half_precision = self.dtype == 'fp16'
        cfg.use_bf16 = self.dtype == 'bf16'
        cfg.gradient_checkpointing = False
        self.backend_report = base_adapter._configure_optional_cuda_extensions(cfg, self.attention_backend)
        original_cls = getattr(models, cfg.model.get('model_cls','InternVideo2_CLIP'))

        class VisualCLIP(original_cls):
            def build_text_encoder(self):
                return torch.nn.Identity()
            def load_checkpoint(self, vision_ckpt_path=None, *args, **kwargs):
                # Same official visual stage-2 initialization/filtering and
                # position interpolation as InternVideo2_CLIP.load_checkpoint.
                ckpt = torch.load(vision_ckpt_path, map_location='cpu', mmap=True, weights_only=False)
                ckpt = ckpt.get('module', ckpt.get('model', ckpt))
                if self.config.model.get('load_vision_ckpt_from_internvideo2_stage2',False):
                    module = importlib.import_module(prefix+'.models.backbones.internvideo2.pos_embed')
                    module.interpolate_pos_embed(ckpt,self.vision_encoder,orig_t_size=self.config.model.get('vision_ckpt_t_size',4))
                    visual = {k:v for k,v in ckpt.items() if k.startswith('vision_encoder.')
                              and not any(x in k for x in ('clip_decoder','final_clip_decoder','clip_pos_embed','clip_img_pos_embed','img_pos_embed'))}
                else:
                    visual = {'vision_encoder.'+k:v for k,v in ckpt.items()
                              if not k.startswith(('clip_decoder.','mae_decoder.','final_clip_decoder.'))
                              and k not in ('clip_pos_embed','mae_pos_embed')}
                self.visual_init_keys = set(visual)
                self.load_state_dict(visual,strict=False)

        model = VisualCLIP(config=cfg, is_pretrain=True)
        checkpoint = torch.load(str(pretrained), map_location='cpu', mmap=True, weights_only=False)
        state = checkpoint.get('model', checkpoint.get('module', checkpoint))
        visual = {k:v for k,v in state.items() if k.startswith('vision_encoder.')}
        expected = {k:v for k,v in model.state_dict().items() if k.startswith('vision_encoder.')}
        missing = [k for k,v in expected.items() if k not in visual and k not in model.visual_init_keys
                   and 'pos_embed' not in k]
        if missing:
            raise RuntimeError(f'Official CLIP visual checkpoint incomplete: {missing[:20]}')
        model.load_state_dict(visual, strict=False)
        del checkpoint, state, visual
        gc.collect()
        for param in model.parameters():
            param.requires_grad_(False)
        dtype = torch.bfloat16 if self.dtype == 'bf16' else torch.float16 if self.dtype == 'fp16' else torch.float32
        model.to(dtype=dtype).to(torch.device(cfg.device)).eval()
        self.backend_report.update(base_adapter._inspect_runtime_backend(model, self.attention_backend))
        self.model, self.config, self.tokenizer = model, cfg, model.tokenizer
        self.load_report = {'vision_missing':0,'vision_tensors':len(expected),'text_parameters':0,
                            'stage2_visual_checkpoint':cfg.model.vision_ckpt_path,'clip_checkpoint':str(pretrained)}
        print('VISION_ONLY '+json.dumps(self.load_report),flush=True)
        import numpy as np
        probe = self.encode_video(np.zeros((self.num_frames,self.image_size,self.image_size,3),dtype=np.uint8))
        self.feature_dim = int(probe.shape[-1])


def install_vision_only():
    """Only the caller's process is patched; default resident servers unchanged."""
    import fine_turn.models.internvideo2_adapter as encoder_module
    if not getattr(encoder_module.build_video_encoder, '_vision_only', False):
        original = encoder_module.build_video_encoder
        def build_encoder(resources, config):
            # Teacher evicts the base adapter module before each load.
            importlib.import_module('adapters.internvideo2_adapter').InternVideo2Adapter = VisionOnlyAdapter
            return original(resources, config)
        build_encoder._vision_only = True
        encoder_module.build_video_encoder = build_encoder
    import fine_turn.models.lora_adapter as inference_lora
    import fine_turn.models.lora_adapter_v3 as visual_lora
    inference_lora.LoRAConfig = visual_lora.LoRAConfig
    inference_lora.apply_lora_to_named_linears = visual_lora.apply_lora_to_named_linears
