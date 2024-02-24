from transformers import BatchFeature

"""
reference: https://huggingface.co/microsoft/kosmos-2-patch14-224/discussions/1

Text format in Kosmos-1:
- "<s> KOSMOS -1 can perceive multimodal input, learn in context, and generate output. </s>"

Image-Caption format in Kosmos-1:
- "<s> <image> Image Embedding </image> WALL-E giving potted plant to EVE. </s>"

Multimodal format in Kosmos-1:
- "<s> <image> Image Embedding </image> This is WALL-E. <image> Image Embedding </image> This is EVE. </s>"

Multimodal Grounding format in Kosmos-2:
- “<s><image> Image Embedding </image><grounding>...”, where “<grounding>” is used to prompt the model to generate locations tokens.

"""

def process_interleaved_example(processor, prompt, images, placeholder="<i>", num_image_tokens=64, add_special_tokens=True, add_eos_token=False, return_tensors=None):

    first_image_token_id = processor.tokenizer.unk_token_id + 1

    image_input_ids = [processor.tokenizer.convert_tokens_to_ids(processor.boi_token)] + list(range(first_image_token_id, num_image_tokens + first_image_token_id)) + [processor.tokenizer.convert_tokens_to_ids(processor.eoi_token)]
    image_attention_mask = [1] * len(image_input_ids)
    # `-2`: not including `boi` and `eoi`
    image_embeds_position_mask = [0] + [1] * (len(image_input_ids) - 2) + [0]

    import re
    components = re.split(rf"({placeholder})", prompt)

    outputs = {"input_ids": [], "attention_mask": [], "image_embeds_position_mask": []}
    for component in components:
        if component != "<i>":
            # add text tokens: no special tokens -> add them at the end
            encoded = processor(text=component, add_special_tokens=False)
            for key in ["input_ids", "attention_mask"]:
                outputs[key].extend(encoded[key])
            outputs["image_embeds_position_mask"].extend([0] * len(encoded["input_ids"]))
        else:
            # add tokens to indicate image placeholder
            outputs["input_ids"].extend(image_input_ids)
            outputs["attention_mask"].extend(image_attention_mask)
            outputs["image_embeds_position_mask"].extend(image_embeds_position_mask)

    if add_special_tokens:
        outputs["input_ids"] = [processor.tokenizer.bos_token_id] + outputs["input_ids"] + ([processor.tokenizer.eos_token_id] if add_eos_token else [])
        outputs["attention_mask"] = [1] + outputs["attention_mask"] + ([1] if add_eos_token else [])
        outputs["image_embeds_position_mask"] = [0] + outputs["image_embeds_position_mask"] + ([0] if add_eos_token  else [])

    outputs["pixel_values"] = processor.image_processor(images).pixel_values

    for k in ["input_ids", "attention_mask", "image_embeds_position_mask"]:
        outputs[k] = [outputs[k]]
    outputs = BatchFeature(data=outputs,tensor_type=return_tensors)

    return outputs


def get_answer_prob(model, inputs, device, max_new_tokens=128):
    inputs = {k: v.to(device) for k, v in inputs.items()}
    generated_ids = model.generate(
        pixel_values=inputs['pixel_values'],
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        image_embeds=None,
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=max_new_tokens,
    )
    return generated_ids