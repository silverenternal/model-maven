import torch.nn.functional as F


def multitask_step(model, batch, context, alignment_weight=0.1, temporal_weight=0.05):
    outputs = model(*batch["args"], **batch["kwargs"])
    targets = batch["loss_inputs"]["targets"]
    classification = F.cross_entropy(outputs["logits"], targets)

    alignment = 1 - F.cosine_similarity(
        outputs["video_embedding"], outputs["text_embedding"], dim=-1
    ).mean()

    valid = outputs["valid_frames"]
    frame_targets = targets[:, None].expand_as(valid)
    frame_losses = F.cross_entropy(
        outputs["frame_logits"].transpose(1, 2), frame_targets, reduction="none"
    )
    temporal = (frame_losses * valid).sum() / valid.sum().clamp_min(1)
    loss = classification + alignment_weight * alignment + temporal_weight * temporal

    context.log("video_frames", batch["args"][0].shape[1])
    context.log("text_tokens", batch["args"][1].shape[1])
    return {
        "loss": loss,
        "losses": {
            "classification": classification.detach(),
            "alignment": alignment.detach(),
            "temporal": temporal.detach(),
        },
        "metrics": {"accuracy": (outputs["logits"].detach().argmax(-1) == targets).float().mean()},
        "outputs": outputs,
    }


def manual_multitask_step(model, batch, context):
    """User-owned backward and two disjoint optimizer groups with a freeze phase."""
    video_trainable = context.step > 2
    for parameter in model.video_encoder.parameters():
        parameter.requires_grad_(video_trainable)
    context.log("video_trainable", int(video_trainable))

    encoders = context.optimizer(
        "encoders",
        list(model.video_encoder.parameters()) + list(model.text_encoder.parameters()),
        lr=1e-4,
    )
    fusion_heads = context.optimizer(
        "fusion_heads",
        list(model.fusion.parameters())
        + list(model.classification_head.parameters())
        + list(model.temporal_head.parameters()),
        lr=3e-4,
    )
    encoders.zero_grad()
    fusion_heads.zero_grad()

    outputs = model(*batch["args"], **batch["kwargs"])
    targets = batch["loss_inputs"]["targets"]
    classification = F.cross_entropy(outputs["logits"], targets)
    alignment = 1 - F.cosine_similarity(
        outputs["video_embedding"], outputs["text_embedding"], dim=-1
    ).mean()
    valid = outputs["valid_frames"]
    frame_targets = targets[:, None].expand_as(valid)
    frame_losses = F.cross_entropy(
        outputs["frame_logits"].transpose(1, 2), frame_targets, reduction="none"
    )
    temporal = (frame_losses * valid).sum() / valid.sum().clamp_min(1)
    total = classification + 0.1 * alignment + 0.05 * temporal

    context.backward(total)
    encoders.step()
    fusion_heads.step()
    context.log("video_frames", batch["args"][0].shape[1])
    context.log("text_tokens", batch["args"][1].shape[1])
    return {
        "loss": total.detach(),
        "losses": {
            "classification": classification.detach(),
            "alignment": alignment.detach(),
            "temporal": temporal.detach(),
        },
        "outputs": outputs,
    }


def overlapping_optimizer_step(model, batch, context):
    context.optimizer("all", model.parameters())
    context.optimizer("fusion", model.fusion.parameters())
    return {"loss": 0.0}


# Both steps derive phase state from context.step and keep no hidden Python state.
multitask_step.MODEL_MAVEN_STATELESS = True
manual_multitask_step.MODEL_MAVEN_STATELESS = True
