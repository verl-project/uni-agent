from __future__ import annotations

import pytest
import torch

from tests.uni_agent.support import FakeProcessor


@pytest.mark.cpu
@pytest.mark.level0
def test_compute_multi_modal_inputs_returns_empty_dict_without_processor():
    """Without a processor, multimodal postprocess is a no-op and returns an
    empty field map even when raw image data is present."""
    from uni_agent.framework.multi_modal_postprocess import compute_multi_modal_inputs

    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    assert compute_multi_modal_inputs(None, input_ids, {"images": ["image://a.png"]}) == {}


@pytest.mark.cpu
@pytest.mark.level0
def test_compute_multi_modal_inputs_returns_image_tensors_and_images_seqlens():
    """Image inputs are converted through the processor while generated
    ``input_ids`` and ``attention_mask`` are stripped from the TQ payload."""
    from uni_agent.framework.multi_modal_postprocess import compute_multi_modal_inputs

    processor = FakeProcessor()
    input_ids = torch.tensor([[11, processor.image_token_id, 12]], dtype=torch.long)

    multi_modal_inputs = compute_multi_modal_inputs(
        processor,
        input_ids,
        {"images": ["image://a.png"]},
        {"max_pixels": 1024},
    )

    assert "input_ids" not in multi_modal_inputs
    assert "attention_mask" not in multi_modal_inputs
    assert tuple(multi_modal_inputs["pixel_values"].shape) == (1, 3, 2, 2)
    assert multi_modal_inputs["image_grid_thw"].tolist() == [[1, 2, 3]]
    assert multi_modal_inputs["images_seqlens"].tolist() == [6]
    assert "mm_token_type_ids" in multi_modal_inputs
    assert processor.last_processor_call["extra_kwargs"] == {"max_pixels": 1024}


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("processor", [None, object()])
@pytest.mark.parametrize("has_media", [False, True])
def test_compute_position_ids_uses_text_path_without_mrope(processor, has_media):
    """The model capability, not the sample's media, selects ordinary RoPE."""
    from uni_agent.framework.multi_modal_postprocess import compute_position_ids

    input_ids = torch.tensor([[7, 8, 9, 10]], dtype=torch.long)
    attention_mask = torch.tensor([[0, 1, 1, 1]], dtype=torch.long)

    media = {"pixel_values": torch.zeros(1, 3, 2, 2)} if has_media else {}
    position_ids = compute_position_ids(processor, input_ids, attention_mask, media)

    assert tuple(position_ids.shape) == (1, 4)
    assert position_ids.tolist() == [[0, 0, 1, 2]]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("image_first", [False, True])
@pytest.mark.parametrize("legacy_signature", [False, True])
def test_mixed_text_and_image_positions_round_trip_through_tq(image_first, legacy_signature):
    from uni_agent.framework.framework import _list_of_tq_fields_to_tensordict
    from uni_agent.framework.multi_modal_postprocess import compute_multi_modal_inputs, compute_position_ids

    class RequiredModalityProcessor(FakeProcessor):
        def get_rope_index(
            self, *, input_ids, attention_mask, mm_token_type_ids, image_grid_thw=None, video_grid_thw=None
        ):
            return super().get_rope_index(
                input_ids=input_ids,
                attention_mask=attention_mask,
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
            )

    class LegacyProcessor(FakeProcessor):
        def __call__(self, **kwargs):
            output = super().__call__(**kwargs)
            output.pop("mm_token_type_ids", None)
            return output

        # Deliberately omit **kwargs so unexpected modality IDs raise TypeError.
        def get_rope_index(self, *, input_ids, attention_mask, image_grid_thw=None, video_grid_thw=None):
            return super().get_rope_index(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
            )

    processor = LegacyProcessor() if legacy_signature else RequiredModalityProcessor()
    fields = []
    samples = [([7, 8], None), ([11, processor.image_token_id, 12], {"images": ["image://a.png"]})]
    if image_first:
        samples.reverse()
    for tokens, media in samples:
        input_ids = torch.tensor([tokens], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        mm_inputs = compute_multi_modal_inputs(processor, input_ids, media)
        fields.append(
            {
                "input_ids": input_ids.squeeze(0),
                "position_ids": compute_position_ids(processor, input_ids, attention_mask, mm_inputs).squeeze(0),
                "multi_modal_inputs": mm_inputs,
            }
        )
        assert tuple(fields[-1]["position_ids"].shape) == (4, len(tokens))
        if media is None:
            modality_ids = processor.last_get_rope_index_call["mm_token_type_ids"]
            if legacy_signature:
                assert modality_ids is None
            else:
                torch.testing.assert_close(modality_ids, torch.zeros_like(input_ids))

    td = _list_of_tq_fields_to_tensordict(fields)

    assert td["position_ids"].is_nested
    for actual, expected in zip(td["position_ids"].unbind(), fields, strict=True):
        torch.testing.assert_close(actual, expected["position_ids"])


@pytest.mark.cpu
@pytest.mark.level0
def test_compute_position_ids_returns_multimodal_shape_with_processor():
    """With multimodal processor outputs, position ids combine the text row
    with the processor-provided vision RoPE rows."""
    from uni_agent.framework.multi_modal_postprocess import compute_position_ids

    processor = FakeProcessor()
    input_ids = torch.tensor(
        [[11, processor.image_token_id, processor.video_token_id, 12]],
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)
    multi_modal_inputs = {
        "image_grid_thw": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "video_grid_thw": torch.tensor([[1, 3, 4]], dtype=torch.long),
        "mm_token_type_ids": torch.ones_like(input_ids),
    }

    position_ids = compute_position_ids(processor, input_ids, attention_mask, multi_modal_inputs)

    assert tuple(position_ids.shape) == (1, 4, 4)
    assert position_ids[0, 0].tolist() == [0, 1, 2, 3]
    assert processor.last_get_rope_index_call["mm_token_type_ids"].tolist() == [[0, 1, 2, 0]]
